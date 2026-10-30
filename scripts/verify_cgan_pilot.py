"""Verify a downloaded pilot archive without TensorFlow or raw image access."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    frozen = pd.read_csv(args.manifest)
    with zipfile.ZipFile(args.archive) as archive:
        assert archive.testzip() is None, 'Archive CRC failed'
        names = archive.namelist()
        def member(suffix):
            matches = [n for n in names if n == suffix or n.endswith('/' + suffix)]
            assert len(matches) == 1, (suffix, matches)
            return matches[0]
        def js(name):
            return json.loads(archive.read(member(name)))
        def csv(name):
            with archive.open(member(name)) as file:
                return pd.read_csv(file)
        status, config = js('status.json'), js('config.json')
        assert status['steps'] == config['steps'] == 5000
        assert status['status'] == 'completed; quality review pending'
        assert config['seed'] == 142 and config['tensorflow'] == '2.20.0'
        assert config['split_sha256'] == '4f3037e5dc1726e5a534e3a6f155df3e0e0be52229b057659e097483fd0800cd'
        history = csv('history.csv')
        assert history.step.tolist() == list(range(1, 5001))
        assert np.isfinite(history[['g_loss', 'd_loss', 'elapsed_seconds']]).all().all()
        assert (history.elapsed_seconds.diff().dropna() >= 0).all()
        logged = [json.loads(line) for line in archive.read(member('history.jsonl')).decode().splitlines()]
        assert len(logged) == 5000
        assert [r['step'] for r in logged] == history.step.tolist()
        assert np.allclose([r['g_loss'] for r in logged], history.g_loss)
        assert np.allclose([r['d_loss'] for r in logged], history.d_loss)
        train = csv('training_manifest.csv')
        expected = frozen[frozen.split == 'train'].sort_values('image_id')
        fields = ['image_id', 'label', 'group', 'split', 'original_label', 'target']
        assert train[fields].reset_index(drop=True).equals(expected[fields].reset_index(drop=True))
        assert len(train) == 972
        for step in range(0, 5001, 500):
            for target in [0, 1]:
                with archive.open(member(f'grid_class{target}_{step:05d}.png')) as f:
                    im = Image.open(f)
                    im.load()
                    assert im.size == (512, 512) and im.mode == 'RGB'
        for model in ['generator', 'discriminator']:
            with archive.open(member(f'{model}_step05000.keras')) as file:
                with zipfile.ZipFile(file) as keras:
                    assert keras.testzip() is None
                    assert 'model.weights.h5' in keras.namelist()
                    json.loads(keras.read('config.json'))
        pool = csv('audit_pool_manifest.csv')
        assert len(pool) == 972 and pool.file.is_unique
        assert pool.requested_target.value_counts().to_dict() == {0: 530, 1: 442}
        assert pool.latent_index.tolist() == list(range(972))
        assert (pool.sampling_seed == 1142).all() and (pool.checkpoint_step == 5000).all()
        pixels = []
        for row in pool.itertuples():
            with archive.open(member('audit_pool/' + row.file)) as file:
                im = Image.open(file)
                im.load()
                assert im.mode == 'RGB' and im.size == (128, 128)
                arr = np.asarray(im)
                assert hashlib.sha256(arr.tobytes()).hexdigest() == row.pixel_sha256
                pixels.append(np.asarray(im.resize((16, 16), Image.Resampling.BILINEAR), dtype=np.float32) / 255.)
        audit = js('exact_copy_audit.json')
        assert audit['pool_size'] == 972
        # Count outputs repeated within the pool independently of the archived report.
        repeats = int(pool.pixel_sha256.duplicated().sum())
        reported_repeats = sum(bool(r['earlier_synthetic_matches']) for r in audit['duplicate_outputs'])
        assert repeats == reported_repeats
        summaries = {}
        small = np.stack(pixels)
        for target in [0, 1]:
            a = small[pool.requested_target.to_numpy() == target]
            summaries[str(target)] = {
                'images': len(a),
                'mean_pixel_std_across_images_at_16px': float(a.std(axis=0).mean()),
                'unique_pixel_hashes': int(pool.loc[pool.requested_target == target, 'pixel_sha256'].nunique())}
        print(json.dumps({'verification': 'passed', 'steps': 5000,
                          'elapsed_seconds': float(history.elapsed_seconds.iloc[-1]),
                          'final_g_loss': float(history.g_loss.iloc[-1]),
                          'final_d_loss': float(history.d_loss.iloc[-1]),
                          'exact_repeated_pool_outputs': repeats,
                          'outputs_matching_training_pixels': sum(bool(r['training_matches']) for r in audit['duplicate_outputs']),
                          'descriptive_diversity': summaries,
                          'limitations': 'No local TensorFlow restore, raw-training-image hash recomputation, feature-based near-copy audit or clinical label verification.'}, indent=2))


if __name__ == '__main__':
    main()
