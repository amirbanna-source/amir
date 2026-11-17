"""Full conditional StyleGAN2-ADA and paired CNN comparisons. No pilot run."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REAL_ROOT = Path(os.environ.get('ISIC_IMAGE_DIR', str(ROOT / 'data/real'))).expanduser().resolve()
RESULTS = ROOT / 'results'
SPLIT_SHA = '4f3037e5dc1726e5a534e3a6f155df3e0e0be52229b057659e097483fd0800cd'
KIMG = 2000


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def logged(command, path, cwd=None, env=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            rc = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    need(rc == 0, f'Command failed; see {path.name}. Existing outputs are preserved.')


def read_split():
    import pandas as pd
    frame = pd.read_csv(ROOT / 'data/frozen_binary_split.csv')
    fingerprint = hashlib.sha256(frame[['image_id', 'label', 'group', 'split']].sort_values('image_id').to_csv(index=False).encode()).hexdigest()
    need(fingerprint == SPLIT_SHA, 'Frozen split metadata was modified')
    need(len(frame) == 1632 and frame.image_id.is_unique, 'Wrong cohort')
    need(frame.groupby('split').size().to_dict() == {'train': 972, 'val': 329, 'test': 331}, 'Wrong split counts')
    mapping = dict.fromkeys(['NV', 'BKL', 'DF', 'VASC'], 0) | dict.fromkeys(['MEL', 'BCC', 'SCC'], 1)
    need(frame.original_label.map(mapping).equals(frame.target), 'Targets or included diagnoses differ')
    need(frame.groupby('group').split.nunique().max() == 1, 'Recorded groups overlap splits')
    return frame


def original_images(frame):
    """Resolve originals by ID, comparing train bytes with the saved GAN export audit."""
    from PIL import Image
    wanted = set(frame.loc[frame.split.isin(['train', 'val']), 'image_id'])
    found = {}
    for path in sorted(REAL_ROOT.rglob('*')):
        if path.suffix.lower() in {'.png', '.jpg', '.jpeg'} and path.stem.upper() in wanted:
            found.setdefault(path.stem.upper(), []).append(path)
    missing = sorted(wanted - set(found))
    if missing:
        report = RESULTS / 'missing_original_images.csv'
        report.parent.mkdir(parents=True, exist_ok=True)
        frame[frame.image_id.isin(missing)][['image_id', 'split', 'target']].to_csv(report, index=False)
        raise RuntimeError(f'{len(missing)} original training/validation images are missing. Set ISIC_IMAGE_DIR to the original image folder, or use data/real/. See results/missing_original_images.csv. Test-image contents are not needed.')
    expected = json.loads((ROOT / 'data/training_source_fingerprint.json').read_text())
    train_ids = set(frame.loc[frame.split == 'train', 'image_id'])
    source_hashes = {}
    for image_id, copies in found.items():
        hashes = {sha(path) for path in copies}
        need(len(hashes) == 1, f'Conflicting source copies for {image_id}')
        if image_id in train_ids:
            source_hashes[image_id] = next(iter(hashes))
        with Image.open(copies[0]) as image:
            need(image.mode == 'RGB', f'Expected original RGB image: {image_id}')
            image.load()
    canonical = ''.join(f'{image_id},{source_hashes[image_id]}\n' for image_id in sorted(source_hashes))
    need(len(source_hashes) == 972 and hashlib.sha256(canonical.encode()).hexdigest() == expected['source_fingerprint_sha256'], 'Original training bytes differ; do not substitute resized GAN images')
    return found


def preflight():
    frame = read_split()
    original_images(frame)
    import tensorflow as tf
    import classifier
    need(tf.__version__ == '2.20.0', 'Classifier runtime must be TensorFlow 2.20.0')
    hashes = {}
    for seed in [42, 43, 44]:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        model = classifier.build_reference_cnn()
        hashes[str(seed)] = classifier.load_reference_weights(model, seed)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'input_preflight.json').write_text(json.dumps(dict(
        status='passed', split_fingerprint=SPLIT_SHA, training_images=972,
        validation_images=329, test_image_content_read=False, initialization_hashes=hashes,
        tensorflow=tf.__version__, checked_at=datetime.now(timezone.utc).isoformat()), indent=2))
    from prepare_gan_input import export_dataset
    dataset = ROOT / 'data/isic_train128.zip'
    if not dataset.exists():
        export_dataset(ROOT / 'data/frozen_binary_split.csv', REAL_ROOT, dataset)
    validate_export()
    print('Original images, fixed split, classifier weights and rebuilt GAN input verified. No model fitting in this check.')


def validate_export():
    import pandas as pd
    dataset = ROOT / 'data/isic_train128.zip'
    audit = json.loads((ROOT / 'data/isic_train128.audit.json').read_text())
    expected = json.loads((ROOT / 'data/training_source_fingerprint.json').read_text())
    canonical = ''.join(f"{r['image_id']},{r['source_sha256']}\n" for r in sorted(audit['training_sources'], key=lambda r: r['image_id']))
    need(hashlib.sha256(canonical.encode()).hexdigest() == expected['source_fingerprint_sha256'], 'GAN export uses different originals')
    digest = sha(dataset)
    need(audit['dataset_sha256'] == digest and audit['split_sha256'] == SPLIT_SHA, 'Export hash or split differs')
    train = read_split().query("split == 'train'")
    with zipfile.ZipFile(dataset) as z:
        need(z.testzip() is None and len(z.namelist()) == 973, 'Incomplete GAN input')
        labels = json.loads(z.read('dataset.json'))['labels']
        observed = {Path(name).stem: target for name, target in labels}
        need(observed == dict(zip(train.image_id, train.target)) and len(labels) == 972, 'GAN input labels differ')
    return digest


def gan():
    import torch
    need(torch.__version__.split('+')[0] == '2.11.0', 'Use supplied PyTorch 2.11.0 environment')
    need(torch.cuda.is_available() and (torch.version.cuda or '').startswith('13.'), 'GPU-enabled CUDA 13 PyTorch runtime is required')
    dataset = ROOT / 'data/isic_train128.zip'
    dataset_sha = validate_export()
    with zipfile.ZipFile(dataset) as z:
        need(z.testzip() is None, 'GAN dataset ZIP is corrupt')
    read_split()
    out = RESULTS / 'stylegan2ada_full_seed142'
    out.mkdir(parents=True, exist_ok=True)
    statepath = out / 'state.json'
    if statepath.exists():
        state = json.loads(statepath.read_text())
        if state.get('status') == 'complete':
            checkpoint = Path(state['checkpoint'])
            # The bundle may have moved between runs or container mounts.
            checkpoint = out / state['checkpoint_relative']
            need(state.get('kimg') == KIMG and state.get('dataset_sha256') == dataset_sha and sha(checkpoint) == state['checkpoint_sha256'], 'Completed GAN provenance/checkpoint differs')
            print('Reusing completed full GAN; no repeat training.')
            return
        raise RuntimeError('A full GAN attempt already started or failed. Preserve it. An interrupted run is not silently restarted or treated as complete.')
    runtime = out / 'runtime'
    if not runtime.exists():
        logged([sys.executable, '-u', str(ROOT / 'verify_runtime.py'), '--output', str(runtime)], out / 'runtime_checks.log')
    report = json.loads((runtime / 'stylegan_gpu_runtime_report.json').read_text())
    need(report['status'] == 'gpu_and_targeted_runtime_checks_passed', 'GPU compatibility checks failed')
    source = runtime / 'source_checkout/stylegan2-ada-pytorch'
    env = os.environ.copy()
    env.update(MAX_JOBS='2', TORCH_CUDA_ARCH_LIST='.'.join(map(str, torch.cuda.get_device_capability(0))),
               TORCH_EXTENSIONS_DIR=str(runtime / 'cuda_extensions'))
    command = [sys.executable, '-u', str(source / 'train.py'), f'--outdir={out / "runs"}',
               f'--data={dataset}', '--gpus=1', '--cfg=auto', '--cond=1', '--aug=ada',
               '--augpipe=bg', '--mirror=0', '--batch=16', f'--kimg={KIMG}', '--tick=4',
               '--snap=10', '--metrics=none', '--seed=142', '--workers=2']
    need(not (out / 'runs').exists(), 'Training output exists; refusing overwrite')
    logged(command + ['--dry-run'], out / 'configuration_check.log', cwd=source, env=env)
    state = dict(status='running', kimg=KIMG, dataset_sha256=dataset_sha,
                 split_fingerprint=SPLIT_SHA, command=command, started_at=datetime.now(timezone.utc).isoformat())
    statepath.write_text(json.dumps(state, indent=2))
    try:
        logged(command, out / 'training.log', cwd=source, env=env)
        checkpoints = list((out / 'runs').rglob(f'network-snapshot-{KIMG:06d}.pkl'))
        need(len(checkpoints) == 1, 'Expected final full-run checkpoint missing')
        state.update(status='complete', checkpoint=str(checkpoints[0]), checkpoint_relative=str(checkpoints[0].relative_to(out)),
                     checkpoint_sha256=sha(checkpoints[0]), completed_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        statepath.write_text(json.dumps(state, indent=2))


def generate_pool():
    import pandas as pd
    import classifier
    out = RESULTS / 'stylegan2ada_full_seed142'
    state = json.loads((out / 'state.json').read_text())
    need(state.get('status') == 'complete' and state.get('kimg') == KIMG, 'Full GAN must finish before generation')
    checkpoint = out / state['checkpoint_relative']
    need(sha(checkpoint) == state['checkpoint_sha256'], 'Full GAN checkpoint hash differs')
    folder = RESULTS / 'synthetic_pool'
    manifest, provenance = folder / 'pool_manifest.csv', folder / 'pool_provenance.json'
    if manifest.exists() and provenance.exists():
        old = json.loads(provenance.read_text())
        need(old['generator_checkpoint_sha256'] == sha(checkpoint) and old['pool_manifest_sha256'] == sha(manifest), 'Existing pool differs')
        print('Reusing recorded synthetic pool.')
        return
    need(not folder.exists(), 'Incomplete pool exists; preserve it rather than overwrite')
    folder.mkdir(parents=True)
    source = out / 'runtime/source_checkout/stylegan2-ada-pytorch'
    rows = []
    try:
        for target, count in [(0, 530), (1, 442)]:
            images = folder / f'class{target}'
            logged([sys.executable, str(source / 'generate.py'), f'--network={checkpoint}',
                    f'--seeds=0-{count-1}', f'--class={target}', '--trunc=1', f'--outdir={images}'],
                   folder / f'generation_class{target}.log', cwd=source)
            for seed in range(count):
                path = images / f'seed{seed:04d}.png'
                rows.append(dict(sample_id=f'SYN_CLASS{target}_SEED{seed:04d}', path=str(path.relative_to(folder)),
                                 target=target, seed=seed, pixel_sha256=classifier.pixel_hash(path)))
            from PIL import Image
            grid = Image.new('RGB', (512, 512))
            for seed in range(16):
                with Image.open(images / f'seed{seed:04d}.png') as image:
                    grid.paste(image, ((seed % 4) * 128, (seed // 4) * 128))
            grid.save(folder / f'grid_class{target}.png')
        pd.DataFrame(rows).to_csv(manifest, index=False)
        record = dict(generator_type='official_stylegan2ada', generator_checkpoint_sha256=sha(checkpoint),
                      split_fingerprint=SPLIT_SHA, validation_test_images_used=False,
                      pool_manifest_sha256=sha(manifest), generated_at=datetime.now(timezone.utc).isoformat(),
                      requested_labels_not_clinical_diagnoses=True, quality_review='not assessed',
                      near_copy_review='not assessed', automatic_record_not_human_approval=True)
        provenance.write_text(json.dumps(record, indent=2))
    except BaseException:
        print('Generation interrupted; partial pool retained.', file=sys.stderr)
        raise


def cnn(conventional):
    import classifier
    folder = RESULTS / 'synthetic_pool'
    train, val, pool, provenance = classifier.prepare_inputs(ROOT / 'data/frozen_binary_split.csv',
        REAL_ROOT, folder / 'pool_manifest.csv', folder / 'pool_provenance.json')
    out = RESULTS / ('cnn_gan_conventional' if conventional else 'cnn_gan')
    if (out / 'config.json').exists():
        old = json.loads((out / 'config.json').read_text())
        need(old.get('status') == 'complete' and old.get('pool_manifest_sha256') == provenance['pool_manifest_sha256']
             and old.get('split_fingerprint') == SPLIT_SHA, 'Existing classifier attempt incomplete or incompatible; refusing overwrite')
        need((out / 'results.csv').is_file(), 'Completed classifier result table missing')
        print('This classifier condition is already complete; leaving it unchanged.')
        return
    classifier.run_experiment(train, val, pool, provenance, out, conventional)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['preflight', 'gan', 'pool', 'cnn', 'cnn-conventional'])
    args = parser.parse_args()
    {'preflight': preflight, 'gan': gan, 'pool': generate_pool,
     'cnn': lambda: cnn(False), 'cnn-conventional': lambda: cnn(True)}[args.stage]()
