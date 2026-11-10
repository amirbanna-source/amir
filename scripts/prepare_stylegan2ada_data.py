"""Export the frozen training subset to the official StyleGAN dataset format.

Data adapter only: no GAN architecture, loss, optimizer or training loop.
"""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image

FINGERPRINT = '4f3037e5dc1726e5a534e3a6f155df3e0e0be52229b057659e097483fd0800cd'
MAPPING = {'NV': 0, 'BKL': 0, 'DF': 0, 'VASC': 0, 'MEL': 1, 'BCC': 1, 'SCC': 1}


def verified_train(frame):
    required = {'image_id', 'label', 'group', 'split', 'original_label', 'target'}
    assert required <= set(frame.columns), 'Manifest columns missing'
    digest = hashlib.sha256(frame[['image_id', 'label', 'group', 'split']]
                           .sort_values('image_id').to_csv(index=False).encode()).hexdigest()
    assert digest == FINGERPRINT, 'Frozen split differs'
    assert len(frame) == 1632 and frame.image_id.is_unique
    assert frame.group.notna().all() and frame.groupby('group').split.nunique().max() == 1
    assert frame.original_label.map(MAPPING).equals(frame.target), 'Target mapping differs'
    assert frame.target.map({0: 'benign', 1: 'malignant'}).equals(frame.label)
    assert frame.split.value_counts().to_dict() == {'train': 972, 'test': 331, 'val': 329}
    train = frame[frame.split == 'train'].sort_values('image_id').reset_index(drop=True)
    assert train.target.value_counts().to_dict() == {0: 530, 1: 442}
    return train


def export_dataset(manifest, input_root, output):
    train = verified_train(pd.read_csv(manifest))
    assert not output.exists(), 'Refusing to overwrite existing dataset'
    wanted = set(train.image_id)
    copies = {}
    for p in sorted(input_root.rglob('*')):
        if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'} and p.stem.upper() in wanted:
            copies.setdefault(p.stem.upper(), []).append(p)
    assert set(copies) == wanted, 'Training image files missing'
    selected = {}
    for image_id, paths in copies.items():
        if len(paths) > 1:
            assert len({hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}) == 1, image_id
        selected[image_id] = paths[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + '.partial')
    assert not partial.exists(), 'Incomplete previous export exists; choose a new output path'
    labels, evidence = [], []
    with zipfile.ZipFile(partial, 'x', compression=zipfile.ZIP_STORED) as archive:
        for row in train.itertuples(index=False):
            path = selected[row.image_id]
            with Image.open(path) as im:
                im.load()
                # Explicit recipe: whole-image resize, no crop or augmentation.
                resized = im.convert('RGB').resize((128, 128), Image.Resampling.BILINEAR)
            name = f'images/{row.image_id}.png'
            data = io.BytesIO()
            resized.save(data, format='PNG')
            archive.writestr(name, data.getvalue())
            labels.append([name, int(row.target)])
            evidence.append({'image_id': row.image_id, 'target': int(row.target),
                             'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        archive.writestr('dataset.json', json.dumps({'labels': labels}))
    with zipfile.ZipFile(partial) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == 973
        assert json.loads(archive.read('dataset.json'))['labels'] == labels
    partial.rename(output)
    report = {'dataset': str(output), 'image_count': 972, 'class_counts': {'0': 530, '1': 442},
              'split_sha256': FINGERPRINT,
              'dataset_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
              'preprocessing': 'Pillow RGB; whole-image bilinear resize to 128x128; uint8 PNG',
              'note': 'Official GAN loader performs its own numeric scaling; CNN preprocessing unchanged',
              'validation_test_image_content_read': False, 'training_sources': evidence}
    output.with_suffix('.audit.json').write_text(json.dumps(report, indent=2))
    print(f'Exported {len(labels)} training images (530 benign / 442 malignant): {output}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--input-root', type=Path, default=Path('/kaggle/input'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    export_dataset(args.manifest, args.input_root, args.output)


if __name__ == '__main__':
    main()
