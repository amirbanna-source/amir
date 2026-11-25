"""Export the frozen training subset to the official StyleGAN dataset format.

Data adapter only: no GAN architecture, loss, optimizer or training loop.
"""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

from PIL import Image

from study_split import SPLIT_SHA, TRAIN_COUNTS, TRAIN_SIZE, read_split


def export_dataset(manifest, input_root, output):
    train = read_split(manifest).query("split == 'train'").sort_values('image_id')
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
        assert len(archive.namelist()) == TRAIN_SIZE + 1
        assert json.loads(archive.read('dataset.json'))['labels'] == labels
    partial.rename(output)
    report = {'dataset': str(output), 'image_count': TRAIN_SIZE, 'class_counts': TRAIN_COUNTS,
              'split_sha256': SPLIT_SHA,
              'dataset_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
              'preprocessing': 'Pillow RGB; whole-image bilinear resize to 128x128; uint8 PNG',
              'note': 'Official GAN loader performs its own numeric scaling; CNN preprocessing unchanged',
              'validation_test_image_content_read': False, 'training_sources': evidence}
    output.with_suffix('.audit.json').write_text(json.dumps(report, indent=2))
    print(f'Exported {len(labels)} training images ({TRAIN_COUNTS[0]} benign / {TRAIN_COUNTS[1]} malignant): {output}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    export_dataset(args.manifest, args.input_root, args.output)


if __name__ == '__main__':
    main()
