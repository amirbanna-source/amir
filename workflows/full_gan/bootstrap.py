"""Verify the GitHub manifest and the three image-free AK-inclusive initializations."""
import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

from study_split import CLASS_COUNTS, MANIFEST_NAME, MANIFEST_SHA256, read_split

ROOT = Path(__file__).resolve().parent
REFERENCE_URL = ('https://raw.githubusercontent.com/amirbanna-source/amir/'
                 '5086c451836a739d0758ee906af7051cd0e54d24/artifacts/')


def reference_hashes():
    return json.loads((ROOT / 'reference_hashes.json').read_text())


def fetch_reference(reference):
    """Fill the standalone ZIP's cache from the pinned, image-free GitHub artifacts."""
    reference.mkdir(parents=True, exist_ok=True)
    for name, expected in {MANIFEST_NAME: MANIFEST_SHA256, **reference_hashes()}.items():
        destination = reference / name
        if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == expected:
            continue
        partial = destination.with_suffix('.partial')
        try:
            with urllib.request.urlopen(REFERENCE_URL + name, timeout=120) as response, partial.open('wb') as out:
                shutil.copyfileobj(response, out)
            if hashlib.sha256(partial.read_bytes()).hexdigest() != expected:
                raise ValueError(f'GitHub reference file hash differs: {name}')
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)


def setup_reference(reference):
    frame = read_split(reference / MANIFEST_NAME)
    for name, digest in reference_hashes().items():
        if hashlib.sha256((reference / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Classifier initialization file hash differs: {name}')
    print(f'Verified {len(frame)} images in the GitHub manifest and all three initialization archives.')
    for split, counts in CLASS_COUNTS.items():
        print(f'{split}: {sum(counts.values())} ({counts[0]} benign, {counts[1]} malignant)')
    print('No image files were read.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('--fetch', action='store_true')
    args = parser.parse_args()
    if args.fetch:
        fetch_reference(args.reference)
    else:
        setup_reference(args.reference)
