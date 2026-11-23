"""Fetch existing image-free reference assets; never downloads source images."""
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REFERENCE_URL = 'https://raw.githubusercontent.com/amirbanna-source/amir/18447b4faf5ffda3f5653bf05ac976a7bfe3c143/artifacts/binary_lr0001_all_artifacts.zip'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup_reference():
    import pandas as pd
    expected = json.loads((ROOT / 'reference_hashes.json').read_text())
    data = ROOT / 'data'
    data.mkdir(exist_ok=True)
    names = dict(expected['files'])
    names['isic2019_binary_split.csv'] = None
    destinations = {name: data / ('frozen_binary_split.csv' if name.endswith('.csv') else name) for name in names}
    ready = all(path.is_file() for path in destinations.values())
    if ready:
        ready = all(digest(destinations[name]) == sha for name, sha in expected['files'].items())
    if not ready:
        cache = ROOT / '.cache'
        cache.mkdir(exist_ok=True)
        archive = cache / 'classifier_reference.zip'
        if not archive.exists() or digest(archive) != expected['archive_sha256']:
            partial = archive.with_suffix('.partial')
            print('Downloading the existing image-free classifier reference archive (about 19 MB).')
            with urllib.request.urlopen(REFERENCE_URL, timeout=120) as response, partial.open('wb') as out:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    out.write(block)
            if digest(partial) != expected['archive_sha256']:
                raise RuntimeError('Reference archive SHA256 differs; no assets installed')
            partial.replace(archive)
        with zipfile.ZipFile(archive) as z:
            for name, path in destinations.items():
                if z.namelist().count(name) != 1:
                    raise RuntimeError('Ambiguous reference archive member')
                payload = z.read(name)
                if names[name] and hashlib.sha256(payload).hexdigest() != names[name]:
                    raise RuntimeError('Initial-weight archive hash differs')
                path.write_bytes(payload)
    frame = pd.read_csv(data / 'frozen_binary_split.csv')
    fingerprint = hashlib.sha256(frame[['image_id', 'label', 'group', 'split']].sort_values('image_id').to_csv(index=False).encode()).hexdigest()
    if fingerprint != '4f3037e5dc1726e5a534e3a6f155df3e0e0be52229b057659e097483fd0800cd':
        raise RuntimeError('Reference split fingerprint differs')
    print('Fixed split and all three archived classifier initializations are available. No images downloaded.')


if __name__ == '__main__':
    setup_reference()
