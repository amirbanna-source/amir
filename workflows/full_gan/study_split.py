"""Verified AK-inclusive ISIC cohort and shared presentation counts."""
import hashlib
from pathlib import Path

MANIFEST_NAME = 'isic2019_binary_split.csv'
MANIFEST_SHA256 = 'f6259aeb0a604f6927cdf5d2307ab195367b975f9c1dc9ecea4cb8ac36a86f3b'
SPLIT_SHA = '4c99045e3d79110b30123e26eccf05462793cf080ee9090d692a3774fc2d3333'
CLASS_COUNTS = {'train': {0: 538, 1: 518}, 'val': {0: 179, 1: 174}, 'test': {0: 177, 1: 175}}
TRAIN_COUNTS = CLASS_COUNTS['train']
TRAIN_SIZE = sum(TRAIN_COUNTS.values())
SOURCE_COUNTS = {target: count // 2 for target, count in TRAIN_COUNTS.items()}
MAPPING = dict.fromkeys(['NV', 'BKL', 'DF', 'VASC'], 0) | dict.fromkeys(['MEL', 'BCC', 'SCC', 'AK'], 1)


def read_split(path):
    import pandas as pd

    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != MANIFEST_SHA256:
        raise ValueError('Expected the verified AK-inclusive GitHub manifest at artifacts/isic2019_binary_split.csv')
    frame = pd.read_csv(path)
    if frame.image_id.isna().any() or not frame.image_id.is_unique or frame.group.isna().any():
        raise ValueError('Missing image/group IDs or duplicate images')
    if frame.groupby('group').split.nunique().max() != 1 or frame.groupby('lesion_id').split.nunique().max() != 1:
        raise ValueError('A recorded group or lesion spans multiple splits')
    if not frame.original_label.map(MAPPING).equals(frame.target) or not frame.target.map({0: 'benign', 1: 'malignant'}).equals(frame.label):
        raise ValueError('Diagnosis, binary label or target mismatch')
    counts = {name: part.target.value_counts().to_dict() for name, part in frame.groupby('split')}
    if counts != CLASS_COUNTS:
        raise ValueError('Split/class counts differ from the verified cohort')
    columns = ['image_id', 'label', 'group', 'split']
    fingerprint = hashlib.sha256(frame[columns].sort_values('image_id').to_csv(index=False).encode()).hexdigest()
    if fingerprint != SPLIT_SHA:
        raise ValueError('Split fingerprint differs')
    return frame
