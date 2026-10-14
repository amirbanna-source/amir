# Exploratory optimizer revision: only Adam LR changes from 0.001 to 0.0001.
# Reconstruct audited source split; abort if membership differs from archive.
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

root = Path("/kaggle/input")
gt_file = next(root.rglob("ISIC_2019_Training_GroundTruth.csv"))
meta_file = next(root.rglob("ISIC_2019_Training_Metadata.csv"))

gt = pd.read_csv(gt_file)
meta = pd.read_csv(meta_file)
classes = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

gt["image"] = gt["image"].str.upper()
meta["image"] = meta["image"].str.upper()
gt = gt.set_index("image")
meta = meta.drop_duplicates("image").set_index("image")

# Retain one copy per official image ID.
paths = {}
for p in sorted(root.rglob("*")):
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        image_id = p.stem.upper()
        if image_id in gt.index and image_id not in paths:
            paths[image_id] = str(p)

rows = []
for image_id, path in sorted(paths.items()):
    positives = [c for c in classes if gt.at[image_id, c] == 1]
    if len(positives) != 1:
        continue
    lesion = meta.at[image_id, "lesion_id"] if image_id in meta.index else None
    group = (
        f"lesion:{lesion}"
        if pd.notna(lesion) and str(lesion).strip()
        else f"image:{image_id}"
    )
    rows.append((image_id, path, positives[0], lesion, group))

df = pd.DataFrame(rows, columns=["image_id", "path", "label", "lesion_id", "group"])
assert len(df) == 1761, f"Expected 1761 images, found {len(df)}"

outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
trainval_idx, test_idx = next(outer.split(df, df["label"], df["group"]))
trainval, test = df.iloc[trainval_idx].copy(), df.iloc[test_idx].copy()

inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
train_idx, val_idx = next(inner.split(
    trainval, trainval["label"], trainval["group"]
))
train = trainval.iloc[train_idx].copy()
val = trainval.iloc[val_idx].copy()

train["split"], val["split"], test["split"] = "train", "val", "test"
splits = pd.concat([train, val, test], ignore_index=True)

assert not (set(train.group) & set(val.group))
assert not (set(train.group) & set(test.group))
assert not (set(val.group) & set(test.group))

output = Path("/kaggle/working/isic2019_verified_split.csv")
splits.to_csv(output, index=False)

print(pd.crosstab(splits["label"], splits["split"], margins=True))
print("Saved:", output)

from hashlib import sha256
assert sha256(splits[['image_id', 'label', 'group', 'split']].sort_values("image_id").to_csv(index=False).encode()).hexdigest() == '900aed297caf894a0a5b0001b3ca974ea954d96104d70781b2eb7a71306227c1', "Split differs from archived baseline"

# Binary task: preserve membership, exclude AK, keep original diagnoses.
import json, gc, zipfile
import numpy as np
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix,
    accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score,
    average_precision_score)

MAPPING = {**dict.fromkeys(['NV','BKL','DF','VASC'], 'benign'),
           **dict.fromkeys(['MEL','BCC','SCC'], 'malignant')}
out = Path('/kaggle/working/binary_lr0001')
out.mkdir(exist_ok=True)
excluded = splits[splits.label == 'AK'].copy()
splits = splits[splits.label.isin(MAPPING)].copy()
splits['original_label'] = splits.label
splits['label'] = splits.original_label.map(MAPPING)
splits['target'] = splits.label.map({'benign': 0, 'malignant': 1})
fingerprint = sha256(splits[['image_id','label','group','split']]
    .sort_values('image_id').to_csv(index=False).encode()).hexdigest()
assert fingerprint == '4f3037e5dc1726e5a534e3a6f155df3e0e0be52229b057659e097483fd0800cd'
assert len(splits) == 1632 and splits.image_id.is_unique
assert splits.groupby('group').split.nunique().max() == 1
splits.to_csv(out/'isic2019_binary_split.csv', index=False)
excluded.to_csv(out/'excluded_ak.csv', index=False)
train = splits[splits.split == 'train'].copy()
val = splits[splits.split == 'val'].copy()
assert (len(train), len(val)) == (972, 329)
class_weights = {int(k): len(train)/(2*int(v))
                 for k,v in train.target.value_counts().items()}
print(pd.crosstab(splits.label, splits.split, margins=True))
print('Class weights:', class_weights)
assert tf.config.list_physical_devices('GPU'), 'Enable GPU before training.'

def load_image(path, target):
    x = tf.io.decode_image(tf.io.read_file(path), channels=3,
                           expand_animations=False)
    x.set_shape([None, None, 3])
    return tf.cast(tf.image.resize(x, [128,128]), tf.float32)/255., target

def dataset(part, seed, shuffle):
    ds = tf.data.Dataset.from_tensor_slices((part.path.astype(str).to_numpy(),
                                            part.target.to_numpy('int32')))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(len(part), seed=seed, reshuffle_each_iteration=True)
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    return ds.with_options(opts).batch(32).prefetch(tf.data.AUTOTUNE)

def build_cnn():
    return tf.keras.Sequential([
        tf.keras.Input((128,128,3)),
        tf.keras.layers.Conv2D(32,3,padding='same',activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64,3,padding='same',activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128,3,padding='same',activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(256,3,padding='same',activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dropout(.35,seed=12345),
        tf.keras.layers.Dense(1,activation='sigmoid')])

results = []
for seed in [42,43,44]:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = build_cnn()
    init_hash = sha256(b''.join(w.tobytes() for w in model.get_weights())).hexdigest()
    expected_hashes = {42:'f0b9a0ddce8f8452c7308e6e7ae6c7d37f9d3b7eb850da4831cbb7f2651b3953',43:'f3a606597f9c02b4fbe03928aba8aadb9df21747c08c2b4142b6d4868042002f',44:'c015c7fd952ac5cb0c757a68a82d229d09a0c04101bdf0f3c5b1ca31e759e2b1'}
    assert init_hash == expected_hashes[seed], 'Initialization differs from original baseline'
    # Preserve initial arrays for future paired binary augmentation comparisons.
    np.savez_compressed(out/f'initial_weights_seed{seed}.npz', *model.get_weights())
    model.compile(optimizer=tf.keras.optimizers.Adam(.0001),
                  loss='binary_crossentropy', metrics=['accuracy'])
    run = f'binary_lr0001_no_aug_seed{seed}'
    checkpoint = out/f'{run}.keras'
    train_ds, val_ds = dataset(train, seed, True), dataset(val, seed, False)
    history = model.fit(train_ds, validation_data=val_ds, epochs=25,
        class_weight=class_weights, callbacks=[
            tf.keras.callbacks.ModelCheckpoint(str(checkpoint),
                monitor='val_loss',save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,
                                             restore_best_weights=True)], verbose=2)
    best = tf.keras.models.load_model(checkpoint)
    probability = best.predict(val_ds,verbose=0).reshape(-1)
    y = val.target.to_numpy()
    pred = (probability >= .5).astype(int)
    tn,fp,fn,tp = confusion_matrix(y,pred,labels=[0,1]).ravel()
    row = dict(seed=seed,condition='binary_lr0001_no_aug',
        epochs_trained=len(history.history['loss']),
        selected_epoch=int(np.argmin(history.history['val_loss'])+1),
        validation_loss=float(min(history.history['val_loss'])),
        accuracy=float(accuracy_score(y,pred)),
        macro_f1=float(f1_score(y,pred,labels=[0,1],average='macro',zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y,pred)),
        malignant_sensitivity=float(tp/(tp+fn)), specificity=float(tn/(tn+fp)),
        roc_auc=float(roc_auc_score(y,probability)),
        average_precision=float(average_precision_score(y,probability)),
        threshold=.5,initial_weights_sha256=init_hash,
        split_fingerprint=fingerprint,test_evaluated=False)
    report=classification_report(y,pred,labels=[0,1],
        target_names=['benign','malignant'],output_dict=True,zero_division=0)
    (out/f'{run}_metrics.json').write_text(json.dumps(
        dict(**row,classification_report=report),indent=2))
    pd.DataFrame(history.history).assign(
        epoch=lambda d: np.arange(1,len(d)+1)).to_csv(out/f'{run}_history.csv',index=False)
    predictions=val[['image_id','original_label','label','group']].copy()
    predictions['malignant_probability']=probability
    predictions['prediction']=pred
    predictions.to_csv(out/f'{run}_validation_predictions.csv',index=False)
    pd.DataFrame(confusion_matrix(y,pred,labels=[0,1]),
        index=['benign','malignant'],columns=['benign','malignant']).to_csv(
            out/f'{run}_confusion_matrix.csv')
    results.append(row)
    pd.DataFrame(results).to_csv(out/'binary_lr0001_results.csv',index=False)
    # Checkpoint each seed's complete evidence before moving to the next run.
    with zipfile.ZipFile(f'/kaggle/working/{run}_artifacts.zip','w',
                         zipfile.ZIP_DEFLATED) as archive:
        for p in out.glob(f'{run}*'): archive.write(p,p.name)
        for name in [f'initial_weights_seed{seed}.npz','isic2019_binary_split.csv']:
            archive.write(out/name,name)
    print('BINARY_RUN_COMPLETE',json.dumps(row),flush=True)
    del model,best,train_ds,val_ds
    gc.collect()

config=dict(mapping=MAPPING,class_weights=class_weights,seeds=[42,43,44],
    tensorflow_version=tf.__version__,split_fingerprint=fingerprint,
    threshold=.5,test_evaluated=False,max_epochs=25,patience=5,
    batch_size=32,steps_per_epoch=31,learning_rate=.0001)
(out/'binary_lr0001_config.json').write_text(json.dumps(config,indent=2))
print('ALL_BINARY_BASELINES_COMPLETE')
print(pd.DataFrame(results).to_string(index=False))
# No test image dataset is constructed or evaluated in this script.

with zipfile.ZipFile("/kaggle/working/binary_lr0001_all_artifacts.zip","w",zipfile.ZIP_DEFLATED) as archive:
    for p in out.iterdir(): archive.write(p,p.name)
print("LOWER_LR_ALL_ARTIFACTS_SAVED",flush=True)
