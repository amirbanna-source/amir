# Conventional augmentation experiment: same frozen split and CNN
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

# نختار نسخة واحدة فقط لكل image ID
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
import tensorflow as tf

class_names = sorted(splits["label"].unique())
class_to_num = {name: i for i, name in enumerate(class_names)}
print("Classes:", class_to_num)

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [128, 128])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


import numpy as np, json, gc
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report, confusion_matrix
out_dir = Path('/kaggle/working/paired_seed_study')
out_dir.mkdir(exist_ok=True)
SEEDS = [42, 43, 44]
train_part = splits[splits.split == 'train'].copy()
val_part = splits[splits.split == 'val'].copy()
y_true = val_part.label.map(class_to_num).to_numpy(dtype=int)
weights = compute_class_weight(class_weight='balanced', classes=np.arange(8), y=train_part.label.map(class_to_num))
class_weights = dict(enumerate(map(float, weights)))
assert tf.config.list_physical_devices('GPU'), 'GPU required'

def make_aug(seed):
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal_and_vertical', seed=seed+1000),
        tf.keras.layers.RandomRotation(0.05, fill_mode='reflect', seed=seed+1001),
        tf.keras.layers.RandomZoom(0.10, fill_mode='reflect', seed=seed+1002),
    ])

# Training-only examples, one per class. Preview seeds are separate from training.
examples = train_part.groupby('label', sort=True).head(1)
fig, axes = plt.subplots(8, 2, figsize=(7, 22))
preview_aug = make_aug(999)
for i, (_, row) in enumerate(examples.iterrows()):
    x, _ = load_image(row.path, 0)
    z = tf.clip_by_value(preview_aug(x[None], training=True)[0], 0., 1.)
    axes[i,0].imshow(x.numpy()); axes[i,1].imshow(z.numpy())
    axes[i,0].set_title(f'{row.label}: {row.image_id}')
    axes[i,1].set_title('Augmented training image')
    axes[i,0].axis('off'); axes[i,1].axis('off')
fig.tight_layout()
fig.savefig(out_dir/'augmentation_preview.png', dpi=130)
plt.show()
print('PREVIEW_READY: training images only; split fingerprint matched')

# Paired baseline versus conventional augmentation, seeds fixed before model/pipeline creation

def build_cnn():
    return tf.keras.Sequential([
        tf.keras.Input(shape=(128,128,3)),
        tf.keras.layers.Conv2D(32,3,padding='same',activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64,3,padding='same',activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128,3,padding='same',activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(256,3,padding='same',activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dropout(0.35, seed=12345),
        tf.keras.layers.Dense(8,activation='softmax'),
    ])

def dataset(part, shuffle, seed):
    labels = part.label.map(class_to_num).to_numpy(dtype='int32')
    ds = tf.data.Dataset.from_tensor_slices((part.path.astype(str).to_numpy(), labels))
    # Both conditions use identical preprocessing, explicit shuffle seed and ordering.
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(len(part), seed=seed, reshuffle_each_iteration=True)
    opts = tf.data.Options(); opts.experimental_deterministic = True
    return ds.with_options(opts).batch(32).prefetch(tf.data.AUTOTUNE)

results = []
for seed in SEEDS:
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
    initial_model = build_cnn()
    initial_weights = [w.copy() for w in initial_model.get_weights()]
    init_hash = sha256(b''.join(w.tobytes() for w in initial_weights)).hexdigest()
    del initial_model
    for condition in ['no_aug', 'conventional_aug']:
        tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
        model = build_cnn(); model.set_weights(initial_weights)
        assert all(np.array_equal(a,b) for a,b in zip(initial_weights, model.get_weights()))
        train_ds = dataset(train_part, True, seed)
        val_ds = dataset(val_part, False, seed)
        if condition == 'conventional_aug':
            augmentation = make_aug(seed)
            train_ds = train_ds.map(lambda x,y: (tf.clip_by_value(augmentation(x,training=True),0.,1.),y), num_parallel_calls=1).prefetch(tf.data.AUTOTUNE)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        run_id = f'{condition}_seed{seed}'
        model_path = out_dir/f'{run_id}.keras'
        print('RUN_START', run_id, 'initial_weights_sha256', init_hash, flush=True)
        history = model.fit(train_ds,validation_data=val_ds,epochs=25,class_weight=class_weights,
            callbacks=[tf.keras.callbacks.ModelCheckpoint(str(model_path),monitor='val_loss',save_best_only=True),tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)],verbose=2)
        best = tf.keras.models.load_model(model_path)
        probabilities = best.predict(val_ds,verbose=0); y_pred = probabilities.argmax(axis=1)
        report = classification_report(y_true,y_pred,labels=np.arange(8),target_names=class_names,output_dict=True,zero_division=0)
        row = dict(seed=seed,condition=condition,epochs_trained=len(history.history['loss']),best_epoch=int(np.argmin(history.history['val_loss'])+1),best_val_loss=float(min(history.history['val_loss'])),accuracy=float(accuracy_score(y_true,y_pred)),macro_f1=float(f1_score(y_true,y_pred,labels=np.arange(8),average='macro',zero_division=0)),balanced_accuracy=float(balanced_accuracy_score(y_true,y_pred)),initial_weights_sha256=init_hash)
        results.append(row)
        pd.DataFrame(history.history).assign(epoch=lambda d: np.arange(1,len(d)+1)).to_csv(out_dir/f'{run_id}_history.csv',index=False)
        preds=val_part[['image_id','label']].copy(); preds['predicted_label']=[class_names[j] for j in y_pred];preds['confidence']=probabilities.max(axis=1)
        preds.to_csv(out_dir/f'{run_id}_predictions.csv',index=False)
        pd.DataFrame(confusion_matrix(y_true,y_pred,labels=np.arange(8)),index=class_names,columns=class_names).to_csv(out_dir/f'{run_id}_confusion_matrix.csv')
        (out_dir/f'{run_id}_metrics.json').write_text(json.dumps(dict(**row,classification_report=report,test_evaluated=False),indent=2))
        pd.DataFrame(results).to_csv(out_dir/'paired_seed_results.csv',index=False)
        print('RUN_COMPLETE',json.dumps(row),flush=True)
        del best,model,train_ds,val_ds;gc.collect()
summary=pd.DataFrame(results)
print('ALL_PAIRED_RUNS_COMPLETE')
print(summary.to_string(index=False))
print('MEAN_AND_SAMPLE_SD')
print(summary.groupby('condition')[['accuracy','macro_f1','balanced_accuracy']].agg(['mean','std']).to_string())
config=dict(seeds=SEEDS,split_fingerprint='900aed297caf894a0a5b0001b3ca974ea954d96104d70781b2eb7a71306227c1',tensorflow_version=tf.__version__,augmentation=dict(flip='horizontal_and_vertical',rotation_factor=0.05,zoom_factor=0.10,fill_mode='reflect'),batch_size=32,max_epochs=25,patience=5,learning_rate=0.001,dropout_seed=12345,test_evaluated=False)
(out_dir/'paired_seed_config.json').write_text(json.dumps(config,indent=2))
# Preserve all run evidence; archive each condition separately for GitHub upload limits.
import zipfile
for condition in ['no_aug','conventional_aug']:
    with zipfile.ZipFile(f'/kaggle/working/paired_{condition}_artifacts.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out_dir.glob(f'{condition}_seed*')): z.write(p,p.name)
        for name in ['paired_seed_results.csv','paired_seed_config.json','augmentation_preview.png']: z.write(out_dir/name,name)
print('ARTIFACT_ARCHIVES_READY')
