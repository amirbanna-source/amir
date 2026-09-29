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

def make_dataset(split_name, shuffle=False):
    part = splits[splits["split"] == split_name]
    labels = part["label"].map(class_to_num).astype("int32").to_numpy()
    ds = tf.data.Dataset.from_tensor_slices(
        (part["path"].astype(str).to_numpy(), labels)
    )
    if shuffle:
        ds = ds.shuffle(len(part), seed=42, reshuffle_each_iteration=True)
    return ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset("train", shuffle=True)
val_ds = make_dataset("val")
test_ds = make_dataset("test")

images_batch, labels_batch = next(iter(train_ds))
print("Batch shape:", images_batch.shape)
print("Labels shape:", labels_batch.shape)
assert tf.config.list_physical_devices("GPU"), "Enable GPU before training"

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

tf.keras.utils.set_random_seed(42)

train_labels = splits.loc[splits["split"] == "train", "label"].map(class_to_num)
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(class_names)),
    y=train_labels.to_numpy(),
)
class_weights = {i: float(w) for i, w in enumerate(weights)}
print("Class weights:", class_weights)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(128, 128, 3)),

    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu"),
    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.35),
    tf.keras.layers.Dense(len(class_names), activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "/kaggle/working/cnn_conventional_aug.keras",
        monitor="val_loss",
        save_best_only=True,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    ),
]

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=101),
    tf.keras.layers.RandomRotation(0.05, fill_mode="reflect", seed=102),
    tf.keras.layers.RandomZoom(0.10, fill_mode="reflect", seed=103),
], name="conventional_augmentation")
aug_train_ds = train_ds.map(
    lambda x, y: (tf.clip_by_value(augmentation(x, training=True), 0., 1.), y),
    num_parallel_calls=1,
).prefetch(tf.data.AUTOTUNE)
history = model.fit(
    aug_train_ds,
    validation_data=val_ds,
    epochs=25,
    verbose=2,
    class_weight=class_weights,
    callbacks=callbacks,
)

print("Best validation loss:", min(history.history["val_loss"]))
print("Epochs trained:", len(history.history["loss"]))

# Evaluate the best saved baseline on validation only (leave test untouched)
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score

best_model = tf.keras.models.load_model('/kaggle/working/cnn_conventional_aug.keras')
val_part = splits.loc[splits['split'] == 'val'].copy()
y_true = val_part['label'].map(class_to_num).to_numpy(dtype=int)
probabilities = best_model.predict(val_ds, verbose=0)
y_pred = probabilities.argmax(axis=1)
assert len(y_true) == len(y_pred) == 355

print('Validation accuracy:', round(accuracy_score(y_true, y_pred), 4))
print('Validation macro F1:', round(f1_score(y_true, y_pred, labels=np.arange(len(class_names)), average='macro', zero_division=0), 4))
print('Validation balanced accuracy:', round(balanced_accuracy_score(y_true, y_pred), 4))
print('\nPer-class report:')
print(classification_report(y_true, y_pred, labels=np.arange(len(class_names)), target_names=class_names, zero_division=0))
cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names))), index=class_names, columns=class_names)
print('Confusion matrix (rows=true, columns=predicted):')
print(cm.to_string())
cm.to_csv('/kaggle/working/conventional_aug_validation_confusion_matrix.csv')
val_part['predicted_label'] = [class_names[i] for i in y_pred]
val_part['confidence'] = probabilities.max(axis=1)
val_part[['image_id', 'label', 'predicted_label', 'confidence']].to_csv('/kaggle/working/conventional_aug_validation_predictions.csv', index=False)
print('Saved validation metrics files in /kaggle/working')
import json
from sklearn.metrics import classification_report
pd.DataFrame(history.history).assign(epoch=lambda d: np.arange(1, len(d)+1)).to_csv('/kaggle/working/conventional_aug_history.csv', index=False)
metrics = {
    'experiment': 'conventional_augmentation', 'seed': 42,
    'epochs_trained': len(history.history['loss']),
    'best_epoch': int(np.argmin(history.history['val_loss']) + 1),
    'best_validation_loss': float(min(history.history['val_loss'])),
    'validation_accuracy': float(accuracy_score(y_true, y_pred)),
    'validation_macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
    'validation_balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
    'augmentation': {'flip': 'horizontal_and_vertical', 'rotation_turn_fraction': 0.05, 'zoom_factor': 0.10, 'fill_mode': 'reflect'},
    'tensorflow_version': tf.__version__,
    'classification_report': classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0),
    'test_evaluated': False,
}
Path('/kaggle/working/conventional_aug_metrics.json').write_text(json.dumps(metrics, indent=2))
print('EXPERIMENT_COMPLETE', json.dumps(metrics))
