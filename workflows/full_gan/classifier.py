"""Local classifier comparisons. Importing definitions never trains a model."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

from study_split import SOURCE_COUNTS, SPLIT_SHA, TRAIN_COUNTS, TRAIN_SIZE

INITIAL_HASHES = {
    42: '58664884d2acf89bb923ccd2b745a10c5499e3998bd4ffa6ea96e534e8bc564c',
    43: 'fb2910c5637b9951cc5e54ac36003c23da39759070ebfd9d0517cf7b89f84e07',
    44: 'eb6da837326d0d9adbde573db1f111f3285086c5b9f055645ff3dc4823b609dd',
}
BATCH_SIZE = 32
LEARNING_RATE = 0.0001


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pixel_hash(path):
    with Image.open(path) as image:
        require(image.mode == 'RGB' and image.size == (128, 128), 'Synthetic images must be RGB128 PNG files')
        return hashlib.sha256(np.asarray(image, dtype=np.uint8).tobytes()).hexdigest()


def prepare_pool(frame, pool_manifest_path, review_path):
    """Validate a generated pool against the real cohort and recorded provenance."""
    require(pool_manifest_path.is_file() and review_path.is_file(), 'Generated pool and its automatic provenance record are required')
    review = json.loads(review_path.read_text())
    require(review.get('generator_type') == 'official_stylegan2ada', 'Old custom GAN pool is not accepted')
    require(review.get('split_fingerprint') == SPLIT_SHA, 'Generator training split differs')
    require(review.get('validation_test_images_used') is False, 'Generator held-out-data exclusion must be recorded')
    checkpoint_sha = review.get('generator_checkpoint_sha256', '')
    require(isinstance(checkpoint_sha, str) and len(checkpoint_sha) == 64 and all(c in '0123456789abcdef' for c in checkpoint_sha), 'Missing generator checkpoint SHA256')
    require(review.get('pool_manifest_sha256') == sha256_file(pool_manifest_path), 'Pool manifest hash differs from recorded pool')
    pool = pd.read_csv(pool_manifest_path)
    require({'sample_id', 'path', 'target', 'seed', 'pixel_sha256'} <= set(pool), 'Synthetic manifest columns are incomplete')
    require(len(pool) == TRAIN_SIZE and pool.sample_id.is_unique, 'Synthetic pool size or unique IDs differ')
    require(pool.target.value_counts().to_dict() == TRAIN_COUNTS, 'Synthetic pool class counts differ')
    require(not pool.duplicated(['target', 'seed']).any(), 'Repeated class/seed in synthetic pool')
    require(not set(pool.sample_id.astype(str)) & set(frame.image_id.astype(str)), 'Synthetic IDs overlap real IDs')
    require(pool.pixel_sha256.is_unique, 'Exact duplicated synthetic outputs; review the pool')
    pool = pool.sort_values('sample_id').reset_index(drop=True)
    resolved = []
    for row in pool.itertuples():
        relative = Path(str(row.path))
        require(not relative.is_absolute() and '..' not in relative.parts, 'Synthetic paths must stay relative to the pool manifest folder')
        path = (pool_manifest_path.parent / relative).resolve()
        require(path.is_relative_to(pool_manifest_path.parent.resolve()), 'Synthetic path escapes pool folder')
        require(path.suffix.lower() == '.png' and path.is_file(), 'Synthetic PNG is missing')
        require(pixel_hash(path) == row.pixel_sha256, 'Synthetic image pixels differ from recorded manifest')
        resolved.append(str(path))
    require(len(set(resolved)) == TRAIN_SIZE, 'Synthetic file paths repeat')
    pool['path'] = resolved
    pool['image_id'] = pool.sample_id.astype(str)

    pool['source'] = 'synthetic'
    provenance = dict(pool_manifest_sha256=sha256_file(pool_manifest_path),
                      pool_provenance_sha256=sha256_file(review_path), generator_checkpoint_sha256=checkpoint_sha,
                      synthetic_labels='requested conditions, not clinically verified',
                      quality_review='not assessed by automated execution',
                      memorization_review='exact duplicates only; near-copy review not performed')
    return pool, provenance


def epoch_schedule(train, pool, seed, epoch):
    """Shuffle all reals for the baseline, or replace half of each class with synthetic samples."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch), 7361]))
    if pool is None:
        return train.iloc[rng.permutation(len(train))].reset_index(drop=True)
    chosen = []
    for part in (train, pool):
        for target, count in SOURCE_COUNTS.items():
            subset = part[part.target == target]
            chosen.append(subset.iloc[rng.choice(len(subset), count, replace=False)])
    selected = pd.concat(chosen, ignore_index=True)
    return selected.iloc[rng.permutation(TRAIN_SIZE)].reset_index(drop=True)


def run_experiment(train, val, pool, provenance, output, condition, reference):
    """Fit one classifier condition using the shared CNN recipe."""
    import gc
    import platform
    import subprocess
    import sys
    import tensorflow as tf
    from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                                 confusion_matrix, roc_auc_score, average_precision_score,
                                 classification_report)
    require(tf.__version__ == '2.20.0', 'Use TensorFlow 2.20.0 for all matched classifier runs.')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    conventional = condition == 'real_gan_conventional'
    synthetic_count = sum(SOURCE_COUNTS.values()) if pool is not None else 0
    class_weights = {int(target): len(train) / (2 * count) for target, count in train.target.value_counts().items()}
    config = dict(condition=condition, **provenance, seeds=[42, 43, 44],
                  tensorflow=tf.__version__, python=platform.python_version(), platform=platform.platform(),
                  numpy=np.__version__, pandas=pd.__version__,
                  devices=[d.device_type for d in tf.config.list_physical_devices()],
                  learning_rate=LEARNING_RATE, batch=BATCH_SIZE, steps_per_epoch=(len(train) + BATCH_SIZE - 1) // BATCH_SIZE,
                  max_epochs=25, patience=5,
                  real_per_epoch=len(train) - synthetic_count, synthetic_per_epoch=synthetic_count, threshold=0.5,
                  augmentation=dict(enabled=bool(conventional), flip='horizontal_and_vertical', rotation_factor=0.05, zoom=[-0.1, 0.1], fill_mode='reflect'),
                  class_weights=class_weights, test_evaluated=False, status='started')
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    (out / 'pip_freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))

    def load_image(path, target):
        x = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        x.set_shape([None, None, 3])
        return tf.cast(tf.image.resize(x, [128, 128]), tf.float32) / 255., target

    def dataset(part):
        ds = tf.data.Dataset.from_tensor_slices((part.path.astype(str).to_numpy(), part.target.to_numpy('int32')))
        ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        opts = tf.data.Options()
        opts.experimental_deterministic = True
        return ds.with_options(opts).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


    results = []
    try:
        for seed in [42, 43, 44]:
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed)
            model = build_reference_cnn()
            load_reference_weights(model, seed, reference)
            init_hash = hashlib.sha256(b''.join(w.tobytes() for w in model.get_weights())).hexdigest()
            require(init_hash == INITIAL_HASHES[seed], 'Initialization differs from archived baseline; stop and investigate runtime pairing')
            np.savez_compressed(out / f'initial_weights_seed{seed}.npz', *model.get_weights())
            model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
            aug = None
            if conventional:
                aug = tf.keras.Sequential([
                    tf.keras.layers.RandomFlip('horizontal_and_vertical', seed=seed+100),
                    tf.keras.layers.RandomRotation(.05, fill_mode='reflect', seed=seed+200),
                    tf.keras.layers.RandomZoom(height_factor=(-.1,.1), width_factor=(-.1,.1), fill_mode='reflect', seed=seed+300)])
            validation = dataset(val)
            history, best_loss, wait = [], float('inf'), 0
            checkpoint = out / f'{condition}_seed{seed}.keras'
            for epoch in range(25):
                schedule = epoch_schedule(train, pool, seed, epoch)
                schedule[['image_id', 'target', 'source']].to_csv(out / f'schedule_seed{seed}_epoch{epoch+1:02d}.csv', index=False)
                training = dataset(schedule)
                if aug is not None:
                    training = training.map(lambda x,y: (aug(x, training=True),y), num_parallel_calls=1, deterministic=True)
                # One epoch per call preserves model/optimizer/augmentation state.
                h = model.fit(training, validation_data=validation, initial_epoch=epoch, epochs=epoch+1,
                              class_weight=class_weights, verbose=2)
                record = {k: float(v[0]) for k,v in h.history.items()}
                require(np.isfinite(list(record.values())).all(), 'Non-finite training or validation metric')
                record.update(epoch=epoch+1, optimizer_updates=int(model.optimizer.iterations.numpy()))
                history.append(record)
                pd.DataFrame(history).to_csv(out / f'history_seed{seed}.csv', index=False)
                if record['val_loss'] < best_loss:
                    best_loss, selected_epoch, wait = record['val_loss'], epoch+1, 0
                    model.save(checkpoint)
                else:
                    wait += 1
                    if wait >= 5:
                        break
            best = tf.keras.models.load_model(checkpoint)
            score = best.predict(validation, verbose=0).reshape(-1)
            y = val.target.to_numpy('int32')
            pred = (score >= 0.5).astype('int32')
            cm = confusion_matrix(y, pred, labels=[0,1])
            tn, fp, fn, tp = cm.ravel()
            row = dict(seed=seed, condition=condition, epochs_trained=len(history), selected_epoch=selected_epoch,
                       optimizer_updates=int(model.optimizer.iterations.numpy()), validation_loss=best_loss,
                       accuracy=float(accuracy_score(y,pred)), macro_f1=float(f1_score(y,pred,labels=[0,1],average='macro',zero_division=0)),
                       balanced_accuracy=float(balanced_accuracy_score(y,pred)), malignant_sensitivity=float(tp/(tp+fn)),
                       specificity=float(tn/(tn+fp)), roc_auc=float(roc_auc_score(y,score)),
                       average_precision=float(average_precision_score(y,score)), threshold=0.5,
                       initial_weights_sha256=init_hash, **provenance, test_evaluated=False)
            (out / f'metrics_seed{seed}.json').write_text(json.dumps(dict(**row, classification_report=classification_report(y,pred,labels=[0,1],target_names=['benign','malignant'],output_dict=True,zero_division=0)), indent=2))
            predictions = val[['image_id', 'original_label', 'label', 'group']].copy()
            predictions['malignant_probability'], predictions['prediction'] = score, pred
            predictions.to_csv(out / f'validation_predictions_seed{seed}.csv', index=False)
            pd.DataFrame(cm,index=['benign','malignant'],columns=['benign','malignant']).to_csv(out / f'confusion_matrix_seed{seed}.csv')
            results.append(row)
            pd.DataFrame(results).to_csv(out / 'results.csv', index=False)
            del model, best, training, validation, aug
            gc.collect()
        config['status'] = 'complete'
    except BaseException as error:
        config.update(status='failed', error=repr(error))
        raise
    finally:
        (out / 'config.json').write_text(json.dumps(config, indent=2))
    print('COMPLETE:', out.resolve())
    return pd.DataFrame(results)


def build_reference_cnn():
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.Input((128,128,3)),
        tf.keras.layers.Conv2D(32,5,padding='same',activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64,5,padding='same',activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128,5,padding='same',activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(256,5,padding='same',activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dropout(.45),
        tf.keras.layers.Dense(1,activation='sigmoid')])



def load_reference_weights(model, seed, reference):
    path = reference / f'initial_weights_seed{seed}.npz'
    with np.load(path, allow_pickle=False) as arrays:
        model.set_weights([arrays[f'arr_{i}'] for i in range(len(arrays.files))])
    digest = hashlib.sha256(b''.join(w.tobytes() for w in model.get_weights())).hexdigest()
    require(digest == INITIAL_HASHES[seed], 'Archived classifier initialization hash differs')
    return digest
