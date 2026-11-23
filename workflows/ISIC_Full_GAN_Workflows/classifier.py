"""Local classifier comparisons. Importing definitions never trains a model."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

SPLIT_SHA = '4f3037e5dc1726e5a534e3a6f155df3e0e0be52229b057659e097483fd0800cd'
MAPPING = dict.fromkeys(['NV', 'BKL', 'DF', 'VASC'], 0) | dict.fromkeys(['MEL', 'BCC', 'SCC'], 1)
INITIAL_HASHES = {42: 'f0b9a0ddce8f8452c7308e6e7ae6c7d37f9d3b7eb850da4831cbb7f2651b3953', 43: 'f3a606597f9c02b4fbe03928aba8aadb9df21747c08c2b4142b6d4868042002f', 44: 'c015c7fd952ac5cb0c757a68a82d229d09a0c04101bdf0f3c5b1ca31e759e2b1'}


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


def prepare_inputs(manifest_path, real_root, pool_manifest_path, review_path):
    """Validate metadata first; resolve/decode only real train and validation images."""
    manifest_path, real_root = Path(manifest_path), Path(real_root)
    pool_manifest_path, review_path = Path(pool_manifest_path), Path(review_path)
    require(manifest_path.is_file(), 'Missing frozen binary split CSV')
    frame = pd.read_csv(manifest_path)
    needed = {'image_id', 'label', 'group', 'split', 'target', 'original_label'}
    require(needed <= set(frame), 'Frozen manifest lacks required columns')
    fingerprint = hashlib.sha256(frame[['image_id', 'label', 'group', 'split']].sort_values('image_id').to_csv(index=False).encode()).hexdigest()
    require(fingerprint == SPLIT_SHA, 'Frozen split fingerprint differs')
    require(len(frame) == 1632 and frame.image_id.is_unique, 'Wrong binary cohort')
    require(frame.groupby('group').split.nunique().max() == 1, 'Groups overlap splits')
    require(frame.original_label.map(MAPPING).equals(frame.target), 'Binary targets differ from original diagnoses')
    require(frame.groupby('split').size().to_dict() == {'test': 331, 'train': 972, 'val': 329}, 'Split counts differ')
    train = frame[frame.split == 'train'].copy().sort_values('image_id').reset_index(drop=True)
    val = frame[frame.split == 'val'].copy().sort_values('image_id').reset_index(drop=True)
    require(train.target.value_counts().to_dict() == {0: 530, 1: 442}, 'Training class counts differ')
    require(val.target.value_counts().to_dict() == {0: 179, 1: 150}, 'Validation class counts differ')

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
    require(len(pool) == 972 and pool.sample_id.is_unique, 'Need a fixed 972-image synthetic pool with unique sample IDs')
    require(pool.target.value_counts().to_dict() == {0: 530, 1: 442}, 'Synthetic pool must contain 530 benign-requested and 442 malignant-requested images')
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
    require(len(set(resolved)) == 972, 'Synthetic file paths repeat')
    pool['path'] = resolved
    pool['image_id'] = pool.sample_id.astype(str)

    wanted = set(train.image_id) | set(val.image_id)
    require(real_root.is_dir(), 'Real-image root is missing')
    candidates = {}
    # Filenames are inventoried; test-image CONTENT is never read.
    for path in sorted(real_root.rglob('*')):
        if path.suffix.lower() in {'.png', '.jpg', '.jpeg'} and path.stem.upper() in wanted:
            candidates.setdefault(path.stem.upper(), []).append(path)
    require(set(candidates) == wanted, 'Some real training/validation images are missing from REAL_IMAGE_ROOT')
    real_paths = {}
    for image_id, copies in candidates.items():
        if len(copies) > 1:
            require(len({sha256_file(p) for p in copies}) == 1, 'Conflicting file copies for a real image ID')
        with Image.open(copies[0]) as im:
            require(im.mode == 'RGB', 'Real image is not RGB; preserve original classifier preprocessing')
            im.load()
        real_paths[image_id] = str(copies[0].resolve())
    for part in (train, val):
        part['path'] = part.image_id.map(real_paths)
    train['source'], pool['source'] = 'real', 'synthetic'
    provenance = dict(split_fingerprint=fingerprint, pool_manifest_sha256=sha256_file(pool_manifest_path),
                      pool_provenance_sha256=sha256_file(review_path), generator_checkpoint_sha256=checkpoint_sha,
                      test_image_content_read=False, synthetic_labels='requested conditions, not clinically verified', quality_review='not assessed by automated execution', memorization_review='exact duplicates only; near-copy review not performed')
    return train, val, pool, provenance


def epoch_schedule(train, pool, seed, epoch):
    """Exactly 486 real + 486 synthetic, with each source containing 265/221 labels."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch), 7361]))
    chosen = []
    for part in (train, pool):
        for target, count in [(0, 265), (1, 221)]:
            subset = part[part.target == target]
            chosen.append(subset.iloc[rng.choice(len(subset), count, replace=False)])
    selected = pd.concat(chosen, ignore_index=True)
    return selected.iloc[rng.permutation(972)].reset_index(drop=True)


def run_experiment(train, val, pool, provenance, output, conventional):
    """Train the CNN on a fixed synthetic pool. Does not train or load a GAN."""
    import gc
    import platform
    import subprocess
    import sys
    import tensorflow as tf
    from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                                 confusion_matrix, roc_auc_score, average_precision_score,
                                 classification_report)
    require(tf.__version__ == '2.20.0', 'Use TensorFlow 2.20.0 for pairing with the archived reference. A different runtime requires a separately documented paired baseline.')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    condition = 'real_gan_conventional' if conventional else 'real_gan'
    config = dict(condition=condition, **provenance, seeds=[42, 43, 44],
                  tensorflow=tf.__version__, python=platform.python_version(), platform=platform.platform(),
                  numpy=np.__version__, pandas=pd.__version__,
                  devices=[d.device_type for d in tf.config.list_physical_devices()],
                  learning_rate=0.0001, batch=32, steps_per_epoch=31, max_epochs=25, patience=5,
                  real_per_epoch=486, synthetic_per_epoch=486, threshold=0.5,
                  augmentation=dict(enabled=bool(conventional), flip='horizontal_and_vertical', rotation_factor=0.05, zoom=[-0.1, 0.1], fill_mode='reflect'),
                  class_weights={'0': 972 / 1060, '1': 972 / 884}, test_evaluated=False,
                  status='started', note='Fixed presentation/update ceiling; half of real presentations are replaced with synthetic images.')
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
        return ds.with_options(opts).batch(32).prefetch(tf.data.AUTOTUNE)


    results = []
    try:
        for seed in [42, 43, 44]:
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed)
            model = build_reference_cnn()
            load_reference_weights(model, seed)
            init_hash = hashlib.sha256(b''.join(w.tobytes() for w in model.get_weights())).hexdigest()
            require(init_hash == INITIAL_HASHES[seed], 'Initialization differs from archived baseline; stop and investigate runtime pairing')
            np.savez_compressed(out / f'initial_weights_seed{seed}.npz', *model.get_weights())
            model.compile(optimizer=tf.keras.optimizers.Adam(.0001), loss='binary_crossentropy', metrics=['accuracy'])
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
                              class_weight={0: 972/1060, 1: 972/884}, verbose=2)
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



def load_reference_weights(model, seed):
    path = Path(__file__).parent / 'data' / f'initial_weights_seed{seed}.npz'
    with np.load(path, allow_pickle=False) as arrays:
        model.set_weights([arrays[f'arr_{i}'] for i in range(len(arrays.files))])
    digest = hashlib.sha256(b''.join(w.tobytes() for w in model.get_weights())).hexdigest()
    require(digest == INITIAL_HASHES[seed], 'Archived classifier initialization hash differs')
    return digest
