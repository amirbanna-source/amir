"""Exploratory conditional GAN pilot. Default: preflight only; no training.

Kaggle inputs: original ISIC images and the archived binary split CSV.
Use --train explicitly to run the fixed 5,000-step pilot in a GPU session.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

FINGERPRINT = '4f3037e5dc1726e5a534e3a6f155df3e0e0be52229b057659e097483fd0800cd'
MAPPING = {'NV': 0, 'BKL': 0, 'DF': 0, 'VASC': 0,
           'MEL': 1, 'BCC': 1, 'SCC': 1}


def verify_manifest(frame):
    required = {'image_id', 'label', 'group', 'split', 'original_label', 'target'}
    assert required <= set(frame.columns), 'Missing manifest columns'
    digest = hashlib.sha256(frame[['image_id', 'label', 'group', 'split']]
        .sort_values('image_id').to_csv(index=False).encode()).hexdigest()
    assert digest == FINGERPRINT, 'Manifest differs from frozen split'
    assert len(frame) == 1632 and frame.image_id.is_unique
    assert frame.groupby('group').split.nunique().max() == 1
    assert frame.original_label.map(MAPPING).equals(frame.target)
    assert frame.target.map({0: 'benign', 1: 'malignant'}).equals(frame.label)
    assert frame.split.value_counts().to_dict() == {'train': 972, 'test': 331, 'val': 329}
    train = frame[frame.split == 'train'].sort_values('image_id').reset_index(drop=True)
    assert train.target.value_counts().to_dict() == {0: 530, 1: 442}
    return train


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--input-root', type=Path, default=Path('/kaggle/input'))
    parser.add_argument('--output', type=Path, default=Path('/kaggle/working/binary_cgan_pilot'))
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args(argv)
    manifests = ([args.manifest] if args.manifest else
                 sorted(args.input_root.rglob('isic2019_binary_split.csv')))
    assert len(manifests) == 1, 'Attach exactly one archived isic2019_binary_split.csv or pass --manifest'
    train = verify_manifest(pd.read_csv(manifests[0]))
    # Resolve mount changes by image ID. Only training image content is read.
    candidates = {}
    wanted = set(train.image_id)
    for p in sorted(args.input_root.rglob('*')):
        if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'} and p.stem.upper() in wanted:
            candidates.setdefault(p.stem.upper(), []).append(p)
    assert set(candidates) == wanted, 'Some training images are missing'
    paths = []
    for image_id in train.image_id:
        copies = candidates[image_id]
        hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in copies}
        assert len(hashes) == 1, f'Conflicting image copies: {image_id}'
        paths.append(copies[0])
    import tensorflow as tf
    from PIL import Image
    images = []
    for p in paths:
        with Image.open(p) as im:
            im.load()
            x = np.asarray(im.convert('RGB'), dtype=np.float32)
        x = tf.image.resize(x, [128, 128]).numpy() / 127.5 - 1
        assert x.shape == (128, 128, 3) and np.isfinite(x).all()
        images.append(x)
    images = np.stack(images)
    labels = train.target.to_numpy(dtype=np.int32)
    print('Preflight passed:', len(images), 'training images; counts', train.target.value_counts().to_dict())
    print('TensorFlow:', tf.__version__, 'GPUs:', tf.config.list_physical_devices('GPU'))
    if not args.train:
        print('Preflight only. Run main(["--train"]) explicitly after review to start the pilot.')
        return
    assert tf.__version__ == '2.20.0', 'Runtime version changed; review before training'
    assert tf.config.list_physical_devices('GPU'), 'Enable a Kaggle GPU'
    assert not args.output.exists(), 'Output exists: preserve it and use a fresh output directory'
    args.output.mkdir(parents=True)
    tf.keras.utils.set_random_seed(142)
    tf.config.experimental.enable_op_determinism()
    L = tf.keras.layers
    init = lambda: tf.keras.initializers.RandomNormal(0., .02)
    z = L.Input((128,), name='noise')
    y = L.Input((), dtype='int32', name='requested_class')
    embedding = L.Flatten()(L.Embedding(2, 16)(y))
    x = L.Concatenate()([z, embedding])
    x = L.Dense(4 * 4 * 256, use_bias=False, kernel_initializer=init())(x)
    x = L.Reshape((4, 4, 256))(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    for channels in [256, 128, 64, 32, 16]:
        x = L.Conv2DTranspose(channels, 4, strides=2, padding='same',
                              use_bias=False, kernel_initializer=init())(x)
        x = L.BatchNormalization()(x)
        x = L.ReLU()(x)
    output = L.Conv2D(3, 3, padding='same', activation='tanh', kernel_initializer=init())(x)
    generator = tf.keras.Model([z, y], output, name='conditional_generator')
    image = L.Input((128, 128, 3), name='image')
    condition = L.Input((), dtype='int32', name='condition')
    plane = L.Reshape((128, 128, 1))(L.Embedding(2, 128 * 128)(condition))
    x = L.Concatenate()([image, plane])
    for channels in [32, 64, 128, 256, 256]:
        x = L.Conv2D(channels, 4, strides=2, padding='same', kernel_initializer=init())(x)
        x = L.LeakyReLU(negative_slope=.2)(x)
    # No discriminator BatchNorm or dropout; full architecture archived below.
    logits = L.Dense(1, kernel_initializer=init())(L.Flatten()(x))
    discriminator = tf.keras.Model([image, condition], logits, name='conditional_discriminator')
    g_opt = tf.keras.optimizers.Adam(.0002, beta_1=.5, beta_2=.999)
    d_opt = tf.keras.optimizers.Adam(.0002, beta_1=.5, beta_2=.999)
    g_opt.build(generator.trainable_variables)
    d_opt.build(discriminator.trainable_variables)
    rng = tf.random.Generator.from_seed(3142)
    order_rng = np.random.default_rng(2142)
    fixed_z = tf.constant(np.random.default_rng(4142).normal(size=(32, 128)), tf.float32)
    fixed_y = tf.constant([0] * 16 + [1] * 16, tf.int32)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    step = tf.Variable(0, dtype=tf.int64, trainable=False)
    checkpoint = tf.train.Checkpoint(step=step, generator=generator, discriminator=discriminator,
                                     g_optimizer=g_opt, d_optimizer=d_opt, latent_rng=rng)
    config = {'seed': 142, 'data_order_seed': 2142, 'latent_seed': 3142,
              'grid_seed': 4142, 'audit_seed': 1142, 'steps': 5000, 'batch_size': 32,
              'tensorflow': tf.__version__, 'python': platform.python_version(),
              'numpy': np.__version__, 'split_sha256': FINGERPRINT,
              'gpus': [str(g) for g in tf.config.list_physical_devices('GPU')],
              'generator': json.loads(generator.to_json()),
              'discriminator': json.loads(discriminator.to_json()),
              'g_optimizer': g_opt.get_config(), 'd_optimizer': d_opt.get_config(),
              'normalization': 'PIL RGB; TensorFlow bilinear resize; x/127.5-1',
              'discriminator_loss': '0.5*(real_BCE+fake_BCE)',
              'resume': 'States archived; automatic resume is not implemented or validated'}
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    train.assign(resolved_path=[str(p) for p in paths]).to_csv(args.output / 'training_manifest.csv', index=False)
    order, cursor = order_rng.permutation(972), 0
    histories = []

    def save_state():
        n = int(step.numpy())
        checkpoint.write(str(args.output / f'checkpoint_{n:05d}'))
        (args.output / f'data_state_{n:05d}.json').write_text(json.dumps({
            'order': order.tolist(), 'cursor': cursor, 'rng': order_rng.bit_generator.state}))
        pd.DataFrame(histories).to_csv(args.output / 'history.csv', index=False)
        grid = generator([fixed_z, fixed_y], training=False).numpy()
        assert np.isfinite(grid).all() and grid.min() >= -1.001 and grid.max() <= 1.001
        grid = np.clip((grid + 1) * 127.5, 0, 255).round().astype('uint8')
        for target in [0, 1]:
            canvas = Image.new('RGB', (512, 512))
            for j, a in enumerate(grid[target * 16:(target + 1) * 16]):
                canvas.paste(Image.fromarray(a), ((j % 4) * 128, (j // 4) * 128))
            canvas.save(args.output / f'grid_class{target}_{n:05d}.png')

    def checked_apply(tape, objective, variables, optimizer):
        tf.debugging.assert_all_finite(objective, 'Non-finite loss')
        gradients = tape.gradient(objective, variables)
        assert all(g is not None for g in gradients), 'Disconnected gradient'
        for g in gradients:
            tf.debugging.assert_all_finite(tf.convert_to_tensor(g), 'Non-finite gradient')
        optimizer.apply_gradients(zip(gradients, variables))

    save_state()
    started = time.monotonic()
    try:
        for n in range(1, 5001):
            indices = []
            while len(indices) < 32:
                take = min(32 - len(indices), 972 - cursor)
                indices.extend(order[cursor:cursor + take])
                cursor += take
                if cursor == 972:
                    order, cursor = order_rng.permutation(972), 0
            real = tf.constant(images[indices])
            batch_y = tf.constant(labels[indices])
            # Inference mode avoids updating generator BatchNorm during D updates.
            fake = tf.stop_gradient(generator([rng.normal((32, 128)), batch_y], training=False))
            with tf.GradientTape() as tape:
                real_logits = discriminator([real, batch_y], training=True)
                fake_logits = discriminator([fake, batch_y], training=True)
                d_loss = .5 * (loss(tf.ones_like(real_logits), real_logits) +
                                loss(tf.zeros_like(fake_logits), fake_logits))
            checked_apply(tape, d_loss, discriminator.trainable_variables, d_opt)
            with tf.GradientTape() as tape:
                fake = generator([rng.normal((32, 128)), batch_y], training=True)
                fake_logits = discriminator([fake, batch_y], training=False)
                g_loss = loss(tf.ones_like(fake_logits), fake_logits)
            checked_apply(tape, g_loss, generator.trainable_variables, g_opt)
            step.assign(n)
            record = {'step': n, 'g_loss': float(g_loss.numpy()), 'd_loss': float(d_loss.numpy()),
                      'benign_in_batch': int(np.sum(labels[indices] == 0)),
                      'elapsed_seconds': time.monotonic() - started}
            histories.append(record)
            with (args.output / 'history.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            if n % 500 == 0:
                save_state()
                print(record, flush=True)
        generator.save(args.output / 'generator_step05000.keras')
        discriminator.save(args.output / 'discriminator_step05000.keras')
        # Preserve the first unfiltered pool, including any duplicates.
        audit_rng = np.random.default_rng(1142)
        pool = args.output / 'audit_pool'
        pool.mkdir()
        rows = []
        real_hashes = {}
        for image_id, a in zip(train.image_id, images):
            pixels = np.clip((a + 1) * 127.5, 0, 255).round().astype('uint8')
            real_hashes.setdefault(hashlib.sha256(pixels.tobytes()).hexdigest(), []).append(image_id)
        seen = {}
        duplicates = []
        for target, count in [(0, 530), (1, 442)]:
            for start in range(0, count, 32):
                size = min(32, count - start)
                noise = audit_rng.normal(size=(size, 128)).astype('float32')
                outputs = generator([tf.constant(noise), tf.constant([target] * size)], training=False).numpy()
                assert np.isfinite(outputs).all() and outputs.min() >= -1.001 and outputs.max() <= 1.001
                for j, a in enumerate(outputs):
                    pixels = np.clip((a + 1) * 127.5, 0, 255).round().astype('uint8')
                    name = f'class{target}_{start+j:04d}.png'
                    Image.fromarray(pixels).save(pool / name)
                    digest = hashlib.sha256(pixels.tobytes()).hexdigest()
                    if digest in real_hashes or digest in seen:
                        duplicates.append({'file': name, 'training_matches': real_hashes.get(digest, []),
                                           'earlier_synthetic_matches': seen.get(digest, []).copy()})
                    seen.setdefault(digest, []).append(name)
                    rows.append({'file': name, 'requested_target': target, 'sampling_seed': 1142,
                                 'latent_index': len(rows), 'checkpoint_step': 5000,
                                 'pixel_sha256': hashlib.sha256(pixels.tobytes()).hexdigest()})
        pd.DataFrame(rows).to_csv(args.output / 'audit_pool_manifest.csv', index=False)
        (args.output / 'exact_copy_audit.json').write_text(json.dumps({
            'representation': 'RGB 128x128 uint8 after declared resize and rounding',
            'duplicate_outputs': duplicates, 'pool_size': len(rows),
            'near_copy_review': 'pending; exact hashes do not establish novelty'}, indent=2))
        (args.output / 'status.json').write_text(json.dumps({'status': 'completed; quality review pending', 'steps': 5000}))
    except Exception as error:
        (args.output / 'failure.json').write_text(json.dumps({
            'last_completed_step': int(step.numpy()), 'error': repr(error)}))
        raise


if __name__ == '__main__':
    main()
