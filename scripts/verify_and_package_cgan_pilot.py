"""Run after the GAN pilot finishes; verifies and packages existing outputs only."""
from pathlib import Path
import hashlib
import io
import json
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

pilot = Path('/kaggle/working/binary_cgan_pilot')
status = json.loads((pilot / 'status.json').read_text())
assert status['steps'] == 5000 and status['status'] == 'completed; quality review pending'
assert not (pilot / 'failure.json').exists(), 'Run recorded a failure; review before packaging'
history = pd.read_csv(pilot / 'history.csv')
assert history.step.tolist() == list(range(1, 5001))
assert np.isfinite(history[['g_loss', 'd_loss', 'elapsed_seconds']]).all().all()
pool = pd.read_csv(pilot / 'audit_pool_manifest.csv')
assert len(pool) == 972 and pool.file.is_unique
assert pool.requested_target.value_counts().to_dict() == {0: 530, 1: 442}
reloaded_generator = tf.keras.models.load_model(pilot / 'generator_step05000.keras', compile=False)
reloaded_discriminator = tf.keras.models.load_model(pilot / 'discriminator_step05000.keras', compile=False)
assert reloaded_generator.output_shape == (None, 128, 128, 3)
assert reloaded_discriminator.output_shape == (None, 1)
audit_rng = np.random.default_rng(1142)
checked = 0
for target, count in [(0, 530), (1, 442)]:
    for start in range(0, count, 32):
        size = min(32, count - start)
        noise = audit_rng.normal(size=(size, 128)).astype('float32')
        generated = reloaded_generator([tf.constant(noise), tf.constant([target] * size)], training=False).numpy()
        for j, array in enumerate(generated):
            name = f'class{target}_{start+j:04d}.png'
            row = pool[pool.file == name].iloc[0]
            pixels = np.clip((array + 1) * 127.5, 0, 255).round().astype('uint8')
            assert hashlib.sha256(pixels.tobytes()).hexdigest() == row.pixel_sha256, name
            with Image.open(pilot / 'audit_pool' / name) as image:
                assert np.array_equal(np.asarray(image), pixels), name
            checked += 1
reader = tf.train.load_checkpoint(str(pilot / 'checkpoint_05000'))
iterations = {name: int(reader.get_tensor(name)) for name in reader.get_variable_to_shape_map()
              if 'iterations' in name and name.endswith('VARIABLE_VALUE')}
assert len(iterations) == 2 and set(iterations.values()) == {5000}, iterations
saved_step = int(reader.get_tensor('step/.ATTRIBUTES/VARIABLE_VALUE'))
assert saved_step == 5000
for step in range(0, 5001, 500):
    assert (pilot / f'checkpoint_{step:05d}.index').exists()
    assert (pilot / f'checkpoint_{step:05d}.data-00000-of-00001').exists()
    for target in [0, 1]:
        with Image.open(pilot / f'grid_class{target}_{step:05d}.png') as image:
            image.load()
            assert image.size == (512, 512)
report = {'completed_adversarial_steps': saved_step, 'optimizer_iterations': iterations,
          'regenerated_images_matching_saved_pixel_hashes': checked,
          'generator_parameters': reloaded_generator.count_params(),
          'discriminator_parameters': reloaded_discriminator.count_params(),
          'quality_review': 'pending; successful execution does not establish image usefulness',
          'resume': 'No resumed training or complete optimizer-state restoration test performed'}
(pilot / 'execution_verification.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
for kind in ['all', 'review']:
    destination = Path(f'/kaggle/working/binary_cgan_pilot_{kind}_artifacts.zip')
    assert not destination.exists(), 'Preserve previous archive rather than overwrite it'
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for file in sorted(pilot.rglob('*')):
            if not file.is_file():
                continue
            if kind == 'review' and file.name.startswith('checkpoint_') and not file.name.startswith('checkpoint_05000'):
                continue
            archive.write(file, str(file.relative_to(pilot)))
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
    print('Verified archive:', destination.name, destination.stat().st_size, 'bytes')
