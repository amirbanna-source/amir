"""Compare saved generator BatchNorm modes; no optimizer or training-data access.

Run in the original TensorFlow 2.20.0 Kaggle environment:
python diagnose_cgan_modes.py --model /path/generator_step05000.keras \
    --output /kaggle/working/cgan_mode_diagnostic
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    import numpy as np
    import tensorflow as tf
    from PIL import Image

    assert tf.__version__ == '2.20.0', 'Use the pilot runtime for comparison'
    assert not args.output.exists(), 'Choose a fresh output directory'
    model = tf.keras.models.load_model(args.model, compile=False)
    original = model.get_weights()
    digest = lambda weights: hashlib.sha256(b''.join(a.tobytes() for a in weights)).hexdigest()
    before = digest(original)
    noise = np.random.default_rng(4142).normal(size=(32, 128)).astype('float32')
    cases = {'all_benign': [0] * 32, 'all_malignant': [1] * 32,
             'mixed': [0] * 16 + [1] * 16}
    arrays, results = {}, {}
    try:
        for name, labels in cases.items():
            outputs = {}
            for mode in [False, True]:
                # Reset all state before each call: training=True updates BN statistics.
                model.set_weights(original)
                a = model([tf.constant(noise), tf.constant(labels, tf.int32)],
                          training=mode).numpy()
                assert a.shape == (32, 128, 128, 3) and np.isfinite(a).all()
                outputs[mode] = (a + 1) / 2  # Metrics use floating pixels on [0, 1].
                arrays[f'{name}_{"training" if mode else "inference"}'] = outputs[mode]
            results[name] = {
                'mode_mean_absolute_difference': float(np.abs(outputs[True] - outputs[False]).mean()),
                'inference_mean_pixel_sd': float(outputs[False].std(axis=0).mean()),
                'training_mean_pixel_sd': float(outputs[True].std(axis=0).mean()),
            }
    finally:
        model.set_weights(original)
    assert digest(model.get_weights()) == before, 'Model state was not restored'
    results['same_noise_class_change_mae'] = {
        mode: float(np.abs(arrays[f'all_benign_{mode}'] - arrays[f'all_malignant_{mode}']).mean())
        for mode in ['training', 'inference']
    }
    args.output.mkdir(parents=True)
    for name, a in arrays.items():
        canvas = Image.new('RGB', (512, 512))
        for j, pixels in enumerate(a[:16]):
            canvas.paste(Image.fromarray(np.clip(pixels * 255, 0, 255).round().astype('uint8')),
                         ((j % 4) * 128, (j // 4) * 128))
        canvas.save(args.output / f'{name}.png')
    report = {'tensorflow': tf.__version__, 'model_file': str(args.model),
              'model_file_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
              'noise_seed': 4142, 'batch_size': 32, 'model_state_restored': True,
              'optimizer_updates': 0, 'results': results,
              'limitations': 'Descriptive only; no pass threshold, causal proof, clinical validity, '
                             'or memorization assessment. Training-mode output depends on batch composition.'}
    (args.output / 'mode_diagnostic.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
