# GAN mode diagnostic — 6 November 2026

## Confirmed code finding

Review of the archived 5,000-step training script confirms that fake images for discriminator updates use `generator(..., training=False)`, while generator updates use `generator(..., training=True)`. The generator has six BatchNormalization layers. Inference uses moving statistics, whereas training uses batch statistics and updates moving statistics. `tf.stop_gradient` prevents gradient propagation but does not require inference mode.

This creates different generator computation paths for the two adversarial objectives. It is a plausible contributor to the very low generator loss despite poor inference grids. It is **not proof of the cause** of collapse. Checkerboard patterns and weak conditioning need separate evaluation. Kernel 4 / stride 2 in the transposed convolutions has even overlap; artifacts alone do not establish uneven overlap as the cause.

## Prepared diagnostic

[scripts/diagnose_cgan_modes.py](../scripts/diagnose_cgan_modes.py) loads the final generator and compares identical noise and labels in both modes. It tests all-benign, all-malignant and mixed batches, saves six grids, and reports mode differences, within-batch pixel variability and same-noise class differences. Every call starts from the original weights; all model state is restored and verified. No optimizer is constructed, no training images are read, and no revised model is saved. Choose a fresh output directory.

Run in the original TensorFlow 2.20.0 Kaggle runtime with the saved Version 2 model:

```bash
python diagnose_cgan_modes.py \
  --model /path/to/generator_step05000.keras \
  --output /kaggle/working/cgan_mode_diagnostic
```

The model path must refer to a copy of the archived model available in the session. Do not use Run All on notebook 09: its enabled training cell repeats the original pilot.

## Verification and remaining work

Python compilation and CLI help passed locally. **TensorFlow execution has not been performed**: this workspace lacks TensorFlow. No diagnostic metrics or image comparisons are claimed yet. Pixel differences are descriptive, with no acceptance threshold; they do not establish novelty or diagnostic validity.

After reviewing the diagnostic, use a separate, short from-scratch ablation with consistent generator training mode in both updates, fixed data order/seeds, and additional inference/training monitoring. Keep the architecture and other settings fixed for this first comparison. Track the extra BatchNorm moving-statistic update explicitly. Only then consider a separate upsampling change. A final-checkpoint comparison alone cannot prove causation during earlier training.

The failed pilot remains preserved. No new training or classifier evaluation was launched, and validation/test image content was not accessed.

References: [TensorFlow BatchNormalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization), [original pilot report](binary-cgan-pilot-results.md).
