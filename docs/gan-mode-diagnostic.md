# GAN BatchNorm mode diagnostic — 10 November 2026

## Outcome

The saved generator behaves differently in training and inference modes, while changing the requested class has a much smaller pixel effect. Assistant inspection of all-benign and all-malignant grids in both modes still shows strongly repeated structures and checkerboard patterns. Neither mode produces an acceptable augmentation pool. This supports investigating the training-mode mismatch and weak conditioning; it does not prove the cause of collapse or establish that a mode change will fix it.

## What was executed

- Loaded the original step-5000 generator, verified SHA256 `a67c5a17e94f8e07540658818c071ddf9481203321fa4b7f279751097a88fdbb`.
- Compared identical latent vectors (seed 4142, batch 32) and labels in both modes, with all-benign, all-malignant and mixed batches.
- Restored original weights and BatchNorm state before each call and after the comparison; the state digest matched.
- No optimizer was constructed and no gradient updates were performed. No training, validation or test image dataset was read. The replay check reads archived synthetic grids only.
- Current Kaggle CPU runtime: TensorFlow 2.21.0, Python 3.13.18. Pilot: TensorFlow 2.20.0, Python 3.13.15. The first attempt stopped at the strict version guard before evaluation. A revised script requires explicit `--allow-runtime-drift` and records the mismatch. This is a cross-runtime exploratory diagnostic, not exact reproduction.

## Measured results

All metrics below use floating RGB values on a 0–1 scale and 32 samples per batch. Pixel SD is a descriptive across-sample measure, not diagnostic diversity or a quality threshold.

| Label batch | Mean absolute training/inference difference | Inference mean pixel SD | Training mean pixel SD |
| --- | ---: | ---: | ---: |
| All benign | 0.105833 | 0.033548 | 0.046490 |
| All malignant | 0.105182 | 0.034056 | 0.047614 |
| Mixed | 0.105519 | 0.033891 | 0.047190 |

With noise fixed, changing all labels from benign to malignant produced mean absolute pixel differences of **0.002337 in inference mode** and **0.001873 in training mode**. These small effects are consistent with weak class conditioning, not a test of clinical diagnosis validity.

Replaying the original inference grids under the current runtime produced maximum channel-value differences of 1/255, affecting 4 of 786,432 channel values for benign and 7 of 786,432 for malignant. Mean uint8 differences were 0.000005086 and 0.000008901. Thus those grids are near-identical, but not byte-exact reproduction. This limited replay does not establish full equivalence of runtimes.

## Confirmed code issue to investigate

The original training script feeds the discriminator fake batches generated with `training=False`, but computes generator updates with `training=True`. The generator has six BatchNormalization layers, which use moving statistics during inference and batch statistics during training. `tf.stop_gradient` blocks gradients without requiring inference mode.

The objectives therefore use different generator paths. This may contribute to the low generator loss despite poor inference output. It remains a hypothesis. Transposed-convolution kernel 4 / stride 2 has even overlap; observed checkerboards alone do not identify their cause.

## Saved evidence

- [Executed notebook 10](../notebooks/10_isic2019_gan_mode_diagnostic.ipynb): source and numerical outputs; image display payloads omitted explicitly for readability.
- [Mode report JSON](../artifacts/cgan_mode_diagnostic.json) and [runtime replay JSON](../artifacts/cgan_mode_runtime_replay.json).
- [Diagnostic script](../scripts/diagnose_cgan_modes.py).
- [Kaggle notebook](https://www.kaggle.com/code/amirsalahk/10-isic-gan-batchnorm-diagnostic/edit), Version 1: quick save completed with output preservation selected. Six grids, JSONs and diagnostic ZIP were saved. Full executed notebook and ZIP were also downloaded; archive CRC passed.
- Original 5,000-step pilot and its model files were not modified. The diagnostic CPU session was stopped after saving.

Python compilation passed. The saved model hash matches the earlier model archive. No feature-based memorization or clinical validity assessment was performed.

## Next experiment

Prepare a separate short, from-scratch matched ablation: original generator mode path versus consistent training mode for both adversarial updates. Keep architecture, optimizers, seed, data order and noise sequence fixed, and compare both training- and inference-mode grids. Record the additional BatchNorm moving-statistics update explicitly. Do not resume the collapsed checkpoint as if it were a clean experiment. Change upsampling only in a separate later comparison if needed. This diagnostic does not authorize acceptance of any synthetic image pool or test-set evaluation.

References: [TensorFlow BatchNormalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization), [original pilot report](binary-cgan-pilot-results.md).
