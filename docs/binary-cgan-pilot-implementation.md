# Conditional GAN pilot implementation — 30 October 2026

Status: prepared, not executed. No GAN model or generated images exist from this implementation yet.

Use [notebook 09](../notebooks/09_isic2019_binary_cgan_pilot.ipynb) in Kaggle. It contains the complete [training script](../scripts/binary_cgan_pilot.py), setup instructions, a preflight cell and a commented explicit training launch. See the [protocol](gan-protocol-draft.md).

The first pilot fixes seed 142 and 5,000 adversarial steps on the 972 frozen training images. It saves scalar logs each step, both-class fixed grids and full TensorFlow checkpoint states every 500 steps. Model, data-order, latent, grid and audit streams are recorded separately. The final unfiltered pool has 530 requested benign and 442 requested malignant images. An exact decoded/resized-pixel copy audit is included. Feature-based near-copy analysis, diversity review and label plausibility review remain work to complete before any downstream classifier experiment.

The notebook specifies architecture widths, conditioning, initialization, normalization, activations, optimizer settings and BatchNorm update behavior. Full model configurations and runtime versions are archived at execution. No discriminator dropout is used. Training is gated on TensorFlow 2.20.0 and a GPU. It refuses an existing output directory. Checkpoint/data-order states are preserved, but automatic resume and restoration are not implemented or tested.

Local verification passed: Python syntax; archived manifest accepted with 972 training images (530/442); changed split and binary target rejected. TensorFlow is unavailable locally: model construction, gradients, saving/restoration and full GPU execution remain unverified. Run the notebook preflight in Kaggle first. The supplied Run All workflow performs preflight only; training requires uncommenting the explicit launch.

No validation or test image content is read for generation, no classifier is fitted and no validation/test score is computed. Manifest split/label metadata are read solely to verify the frozen cohort. Requested synthetic labels are not confirmed clinical diagnoses.
