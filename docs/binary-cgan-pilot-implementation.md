# Conditional GAN pilot implementation — 30 October 2026

Status: training-image preflight executed successfully in Kaggle on 30 October 2026. No GAN model has been trained and no synthetic image pool has been generated.

Use [notebook 09](../notebooks/09_isic2019_binary_cgan_pilot.ipynb) in Kaggle. It contains the complete [training script](../scripts/binary_cgan_pilot.py), setup instructions, a preflight cell and a commented explicit training launch. See the [protocol](gan-protocol-draft.md).

The first pilot fixes seed 142 and 5,000 adversarial steps on the 972 frozen training images. It saves scalar logs each step, both-class fixed grids and full TensorFlow checkpoint states every 500 steps. Model, data-order, latent, grid and audit streams are recorded separately. The final unfiltered pool has 530 requested benign and 442 requested malignant images. An exact decoded/resized-pixel copy audit is included. Feature-based near-copy analysis, diversity review and label plausibility review remain work to complete before any downstream classifier experiment.

The notebook specifies architecture widths, conditioning, initialization, normalization, activations, optimizer settings and BatchNorm update behavior. Full model configurations and runtime versions are archived at execution. No discriminator dropout is used. Training is gated on TensorFlow 2.20.0 and a GPU. It refuses an existing output directory. Checkpoint/data-order states are preserved, but automatic resume and restoration are not implemented or tested.

Local verification passed: Python syntax; archived manifest accepted with 972 training images (530/442); changed split and binary target rejected. The Kaggle preflight accepted the frozen split, verified agreement for IDs with multiple file copies, and successfully decoded all 972 training images (530 benign / 442 malignant). TensorFlow 2.20.0 detected two Tesla T4 GPUs. Model construction, gradients, saving/restoration and full GAN training remain unverified. The supplied Run All workflow performs preflight only; training requires uncommenting the explicit launch.

No validation or test image content is read for generation, no classifier is fitted and no validation/test score is computed. Manifest split/label metadata are read solely to verify the frozen cohort. Requested synthetic labels are not confirmed clinical diagnoses.

## Kaggle execution record

[Open the pilot notebook](https://www.kaggle.com/code/amirsalahk/notebook5a45a9891c/edit#Preflight-only). The source images and a private input named ISIC 2019 Verified Binary Split are attached. The CSV is the archived manifest, not a new split. Notebook 09 in this repository preserves the actual successful preflight output and execution counts; its training cell remains commented and unexecuted.

The first preflight attempt was interrupted during redundant file-byte reads. The implementation now compares bytes only for IDs with multiple copies and logs decoding progress every 100 images. Every selected training image is still decoded and checked. The successful retry used the exact updated script; the exported code was compared against the prepared source. No model hyperparameters or data memberships changed.
