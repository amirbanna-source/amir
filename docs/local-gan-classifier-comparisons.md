# Full GAN classifier comparisons

[Workflow 14](../notebooks/14_isic2019_real_plus_gan_local.ipynb) trains the full conditional StyleGAN2-ADA model and a real + GAN classifier. [Workflow 15](../notebooks/15_isic2019_real_plus_gan_conventional_local.ipynb) uses the same generator, synthetic pool and real/synthetic schedules with conventional image transformations.

Both workflows use the [AK-inclusive GitHub manifest](../artifacts/isic2019_binary_split.csv). It contains 1,056 training, 353 validation and 352 test images, with AK assigned to the study's malignant group. Setup builds the GAN input from the locally supplied training originals. Preflight checks train and validation images, and test contents remain unopened.

The no-augmentation predictions for seeds 42, 43 and 44 match the 353 validation IDs, original diagnoses, binary labels and groups. The archived baseline configuration records Adam 0.001, TensorFlow 2.19.0 and 31 batches per epoch. The current baseline notebook source has a separate Adam 0.0001 setting. Use the matched real-only classifier at Adam 0.0001, TensorFlow 2.20.0 and 33 batches per epoch to estimate the augmentation effect under the current recipe.

See the [setup and execution instructions](../workflows/full_gan/START_HERE.md) or download the [code-only ZIP](../workflows/ISIC_Full_GAN_Workflows_Code_Only.zip). Training images, checkpoints and generated pools stay local.
