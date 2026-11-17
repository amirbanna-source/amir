# Full local GAN classifier workflows

Updated 17 November 2026. The earlier draft requiring a manually reviewed pool/approval JSON is superseded.

Use the two separate files:

- [14 — GAN](../notebooks/14_isic2019_real_plus_gan_local.ipynb): full 2,000-kimg StyleGAN2-ADA training, automatic 972-image pool generation and the real + GAN CNN.
- [15 — GAN + conventional](../notebooks/15_isic2019_real_plus_gan_conventional_local.ipynb): the same GAN/pool, with conventional transformations in CNN training.

The complete private package includes the fixed original 1,632-image split metadata and all three archived classifier initializations, so there is no manual frozen-file repair. All original 972 training +329 validation images are now included privately under `data/real/`, verified against frozen membership, training source hashes and RGB decoding. No additional image download or manual data copying is needed after extracting the updated complete package. The package has no pilot run and no manual approval JSON requirement. Automatic integrity and compatibility checks remain.

Read the [full setup and experiment recipe](../workflows/full_gan/START_HERE.md). Campus hardware, container setup and full model training have not been tested here. These are prepared workflows, not completed results. Image quality, clinical labels and near-copying are not established by automated execution.

The different current 1,761-image split is preserved unchanged and is not used by these workflows. No private CSV, images, starting-weight archives or complete package is added to this public repository by this update.
