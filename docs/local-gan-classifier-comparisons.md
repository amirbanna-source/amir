# Full GAN and GAN + conventional: image-free GitHub handoff

The complete [source-only workflow](../workflows/full_gan/START_HERE.md) and [code-only ZIP](../workflows/ISIC_Full_GAN_Workflows_Code_Only.zip) are now available on GitHub. No images, resized-image archives or generated outputs are included.

- [14 — GAN](../notebooks/14_isic2019_real_plus_gan_local.ipynb): train the full conditional StyleGAN2-ADA run, generate a fixed pool and train the real + GAN CNN.
- [15 — GAN + conventional](../notebooks/15_isic2019_real_plus_gan_conventional_local.ipynb): reuse the same GAN/pool and train the CNN with conventional transformations on both sources.

Sandra points the notebook's `IMAGE_DIR` or shell `ISIC_IMAGE_DIR` to her existing original image folder. Setup restores the original fixed 1,632-image split and archived CNN starting weights from the existing pinned image-free reference archive, then builds the GAN training ZIP locally from the 972 original training images. The frozen split does not need manual repair. Only training/validation image contents are read.

There is no pilot stage or human-approval JSON requirement. Automatic integrity/runtime checks remain. The two conditions share a 2,000-kimg GAN and fixed 530/442 synthetic pool, with classifier seeds 42/43/44 and matching initial weights and source schedules. Full details are in the setup guide.

Verified: all required original inputs, training-source fingerprint, restored reference assets, source hashes, notebook/shell syntax and rebuilt GAN pixels identical to the previous export. Full GPU training, container pulls and actual campus execution are untested. Automated completion does not establish image quality or clinical validity.
