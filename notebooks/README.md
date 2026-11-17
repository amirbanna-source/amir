# Notebooks

## Current workflows: source only, no pilot

| Notebook | Full Linux/NVIDIA workflow |
| --- | --- |
| [14 — GAN](14_isic2019_real_plus_gan_local.ipynb) | Full StyleGAN2-ADA → synthetic pool → real + GAN CNN |
| [15 — GAN + conventional](15_isic2019_real_plus_gan_conventional_local.ipynb) | Reuse the same GAN/pool → CNN with conventional transformations |

Download the [code-only ZIP](../workflows/ISIC_Full_GAN_Workflows_Code_Only.zip), or clone this repository and use [workflows/full_gan](../workflows/full_gan/START_HERE.md). **No images or image-containing archives are included.** Sandra supplies the existing original image folder by setting `IMAGE_DIR` in the notebook or `ISIC_IMAGE_DIR` for the shell launcher.

Setup automatically retrieves the already published image-free classifier reference assets. The original fixed split and starting weights are restored; GAN-format resized inputs are built locally from Sandra's images. No manual frozen-file repair, 20-kimg pilot or approval JSON is required.

The full GAN budget is 2,000 kimg. Both classifier conditions reuse its pool and paired initializations. Run sequentially inside an allocated GPU session. The source-only data adapter was verified against all 1,301 originals: all 972 resized GAN images match the previous export pixel-for-pixel. Actual campus GPU/container execution remains untested.

## Supporting historical setup

- [11 — Official setup record](11_isic2019_official_stylegan2ada.ipynb): chronological export and compatibility evidence.
- [12 — Runtime verification](12_stylegan2ada_runtime_verification.ipynb): standalone diagnostic reference; not an additional required notebook before the full workflows.

## Archive

[Archive index](archive/README.md): the old custom GAN notebooks 09/10 and superseded 20-kimg pilot notebook 13 are retained as history, outside the active workflow.

## Other notebooks

- [01 — Binary split and baseline record](01_isic2019_binary_split.ipynb)
- [02 — CNN creation and unaugmented-data workflow](02_isis2019_Cnn_creation_and_execution_on_unaugmented_dataset.ipynb)
- [03 — Conventional augmentation experiment](03_isic2019_conventional_augmentation.ipynb)

The `Random_*_test.ipynb` files are individual transformation experiments. Filenames alone do not establish that their settings match the current binary comparison; consult the [Methods](../docs/methodology.md) and [Results](../docs/results.md) for the consolidated study record.

Training images and the complete image-containing campus ZIP must remain off GitHub and be shared separately.
