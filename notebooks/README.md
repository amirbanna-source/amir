# Notebooks

## Current workflows: no pilot stage

| Notebook | Full local/cluster workflow |
| --- | --- |
| [14 — GAN](14_isic2019_real_plus_gan_local.ipynb) | Train official conditional StyleGAN2-ADA for 2,000 kimg, generate a fixed pool, then train the real + GAN CNN |
| [15 — GAN + conventional](15_isic2019_real_plus_gan_conventional_local.ipynb) | Reuse that GAN/pool and train the CNN with conventional transformations on both real and synthetic images |

Run either notebook from the complete extracted private **ISIC_Full_GAN_Workflows** package inside an allocated Linux/NVIDIA GPU session. The equivalent shell entry points are `GAN.sh` and `GAN_PLUS_CONVENTIONAL.sh`. The second condition can create the same full GAN/pool if absent. Run sequentially.

There is no 20-kimg pilot stage and no manual approval JSON requirement. Integrity/runtime checks run automatically. The private package supplies the original fixed 1,632-image split and archived CNN starting weights; all original 972 training +329 validation images are now included under `data/real/` in the private ZIP. The data handoff is complete; all 972 training source hashes and all 1,301 original image decodes were verified. Cluster allocation/container support remain site-specific and untested here.

[Full setup, inputs, recipe and limitations](../workflows/full_gan/START_HERE.md). These workflows are prepared, not executed. They do not establish image quality or clinical validity by completing. Validation stays real-only and test-image contents are never loaded.

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
