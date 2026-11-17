# Notebooks

## Current GAN workflow

**[Notebook 13: StyleGAN2-ADA pilot](13_isic2019_stylegan2ada_pilot.ipynb) is the main entry point.** Run it inside the complete extracted campus package on the documented GPU environment. The repository copy alone does not include its training images or bundled source. See the [campus run instructions](../handoff/stylegan_pilot/README.md).

| Notebook | Role | When to use |
| --- | --- | --- |
| [13 — StyleGAN2-ADA pilot](13_isic2019_stylegan2ada_pilot.ipynb) | Current packaged pilot | Run from the complete campus package; automatic checks precede training |
| [11 — Official setup record](11_isic2019_official_stylegan2ada.ipynb) | Historical export, dry-run and compatibility evidence | Consult for provenance; historical cell order is not a clean launch workflow |
| [12 — Runtime verification](12_stylegan2ada_runtime_verification.ipynb) | Standalone troubleshooting tool | Consult for separate CUDA checks; not an additional required step before notebook 13 |

StyleGAN2-ADA still uses benign/malignant class conditioning. It replaces the earlier custom GAN implementation. You do not need to run all GAN notebooks in sequence.

## Local classifier comparisons using synthetic images

These are separate classifier conditions, after a reviewed StyleGAN2-ADA pool is available. They do not retrain the GAN.

| Notebook | Local training condition |
| --- | --- |
| [14 — Real + GAN](14_isic2019_real_plus_gan_local.ipynb) | Mixed real/synthetic training images, no conventional transformations |
| [15 — Real + GAN + conventional](15_isic2019_real_plus_gan_conventional_local.ipynb) | The same mixed pool and schedule, with conventional transformations on both sources |

Both are prepared, not executed, and run in local Jupyter or VS Code. See [local setup, frozen split and pool requirements](../docs/local-gan-classifier-comparisons.md). They use the separately named frozen 1,632-image manifest matching the completed study, not the different current split CSV. Real validation stays unaugmented; test-image contents are never loaded.

## Archived custom GAN experiments

[Archive index](archive/README.md): notebooks 09 and 10 preserve the completed custom conditional GAN pilot and its mode diagnostic. They are not part of the active StyleGAN2-ADA workflow. Code cells and recorded outputs have been preserved. Do not rerun them as setup for the current pilot or use the old synthetic pool for classifier augmentation.

## Other notebooks

- [01 — Binary split and baseline record](01_isic2019_binary_split.ipynb)
- [02 — CNN creation and unaugmented-data workflow](02_isis2019_Cnn_creation_and_execution_on_unaugmented_dataset.ipynb)
- [03 — Conventional augmentation experiment](03_isic2019_conventional_augmentation.ipynb)

The `Random_*_test.ipynb` files are individual transformation experiments. Filenames alone do not establish that their settings match the current binary comparison; consult the [Methods](../docs/methodology.md) and [Results](../docs/results.md) for the consolidated study record.

Training images and the complete image-containing campus ZIP must remain off GitHub and be shared separately.
