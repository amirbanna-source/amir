# Skin lesion classification: augmentation study

Master's thesis work by **Amir Albana**. This repository documents a controlled comparison of a CNN trained on verified ISIC 2019 skin lesion images with no augmentation, conventional image augmentation, GAN-generated images, and their combination. The latter three experiments are **planned**; no comparative results are claimed yet.

## Current status (29 September 2026)

- Audited the Kaggle [Skin Cancer ISIC dataset](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic), matched image IDs to ISIC 2019 Training Ground Truth, and retained 1,761 verified unique images in eight classes.
- Built a lesion-grouped training/validation/test split (1,050/355/356 images).
- Trained an initial CNN without augmentation. The checkpoint with lowest validation loss achieved **1.680 validation loss and 28.17% validation accuracy** at epoch 11; training stopped after epoch 16.
- Macro F1, per-class metrics, and the held-out test result have **not yet been computed**.

## Documentation

| File | Contents |
| --- | --- |
| [Data audit](docs/data-audit.md) | Duplicate/conflicting labels, official matching, and limitations |
| [Methodology](docs/methodology.md) | Verified subset, split, preprocessing, planned comparison, metrics |
| [Baseline](docs/baseline.md) | CNN setup and observed training outcome |
| [Reproduction and next steps](docs/reproduction.md) | Inputs, artifacts, notebook export, and work remaining |

## Data and scope

The original Kaggle folders are **not** used as experimental labels or as the final train/test split. Their copies can overlap and some image IDs occur with contradictory folder labels. Official labels come from the [ISIC 2019 challenge ground truth](https://challenge.isic-archive.com/data/#2019); lesion IDs come from its training metadata. Only image IDs matched to that ground truth were retained.

The project's initial proposal described nine classes. The verified subset currently has **eight classes**: AK, BCC, BKL, DF, MEL, NV, SCC, and VASC. This scope change remains to be discussed with the supervisor before finalizing the study design.

## Repository boundaries

Raw images and uploaded CSV datasets are not included here. The Kaggle notebook is still in Kaggle and needs to be exported into `notebooks/` before the full experiment is reproducible from this repository. The current documents record results supplied by the running Kaggle notebook; they are not a substitute for its executable code and saved split/model outputs.

Research use only; this model is not a clinical diagnostic tool.
