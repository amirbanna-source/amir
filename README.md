# Skin lesion classification: augmentation study

Master's thesis work by **Amir Albana**. This repository documents a controlled comparison of a CNN trained on verified ISIC 2019 skin lesion images with no augmentation, conventional image augmentation, GAN-generated images, and their combination. The no-augmentation baseline and first conventional augmentation experiment are complete. GAN-generated augmentation and the combined approach remain **planned**.

## Current status (29 September 2026)

- Audited the Kaggle [Skin Cancer ISIC dataset](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic), matched image IDs to ISIC 2019 Training Ground Truth, and retained 1,761 verified unique images in eight classes.
- Built a lesion-grouped training/validation/test split (1,050/355/356 images).
- Trained a CNN without augmentation. The original exploratory run selected epoch 11 (validation loss 1.6795, accuracy 28.17%) before its Kaggle checkpoint was lost after a session reset.
- Recreated the split and retrained the baseline in Kaggle. The new checkpoint selected epoch 24 by minimum validation loss **1.5410**. On 355 validation images, its accuracy is **41.41%**, macro F1 **0.4063**, and balanced accuracy **0.4611**. See [per-class results](docs/baseline.md).
- Saved the recovered [model checkpoint, split, and validation outputs](artifacts/). The **held-out test set has not been evaluated**.
- The [archived exploratory Kaggle notebook](notebooks/01_isic2019_audit_split_baseline.ipynb) records the earlier audit and run. The [recovered rerun notebook](notebooks/02_isic2019_baseline_rerun_validation.ipynb) preserves the baseline evaluation; the [Kaggle notebook](https://www.kaggle.com/code/amirsalahk/notebooke017076bfb/edit), saved as Version 1.

- Completed the first conventional augmentation run on the identical split: validation accuracy **29.01%**, macro F1 **0.2576**, balanced accuracy **0.3365**. The checkpoint selected epoch 12 by validation loss **1.6548**; early stopping ended training at epoch 17. This run scored below the baseline; the paired multi-seed comparison below is now complete. See [experiment report](docs/conventional-augmentation.md) and [executed notebook](notebooks/03_isic2019_conventional_augmentation.ipynb).

## Documentation

| File | Contents |
| --- | --- |
| [Data audit](docs/data-audit.md) | Duplicate/conflicting labels, official matching, and limitations |
| [Methodology](docs/methodology.md) | Verified subset, split, preprocessing, planned comparison, metrics |
| [Baseline](docs/baseline.md) | CNN setup and both training runs, validation metrics |
| [Conventional augmentation](docs/conventional-augmentation.md) | Protocol, validation comparison, per-class results and next step |
| [Artifacts](artifacts/README.md) | Recovered model, split, and validation outputs |
| [Reproduction and next steps](docs/reproduction.md) | Inputs, saved outputs, and work remaining |

## Data and scope

The original Kaggle folders are **not** used as experimental labels or as the final train/test split. Their copies can overlap and some image IDs occur with contradictory folder labels. Official labels come from the [ISIC 2019 challenge ground truth](https://challenge.isic-archive.com/data/#2019); lesion IDs come from its training metadata. Only image IDs matched to that ground truth were retained.

The project's initial proposal described nine classes. The verified subset currently has **eight classes**: AK, BCC, BKL, DF, MEL, NV, SCC, and VASC. This scope change remains to be discussed with the supervisor before finalizing the study design.

## Repository boundaries

Raw images and uploaded source CSV datasets are not included here. The archived notebook retains historical failed and restarted cells, so it is not a clean `Run All` workflow. The recovered split CSV contains Kaggle mount paths and may need path repair on a different mount. See [reproduction notes](docs/reproduction.md) before rerunning.

Research use only; this model is not a clinical diagnostic tool.

## Paired three-seed comparison

Completed six controlled runs (seeds 42, 43, 44). Mean validation accuracy: no augmentation **34.46%** versus conventional augmentation **36.62%**. Mean macro F1: **0.3156** versus **0.3477**. Augmentation improved macro F1 in two of three pairs; these preliminary results show substantial seed variation. Earlier exploratory runs are analyzed separately. All six checkpoints and run evidence are preserved. The test set remains untouched.

See [full paired study results](docs/paired-seed-study.md) and [executed notebook](notebooks/04_isic2019_paired_seed_study.ipynb).
