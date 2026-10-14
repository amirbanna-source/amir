# Skin lesion classification: benign versus malignant augmentation study

Master's thesis work by **Amir Albana**. The current task is **binary classification**: benign (NV/BKL/DF/VASC) versus malignant (MEL/BCC/SCC), excluding AK. The study compares no augmentation, conventional augmentation, GAN augmentation and their combination.

## Current binary status (14 October 2026)

The verified binary subset contains **1,632 images**: **972 training / 329 validation / 331 test**. Original split membership is preserved. Class counts are 894 benign and 738 malignant. The same 324 images lack lesion IDs; patient IDs remain unavailable. The test set has not been evaluated.

The [binary methodology](docs/methodology.md), [split manifest](artifacts/isic2019_binary_split.csv), [exclusion manifest](artifacts/isic2019_binary_excluded_ak.csv) and [baseline notebook](notebooks/05_isic2019_binary_baseline.ipynb) are prepared. **Three binary baseline runs are complete.** Mean validation accuracy was 54.51%, macro F1 0.3548 and malignant sensitivity 0.22% at threshold 0.5. These checkpoints largely predict benign and require diagnosis before augmentation. See the [full binary results](docs/binary-baseline.md); all three checkpoints and validation evidence are [archived](artifacts/binary_baseline_all_artifacts.zip). The [GAN protocol](docs/gan-protocol-draft.md) is revised for two classes and remains a proposal. The project-plan scope change should be discussed with the supervisor.

Training-only [CNN diagnostics](docs/binary-training-diagnostics.md) are complete. On a fixed balanced 32-image training subset, reducing Adam LR from 0.001 to 0.0001 improved final training accuracy from 81.25% to 93.75% after 300 updates. Batch and initial gradient checks passed. These are subset training metrics, not validation improvements. Next: controlled three-seed full-data baseline revision at the lower rate; test remains untouched.\n\n## Historical eight-class experiments

The following records describe the previous task. Their performance values are not binary results.

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
