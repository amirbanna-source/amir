# Reproduction and next steps

## Kaggle inputs

The working Kaggle notebook uses these attached datasets:

- `nodoubttome/skin-cancer9-classesisic` — image files.
- `amirsalahk/isic-2019-ground-truth` — ISIC 2019 official diagnosis CSV.
- `amirsalahk/isic-2019-metadata` — lesion metadata CSV.

The image dataset's mount path contains additional directory components. Discover paths from `/kaggle/input` rather than hard-coding the image root. The official CSV names are `ISIC_2019_Training_GroundTruth.csv` and `ISIC_2019_Training_Metadata.csv`.

## Saved rerun outputs (Sep 29)

The original session's `/kaggle/working` files disappeared after a reset. The split construction, image loading, and baseline training cells were rerun in [the Kaggle notebook](https://www.kaggle.com/code/amirsalahk/notebooke017076bfb/edit), then a new validation-only evaluation cell was added. Kaggle **Version 1, “ISIC 2019 baseline rerun and validation,”** was saved via Quick Save. All four output files were downloaded and committed to [artifacts](../artifacts/):

- `isic2019_verified_split.csv` — 1,761 rows, 1,050 train / 355 validation / 356 held-out test.
- `baseline_no_aug.keras` — best checkpoint at epoch 24 by validation loss.
- `baseline_validation_confusion_matrix.csv` — 355 validation cases.
- `baseline_validation_predictions.csv` — 355 validation predictions.

The first [archived exploratory notebook](../notebooks/01_isic2019_audit_split_baseline.ipynb) still records the *earlier* run. It retains historical attempts and session-reset recovery; the direct-URL ground-truth cell failed and later cells read the attached local CSV. Its `Run All` path has not been cleaned up. The current Kaggle draft/version contains the new evaluation cell. Do not attribute the earlier run's validation curve to the recovered checkpoint.

## Reusing the artifacts

The split CSV contains the original Kaggle mount paths. Attach the same three datasets in Kaggle and check these paths exist, or rebuild paths from `image_id` if mounts change. Keep image IDs and split assignments fixed for comparisons. Evaluate the selected checkpoint on the **validation** set with no augmentation; the test split is reserved for final evaluation. Model input is 128 × 128 RGB scaled to [0, 1], class order `AK, BCC, BKL, DF, MEL, NV, SCC, VASC`.

## Next experiment steps

1. Discuss the eight-class verified subset versus the original nine-class plan with the supervisor and record the decision.
2. Clean the exploratory notebook into a reproducible Run All workflow without the obsolete URL download or session-reset cells.
3. Apply conventional augmentation to training only, retrain the same classifier, and compare on the fixed validation set using accuracy, macro F1, and per-class metrics.
4. Specify GAN architecture, training-only input data, quality checks, number of generated images, and synthetic-to-real ratio before the GAN experiment.
5. Run GAN-only and combined conditions, select by validation, then evaluate the chosen approach once on the untouched test set.

External ISIC Archive images require overlap checks and compatible diagnoses before being treated as independent validation data.
