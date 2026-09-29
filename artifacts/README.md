# Baseline artifacts

This folder contains the 2026-09-29 rerun of the no-augmentation CNN baseline on the 1,761 verified ISIC 2019 images. The split is stratified by diagnosis and grouped by lesion ID when available. Its validation set contains 355 images; the 356 test images have not been evaluated.

- `isic2019_verified_split.csv`: training, validation, and test image ID assignments (with Kaggle input paths and lesion metadata).
- `baseline_no_aug.keras`: best checkpoint by validation loss, trained for 25 epochs; selected at epoch 24.
- `baseline_validation_confusion_matrix.csv`: 8 × 8 confusion matrix, rows true and columns predicted.
- `baseline_validation_predictions.csv`: validation image IDs, official labels, predicted labels, and maximum softmax scores.

This checkpoint is a research baseline, not a clinical diagnostic model. The input image dataset itself is not mirrored here; mount the original Kaggle inputs to use the saved paths or rebuild paths from image IDs.
