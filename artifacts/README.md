# Experiment artifacts

The verified split contains 1,050 training, 355 validation and 356 test images. The test set has not been evaluated.

## No augmentation baseline

- `isic2019_verified_split.csv`: frozen split membership, official diagnoses, groups and Kaggle image paths.
- `baseline_no_aug.keras`: recovered baseline checkpoint; epoch 24 selected by validation loss.
- `baseline_validation_confusion_matrix.csv` and `baseline_validation_predictions.csv`: validation-only outputs.

## Conventional augmentation

- `cnn_conventional_aug.keras`: CNN checkpoint selected at epoch 12; training stopped after 17 epochs. Online augmentation is outside the saved CNN and applied only to training batches.
- `conventional_aug_metrics.json`: exact metrics, per-class report and experiment settings.
- `conventional_aug_history.csv`: all 17 training/validation epoch metrics.
- `conventional_aug_validation_confusion_matrix.csv`: rows true, columns predicted.
- `conventional_aug_validation_predictions.csv`: 355 validation image IDs, labels, predictions and maximum softmax scores.

See the [augmentation report](../docs/conventional-augmentation.md) for the comparison and limitations. Raw images are not mirrored here. Mount the original Kaggle datasets to use their paths or resolve images by ID. These are research checkpoints.
