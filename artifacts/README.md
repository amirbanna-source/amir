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

## Paired three-seed study

- `paired_no_aug_artifacts.zip` and `paired_conventional_aug_artifacts.zip`: all three selected model checkpoints per condition, epoch histories, per-image validation predictions, confusion matrices, per-class metrics, configuration and preview.
- `paired_seed_results.csv`: independently verified results for all six runs.
- `paired_seed_summary.csv`: condition means and sample standard deviations.
- `paired_seed_differences.csv`: paired augmentation-minus-baseline differences.
- `augmentation_preview.png`: visually inspected training-only augmentation examples.

ZIP CRCs, validation IDs and labels, recomputed metrics, selected epochs and paired initialization hashes were checked. See [study report](../docs/paired-seed-study.md). No test predictions were generated.
