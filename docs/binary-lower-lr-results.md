# Lower learning rate binary baseline — 14 October 2026

Reducing Adam learning rate from 0.001 to 0.0001 improved validation accuracy and macro F1 in all three matched seeds. Malignant detection recovered, with a tradeoff in specificity. This is exploratory tuning on the same validation cohort; the final test set remains untouched.

## Controlled comparison

Only the learning rate was changed. Both conditions used the same 972 training / 329 validation images, original split membership, seed-specific initial weights, CNN architecture, Dropout0.35 with seed12345, RGB128×128 /255 preprocessing, class weights, batch32, maximum25 epochs, patience5, minimum-validation-loss checkpoint and threshold0.5. Initialization hashes matched the archived baseline for every seed. TensorFlow2.20.0; Kaggle Tesla T4, ordinary single-device Keras training. No augmentation.

The fixed script was committed before execution. All three revised runs reached the 25-epoch budget; selected epochs were25,23,24. The budget was not extended. In particular, seed42 selected the final epoch, so convergence is not established.

## Selected-checkpoint validation results

| Seed | Selected epoch / trained | BCE | Accuracy | Macro F1 | Balanced accuracy | Malignant sensitivity | Specificity | ROC-AUC | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 25 / 25 | 0.6394 | 0.6505 | 0.6452 | 0.6447 | 0.5800 | 0.7095 | 0.6885 | 0.6555 |
| 43 | 23 / 25 | 0.6509 | 0.6201 | 0.6187 | 0.6195 | 0.6133 | 0.6257 | 0.6642 | 0.6265 |
| 44 | 24 / 25 | 0.6487 | 0.6140 | 0.6140 | 0.6182 | 0.6667 | 0.5698 | 0.6775 | 0.6538 |

## Paired summary

Values below are mean ± sample SD over the same three seeds. Deltas are lower rate minus original rate in metric units, not relative percentages. These SDs are seed variability, not patient-level confidence intervals.

| Metric | Original Adam0.001 | Revised Adam0.0001 | Mean paired delta |
| --- | ---: | ---: | ---: |
| accuracy | 0.5451 ± 0.0018 | 0.6282 ± 0.0195 | +0.0831 |
| macro_f1 | 0.3548 ± 0.0042 | 0.6259 ± 0.0169 | +0.2711 |
| balanced_accuracy | 0.5011 ± 0.0019 | 0.6275 ± 0.0149 | +0.1264 |
| malignant_sensitivity | 0.0022 ± 0.0038 | 0.6200 ± 0.0437 | +0.6178 |
| specificity | 1.0000 ± 0.0000 | 0.6350 ± 0.0703 | -0.3650 |
| roc_auc | 0.5969 ± 0.1575 | 0.6767 ± 0.0122 | +0.0799 |
| average_precision | 0.5772 ± 0.1570 | 0.6453 ± 0.0163 | +0.0680 |

Mean accuracy rose by8.31 percentage points, macro F1 by0.2711 and malignant sensitivity by61.78 percentage points. Mean specificity decreased by36.50 percentage points. Seed44 ROC-AUC decreased despite better thresholded classification; ranking did not improve in every seed. These models now predict both classes, but the results remain modest and are not evidence of clinical suitability or of an augmentation benefit.

## Evidence and verification

- [Executed notebook](../notebooks/07_isic2019_binary_lower_lr.ipynb)
- [Fixed training script](../scripts/binary_lr0001.py) and [protocol](binary-lower-lr-protocol.md)
- [All three model checkpoints, initial weights, histories, validation predictions, metrics, confusion matrices, manifest and configuration](../artifacts/binary_lr0001_all_artifacts.zip)
- [Results CSV](../artifacts/binary_lr0001_results.csv), [paired deltas](../artifacts/binary_lr0001_paired_comparison.csv), [runtime configuration](../artifacts/binary_lr0001_config.json)
- [Original weak baseline](binary-baseline.md) and [training-only diagnostic](binary-training-diagnostics.md)

ZIP CRC and nested Keras archive checks passed. Independently verified all329 validation IDs and original/binary labels, recalculated thresholded predictions and seven metrics, checked confusion matrices, minimum-loss epoch selection, identical initialization hashes and split fingerprint. The exported executed code exactly matches the committed script; architecture, loader and dataset functions also match the original script. No test image dataset was constructed or evaluated.

## Next step

Use this revised configuration as the fixed candidate for a paired conventional-augmentation comparison across seeds42/43/44, keeping initial weights, split, training budget, checkpoint rule and evaluation unchanged between conditions. Record the augmentation protocol before running it; retain the original-rate baseline as a separate historical optimizer condition. GAN experiments have not started. Do not claim a GAN or augmentation improvement from optimizer tuning. Repeated validation-guided development can introduce selection optimism; reserve all331 test images for the final frozen comparison. The324 missing lesion IDs and unavailable patient IDs remain limitations.
