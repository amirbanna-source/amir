# Binary no-augmentation baseline results

Completed 14 October 2026. All three prespecified seeds were retained. The baseline performs poorly at the fixed 0.5 threshold: seeds 42 and 43 predict every validation image as benign, and seed 44 detects only one of 150 malignant images. This is a weak baseline requiring diagnosis before augmentation experiments, not evidence that binary classification or GAN augmentation cannot work.

## Dataset and training

Benign (NV/BKL/DF/VASC) versus malignant (MEL/BCC/SCC), excluding AK. Frozen inherited split: 972 training, 329 validation and 331 untouched test images. Class weights: benign 0.916981, malignant 1.099548. The same 324 images lack lesion IDs and patient IDs are unavailable.

Four convolutional blocks (32/64/128/256 filters), global average pooling, Dense128, Dropout0.35 (seed12345), one sigmoid output. Inputs: 128 x 128 RGB scaled to [0,1]. Binary cross-entropy; Adam0.001; batch32; 31 steps per epoch; at most25 epochs; early-stopping patience5. Minimum unweighted validation loss selects each checkpoint. No augmentation, threshold tuning or test evaluation was performed. TensorFlow 2.20.0. Kaggle detected two Tesla T4 devices; the script used ordinary single-device Keras training, not a multi-GPU distribution strategy.

## Validation results at threshold 0.5

| Seed | Selected epoch / total | Val loss | Accuracy | Macro F1 | Balanced accuracy | Malignant sensitivity | Specificity | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 9 / 14 | 0.6845 | 0.5441 | 0.3524 | 0.5000 | 0.0000 | 1.0000 | 0.6574 |
| 43 | 10 / 15 | 0.6902 | 0.5441 | 0.3524 | 0.5000 | 0.0000 | 1.0000 | 0.4181 |
| 44 | 12 / 17 | 0.6880 | 0.5471 | 0.3597 | 0.5033 | 0.0067 | 1.0000 | 0.7152 |

| Metric | Mean ± sample SD across seeds |
| --- | ---: |
| accuracy | 0.5451 ± 0.0018 |
| macro_f1 | 0.3548 ± 0.0042 |
| balanced_accuracy | 0.5011 ± 0.0019 |
| malignant_sensitivity | 0.0022 ± 0.0038 |
| specificity | 1.0000 ± 0.0000 |
| roc_auc | 0.5969 ± 0.1575 |
| average_precision | 0.5772 ± 0.1570 |

Validation contains 179 benign and 150 malignant images. An always-benign rule gives accuracy 0.5441, macro F1 0.3524 and balanced accuracy0.5; its malignant sensitivity is zero. The three checkpoints are essentially at that classification baseline. Specificity1.0 is a consequence of predicting almost no malignancies and must not be presented as successful detection. Sample SDs describe seed variability on the same validation cohort, not patient-level confidence intervals.

## Probability diagnostics

| Seed | Minimum | Median | Maximum malignant score |
| --- | ---: | ---: | ---: |
| 42 | 0.316230 | 0.494117 | 0.499321 |
| 43 | 0.466137 | 0.472488 | 0.476453 |
| 44 | 0.382857 | 0.491035 | 0.500868 |

Scores cluster close to, and mostly below,0.5. ROC-AUC shows some ranking signal in seeds42 and44 but poor ranking in seed43. Thus this is not simply a universal threshold problem. Threshold optimization after seeing these outputs would be exploratory and must remain separate from the fixed-threshold results. A constant probability equal to training malignant prevalence (442/972) gives validation BCE approximately0.68926; selected model losses are only marginally different. The cause of weak learning is not established by these outputs.

## Verification and evidence

Downloaded ZIP CRC and nested Keras checkpoint integrity checks passed. Independently matched all329 prediction IDs and original/binary labels to the frozen validation manifest; recomputed thresholded predictions, confusion matrices and metrics; checked minimum-loss epoch selection and initial weight hashes. The exported executed cell exactly matches the prepared script.

- [Executed notebook](../notebooks/05_isic2019_binary_baseline.ipynb)
- [Training script](../scripts/binary_baseline.py)
- [All three model checkpoints, initial weights, histories, predictions, metrics and run config](../artifacts/binary_baseline_all_artifacts.zip)
- [Run results CSV](../artifacts/binary_baseline_results.csv)
- [Probability diagnostics](../artifacts/binary_probability_diagnostics.csv)
- [Current methodology](methodology.md)

## Next step before GAN

Perform training-only diagnostics: verify image/target batches and gradients, check whether the CNN can overfit a small fixed training subset, and inspect preprocessing and optimization. Predeclare a modest optimizer or architecture revision if needed, then run new binary baselines across seeds with all changes logged. Preserve this failed/weak baseline; do not overwrite it or choose a lucky seed. The final test set stays sealed. GAN and conventional-augmentation comparisons have not yet run for the binary task.
