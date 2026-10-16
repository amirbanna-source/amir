# Binary conventional augmentation — 16 October 2026

This is the controlled augmentation comparison against the revised Adam0.0001 baseline. All three seeds are retained. These are exploratory validation results on the same cohort; the 331 test images were not evaluated.

## Protocol and pairing

The protocol and script were committed before execution. Only training augmentation changes: horizontal/vertical flips, random rotations up to 18 degrees, and independent height/width zoom factors (-0.1,0.1), with reflect fill. Per-layer seeds use run seed+100/+200/+300. A sequential dataset map applies augmentation to training batches only; validation uses the original loader with no augmentation. The saved classifier contains no augmentation layers.

Preserved: 972 training / 329 validation images, original grouped split, seed-specific initial weights, four-convolution CNN, 128×128 RGB /255, Dropout 0.35 (seed 12345), training-derived class weights, batch 32, 31 steps per epoch, Adam 0.0001, 25 maximum epochs, early-stopping patience 5, minimum-validation-loss checkpoint and threshold 0.5. No resampling, GAN images, threshold tuning or extra training budget. TensorFlow 2.20.0 and Kaggle T4. Initialization and split hashes matched the archived baseline for all three runs.

## Validation results

| Seed | Selected / trained epochs | BCE | Accuracy | Macro F1 | Balanced accuracy | Malignant sensitivity | Specificity | ROC-AUC | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 21 / 25 | 0.6448 | 0.6383 | 0.6348 | 0.6347 | 0.5933 | 0.6760 | 0.6817 | 0.6544 |
| 43 | 23 / 25 | 0.6527 | 0.6353 | 0.6316 | 0.6313 | 0.5867 | 0.6760 | 0.6593 | 0.6283 |
| 44 | 24 / 25 | 0.6516 | 0.6170 | 0.6169 | 0.6205 | 0.6600 | 0.5810 | 0.6737 | 0.6407 |

## Paired comparison against no augmentation

Means ± sample SD across seeds 42/43/44. Paired delta = augmentation minus baseline. The SD describes seed variation on one validation cohort, not patient-level uncertainty. Macro F1 is the primary descriptive comparison.

| Metric | No augmentation | Conventional augmentation | Mean paired delta |
| --- | ---: | ---: | ---: |
| accuracy | 0.6282 ± 0.0195 | 0.6302 ± 0.0115 | +0.0020 |
| macro_f1 | 0.6259 ± 0.0169 | 0.6278 ± 0.0095 | +0.0018 |
| balanced_accuracy | 0.6275 ± 0.0149 | 0.6288 ± 0.0074 | +0.0013 |
| malignant_sensitivity | 0.6200 ± 0.0437 | 0.6133 ± 0.0406 | -0.0067 |
| specificity | 0.6350 ± 0.0703 | 0.6443 ± 0.0548 | +0.0093 |
| roc_auc | 0.6767 ± 0.0122 | 0.6716 ± 0.0113 | -0.0052 |
| average_precision | 0.6453 ± 0.0163 | 0.6412 ± 0.0131 | -0.0041 |

Macro F1 improved in **2/3 matched seeds**. All seed-level differences are retained below; no seed is selected for reporting.

| Seed | Accuracy delta | Macro F1 delta | Sensitivity delta | Specificity delta | ROC-AUC delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | -0.0122 | -0.0104 | +0.0133 | -0.0335 | -0.0069 |
| 43 | +0.0152 | +0.0129 | -0.0267 | +0.0503 | -0.0049 |
| 44 | +0.0030 | +0.0030 | -0.0067 | +0.0112 | -0.0037 |

## Original diagnosis breakdown

Mean fraction correctly classified into benign/malignant across seeds. These are binary predictions within each original diagnosis, not a seven-class classifier. Small subtype counts limit interpretation.

| Original diagnosis | Validation images | No augmentation | Augmentation |
| --- | ---: | ---: | ---: |
| BCC | 72 | 0.6019 | 0.6019 |
| BKL | 94 | 0.6348 | 0.6560 |
| DF | 22 | 0.5909 | 0.6061 |
| MEL | 28 | 0.7738 | 0.7857 |
| NV | 34 | 0.5098 | 0.4902 |
| SCC | 50 | 0.5600 | 0.5333 |
| VASC | 29 | 0.8161 | 0.8161 |

These results do not establish a general augmentation benefit or clinical suitability. Transformations, training budget and checkpoint selection are a single fixed candidate; three seeds do not justify a significance claim. Repeated validation-guided development can introduce selection optimism. Missing lesion IDs for 324 retained images and unavailable patient IDs prevent confirmation of complete patient-level independence. No external validation or GAN experiment has been performed.

## Verification and evidence

Archive CRC and nested Keras archive integrity passed. All 329 prediction IDs and original/binary labels matched the frozen validation manifest. Threshold predictions, confusion matrices and seven metrics were independently recalculated. Epoch histories confirmed the minimum-validation-loss checkpoint selection. Initial arrays and split hashes matched the paired baseline. Exported executed source was checked against the prepared script.

- [Executed notebook](../notebooks/08_isic2019_binary_conventional_augmentation.ipynb)
- [Training script](../scripts/binary_conventional_aug.py) and [predeclared protocol](binary-conventional-augmentation-protocol.md)
- [All three checkpoints and supporting evidence](../artifacts/binary_conventional_aug_all_artifacts.zip)
- [Original-diagnosis results](../artifacts/binary_conventional_aug_diagnosis_subtypes.csv)
- [Run metrics](../artifacts/binary_conventional_aug_results.csv), [paired deltas](../artifacts/binary_conventional_aug_paired_comparison.csv), [summary](../artifacts/binary_conventional_aug_summary.csv), [runtime config](../artifacts/binary_conventional_aug_config.json)
- [Paired no-augmentation baseline](binary-lower-lr-results.md)

## Next step

Review this comparison with the baseline before freezing the classifier protocol for GAN experiments. Preserve these runs regardless of direction. Any subsequent GAN pilot must use training images only; hold validation/test images out of generation, audit synthetic copies and memorization, and match classifier update budgets across augmentation conditions. Test evaluation remains reserved until the final study design is frozen.
