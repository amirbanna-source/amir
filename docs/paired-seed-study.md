# Paired three-seed comparison

Six controlled CNN runs were completed using seeds 42, 43 and 44 on the same 355-image validation set. Augmentation had a higher macro F1 in 2 of the three pairs, with a mean paired difference (augmentation minus baseline) of +0.0321. This is a preliminary comparison of this recipe and CNN on one fixed validation split; three seeds do not establish general statistical significance.

## Protocol and preview

The 1,050/355/356 split membership was checked against the archived fingerprint before training. Every pair started from identical CNN weight arrays, verified byte-for-byte and recorded with SHA-256 hashes. Both conditions use the same architecture, class weights, 128 × 128 RGB preprocessing, [0,1] scaling, batch size 32, Adam learning rate 0.001, sparse categorical cross-entropy, 25-epoch budget, early-stopping patience 5, and minimum-validation-loss selection. Augmentation is training-only: horizontal/vertical flips, ±18-degree rotations and zoom factor 0.10 with reflection filling and bilinear interpolation.

Seeds were set before model and dataset construction. Dropout seed 12345 was explicit in both conditions; augmentation seeds were the run seed +1000/+1001/+1002. Dataset ordering was configured deterministic. GPU kernels and differing stochastic operations may still prevent bitwise reproducibility of full training. TensorFlow version and configuration are saved with the evidence.

One training example per class was inspected before the runs. All eight shown lesions remained visible, without obvious severe clipping; interpolation blur was visible. This small preview is not a comprehensive check of all augmentations. [View the preview](../artifacts/augmentation_preview.png).

## All selected checkpoints

| Seed | Condition | Epochs trained | Selected epoch | Validation loss | Accuracy | Macro F1 | Balanced accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | conventional_aug | 25 | 20 | 1.5416 | 0.3549 | 0.3615 | 0.4095 |
| 42 | no_aug | 17 | 12 | 1.6566 | 0.3127 | 0.2686 | 0.3101 |
| 43 | conventional_aug | 17 | 12 | 1.6450 | 0.3239 | 0.2690 | 0.3155 |
| 43 | no_aug | 17 | 12 | 1.6426 | 0.3437 | 0.2735 | 0.3268 |
| 44 | conventional_aug | 25 | 24 | 1.4803 | 0.4197 | 0.4125 | 0.4582 |
| 44 | no_aug | 25 | 23 | 1.5115 | 0.3775 | 0.4047 | 0.4392 |

## Mean and variability across seeds

| Metric | No augmentation, mean ± sample SD | Conventional augmentation, mean ± sample SD |
| --- | ---: | ---: |
| accuracy | 0.3446 ± 0.0324 | 0.3662 ± 0.0489 |
| macro_f1 | 0.3156 ± 0.0772 | 0.3477 ± 0.0727 |
| balanced_accuracy | 0.3587 ± 0.0702 | 0.3944 ± 0.0725 |

These sample standard deviations describe variability across three training seeds on the same validation set. They are not confidence intervals for performance on new patients. Known lesions were grouped within one split, but 324 images lack lesion IDs and patient IDs are unavailable.

## Paired macro F1 differences

| Seed | Augmentation minus baseline |
| --- | ---: |
| 42 | +0.0929 |
| 43 | -0.0045 |
| 44 | +0.0078 |

The exploratory baseline and first augmentation run remain separate in their earlier reports. The paired workflow changed random initialization controls and the dropout seed, so its baselines should not be pooled with those exploratory runs. Every run was retained; no seed was excluded and no recipe was changed after inspecting these results. The 356-image held-out test set remains unevaluated.

## Reproduction and evidence

- [Executed two-cell notebook](../notebooks/04_isic2019_paired_seed_study.ipynb)
- [Study script](../scripts/paired_seed_study.py)
- [All six results](../artifacts/paired_seed_results.csv)
- [Mean and sample SD](../artifacts/paired_seed_summary.csv)
- [Paired differences](../artifacts/paired_seed_differences.csv)
- [No-augmentation archive](../artifacts/paired_no_aug_artifacts.zip)
- [Conventional augmentation archive](../artifacts/paired_conventional_aug_artifacts.zip)

Each archive contains three selected model checkpoints, each run's epoch history, validation image predictions, confusion matrix and per-class metrics, plus the common configuration, results and preview. Archive CRCs, nested Keras ZIPs, prediction IDs and labels, metrics recalculation, history selection and paired initial-weight hashes were checked independently after download.

## Next step

Review the full per-class results and learning curves with Sandra, agree how to report the observed variability, and finalize the eight-class scope with the supervisor. Define the GAN architecture, training-only data, synthetic image count, quality checks and synthetic-to-real ratio before starting the GAN comparison. Keep the final test set reserved for the planned final evaluation.
