# Per-class review of paired augmentation study

Reviewed 6 October 2026. Results refer only to the fixed 355-image validation set and seeds 42, 43, 44. Means and sample SDs summarize training-seed variability, not independent patient cohorts.

| Class | Validation images | Baseline F1 mean ± SD | Augmentation F1 mean ± SD | Mean paired change |
| --- | ---: | ---: | ---: | ---: |
| AK | 26 | 0.0866 ± 0.0864 | 0.1533 ± 0.1403 | +0.0667 |
| BCC | 72 | 0.4567 ± 0.0432 | 0.4830 ± 0.0384 | +0.0263 |
| BKL | 94 | 0.1413 ± 0.0846 | 0.1668 ± 0.1014 | +0.0255 |
| DF | 22 | 0.0641 ± 0.1110 | 0.0936 ± 0.0836 | +0.0295 |
| MEL | 28 | 0.4713 ± 0.0316 | 0.6041 ± 0.0471 | +0.1328 |
| NV | 34 | 0.6016 ± 0.0852 | 0.4614 ± 0.1391 | -0.1402 |
| SCC | 50 | 0.2614 ± 0.0865 | 0.2217 ± 0.0531 | -0.0397 |
| VASC | 29 | 0.4420 ± 0.3157 | 0.5976 ± 0.2720 | +0.1556 |

## Interpretation

Mean F1 increased most for VASC (+0.1556) and MEL (+0.1328), while NV (-0.1402) and SCC (-0.0396) decreased. These are descriptive changes across only three seeds. VASC has substantial variability; AK, BKL and DF remain low-performing in both conditions. MEL mean recall increased from 0.4048 to 0.6905, while NV fell from 0.6863 to 0.4902 and SCC from 0.3467 to 0.2133. The overall macro-F1 gain therefore hides class-specific tradeoffs.

## Learning-curve observations

| Condition | Seed | Selected epoch / total | Training loss at selection | Validation loss at selection | Final training / validation loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_aug | 42 | 12 / 17 | 1.6279 | 1.6566 | 1.5309 / 1.8182 |
| no_aug | 43 | 12 / 17 | 1.6290 | 1.6426 | 1.5762 / 1.6751 |
| no_aug | 44 | 23 / 25 | 1.3641 | 1.5115 | 1.3626 / 1.6821 |
| conventional_aug | 42 | 20 / 25 | 1.5729 | 1.5416 | 1.4113 / 1.6991 |
| conventional_aug | 43 | 12 / 17 | 1.6257 | 1.6450 | 1.5721 / 1.7120 |
| conventional_aug | 44 | 24 / 25 | 1.3846 | 1.4803 | 1.4082 / 1.5890 |

Training loss uses class weighting; validation loss does not. Direct train-versus-validation loss gaps should not be interpreted as pure overfitting measurements. Both seed-44 runs and augmented seed 42 reached the 25-epoch budget; this does not establish convergence. No model or augmentation settings were changed during this review.

## Next decisions

Keep the existing split and test set reserved. Confirm the eight-class scope with the supervisor. Specify the GAN protocol before training: architecture, compute budget, training-only fitting, class-conditional sampling, synthetic-to-real ratios, duplicate/memorization checks, diversity and label-quality assessment, and matched classifier comparisons across seeds. Inspect results for every class rather than assuming synthetic data helps all minority classes. Any image-quality threshold must be defined independently of the held-out test set.

Source evidence: [paired study report](paired-seed-study.md), including both archived run bundles.
