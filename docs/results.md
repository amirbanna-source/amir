# Results: binary skin-lesion augmentation study

Working thesis Results chapter, prepared 13 November 2026. This chapter consolidates completed experiment records; it reports no new training or test evaluation. Procedures are described in the [Methods chapter](methodology.md). Pending experiments are marked explicitly.

## 1. Evaluated cohort and reporting scope

The audited source cohort contained 1,761 unique images with verified ISIC 2019 diagnoses. Excluding 129 AK images produced the binary cohort of 1,632 images. The inherited split contained 972 training images, 329 validation images and 331 reserved test images. The validation set comprised 179 benign and 150 malignant images.

All classifier results below refer to this same real-image validation cohort. The malignant-score threshold was fixed at 0.5. All three seeds (42, 43 and 44) were retained; each run was represented by its minimum-validation-loss checkpoint. The 331 test images were not evaluated. Earlier eight-class results are historical and are not pooled with these binary results.

Mean ± sample standard deviation summarizes training-seed variability on this fixed cohort. It does not represent patient-level confidence intervals. Repeated validation-guided development makes these exploratory estimates.

## 2. Classifier performance overview

Table 1 summarizes the original optimizer condition, the revised no-augmentation reference, and the paired conventional-augmentation condition. The original Adam 0.001 condition is included to document optimizer development; the revised Adam 0.0001 condition is the reference for the augmentation comparison.

**Table 1. Selected-checkpoint validation performance, mean ± sample SD over three seeds.**

| Metric | Original no augmentation, Adam 0.001 | Revised no augmentation, Adam 0.0001 | Conventional augmentation, Adam 0.0001 |
| --- | ---: | ---: | ---: |
| Accuracy | 0.5451 ± 0.0018 | 0.6282 ± 0.0195 | 0.6302 ± 0.0115 |
| Macro F1 | 0.3548 ± 0.0042 | 0.6259 ± 0.0169 | 0.6278 ± 0.0095 |
| Balanced accuracy | 0.5011 ± 0.0019 | 0.6275 ± 0.0149 | 0.6288 ± 0.0074 |
| Malignant sensitivity | 0.0022 ± 0.0038 | 0.6200 ± 0.0437 | 0.6133 ± 0.0406 |
| Specificity | 1.0000 ± 0.0000 | 0.6350 ± 0.0703 | 0.6443 ± 0.0548 |
| ROC-AUC | 0.5969 ± 0.1575 | 0.6767 ± 0.0122 | 0.6716 ± 0.0113 |
| Average precision | 0.5772 ± 0.1570 | 0.6453 ± 0.0163 | 0.6412 ± 0.0131 |

Values use a 0–1 scale. Each column contains the same three seeds. The table was calculated from the saved [original baseline CSV](../artifacts/binary_baseline_results.csv), [revised baseline CSV](../artifacts/binary_lr0001_results.csv), and [augmentation CSV](../artifacts/binary_conventional_aug_results.csv). SDs use n − 1 in the denominator.

## 3. Original no-augmentation baseline

The initial Adam 0.001 models performed close to an always-benign decision rule at threshold 0.5. Seeds 42 and 43 classified every validation image as benign; seed 44 detected one of the 150 malignant images. Mean validation accuracy was 0.5451, macro F1 was 0.3548, and malignant sensitivity was 0.0022. Specificity was 1.0000 because almost no images were assigned to the malignant class.

For context, assigning all 329 validation images to benign gives accuracy 179/329 = 0.5441, macro F1 0.3524, balanced accuracy 0.5000 and zero malignant sensitivity. The observed specificity therefore did not indicate successful malignant-lesion detection.

The continuous-score ROC-AUC varied from 0.4181 to 0.7152 across seeds. Thus the fixed-threshold failures did not correspond to identical ranking behavior in every model. The selected epochs were 9, 10 and 12, with training stopping after 14, 15 and 17 epochs respectively. These runs were retained rather than replaced by the later optimizer condition. See the [original baseline report](binary-baseline.md).

## 4. Training-only diagnostic and learning-rate revision

A balanced 32-image training subset was used to investigate optimization. Both learning-rate conditions used identical seed-42 initialization and 300 full-batch updates. No validation or test images entered this diagnostic.

**Table 2. Final inference-mode performance on the repeatedly used training subset.**

| Adam learning rate | Final subset BCE | Correct / 32 | Training accuracy |
| --- | ---: | ---: | ---: |
| 0.001 | 0.34105 | 26 | 81.25% |
| 0.0001 | 0.27943 | 30 | 93.75% |

Both conditions learned the subset, but neither reached 95% accuracy within the fixed update budget. The lower-rate condition had a more stable sampled loss trajectory and higher final subset accuracy. These are optimization observations on reused training images, not validation performance. They motivated the separate full-cohort optimizer revision documented in the [diagnostic report](binary-training-diagnostics.md).

With Adam reduced to 0.0001 and other settings preserved, validation accuracy and macro F1 increased in all three matched seeds. Mean accuracy increased from 0.5451 to 0.6282, an increase of 8.31 percentage points. Mean macro F1 increased from 0.3548 to 0.6259. Malignant sensitivity rose from 0.0022 to 0.6200, while specificity decreased from 1.0000 to 0.6350.

Mean ROC-AUC increased from 0.5969 to 0.6767, although seed 44's ROC-AUC decreased. The revision therefore improved thresholded classification in every seed without improving ranking in every seed. This comparison supports the revised optimizer as the subsequent reference candidate; it is not evidence of an augmentation effect.

All revised runs used the full 25-epoch budget, selecting epochs 25, 23 and 24. Selection of the final epoch in seed 42 means convergence was not established within that budget. The [optimizer comparison](binary-lower-lr-results.md) preserves all paired outcomes.

## 5. Conventional augmentation versus the revised reference

Adding conventional augmentation changed mean validation accuracy from 0.6282 to 0.6302. The mean paired difference, calculated before rounding, was +0.002026 (approximately +0.20 percentage points). Macro F1 changed from 0.6259 to 0.6278, with a mean paired difference of +0.001832. The displayed rounded means need not subtract to the rounded mean difference.

The direction was mixed across metrics: mean malignant sensitivity decreased from 0.6200 to 0.6133, specificity increased from 0.6350 to 0.6443, ROC-AUC decreased from 0.6767 to 0.6716, and average precision decreased from 0.6453 to 0.6412. Macro F1 improved in two seeds, whereas ROC-AUC decreased in every seed.

**Table 3. Paired differences: conventional augmentation minus revised no augmentation.**

| Seed | Accuracy difference | Macro F1 difference | Sensitivity difference | Specificity difference | ROC-AUC difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | -0.0122 | -0.0104 | +0.0133 | -0.0335 | -0.0069 |
| 43 | +0.0152 | +0.0129 | -0.0267 | +0.0503 | -0.0049 |
| 44 | +0.0030 | +0.0030 | -0.0067 | +0.0112 | -0.0037 |

Differences are in metric units, not relative percentages. Seed 42 had lower accuracy and macro F1 with augmentation; seeds 43 and 44 had higher values. All augmentation runs completed 25 epochs and selected epochs 21, 23 and 24 respectively.

The single fixed augmentation candidate produced a small improvement in mean thresholded summary scores with losses in sensitivity and ranking measures. These observations do not establish a consistent general benefit or statistical significance. No GAN images were used in this comparison. Subtype summaries in the [full augmentation report](binary-conventional-augmentation.md) describe binary correctness within original diagnoses, not seven-class predictions.

## 6. Seed-level classifier results

**Table 4. All nine selected-checkpoint validation runs.**

| Condition | Seed | Selected / trained epochs | Validation BCE | Accuracy | Macro F1 | Malignant sensitivity | Specificity | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original baseline | 42 | 9 / 14 | 0.6845 | 0.5441 | 0.3524 | 0.0000 | 1.0000 | 0.6574 |
| Original baseline | 43 | 10 / 15 | 0.6902 | 0.5441 | 0.3524 | 0.0000 | 1.0000 | 0.4181 |
| Original baseline | 44 | 12 / 17 | 0.6880 | 0.5471 | 0.3597 | 0.0067 | 1.0000 | 0.7152 |
| Revised baseline | 42 | 25 / 25 | 0.6394 | 0.6505 | 0.6452 | 0.5800 | 0.7095 | 0.6885 |
| Revised baseline | 43 | 23 / 25 | 0.6509 | 0.6201 | 0.6187 | 0.6133 | 0.6257 | 0.6642 |
| Revised baseline | 44 | 24 / 25 | 0.6487 | 0.6140 | 0.6140 | 0.6667 | 0.5698 | 0.6775 |
| Conventional augmentation | 42 | 21 / 25 | 0.6448 | 0.6383 | 0.6348 | 0.5933 | 0.6760 | 0.6817 |
| Conventional augmentation | 43 | 23 / 25 | 0.6527 | 0.6353 | 0.6316 | 0.5867 | 0.6760 | 0.6593 |
| Conventional augmentation | 44 | 24 / 25 | 0.6516 | 0.6170 | 0.6169 | 0.6600 | 0.5810 | 0.6737 |

No seed was selected for preferential reporting. Paired initialization hashes and split fingerprints agree across conditions, as recorded in the source CSVs. The numerical summaries in Tables 1 and 3 were recalculated from these archived run records and checked against the saved augmentation summary. This consolidation did not rerun the models or repeat prediction-level verification.

## 7. Custom conditional GAN feasibility pilot

The custom seed-142 conditional GAN completed 5,000 adversarial steps on the frozen training cohort. This corresponded to 160,000 real-image presentations, not 160,000 independent images. Logged training time was 2,024.44 seconds (33.74 minutes), excluding input checks and final packaging.

**Table 5. Completed custom-GAN pilot evidence.**

| Item | Recorded result |
| --- | --- |
| Real training images | 972: 530 benign, 442 malignant |
| Adversarial steps | 5,000 |
| Real-image presentations | 160,000 |
| Saved checkpoint sets | 11, at steps 0, 500, …, 5,000 |
| Saved class grids | 22 |
| Final unfiltered pool | 972 outputs: 530 benign-requested, 442 malignant-requested |
| Reloaded generator audit | All 972 saved image hashes reproduced |
| Exact repeated outputs in the pool | 0 |
| Exact resized/rounded training-pixel matches, in-run audit | 0 |
| Near-copy and clinical label assessment | Not completed |
| Downstream classifier augmentation | Not performed |

The final generator and discriminator losses were 4.4825 × 10⁻⁹ and 0.16707 respectively. These instantaneous optimization values are not image-quality measures.

The archived report records assistant visual inspection of fixed grids at steps 500, 1,000, 2,500 and 5,000. It identified repeated structures, checkerboard artifacts and similar appearance between the two requested classes, consistent with collapse and weak conditioning. This was an exploratory qualitative judgment, not an independent clinical review; Amir/Sandra review remained pending in the source record.

The pool was withheld from classifier training. Zero exact duplicates did not establish visual diversity or absence of memorization, and requested labels were not confirmed diagnoses. The in-run exact-copy audit compared consistently resized/rounded training pixels; the independent local archive checker did not reread the original training images. These scopes must remain distinct. The [custom GAN report](binary-cgan-pilot-results.md) preserves the execution and verification evidence.

## 8. Saved-generator mode diagnostic

A later exploratory diagnostic compared the saved generator in training and inference modes while holding latent inputs fixed and restoring the saved weights and BatchNorm state between comparisons. It performed no gradient updates.

**Table 6. Mean absolute pixel differences between generator training and inference modes, RGB scale 0–1.**

| Requested label batch | Mean absolute mode difference |
| --- | ---: |
| All benign | 0.105833 |
| All malignant | 0.105182 |
| Mixed | 0.105519 |

With the latent inputs fixed, changing labels from benign to malignant produced mean absolute differences of 0.002337 in inference mode and 0.001873 in training mode. In this diagnostic, changing execution mode had a larger pixel effect than changing the requested class. This comparison does not establish diagnosis validity or a causal explanation for collapse.

Recorded visual inspection found repeated structures and artifacts in both modes. The diagnostic ran under TensorFlow 2.21.0 rather than the pilot's 2.20.0; this software drift was explicitly recorded. Replayed grids differed by at most 1/255 in a small number of channel values, so reproduction was near-identical rather than byte-exact. The finding supported further investigation but did not demonstrate that a mode change would repair the generator. See the [mode diagnostic](gan-mode-diagnostic.md).

## 9. Pending StyleGAN2-ADA and mixed-augmentation results

The official StyleGAN2-ADA package has been prepared and sent to Sandra. The campus pilot is planned for Monday, 16 November 2026. No campus training outputs have been received for this chapter. Setup and runtime checks are preparation evidence; they are not trained-generator results.

**Table 7. Remaining experiments and explicit result placeholders.**

| Experiment | Training condition | Status / result |
| --- | --- | --- |
| Official StyleGAN2-ADA pilot | Training images only; fixed 20-kimg feasibility protocol | Pending execution and output review |
| GAN-only classifier augmentation | Real training images + GAN-generated images | Not performed; no scores available |
| Combined classifier augmentation | Real + GAN-generated training images, with conventional transformations | Not performed; no scores available |
| Final held-out comparison | Real test images only | Not performed |
| External validation | Separate compatible cohort after overlap checks | Not performed |

For the two proposed classifier conditions, synthetic counts, ratio, sampling rules and matched training-update budgets still require specification before execution. Both depend on a suitable generator and documented image review. The combined condition has no numerical result yet and must not be represented by the completed conventional-augmentation score.

When campus outputs return, the [evaluation checklist](stylegan2ada-pilot-evaluation.md) and [pilot results template](stylegan2ada-pilot-results-template.md) will distinguish technical completion, sample export, image observations, missing assessments and the next decision. Image-containing evidence and the complete package remain off GitHub.

## 10. Findings supported by the completed work

The revised optimizer improved validation classification relative to the original near-always-benign baseline. Conventional augmentation then produced a small, mixed change relative to that revised reference, rather than a clear improvement across measures. The completed custom GAN pilot produced a reproducible but visually problematic pool that was withheld from classifier augmentation.

The central comparison involving GAN-generated images remains incomplete. No conclusion about GAN-only augmentation, combined augmentation, held-out test performance, external generalization or clinical suitability follows from the current results.

## Source records

- [Original baseline metrics](../artifacts/binary_baseline_results.csv) and [report](binary-baseline.md)
- [Revised baseline metrics](../artifacts/binary_lr0001_results.csv) and [report](binary-lower-lr-results.md)
- [Conventional augmentation metrics](../artifacts/binary_conventional_aug_results.csv), [paired differences](../artifacts/binary_conventional_aug_paired_comparison.csv), [summary](../artifacts/binary_conventional_aug_summary.csv), and [report](binary-conventional-augmentation.md)
- [Training-only diagnostics](binary-training-diagnostics.md)
- [Custom GAN pilot](binary-cgan-pilot-results.md) and [mode diagnostic](gan-mode-diagnostic.md)
