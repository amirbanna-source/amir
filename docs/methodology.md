# Methodology (binary study; working draft)

Updated 14 October 2026 following Amir and Sandra's decision to study benign versus malignant lesions. The earlier eight-class experiments are retained as preliminary work.

Images were obtained from the Skin Cancer ISIC dataset hosted on Kaggle. Because an initial audit found duplicate images, conflicting folder labels, and overlap between the supplied training and test sets, those folders were not used as the experimental split. Image IDs were instead matched to the ISIC 2019 Training Ground Truth, and lesion IDs were obtained from the corresponding metadata. One copy of each matched image was retained, producing 1,761 unique images across eight officially verified diagnostic classes.

For the binary task, NV, BKL, DF and VASC were mapped to benign (target 0), while MEL, BCC and SCC were mapped to malignant (target 1). The 129 AK images were excluded from the primary analysis rather than assigned to either group. This is an explicit study-scope choice: [AAD describes AK as a precancerous growth](https://www.aad.org/public/diseases/skin-cancer/actinic-keratosis-overview). Original diagnoses are retained for traceability. The binary subset contains 1,632 unique images: 894 benign and 738 malignant.

The verified images were originally divided into training (1,050), validation (355), and test (356) sets using stratified group splitting. Images sharing a known lesion ID were kept in the same set. For images without a lesion ID, the image ID served as the grouping identifier. Binary labels were applied after filtering AK, preserving each retained image's original membership; a new binary-stratified split was not created. The resulting sets contain 972 training, 329 validation and 331 test images.

| Class | Training | Validation | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| Benign | 530 | 179 | 185 | 894 |
| Malignant | 442 | 150 | 146 | 738 |
| Total | 972 | 329 | 331 | 1,632 |

Images will be resized to 128 x 128 pixels and scaled to [0,1]. Binary class weights derived exclusively from the training set are 0.916981 for benign and 1.099548 for malignant, using N/(2*n_class).

The planned experiments will compare a convolutional neural network trained without augmentation against the same classification approach trained with conventional augmentation, GAN-generated images, and a combination of both. Augmentation and synthetic-image generation will use training data only. The validation set will guide model selection, while the held-out test set will be used for final evaluation. The binary baseline retains the earlier convolutional backbone but uses one sigmoid output and binary cross-entropy. Baseline seeds were 42, 43 and 44, with Adam at 0.001, batch size 32, at most 25 epochs, early-stopping patience 5 and checkpoint selection by minimum validation loss.

Results will include accuracy, macro-averaged F1 score, balanced accuracy, per-class performance and confusion matrices. For the malignant positive class, sensitivity, specificity, ROC-AUC and average precision will also be reported. The initial classification threshold is fixed at 0.5; any later threshold tuning must use validation data and be disclosed. Three no-augmentation baseline runs were completed on 14 October 2026. Their selected checkpoints largely predicted benign at threshold 0.5; mean accuracy was 0.5451, macro F1 0.3548 and malignant sensitivity 0.0022. These weak results are preserved in the [binary baseline report](binary-baseline.md). A training-only subset diagnostic motivated an exploratory optimizer revision to Adam0.0001. All three revised seeds completed with the other settings fixed; mean validation accuracy was0.6282, macro F1 0.6259, malignant sensitivity0.6200 and specificity0.6350. See the [paired optimizer comparison](binary-lower-lr-results.md). This tuning used the existing validation cohort and does not establish an augmentation effect. The fixed conventional-augmentation comparison completed on 16 October 2026: mean macro F1 was 0.6278 versus 0.6259 without augmentation; sensitivity was 0.6133 versus 0.6200. The small mixed effect and all paired outcomes are documented in the [binary augmentation report](binary-conventional-augmentation.md). GAN experiments remain unexecuted.

The original project plan specified nine classes, and the audit supported eight official diagnoses; the current research task groups seven of these diagnoses into two classes and excludes AK. This change should be reflected in the project plan and discussed with the supervisor. In addition, 324 retained images lack lesion IDs and patient IDs are unavailable, so complete patient-level separation cannot be confirmed. The grouped binary labels contain heterogeneous diagnosis subtypes; subtype-specific errors should be reported when feasible. Any future external validation using the ISIC Archive will first require checks for image overlap and compatible diagnoses.

Repeated development on the same validation set can introduce selection optimism. All earlier eight-class results remain historical; their metrics are not binary baseline results. The 331 retained test images remain reserved for the final evaluation.

## Reproducibility

- [Binary split manifest](../artifacts/isic2019_binary_split.csv)
- [Excluded AK manifest](../artifacts/isic2019_binary_excluded_ak.csv)
- [Mapping, counts and fingerprints](../artifacts/binary_config.json)
- [Executed baseline notebook](../notebooks/05_isic2019_binary_baseline.ipynb)
- [Baseline script](../scripts/binary_baseline.py)

Membership and known-group separation were verified on the saved manifest. GPU training completed for all three seeds. Archive integrity, validation IDs/labels, recalculated metrics, selected epochs and initialization hashes were independently checked after download.
