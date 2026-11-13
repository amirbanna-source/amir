# Methods: binary skin-lesion augmentation study

Working thesis Methods draft, updated 13 November 2026. This version describes completed procedures in the past tense and identifies unexecuted components explicitly. The original multiclass work is preliminary; the current task is binary classification.

## 1. Study design and experimental status

The study investigates whether training-data augmentation improves a convolutional neural network (CNN) for classifying skin-lesion images into the study-defined benign and malignant groups. The intended comparison includes no augmentation, conventional image augmentation, GAN-generated training images, and GAN-generated images combined with conventional augmentation. All conditions use the same frozen image split. Only the no-augmentation and conventional-augmentation classifier conditions have been completed.

| Component | Status as of 13 November 2026 |
| --- | --- |
| Data audit and frozen binary split | Completed |
| Original CNN baseline, Adam learning rate 0.001, seeds 42/43/44 | Completed; retained as the original optimizer condition |
| Training-only optimization diagnostics and revised CNN baseline, Adam 0.0001 | Completed |
| Paired conventional-augmentation comparison, seeds 42/43/44 | Completed |
| Custom conditional GAN, seed 142, 5,000 steps | Completed feasibility experiment; pool withheld from classifier training |
| Saved-generator BatchNorm mode diagnostic | Completed exploratory diagnostic |
| Official StyleGAN2-ADA source/data/runtime preparation | Completed within the recorded validation scope |
| Campus StyleGAN2-ADA 20-kimg pilot | Package sent to Sandra; execution planned for 16 November 2026, outputs pending |
| GAN-only and GAN-plus-conventional classifier comparisons | Planned; not performed |
| Held-out test evaluation and external validation | Not performed |

This is an exploratory development study. Repeated use of the same validation cohort for development limits the independence of validation estimates. GAN feasibility experiments and optimizer revision are distinct from tests of classifier augmentation benefit.

## 2. Image source and audit

Images were obtained from the Kaggle Skin Cancer ISIC dataset. The supplied folder labels and Train/Test directories were not adopted as experimental ground truth or split membership. The initial audit identified 4,714 files representing 2,202 unique image IDs, 155 IDs with conflicting folder labels, and 18 filenames shared between the supplied Train and Test folders.

Image IDs were matched to the official ISIC 2019 training ground truth. Corresponding lesion IDs were obtained from the official training metadata. One copy per matched image ID was retained; 441 unmatched IDs were excluded. The resulting audited cohort contained 1,761 images across eight official diagnoses. Retained copies of the same ID agreed byte-for-byte; no distinct retained IDs had identical image content, no known lesion had conflicting official diagnoses, and no retained image was unreadable. These exact-content checks do not exclude near-duplicate images.

The [dataset audit](data-audit.md) records the sources, counts and checks. Official diagnoses superseded Kaggle folder names throughout the analysis.

## 3. Binary endpoint and inclusion criteria

The binary endpoint was derived from the official diagnosis labels using the following study-specific mapping:

| Target | Group | Included diagnoses |
| --- | --- | --- |
| 0 | Benign | NV, BKL, DF, VASC |
| 1 | Malignant | MEL, BCC, SCC |
| Excluded | AK | All 129 AK images |

AK was excluded rather than forced into either binary group. This defines the study population and limits the interpretation of the endpoint; the model is not intended to classify every possible skin lesion. Original diagnosis labels were retained for traceability and subtype-level error summaries.

After AK exclusion, 1,632 images remained: 894 benign and 738 malignant. This is a verified subset of the source collection, not the complete ISIC 2019 training dataset.

## 4. Frozen training, validation and test split

The original eight-class cohort was partitioned by lesion-aware stratified group splitting. The implementation used the first split from an outer five-fold StratifiedGroupKFold with shuffling and random state 42 to reserve the test set. Within the remaining images, the first split from an inner four-fold StratifiedGroupKFold, also with shuffling and random state 42, reserved validation images. Stratification used the original diagnostic classes.

Images with a known lesion ID shared a group identifier. When the lesion ID was missing, the image ID was used as its fallback group. Group overlap between training, validation and test sets was checked and rejected. This procedure created one fixed train/validation/test partition; it was not a five-fold cross-validation evaluation.

The binary labels were applied after removing AK, preserving the original membership of every retained image. No new binary-stratified split was created.

| Binary group | Training | Validation | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| Benign | 530 | 179 | 185 | 894 |
| Malignant | 442 | 150 | 146 | 738 |
| Total | 972 | 329 | 331 | 1,632 |

The binary cohort contains 324 images without lesion IDs, and patient IDs are unavailable. Consequently, complete lesion-level separation for those images and patient-level independence cannot be confirmed.

The archived [binary manifest](../artifacts/isic2019_binary_split.csv) defines membership. Its fingerprint was checked during reconstruction; changing the split or targets invalidates the frozen protocol. The binary fingerprint is `4f3037e5dc1726e5a534e3a6f155df3e0e0be52229b057659e097483fd0800cd`.

## 5. Classifier preprocessing and architecture

The TensorFlow loader decoded each image into three channels, resized the whole image to 128×128 pixels using TensorFlow image resizing, converted values to float32, and divided by 255. Training images were shuffled with the run seed and reshuffled each epoch. Validation data were loaded without random augmentation. No test-image dataset was constructed for the completed classifier experiments.

The classifier was trained from scratch and used four 3×3, same-padded convolutional layers with ReLU activation and 32, 64, 128 and 256 filters. Each of the first three convolutional layers was followed by max pooling. The fourth convolutional layer was followed by global average pooling, a 128-unit ReLU dense layer, dropout with rate 0.35 and fixed dropout seed 12345, and a single sigmoid output representing the malignant score.

The [baseline implementation](../scripts/binary_baseline.py) specifies the loader and architecture. No transfer-learning backbone was used.

## 6. Classifier optimization, pairing and checkpoint selection

Training minimized binary cross-entropy with training-derived class weights. For class c, the weight was N / (2 n_c), where N = 972 and n_c is the number of training images in that class. This gave weights 0.916981 for benign and 1.099548 for malignant. Validation loss was unweighted.

The original baseline used Adam with learning rate 0.001. Training-only diagnostics motivated a separately recorded reduction to 0.0001, followed by three revised baseline runs. This was an exploratory optimizer revision, not an augmentation effect. The revised baseline provided the reference condition for the conventional-augmentation comparison.

| Setting | Completed classifier experiments |
| --- | --- |
| Run seeds | 42, 43, 44; all retained |
| Batch size | 32; 31 batches per epoch, including the partial final batch |
| Maximum budget | 25 epochs |
| Early stopping | Validation loss, patience 5 |
| Checkpoint selection | Minimum unweighted validation loss |
| Threshold | Malignant score ≥ 0.5 |
| Paired initialization | Seed-specific initial arrays/hashes verified across compared conditions |
| Software / hardware | TensorFlow 2.20.0, Kaggle Tesla T4; ordinary single-device Keras training |

Within the augmentation comparison, architecture, preprocessing, split, initialization, optimizer, class weights, maximum budget, checkpoint rule and threshold were held fixed. Early stopping could produce different realized training lengths; a common maximum budget does not imply identical numbers of updates when stopping differs. Seed variation describes repeated model fitting on the same cohort, not independent patient samples.

See the [original baseline](binary-baseline.md), [training diagnostics](binary-training-diagnostics.md), and [optimizer comparison](binary-lower-lr-results.md).

## 7. Conventional augmentation

Conventional augmentation was applied only to training batches through a separate preprocessing pipeline. It comprised horizontal and vertical flips, random rotations with factor 0.05 (up to 18 degrees in either direction), and independent height and width zoom factors in the interval −0.1 to 0.1, with reflection fill.

Layer seeds were the run seed plus 100, 200 and 300 for flipping, rotation and zoom respectively. The augmentation map was sequential and deterministic in ordering. Validation images were not augmented, and the saved classifier contained no augmentation layers.

No synthetic images, resampling or threshold tuning were introduced in this comparison. The [predeclared protocol](binary-conventional-augmentation-protocol.md), [implementation](../scripts/binary_conventional_aug.py) and [completed comparison](binary-conventional-augmentation.md) preserve the settings and all three paired runs.

## 8. Generative-model feasibility work

### 8.1 Custom conditional GAN and diagnostic

An initial custom conditional GAN was trained from scratch on the 972 real training images at RGB128 resolution. The seed-142 pilot used a 128-dimensional latent vector, batch size 32, and 5,000 adversarial steps, with one discriminator and one generator update per step. Both optimizers used Adam with learning rate 0.0002, beta1 = 0.5 and beta2 = 0.999. The discriminator used neither BatchNorm nor dropout.

The pilot retained checkpoints and fixed class grids every 500 steps. An unfiltered final pool contained 530 benign-requested and 442 malignant-requested images. Requested class labels were not independently verified diagnoses. The archived models were reloaded and the saved audit-image hashes reproduced. The in-run decoded/resized-pixel audit reported no exact training-image matches; this did not establish absence of near-copying or memorization.

Visual inspection identified repeated structures and artifacts; the pool was withheld from classifier training. A subsequent saved-generator diagnostic compared training and inference modes using identical latent inputs and labels, restoring the original weights and BatchNorm state between comparisons. It performed no optimizer updates and recorded software-version drift. The observed mode dependence did not establish the cause of collapse. Further custom-GAN development was paused.

These procedures and their limitations are preserved in the [pilot implementation](binary-cgan-pilot-implementation.md), [pilot report](binary-cgan-pilot-results.md), and [mode diagnostic](gan-mode-diagnostic.md).

### 8.2 Official StyleGAN2-ADA candidate

The subsequent candidate uses NVIDIA's official StyleGAN2-ADA PyTorch implementation, pinned to commit `d72cc7d041b42ec8e806021a205ed9349f87c6a4`, without pretrained weights. The study reuses an established implementation and does not claim a new GAN architecture.

The training-only adapter converted the same frozen 972-image cohort to RGB128 uint8 PNGs using Pillow bilinear resizing of the whole image, without cropping. This image-export implementation is recorded separately from the TensorFlow classifier loader. Labels were stored in NVIDIA's dataset format. The training ZIP, label counts, image dimensions, IDs and hashes were verified.

Two explicit compatibility patches address the extension-loader API and grid-sampling gradients on the tested modern PyTorch environment. No GAN architecture, objective or CUDA kernel source was changed. The preparation evidence includes CUDA extension and derivative checks and production-resolution generator/discriminator, R1 and path-length backward checks on synthetic tensors. These checks did not train a model or establish convergence or image quality.

The campus handoff fixes one GPU, batch 16, seed 142, conditional auto configuration, ADA geometry, mirror disabled, 20 kimg, snapshots every tick, two workers and no automatic metrics. Twenty kimg denotes 20,000 real-image presentations, not 20,000 steps. It is not budget-equivalent to the earlier custom 5,000-step pilot. ADA regularizes GAN training and is distinct from conventional augmentation of classifier inputs.

The launcher validates the data, source, runtime and official dry run before fitting. The complete package was sent separately to Sandra for a planned campus run on 16 November 2026. As of this Methods update, campus execution, final checkpoints and sample quality remain pending. The Docker definition and campus hardware/permissions have not been validated here. No StyleGAN2-ADA training result is claimed.

See the [transition record](stylegan2ada-transition.md), [fixed campus protocol](../handoff/stylegan_pilot/README.md), and [prospective pilot evaluation plan](stylegan2ada-pilot-evaluation.md). Later preparation updates supersede earlier unexecuted statuses within the chronological transition record.

## 9. Performance measures and analysis

Completed classifier evaluations used the 329 real validation images. Macro F1 was the primary descriptive measure for the paired conventional-augmentation comparison. Accuracy, balanced accuracy, per-class precision/recall/F1, confusion matrices, malignant sensitivity, specificity, ROC-AUC and average precision were also recorded.

Malignant was the positive class. With the threshold fixed at 0.5, sensitivity was TP / (TP + FN) and specificity was TN / (TN + FP). ROC-AUC and average precision used continuous malignant scores. No threshold optimization was performed in the completed comparisons.

Each condition reports all three seeds and mean ± sample standard deviation. Paired differences use the augmentation result minus the corresponding no-augmentation result for the same seed. These standard deviations describe training-seed variation; they are not patient-level confidence intervals. Three seeds and one validation cohort do not support a general significance or clinical-utility claim.

Original-diagnosis subgroup summaries were calculated as binary correctness within each diagnosis, not as seven-class classification. Small subtype samples limit interpretation. Detailed numerical findings belong in the Results section and remain linked in the experiment reports.

## 10. Planned synthetic augmentation and final evaluation

GAN-only and GAN-plus-conventional classifier experiments remain unexecuted. Before either proceeds, generated samples will be reviewed for artifacts, diversity, requested-class behavior and possible copying against training images only. The 32 fixed final samples from the short StyleGAN pilot provide a preliminary review, not proof of clinical validity or absence of memorization.

A later experiment must specify the synthetic sample counts, class ratio, generation procedure, selection/filtering rules, checkpoint rule and classifier update budget before execution. Classifier comparisons are intended to preserve the frozen real-image split and use paired seeds 42/43/44. The revised Adam 0.0001 classifier is the current reference candidate; any further change must be documented.

Validation evaluation will remain real-only. The 331 test images remain reserved until the final comparison and model-selection rules are frozen. No test result, external-validation result or synthetic-augmentation benefit is currently available. Any external cohort will require compatible diagnosis mapping and checks for overlap with the development data.

## 11. Reproducibility, data handling and limitations

Membership manifests, original diagnoses, class mappings, source pins, hashes, scripts, executed notebooks and experiment reports provide the audit trail. Saved classifier outputs include initial weights, selected models, histories, per-image validation predictions and configuration. Archive integrity, validation labels, recalculated metrics, initialization pairing and checkpoint selection were checked in the completed classifier studies.

Training images and the complete campus ZIP must remain off GitHub. The handoff and image-containing results are shared separately; public repository updates contain code and reviewed text/aggregate evidence. Applicable source-data usage conditions remain relevant to redistribution.

The principal limitations are the small selected cohort, heterogeneous binary groups, excluded AK category, missing lesion identifiers, unavailable patient identifiers, repeated validation-guided development, limited seed count, and absence of external or held-out test evaluation. Resolution reduction and the limited GAN feasibility budget further restrict conclusions. The study does not establish clinical suitability.

## Evidence index

- [Dataset audit](data-audit.md)
- [Binary mapping, counts and fingerprints](../artifacts/binary_config.json)
- [Frozen binary split](../artifacts/isic2019_binary_split.csv) and [AK exclusions](../artifacts/isic2019_binary_excluded_ak.csv)
- [Original baseline](binary-baseline.md), [revised baseline](binary-lower-lr-results.md), [conventional augmentation](binary-conventional-augmentation.md)
- [Custom GAN pilot](binary-cgan-pilot-results.md) and [mode diagnostic](gan-mode-diagnostic.md)
- [Official GAN transition](stylegan2ada-transition.md), [campus launcher](../handoff/stylegan_pilot/README.md), [evaluation plan](stylegan2ada-pilot-evaluation.md)
