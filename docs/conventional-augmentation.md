# Conventional augmentation experiment — 29 September 2026

The first conventional augmentation run scored below the recovered no-augmentation baseline on the same 355 validation images. This is a preliminary single-run comparison, not evidence that augmentation is generally harmful.

## Protocol

The split membership fingerprint (image ID, official label, group and split) was checked against the archived baseline before training: `900aed297caf894a0a5b0001b3ca974ea954d96104d70781b2eb7a71306227c1`. There were 1,050 training, 355 validation and 356 test images. Known lesions were kept within one split; 324 images lack lesion IDs and patient IDs are unavailable.

Images were resized to 128 × 128 RGB and scaled to [0, 1]. Online augmentation used horizontal/vertical flips, random rotation up to ±18° (factor 0.05), and zoom factor 0.10 with reflection filling and bilinear interpolation. Augmentation was applied only to training batches; validation inputs were unaugmented. No synthetic images were used.

The CNN architecture, training class weights, Adam learning rate 0.001, sparse categorical cross-entropy, batch size 32, 25-epoch budget, and early-stopping patience 5 matched the baseline. Training seed was 42; augmentation layer seeds were 101, 102 and 103. TensorFlow 2.20.0 ran on Kaggle T4 hardware without distributed training. The saved CNN excludes preprocessing augmentation layers, so evaluation uses original resized images.

## Results

| Validation metric | No augmentation | Conventional augmentation |
| --- | ---: | ---: |
| Accuracy | 0.4141 | 0.2901 |
| Macro F1 | 0.4063 | 0.2576 |
| Balanced accuracy | 0.4611 | 0.3365 |
| Best validation loss | 1.5410 | 1.6548 |
| Selected epoch (minimum validation loss) | 24 | 12 |
| Epochs trained | 25 | 17 |

Accuracy decreased by 12.39 percentage points and macro F1 by 0.1487. The augmented run stopped after five epochs without improving validation loss. Its checkpoint was reloaded before evaluation.

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| AK | 26 | 0.00 | 0.00 | 0.00 |
| BCC | 72 | 0.00 | 0.00 | 0.00 |
| BKL | 94 | 0.37 | 0.45 | 0.41 |
| DF | 22 | 0.12 | 0.77 | 0.21 |
| MEL | 28 | 0.47 | 0.61 | 0.53 |
| NV | 34 | 0.30 | 0.38 | 0.33 |
| SCC | 50 | 0.00 | 0.00 | 0.00 |
| VASC | 29 | 0.74 | 0.48 | 0.58 |

The checkpoint predicted no BCC or SCC images, and only one image as AK. It predicted DF for 142 validation images, indicating strong class confusion. This observation motivates investigating stability and augmentation effects; it does not establish their cause.

## Saved evidence and checks

- [Executed notebook](../notebooks/03_isic2019_conventional_augmentation.ipynb)
- [Standalone experiment code](../scripts/conventional_aug_experiment.py)
- [Checkpoint](../artifacts/cnn_conventional_aug.keras)
- [Exact metrics and per-class report](../artifacts/conventional_aug_metrics.json)
- [Learning history](../artifacts/conventional_aug_history.csv)
- [Confusion matrix](../artifacts/conventional_aug_validation_confusion_matrix.csv)
- [Per-image predictions](../artifacts/conventional_aug_validation_predictions.csv)

The downloaded checkpoint is a valid Keras ZIP container. Predictions contain 355 unique image IDs; the confusion matrix sums to 355; independently recalculated accuracy and macro F1 agree with the saved metrics. The learning history has 17 epochs and minimum validation loss at epoch 12. Kaggle Quick Save Version 2 also preserves the executed cell; the GitHub artifacts preserve the downloaded outputs separately.

## Next step

Preserve this result and run a paired multi-seed comparison with explicit, consistent seed initialization before data-pipeline creation and model construction. Check augmented image examples for excessive cropping or distortion. Any altered augmentation recipe should be documented as a new experiment and selected using validation only. Do not use the held-out test set for tuning. GAN experiments remain planned.

[Keras RandomRotation documentation](https://keras.io/api/layers/preprocessing_layers/image_augmentation/random_rotation/) and [RandomZoom documentation](https://keras.io/api/layers/preprocessing_layers/image_augmentation/random_zoom/) describe the augmentation parameters.
