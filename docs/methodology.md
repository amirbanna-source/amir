# Methodology (working draft)

Images were obtained from the Skin Cancer ISIC dataset hosted on Kaggle. An initial audit found repeated images, conflicting folder labels, and overlap between the supplied training and test sets. Image IDs were matched to the ISIC 2019 Training Ground Truth, and lesion IDs were obtained from its metadata. One copy of each matched image was retained, producing 1,761 unique images in eight officially verified diagnostic classes. The 441 unmatched IDs were excluded.

The verified images were divided into training (1,050), validation (355), and test (356) sets using stratified group splitting. Images with a known shared lesion ID were kept in the same set. For images without a lesion ID, the image ID served as the grouping identifier. Inputs were resized to 128 × 128 RGB pixels and scaled to [0, 1]. Class weights computed from the training set addressed class imbalance during baseline training.

The planned comparison uses the same classification architecture and data split across four conditions: no augmentation, conventional augmentation, GAN-generated training images, and conventional augmentation combined with GAN-generated images. Augmentation and synthetic-image generation must use **training data only**. The validation set guides model selection and tuning; the held-out test set is reserved for final evaluation after the method is fixed. Intended metrics include accuracy, macro-averaged F1, per-class precision/recall/F1, and a confusion matrix. The three augmented conditions have not yet been run.

The original project plan specified nine classes, whereas this verified subset supports eight. The scope change still requires discussion with the supervisor. In addition, 324 verified images lack lesion IDs and patient IDs are unavailable, so complete patient-level separation cannot be confirmed. Any later external validation using ISIC Archive images requires overlap checks and diagnosis mapping before evaluation.

## Experimental safeguards

1. Keep the saved image-ID and lesion-group split fixed across all conditions.
2. Generate or augment images from the training fold only; never use validation or test images for GAN training.
3. Apply the same preprocessing and metric definitions to each condition.
4. Choose checkpoints and hyperparameters using validation data only; evaluate test once after selection.
5. Report class counts and results for all eight classes, including any failed or low-performing classes.

This is the current study design, not a claim that all four experiments have been completed.
