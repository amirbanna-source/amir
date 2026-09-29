# Reproduction and next steps

## Kaggle inputs and outputs

The working Kaggle notebook used these input files:

```text
/kaggle/input/datasets/amirsalahk/isic-2019-ground-truth/ISIC_2019_Training_GroundTruth.csv
/kaggle/input/datasets/amirsalahk/isic-2019-metadata/ISIC_2019_Training_Metadata.csv
```

The Kaggle image dataset is `nodoubttome/skin-cancer9-classesisic`; its mounted path contains additional dataset directory components and should be discovered from the attached notebook input rather than hard-coded from this document.

Known working outputs:

```text
/kaggle/working/isic2019_verified_split.csv
/kaggle/working/baseline_no_aug.keras
```

The `/kaggle/working/` files can disappear after a session reset unless saved in a Kaggle notebook version or separately downloaded. Confirm they exist before attempting validation evaluation. The exact reproducible split construction and baseline training code are currently in the Kaggle notebook, not in this archive.

## Bring the notebook into GitHub

1. In Kaggle, save a notebook version with outputs after verifying the split and baseline checkpoint.
2. Export/download the actual notebook as `.ipynb` and put it under `notebooks/`; give it a descriptive name such as `01_isic2019_audit_split_baseline.ipynb`.
3. Remove any secrets, large embedded image outputs, and unnecessary execution traces before committing. Keep the code, chosen random seed, and meaningful numerical outputs.
4. Add a small, stable manifest of image IDs, labels, groups, and split assignments if permitted; do not commit raw images. Check mounted Kaggle paths if the notebook is run in a new environment.

## Next experiment steps

1. Evaluate the selected baseline checkpoint on **validation**: report macro F1, balanced accuracy, per-class metrics, and confusion matrix.
2. Discuss eight versus nine classes with the supervisor and record the decision.
3. Implement conventional augmentation applied to the training set only, then compare on the same validation set.
4. Specify GAN architecture, training-only input data, quality checks, number of generated images, and synthetic-to-real ratio before the GAN experiment.
5. Run the GAN-only and combined augmentation conditions; select the approach using validation results.
6. Perform final evaluation on the untouched test set and document limitations.

Do not describe external ISIC Gallery images as an independent validation cohort until diagnoses and image overlap have been checked.
