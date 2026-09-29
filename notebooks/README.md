# Notebooks

- [Original audit, split, and baseline](01_isic2019_audit_split_baseline.ipynb): preserves the original exploratory work and first training run.
- [Baseline rerun and validation](02_isic2019_baseline_rerun_validation.ipynb): preserves the recovered split, 25-epoch rerun, saved checkpoint checks, and validation-only evaluation. Best validation loss was 1.5410 at epoch 24; validation accuracy was 0.4141 and macro F1 was 0.4063.

These exported Kaggle notebooks include historical failed and restart-related cells and are not yet clean end-to-end `Run All` notebooks. The bulky initial file listing and obsolete network-error traceback were removed from saved outputs; source code and meaningful results are retained. The rerun notebook header identifies the cells to use.

[Recovered artifacts](../artifacts/) include the split CSV, best CNN checkpoint, validation predictions, and confusion matrix. The 356-image test set remains unevaluated.

- [Conventional augmentation](03_isic2019_conventional_augmentation.ipynb): clean single-cell Kaggle workflow with the exact split fingerprint check, training and validation-only evaluation, including executed outputs. This can run independently with the three source datasets attached and GPU enabled.
