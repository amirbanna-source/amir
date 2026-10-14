# Lower learning rate baseline revision

Declared 14 October 2026 after the training-only subset diagnostic. This is exploratory optimizer tuning using the existing validation cohort, not a new confirmatory study.

Only Adam learning rate changes: 0.001 → 0.0001. Retain seeds 42/43/44, identical initial arrays (asserted by SHA256), architecture, Dropout seed, preprocessing, class weights, split membership, batch32, maximum25 epochs, early-stopping patience5, minimum-validation-loss checkpoint and classification threshold0.5. Preserve every seed, including failures. No augmentation is applied.

Compare paired validation accuracy, macro F1, balanced accuracy, malignant sensitivity, specificity, ROC-AUC and average precision against the archived original-rate baseline. Preserve models, initialization, predictions, histories, confusion matrices, run configuration and split manifest under new filenames. Report actual early stopping and epoch budget limits without extending the budget after seeing results. The 331 test images remain sealed.

Run [the fixed script](../scripts/binary_lr0001.py). This revision was committed before execution. The earlier weak baseline remains archived.
