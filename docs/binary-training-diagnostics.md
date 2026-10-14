# Binary CNN training diagnostics — 14 October 2026

The weak three-seed binary baseline was followed by a training-only sanity check. This diagnostic does not replace or improve the reported validation baseline.

## Fixed protocol

The protocol was committed before execution. Select 32 training images (16 benign, 16 malignant), one per recorded group, using deterministic sampling seeds 2026/2027. Keep the original CNN, 128×128 RGB /255 preprocessing, Dropout 0.35 and seed-42 initialization. Compare Adam learning rates 0.001 and 0.0001 for exactly 300 full-batch updates, reporting every 25 updates. Use unweighted binary cross-entropy on this balanced subset, no augmentation, and threshold 0.5. Dropout remains active during optimization; reported metrics use inference mode.

Both runs had identical initial weight hashes matching the archived seed-42 baseline. All 32 IDs belong to the existing training set. No validation or test images were loaded. This is a single-seed optimization diagnostic with repeated exposure to the same 32 images, not a generalization experiment.

## Checks and results

Images and labels matched the manifest. Images were finite, nonconstant and in [0,1], with tensor shape (32,128,128,3). Explicit diagnostic labels had shape (32,1); the existing input pipeline returned labels of shape (32,). Every initial inference-mode trainable-variable gradient was finite and nonzero. No disconnected kernel gradients were found. This does not exhaustively validate every possible pipeline issue.

| Learning rate | Final training BCE | Correct / 32 | Training accuracy | Malignant sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.001 | 0.34105 | 26 | 81.25% | 81.25% | 81.25% |
| 0.0001 | 0.27943 | 30 | 93.75% | 87.50% | 100.00% |

At 0.001, accuracy reached 87.5% at update 150 but later fluctuated; BCE rose to 0.916 at update 175. At 0.0001, sampled BCE decreased from 0.6932 to 0.2794. Neither run achieved at least 95% accuracy within the fixed 300-update budget. Do not describe this as perfect memorization or claim the baseline problem is solved.

## Interpretation and next experiment

The network can learn both labels on this subset. The lower rate produced a more stable sampled loss trajectory and better final subset performance, making 0.0001 a reasonable candidate for a controlled full-training-set revision. This does not establish that learning rate alone explains the weak validation results, nor that the lower rate generalizes better.

Next: repeat the full no-augmentation binary baseline at Adam 0.0001 across seeds 42/43/44, preserving split, initialization, class weights, architecture, batch size, epoch budget, checkpoint rule and threshold. Report all seeds against the original baseline, identify this as exploratory tuning, and inspect validation sensitivity, macro F1, ROC-AUC and predicted probabilities. Keep all 331 test images untouched and postpone GAN comparisons until the classifier is useful.

## Reproduction and evidence

Run [the diagnostic script](../scripts/binary_training_diagnostics.py) after the binary baseline has generated `/kaggle/working/binary_baseline/isic2019_binary_split.csv`. The fingerprint assertion rejects altered split membership. Results, subset IDs, gradient norms, audit configuration, histories and final predictions are in [artifacts/binary_diagnostics](../artifacts/binary_diagnostics/). Kaggle notebook Version 4 preserves the diagnostic cell and its live outputs.

Archive CRC, training-only membership, threshold predictions, confusion counts and final accuracies were independently checked after downloading the evidence.
