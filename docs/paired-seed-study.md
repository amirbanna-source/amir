# Paired seed study

Protocol fixed before examining the six runs: seeds 42, 43, 44; each seed has one no-augmentation run and one conventional-augmentation run. Each pair starts from the same CNN weights, verified byte-for-byte and recorded with a SHA-256 fingerprint. Seeds are set before model and data-pipeline construction. Both conditions use the frozen lesion-grouped split and identical preprocessing, class weights, batch size, optimizer, epoch budget, early stopping and validation-loss selection.

The split remains 1,050 training / 355 validation / 356 test images. The split membership fingerprint is checked before training. Training augmentation retains the initial recipe: horizontal/vertical flips, ±18-degree rotations, zoom factor 0.10, reflection fill, bilinear interpolation. Validation stays unaugmented. No test evaluation is performed.

The paired workflow uses an explicit dropout seed (12345) in both conditions and creates the data pipeline after seed initialization. These controls differ from the exploratory runs, which should remain separate in analysis. Augmentation seeds are training seed +1000/+1001/+1002. Dataset order is deterministic given the pipeline; GPU kernels and different stochastic operations may still prevent bitwise reproducibility of full training.

One training example per class was inspected before the runs. Lesions remained visible in the eight shown examples, with no obvious severe clipping; interpolation blur was visible. This small preview is a visual check, not a comprehensive assessment of every transformation or clinical validity.

[Study script](../scripts/paired_seed_study.py). Run evidence will include every checkpoint, epoch history, per-image validation predictions, confusion matrix, per-class metrics, configuration, and preview. The main comparison is the mean and sample standard deviation of macro F1 over the three seeds, alongside paired per-seed differences, accuracy and balanced accuracy. Three seeds measure initial stability; they do not establish broad statistical significance.
