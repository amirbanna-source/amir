# CNN baseline without augmentation

## Setup recorded from the Kaggle notebook

- Inputs: 128 × 128 RGB, pixel values scaled to [0, 1], batch size 32.
- Labels: `AK=0, BCC=1, BKL=2, DF=3, MEL=4, NV=5, SCC=6, VASC=7`.
- Architecture: convolution blocks with 32, 64, 128, and 256 filters; global average pooling; dense layer of 128 units; dropout 0.35; eight-way softmax.
- Optimizer/loss: Adam at learning rate 0.001 and sparse categorical cross-entropy.
- Training: class weights from the 1,050 training images; up to 25 epochs; early stopping after validation loss failed to improve, with best weights restored. Checkpoint selected by minimum validation loss.
- Hardware: Kaggle session detected two Tesla T4 GPUs. This does not by itself show that distributed two-GPU training was used.

| Class | Training images | Weight |
| --- | ---: | ---: |
| AK | 78 | 1.6827 |
| BCC | 236 | 0.5561 |
| BKL | 295 | 0.4449 |
| DF | 58 | 2.2629 |
| MEL | 96 | 1.3672 |
| NV | 99 | 1.3258 |
| SCC | 110 | 1.1932 |
| VASC | 78 | 1.6827 |

## Observed results

Training ended after **16 epochs**. The checkpoint chosen by validation loss was from epoch **11**, with validation loss **1.6795** and validation accuracy **0.2817**. The greatest *observed* validation accuracy during the run was **0.2930** at epoch 13; that was not the checkpoint-selection criterion. For context, predicting the most frequent validation class (BKL: 94 of 355 images) every time would yield **26.48% validation accuracy**. This comparison is only an accuracy reference, not an eight-class macro-F1 baseline.

The CNN result is preliminary and close to that simple accuracy reference. Macro F1, balanced accuracy, class-wise results, and a confusion matrix are pending. No held-out test result has been reported. Training logs, exact executable notebook, and model checkpoint should be added to this repository if available; no missing outputs are reconstructed here.
