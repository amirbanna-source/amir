# CNN baseline without augmentation

## Setup

- Inputs: 128 × 128 RGB, pixel values scaled to [0, 1], batch size 32.
- Labels: `AK=0, BCC=1, BKL=2, DF=3, MEL=4, NV=5, SCC=6, VASC=7`.
- Architecture: convolution blocks with 32, 64, 128, and 256 filters; global average pooling; dense layer of 128 units; dropout 0.35; eight-way softmax.
- Optimizer/loss: Adam at learning rate 0.001 and sparse categorical cross-entropy.
- Training: class weights from the 1,050 training images; up to 25 epochs; early stopping with patience 5 and best weights restored. Checkpoint selected by minimum validation loss.
- Hardware: Kaggle detected two Tesla T4 GPUs; the code does not use distributed training.

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

## Training runs

| Run | Epochs trained | Best epoch by validation loss | Best validation loss | Accuracy at selected checkpoint | Macro F1 at selected checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original exploratory run (Sep 24) | 16 | 11 | 1.6795 | 0.2817 | Not measured |
| Recovered rerun (Sep 29) | 25 | 24 | 1.5410 | 0.4141 | 0.4063 |

The original Kaggle session lost its split CSV and model checkpoint. On Sep 29 the same split construction and baseline cells were rerun with the three datasets attached. The recovered split again contained 1,050 training, 355 validation, and 356 test images. The selected rerun checkpoint has **validation accuracy 0.4141**, **macro F1 0.4063**, and **balanced accuracy 0.4611**. The saved checkpoint and per-image validation outputs are in [artifacts](../artifacts/). These are two separate training runs; do not combine the old learning curve with the recovered checkpoint.

| Class | Validation support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| AK | 26 | 0.20 | 0.62 | 0.30 |
| BCC | 72 | 0.41 | 0.51 | 0.45 |
| BKL | 94 | 0.44 | 0.17 | 0.25 |
| DF | 22 | 1.00 | 0.05 | 0.09 |
| MEL | 28 | 0.85 | 0.39 | 0.54 |
| NV | 34 | 0.51 | 0.79 | 0.62 |
| SCC | 50 | 0.35 | 0.26 | 0.30 |
| VASC | 29 | 0.58 | 0.90 | 0.70 |

The confusion matrix and image-level predictions were generated only for validation. DF recall is particularly weak (1 of 22), and 15 of 28 melanoma images were predicted as NV. This model is a preliminary research baseline and is not suitable for clinical use. The held-out test set has **not been evaluated**.
