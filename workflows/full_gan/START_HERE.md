# Full GAN workflows on the AK-inclusive split

Both workflows use the verified [GitHub manifest](../../artifacts/isic2019_binary_split.csv). AK maps to malignant (target 1) for this study.

| Split | Benign | Malignant | Total | AK |
| --- | ---: | ---: | ---: | ---: |
| Train | 538 | 518 | 1,056 | 77 |
| Validation | 179 | 174 | 353 | 26 |
| Test | 177 | 175 | 352 | 26 |
| Total | 894 | 867 | 1,761 | 129 |

## Verify setup

Run from `workflows/full_gan/` in the repository or from the extracted code-only ZIP.

```bash
bash SETUP.sh --verify-reference
ISIC_IMAGE_DIR=/absolute/path/to/original-images bash RUN_WORKFLOW.sh preflight
```

The first command uses host Python with pandas to verify the manifest and initialization files. The second prepares the software environments and runs the full input preflight without allocating a GPU. It checks split membership and group isolation, resolves originals by ISIC ID, rejects conflicting copies, decodes all 1,409 train/validation images, reloads all three CNN initializations, and builds and verifies the 1,056-image RGB128 GAN ZIP locally. It records the successful check in `results/ak_inclusive/input_preflight.json`. Test-image contents are never opened.

A repository checkout reads the manifest and initializations directly from `artifacts/`. The standalone ZIP downloads those four files from a pinned GitHub commit and verifies their hashes. Supply originals separately via `ISIC_IMAGE_DIR` or `data/real/`. Split and label assignments come from the manifest. Its source paths and the folder names of the supplied images are ignored.

Setup selects Apptainer/Singularity, then Docker, then host Python interpreters. On clusters with offline compute nodes, run preflight on a network-connected node with access to the originals. GPU training uses pinned PyTorch 2.11.0/CUDA 13 and TensorFlow 2.20.0 environments. Native mode requires frameworks to be installed in the interpreters selected by `GAN_PYTHON` and `CNN_PYTHON`.

## Run in an allocated NVIDIA GPU session

```bash
export ISIC_IMAGE_DIR=/absolute/path/to/original-images
bash GAN.sh
bash GAN_PLUS_CONVENTIONAL.sh
```

The notebook entry points are `GAN.ipynb` and `GAN_PLUS_CONVENTIONAL.ipynb`, or notebooks 14 and 15 in the main repository. Set `IMAGE_DIR` in each notebook. The launchers repeat preflight and check the GPU runtime and official StyleGAN configuration before training.

Workflow 14 trains full conditional StyleGAN2-ADA from scratch for **2,000 kimg** with seed 142, one GPU, batch 16, ADA geometry, mirroring disabled and snapshots every 40 kimg. It then generates a shared pool at truncation 1 (538 benign-requested and 518 malignant-requested) and trains the real + GAN CNN. Workflow 15 reuses the same full checkpoint and pool and applies flips, rotation and zoom to both sources. Run the two launchers sequentially. If started first, workflow 15 builds the same shared GAN and pool. There is no pilot. Matching provenance is required for reuse.

Each GAN classifier epoch uses 528 real and 528 synthetic examples, with 269 benign and 259 malignant examples from each source, sampled without replacement within an epoch. The two conditions use identical seed-specific epoch schedules. The classifiers use seeds 42/43/44, the AK-inclusive baseline CNN (5×5 convolutions, four pooling layers, flattening and dropout 0.45) and its saved starting arrays, Adam 0.0001, training class weights (1056/1076 and 1056/1036), batch 32 (33 batches per epoch), up to 25 epochs, patience 5, and the minimum real-validation BCE checkpoint at threshold 0.5.

## Compare classifier scores

Sandra's validation predictions match all 353 validation records for each seed. Her archived baseline configuration records Adam **0.001**, TensorFlow **2.19.0** and **31** batches per epoch. The current notebook source specifies Adam 0.0001, so the notebook alone cannot establish the archived run's settings. Use these scores as historical results. For a controlled augmentation comparison, run the matched real-only classifier in the same GPU environment as workflows 14 and 15.

```bash
bash RUN_WORKFLOW.sh baseline
```

It uses all 1,056 reals each epoch, with the same Adam rate, runtime, initial arrays, class weights, validation protocol and 33-batch ceiling as the GAN classifiers. Its result table is at `results/ak_inclusive/cnn_real_only/results.csv`.

Data, caches and outputs stay local. The updated outputs use `results/ak_inclusive/` and a cohort-specific GAN ZIP, so earlier cluster outputs can be preserved for review. Interrupted GAN runs require a fresh working directory or moving the old output out of the active path. Assess synthetic image quality and near-copying before interpreting validation results.
