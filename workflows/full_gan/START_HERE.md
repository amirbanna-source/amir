# Public code mirror

This directory contains code only. Use the complete private `ISIC_Full_GAN_Workflows.zip` handoff, now including all 972 original training and 329 validation images, bundled NVIDIA source, fixed split and CNN starting weights. Do not publish the ZIP or data folders.

# Full GAN and GAN + conventional workflows

This replaces the earlier pilot-oriented handoff. There is no 20-kimg pilot and no manual review JSON to fill in.

## Two entry points

Inside a campus GPU allocation, from this extracted folder:

```bash
bash GAN.sh
```

This checks the original images and classifier environment, trains official conditional StyleGAN2-ADA for **2,000 kimg**, generates 972 synthetic images, and trains the real + GAN CNN across seeds 42/43/44.

Then run:

```bash
bash GAN_PLUS_CONVENTIONAL.sh
```

This reuses the same completed GAN and synthetic pool and trains the CNN with conventional flips, rotations and zoom applied to both real and synthetic training images. It can also be run first: it creates the same GAN/pool if absent, then trains its own classifier condition. Run the two entry points sequentially.

`GAN.ipynb` and `GAN_PLUS_CONVENTIONAL.ipynb` provide the equivalent Run All entry points from this same extracted folder. On a managed cluster, the kernel must be in the allocated GPU session; running a notebook on a login node does not allocate a GPU.

## Complete data handoff

Supplied: the frozen 1,632-image split metadata, GAN-prepared 972-image training ZIP and source audit, official pinned NVIDIA source/license, compatibility checks, archived CNN starting weights for all three seeds, the workflow code, dependency definitions, launchers and notebooks.

**Included: all original 972 training and 329 validation image files used by the CNN, already placed under `data/real/`.** No separate dataset download or manual image copying is needed. `data/required_real_images.csv` lists the 1,301 included image IDs. All 972 training-file byte hashes match the saved GAN export source audit, and all required originals passed ZIP CRC and RGB decoding checks. Original CNN images are used rather than substituting the resized GAN export.

The data dependency is now complete: the two workflows use the included originals and fixed metadata automatically. Integrity checks still stop execution if supplied files are removed or changed. Held-out test images are not included, and no test-image contents were opened during extraction. Cluster software/hardware compatibility remains to be checked on the actual campus system.

## Frozen means fixed membership

“Frozen” means that the same images stay in training, validation and test for all experiments. It is not an error or a separate operation Sandra must perform. The chosen original cohort has 972/329/331 images and excludes AK. Its CSV is included privately and used automatically. The different 1,761-image CSV from the current repository is neither needed nor substituted.

## Cluster environment

The default backend is Apptainer, then Singularity, then Docker if available. An allocated compatible NVIDIA GPU, a driver supporting the supplied CUDA 13 environment, compiler support through the devel image, and enough disk space for images/checkpoints are required. Only one GPU is used; multiple allocated GPUs do not automatically accelerate this recipe.

The wrappers keep PyTorch and TensorFlow in separate container environments. The GAN uses the previously pinned PyTorch 2.11.0/CUDA 13 devel image. The classifier uses the official TensorFlow 2.20.0 GPU image. Small additional dependencies are installed into private folders, without changing cluster system packages. Initial setup needs network access and writes sizable container/dependency caches.

If compute nodes are offline, run `bash SETUP.sh` once on a permitted network-connected node before submitting the GPU job. This prepares software only and starts no training. Some sites require `module load apptainer` first. The cluster's account, partition, GPU request and wall-time directives must come from its actual documentation or Sandra's existing working GPU submission. These details are unknown here and are not guessed by this package.

If containers are unavailable, `ISIC_BACKEND=native` uses existing interpreters selected by `GAN_PYTHON` and `CNN_PYTHON`. They must already provide the documented framework versions. This is a fallback for an administrator-configured environment, not a promise that any system Python will work.

The official installation references are https://apptainer.org/docs/user/main/gpu.html and https://www.tensorflow.org/install/docker . Container availability and the campus machine have not been tested here.

## Recorded experiment recipe

- Full GAN: official source pin `d72cc7d041b42ec8e806021a205ed9349f87c6a4`; seed 142; one GPU; batch 16; conditional auto configuration; ADA geometry; mirror off; no pretrained checkpoint; 2,000 kimg = 2,000,000 real-image presentations. Snapshots every 10 ticks, 4 kimg per tick. Automatic FID/KID disabled.
- This is a fixed full-run candidate, not a claim that 2,000 kimg guarantees convergence, good images or clinical correctness. It is 100 times the earlier 20-kimg feasibility budget, and its duration on this cluster is not yet known.
- Pool: 530 benign-requested and 442 malignant-requested RGB128 images from the final checkpoint, fixed generation seeds, truncation 1. Labels are requested conditions, not independently verified diagnoses.
- Both classifiers: same pool and deterministic per-epoch schedule, same archived starting weights, Adam 0.0001, class weights from real training counts, batch 32, at most 25 epochs, patience 5, minimum real-validation BCE, threshold 0.5, seeds 42/43/44. Each epoch contains 486 real + 486 synthetic images (265/221 of each source), 972 presentations and 31 batches. This replaces half of real presentations to preserve the historical budget ceiling. Early stopping can produce different realized update counts, which are recorded.
- Conventional condition: horizontal/vertical flip, rotation factor 0.05, independent zoom ±0.1, reflect fill; the same settings as the completed conventional comparison.

## Integrity checks, interpretation and reruns

There is no separate pilot training stage. Data/source checks and synthetic-tensor GPU compatibility checks happen automatically before fitting. The automatic configuration check performs no optimizer updates. Original-image hashes, fixed membership, archived CNN weight hashes, checkpoint identity, generated-image counts and pixel hashes are verified. Exact duplicated synthetic images block classifier fitting, but absence of exact duplicates does not establish image quality or absence of near-copying.

No human-review approval file is required to run the comparison. Outputs explicitly retain that quality, diagnostic correctness and near-copy/memorization are not established by automatic execution. Interpret the resulting classifier comparisons together with those limitations and inspect the generated images before making thesis claims.

Completed GAN/pool/condition outputs are reused only when recorded provenance matches. Interrupted or failed training is retained and refused on rerun rather than silently resumed. Exact optimizer/RNG resume is not implemented; allocate adequate wall time and discuss any interrupted attempt before restarting. A partial run is never labeled complete.

Keep the entire `results/` directory: full GAN snapshots/logs, generated pool and provenance, each CNN's models, initial weights, epoch schedules, histories, real-validation predictions and metrics. No final test evaluation occurs. All images, split metadata, model archives and private results stay off GitHub.

## Validation scope

Performed here: source/notebook Python compilation, shell syntax, archived initialization-array hashes, fixed split fingerprint/counts, exact classifier architecture match, deterministic source/class sampling, generated-pool input guards, and orchestration tests with synthetic fixtures/stubbed subprocesses. Real-data verification now confirms 972 original training images and 329 original validation images, matching the frozen membership and expected class counts. All 972 training source-byte hashes match the existing audit; all selected archive entries passed CRC and RGB decoding. Twenty-one redundant source copies also agreed byte-for-byte. No test image content was read. See `data/original_image_verification.json`.

Not performed here: container pulls/builds, campus scheduler allocation, GPU training, full GAN optimization, TensorFlow fitting or completed model-quality evaluation. This package cannot honestly be called campus-tested or guaranteed error-free before those checks are run on the actual cluster.
