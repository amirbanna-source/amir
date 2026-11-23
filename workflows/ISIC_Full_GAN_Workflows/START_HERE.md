# Full GAN workflows — images supplied separately

This GitHub folder contains the complete workflow code and NVIDIA source/license. **No images or image-containing archives are included.** Sandra uses the original dataset she already has access to.

## Run

Clone/download this repository, open `workflows/full_gan/`, and run inside an allocated Linux/NVIDIA GPU session:

```bash
ISIC_IMAGE_DIR=/absolute/path/to/original-images bash GAN.sh
```

After that condition finishes:

```bash
ISIC_IMAGE_DIR=/absolute/path/to/original-images bash GAN_PLUS_CONVENTIONAL.sh
```

The image directory can contain subdirectories and the original source labels; images are matched by ISIC filename, not source-folder diagnosis. Only the fixed 972 training and 329 validation image contents are read. Test contents are not opened. Alternatively, place the originals in `data/real/` and omit `ISIC_IMAGE_DIR`.

The same-folder `GAN.ipynb` and `GAN_PLUS_CONVENTIONAL.ipynb` are Run All entry points. Set their `IMAGE_DIR` field to Sandra's existing dataset path. They call the same launchers; there is no pilot stage and no manual approval JSON.

## Automatic setup

1. Select Apptainer/Singularity, then Docker if available. Native framework interpreters are an administrator-configured fallback.
2. Prepare separate PyTorch 2.11.0/CUDA 13 and TensorFlow 2.20.0 environments.
3. Retrieve the already published, image-free reference archive from a pinned project commit and verify its SHA256. Extract only the original fixed split and three CNN initialization arrays; no source images are downloaded.
4. Verify fixed membership, original training-source fingerprint and CNN initial weights.
5. Build the official GAN-format RGB128 training ZIP **locally from Sandra's original images**, using the frozen training IDs only. Its actual hash and source audit are recorded; ZIP timestamps need not match the earlier prepared archive.
6. Run full GAN training, generate the synthetic pool and train the selected classifier condition.

If compute nodes have no internet, run `bash SETUP.sh` once on a permitted network-connected node before the GPU job. Cluster modules, account/partition, GPU allocation and job time limits remain site-specific. The supplied CUDA 13 image needs a compatible NVIDIA driver. The actual campus cluster and containers have not been tested here.

## Two conditions and fixed recipe

| Entry point | Behavior |
| --- | --- |
| GAN.sh / GAN.ipynb | Full official conditional StyleGAN2-ADA, fixed synthetic pool, then real + GAN CNN |
| GAN_PLUS_CONVENTIONAL.sh / GAN_PLUS_CONVENTIONAL.ipynb | Reuse the same completed GAN/pool; train the CNN with conventional transformations on both sources |

Either condition can create the GAN/pool if absent. Run sequentially. Full GAN: seed 142, one GPU, batch 16, conditional auto config, ADA geometry, mirror off, no pretrained weights, **2,000 kimg** (2,000,000 real-image presentations), 4-kimg ticks, snapshots every 10 ticks, automatic metrics disabled. No preliminary 20-kimg run occurs. This budget is not a guarantee of convergence or image quality.

Pool: 530 benign-requested and 442 malignant-requested images at truncation 1. CNN: archived four-convolution architecture and starting weights, seeds 42/43/44, Adam 0.0001, class weights from the real training cohort, 25 maximum epochs, patience 5, minimum real-validation BCE and threshold 0.5. Each epoch samples 486 real +486 synthetic images, with 265/221 labels from each source; 972 presentations and 31 batches. Condition two adds horizontal/vertical flips, rotation factor 0.05 and independent zoom ±0.1 with reflect fill. Matched ceilings do not imply identical realized updates under early stopping; counts are saved.

## Data and results

The fixed cohort has **1,632 images with AK excluded**, matching the completed comparisons. The different later 1,761-image repository CSV is not used. There is no need to repair or recreate a split manually.

Inputs are mounted read-only in the container backends. Generated datasets, caches, models, synthetic images and results are created locally and ignored by Git. No classifier test evaluation runs. Exact duplicated synthetic outputs block classifier fitting, but automatic completion does not establish image quality, clinical label correctness or absence of near-copying. Review those limitations before interpreting results.

Completed outputs are reused only when provenance agrees. Failed/incomplete training is preserved and refused rather than silently resumed. Exact optimizer/RNG resume is not implemented. Allocate sufficient wall time.

## Validation

The original-data package previously verified all 972 training source hashes and all 1,301 required image decodes. The source-only exporter was checked against those same originals and the earlier training export. Shell/Python/notebook syntax, restored weights, source-file hashes, source/class sampling and orchestration guards were checked. Full GPU training, container setup and campus execution remain untested.

Upstream source is unmodified except for the two recorded API compatibility patches applied at runtime. NVIDIA's documentation images are omitted; retained source hashes and license are preserved. References: https://apptainer.org/docs/user/main/gpu.html and https://www.tensorflow.org/install/docker .
