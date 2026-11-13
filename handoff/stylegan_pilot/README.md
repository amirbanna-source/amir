# Sandra's fixed StyleGAN2-ADA pilot

Use the complete **StyleGAN2ADA_Pilot_Ready.zip** handoff shared separately with Sandra. **Do not upload the training images or the complete handoff ZIP to GitHub.** This GitHub directory is a readable code mirror; it omits the training images and bundled NVIDIA source. Copying this directory alone is not the complete runnable package.

## Run

1. Extract the handoff ZIP on the campus Linux GPU machine.
2. Open a terminal in `StyleGAN2ADA_Pilot`.
3. Run `bash START_PILOT.sh`.

No training settings need editing. The launcher uses the supplied pinned official PyTorch/CUDA Docker definition when Docker is installed. GPU-enabled Docker and a compatible NVIDIA driver must already be available. Otherwise it uses the documented host environment: Python, PyTorch 2.11.0/CUDA 13, nvcc and a compatible C++ compiler. On a managed campus cluster, run inside its allocated GPU session.

On an already configured host environment, open `Run_Pilot.ipynb` from the extracted bundle and select **Run All**. [Notebook 13](../../notebooks/13_isic2019_stylegan2ada_pilot.ipynb) mirrors that notebook; it must be run from the complete bundle, not from this repository's notebooks directory.

## Included in the handoff

The exact 972-image training ZIP and audit; pinned original NVIDIA source and license; source hashes; both automatic API compatibility patches; runtime checks; this fixed launcher; a container definition; and English run instructions.

## Fixed experiment

| Setting | Value |
|---|---|
| Images | 972 training-only RGB128, 530 benign / 442 malignant |
| Conditioning | benign 0, malignant 1 |
| NVIDIA commit | d72cc7d041b42ec8e806021a205ed9349f87c6a4 |
| GPU / batch / seed | 1 / 16 / 142 |
| Budget | 20 kimg = 20,000 real-image presentations |
| Configuration | auto, conditional, ADA geometry, mirror off |
| Pretrained weights / metric downloads | none / none |
| Snapshot interval / workers | every tick / 2 |

The launcher checks the data and source hashes, CUDA kernels, first/second derivatives, production mixed-precision G/D backward, ADA+R1, path-length gradients, and official dry-run settings before any optimizer update. A failed check stops launch.

## Outputs

Preserve `results/pilot_seed142/`, including the state file, environment, patches, logs, official snapshots/statistics and final per-class sample grids. Send Amir the grids and training log first; retain the final checkpoint. Existing started or failed runs are protected. No automatic resume is implemented.

## Validation and limits

Data CRC/hash, all image modes/dimensions, labels, IDs, code compilation and shell syntax were checked. Guard checks used mocked GPU metadata only to verify that preparation cannot launch training and started runs are refused. The runtime-check sequence passed on a real Kaggle T4 / PyTorch 2.11.0+cu130 / CUDA 13 environment, including production mixed precision, R1 and path-length double-backward; [the report is here](../../artifacts/stylegan_campus_runtime_report.json). The runtime test cloned the exact pinned source; the bundle's copy was separately verified against that pin and file hashes.

**The 20-kimg pilot has not run.** The Docker definition uses an official published image but has not been built here, and campus hardware/driver/container permissions have not been tested. Full training, checkpoint completion and image quality remain to be evaluated by Sandra's run. Validation and test images are absent from this package.

## Review after execution

Use the [pilot evaluation checklist](../../docs/stylegan2ada-pilot-evaluation.md) and [results template](../../docs/stylegan2ada-pilot-results-template.md) when campus outputs return. These prospective documents do not change the supplied training package. Keep all image evidence and the complete ZIP off GitHub.

## Automatic checks after outputs return

From the repository root, run:

```bash
python scripts/review_stylegan_pilot.py /private/path/to/results/pilot_seed142
```

The read-only checker requires Python and Pillow (`python -m pip install Pillow` if needed). It prints a JSON report and exits 0 only when all implemented technical checks pass; exit 1 means missing, failed, or inconsistent evidence. It checks recorded settings, hashes, runtime status, the logged budget, a uniquely located final checkpoint, all 16 seed images per class, and both grids. It hashes the checkpoint without loading it.

It never starts training or changes the returned results. It does not establish image quality, diversity, clinical correctness or absence of memorization, and is not a substitute for reviewing the full logs. A failed export can coexist with completed training: use the individual checks to diagnose the result. Keep returned files and reports private until reviewed.

Validation: synthetic fixtures covered a complete evidence set, missing outputs, checkpoint modification, corrupt sample images, failed state and malformed state JSON. Campus outputs have not yet been checked.
