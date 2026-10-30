# Conditional GAN pilot — 30 October 2026

## Outcome

The fixed seed-142 pilot completed **5,000 adversarial steps**, but the final images are unsuitable for classifier augmentation. Assistant visual inspection of the fixed grids at steps 500, 1,000, 2,500 and 5,000 found strongly repeated structures and checkerboard artifacts. The final benign-requested and malignant-requested grids look very similar. This is evidence consistent with mode collapse and weak conditioning; it is not a clinical assessment or a quantified diagnosis-validity test.

Decision: **completed but requires revision**. Do not proceed to downstream CNN training with this pool. Independent Amir/Sandra visual review remains pending. No validation/test prediction or classifier experiment was performed.

## Fixed run

- Frozen cohort: 972 real training images, 530 benign and 442 malignant; original split membership preserved. No validation/test image content entered the generator.
- Conditional convolutional GAN, trained from scratch at 128×128 RGB; latent dimension 128. Model seed142, data order2142, training noise3142, fixed grids4142, audit pool1142.
- Batch32; one discriminator and one generator update per step. Both optimizers Adam0.0002, beta1=0.5, beta2=0.999. No discriminator BatchNorm or dropout.
- Natural-class shuffled training stream, with no class balancing or validation-driven checkpoint selection.
- 160,000 real training presentations: 87,227 benign and 72,773 malignant. Repeated presentations are not additional independent images.
- TensorFlow2.20.0, Python3.13.15, Kaggle T4 session. Two T4 GPUs were detected; no explicit multi-GPU distribution strategy was used.
- Logged training duration: 2,024.44 seconds (**33.74 minutes**), excluding initial input checks and final audit/packaging.
- Final step loss: generator4.4825e-9; discriminator0.16707. These are instantaneous optimization losses, not image-quality scores.

The previously documented preflight I/O improvement was already in place. No architecture, optimizer, label mapping or training-budget change was made during this run.

## Preserved evidence and verification

All 11 checkpoint sets (steps0,500,...,5000) and 22 fixed image grids were retained. Each checkpoint includes model/optimizer state and latent RNG, with separately saved data permutation/cursor/RNG state. Automatic resumed training remains unimplemented and untested.

The final generator and discriminator were saved as Keras models and successfully reloaded. Both optimizer counters in the final TensorFlow checkpoint were independently read as5,000. The reloaded generator reproduced **all972 audit images** with the same saved pixel hashes. The unfiltered pool contains530 requested benign and442 requested malignant images.

A separate local verification checked archive CRCs, nested Keras archives, all5,000 CSV/JSONL history records, finite recorded losses, frozen training IDs/labels/groups, all22 grids, all972 pool filenames/counts and image hashes.

- Exact repeated outputs in the pool: **0**.
- Exact matches against consistently resized/rounded training-image pixels reported by the in-run audit: **0**. The local checker did not reread the original training images.
- Mean pixel SD across generated examples after16×16 resizing:0.03098 for requested benign and0.03100 for requested malignant, on a0–1 scale. This is a descriptive statistic without a pass threshold; it does not measure diagnostic diversity.
- Feature-based near-copy analysis and clinical label verification remain pending. Distinct pixel hashes do not rule out memorization or mode collapse.

## Where the files are saved

The [Kaggle notebook](https://www.kaggle.com/code/amirsalahk/notebook5a45a9891c/edit#Explicit-training-launch), **Version2: GAN pilot5000 steps - verified outputs**, completed successfully with output preservation enabled. Its private output contains:

| Archive | Contents | Size |
| --- | --- | ---: |
| binary_cgan_pilot_all_artifacts.zip | All checkpoints, state, grids, pool, logs and verification |537,271,730 bytes |
| binary_cgan_pilot_review_artifacts.zip | Final checkpoint, all grids, complete pool, both models and reports |109,990,191 bytes |
| binary_cgan_pilot_models.zip | Both final models, config, logs, execution verification and exact-copy report |15,708,583 bytes |

The models ZIP SHA256 is `5c979dd3ea25673b038382bb4efaedd2df07975249c4bb740a6c6881d7219e84`. Large archives and the full generated-image pool are kept in Kaggle; repository documentation points to that saved version.

See [executed notebook09](../notebooks/09_isic2019_binary_cgan_pilot.ipynb), [training script](../scripts/binary_cgan_pilot.py), [execution verification/packaging](../scripts/verify_and_package_cgan_pilot.py), [independent archive checker](../scripts/verify_cgan_pilot.py) and [pilot protocol](gan-protocol-draft.md).

## Next investigation

Diagnose collapse before another full run. One concrete implementation detail to test is the different generator BatchNorm modes used for discriminator fake batches (inference mode) and generator updates (training mode). This is a plausible contributor, not an established cause. Review this together with conditioning and upsampling artifacts using a separately documented diagnostic. Preserve this failed pilot; do not silently replace its results or infer that all GAN approaches fail.
