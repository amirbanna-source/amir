# Transition to official StyleGAN2-ADA — 10 November 2026

## Decision and boundaries

Amir requested an existing GAN rather than further development of the project's custom implementation. The next candidate is the **official NVIDIA StyleGAN2-ADA PyTorch implementation**, conditional on benign/malignant labels. Custom GAN notebooks 09 and 10 remain historical feasibility/diagnostic work; further custom architecture and BatchNorm ablations are paused.

Official upstream: https://github.com/NVlabs/stylegan2-ada-pytorch
Paper: https://arxiv.org/abs/2006.06676
Pinned source commit observed and cloned on 10 November 2026: `d72cc7d041b42ec8e806021a205ed9349f87c6a4`.

Selection rationale: published limited-data training method, official implementation, supported class conditioning, adaptive discriminator augmentation and established snapshot/log handling. This is a reasoned candidate choice, not proof of suitability for 972 dermoscopy images or a promise of benefit.

## Existing implementation versus pretrained weights

This preparation reuses NVIDIA's architecture, losses and training loop. It does not claim to supply pretrained benign/malignant dermoscopy weights. The current candidate uses no resume checkpoint: official architecture with newly trained weights on our training set. A future pretrained candidate requires exact source/version, training-data provenance, overlap checks against all frozen held-out IDs/content, domain relevance, and compatible resolution/conditioning. Do not load a public ISIC/HAM-trained checkpoint without those checks.

## Prepared artifacts and status

- [Notebook 11](../notebooks/11_isic2019_official_stylegan2ada.ipynb): source-pinned preparation, runtime inventory, export, official dry-run request and disabled training cell.
- [Data adapter](../scripts/prepare_stylegan2ada_data.py): validates the frozen cohort and writes the official ZIP/PNG + dataset.json format; it defines no GAN.
- Verified locally: source compilation; original frozen manifest accepted; altered split/target rejected; synthetic-fixture export contains exactly 972 training images and correct labels, RGB128x128 dimensions and valid archive CRC; overwrite refused.
- **Not yet performed:** real-image export for this recipe, official loader/dry-run execution, Kaggle/PyTorch/CUDA compatibility test, CUDA extension compilation, model training, image generation or downstream CNN evaluation. Local runtime has no PyTorch. Source clone is not a successful runtime check.
- No dependency environment or GPU session was changed during this preparation.

## Fixed dataset

Use existing frozen split fingerprint `4f3037e5dc1726e5a534e3a6f155df3e0e0be52229b057659e097483fd0800cd`. Training: 972 (530 benign/442 malignant); validation: 329; test: 331. Do not recreate the split or enlarge the training set. AK remains excluded. Known lesions remain grouped; missing lesion/patient IDs remain limitations.

The adapter reads training-image content only, verifies identical file copies for a selected ID, converts to RGB and resizes the whole image to128x128 with Pillow bilinear interpolation, without cropping or augmentation. It stores uint8 PNGs; official StyleGAN preprocessing handles numeric scaling. This interpolation implementation differs from the old TensorFlow GAN pilot and is explicitly part of the new recipe. Existing CNN preprocessing remains unchanged.

## Candidate feasibility settings

- Official source and class-conditional mode `--cond=1`.
- Single GPU; `--cfg=auto`; seed142; batch16.
- 20 kimg:20,000 real-image presentations (not independent images and not20,000 optimizer steps); roughly1,250 main batches, with additional official lazy-regularization phases. This is not budget-equivalent to the custom5,000-step pilot.
- ADA enabled with official geometry pipeline `--augpipe=bg`; dataset mirroring disabled. ADA is internal regularization for GAN training, distinct from conventional augmentation of the classifier.
- Official automatic optimizer/R1 settings; record the fully resolved configuration. No assumption that the CNN's Adam0.0001 is appropriate for this GAN.
- Snapshot every tick; all snapshots retained. `--metrics=none` avoids expensive FID calculations during this feasibility candidate; it does not waive quality review.
- No pretrained resume checkpoint.
- Training launch disabled by default. Runtime inventory and official dry run must pass first. Passing dry run does not prove CUDA kernels compile or training fits memory.

Twenty kimg is a short feasibility budget, not a convergence criterion. Do not automatically extend it or call a poor short run a failure of StyleGAN2-ADA. Review generated grids and training behavior, then predeclare a full run if justified. Older official code may need a compatible environment; do not silently alter architecture or losses to accommodate runtime issues.

## Downstream study

After a generator-quality gate, evaluate a fixed synthetic candidate with the same classifier architecture, frozen split, seeds42/43/44, initialization pairing, Adam0.0001, checkpoint rule and threshold0.5. Specify synthetic counts/ratios and classifier update budget before running. GAN-only and GAN+conventional remain separate planned classifier conditions. Their results do not exist yet.

Check visual artifacts, diversity, requested-class influence and near-copy/memorization against training images. Distinct hashes alone do not demonstrate novelty or class validity. Validation/test images must never enter GAN training. Validation evaluation stays real-only; the331 test images remain reserved until the final comparison is frozen.

The study evaluates **augmentation strategies using an established GAN**. It does not claim to invent a new GAN architecture.
