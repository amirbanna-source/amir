# Proposed GAN experiment protocol

Draft prepared 14 October 2026 for Amir and Sandra. No GAN has been trained under this protocol. Eight-class scope still requires supervisor discussion; the edited methodology remains the authoritative study description. This draft proposes implementation choices rather than recording completed experiments.

## Existing evidence

The verified subset contains 1,761 images: 1,050 training, 355 validation, 356 test. Known lesion groups stay within one split. There are 324 images without lesion IDs and no patient IDs, so complete patient-level independence cannot be established.

The completed paired study used CNN seeds 42, 43, 44. Mean validation macro F1 was 0.3156 without augmentation and 0.3477 with conventional augmentation. Class-specific gains were uneven. See [paired study](paired-seed-study.md) and [per-class review](paired-per-class-review.md). Historical exploratory runs remain separate.

## Research question

Under a fixed CNN training budget and fixed real-data split, does replacing a specified portion of training presentations with class-conditional GAN images improve validation macro F1 relative to real-only training and conventional augmentation?

The experiment measures usefulness for this classifier, not clinical validity of the generated images.

## Generator: proposed feasibility pilot

Start with one shared class-conditional convolutional GAN, trained from scratch on all eight classes rather than eight independent GANs. Proposed pilot: 128 x 128 RGB, 128-dimensional Gaussian latent vector, class conditioning in both generator and discriminator, five upsampling stages from 4 x 4 to 128 x 128, tanh generator output, discriminator logits and binary cross-entropy, Adam learning rate 0.0002 with beta1 0.5 and beta2 0.999, batch size 32, one discriminator and one generator update per step.

Use the natural training class distribution for real batches and match fake-label distribution within batches; do not silently rebalance rare classes. Input normalization for the GAN is [-1,1]; convert generated images back to [0,1] for the archived CNN. This simple architecture is a feasibility starting point, not a claim of optimal sample efficiency.

Proposed pilot budget: 5,000 steps with logs and fixed latent/class grids every 500 steps, then a declared review before any main study. Record wall time, GPU, software versions, full model configuration, losses and all checkpoints. Do not treat loss convergence as image-quality evidence. Pilot revisions must be logged before main runs.

Limited-data GANs can overfit. StyleGAN2-ADA is a literature-supported alternative to assess if the simple pilot fails; changing to it requires a new versioned protocol and runtime feasibility check. Discriminator-side augmentation in an ADA generator must be described separately from downstream classifier augmentation.

No external pretrained generator weights are proposed, avoiding an unquantified additional training corpus.

## Data boundaries and selection

- Export only the frozen 1,050 training images to the GAN input.
- Verify the archived split membership fingerprint before export: 900aed297caf894a0a5b0001b3ca974ea954d96104d70781b2eb7a71306227c1.
- Do not use validation or test images to train the GAN or choose its image-quality settings.
- Use the final fixed-budget generator checkpoint if it passes the predefined quality gate. Do not select GAN checkpoints by downstream validation accuracy. Failed runs remain reported.
- Final quality thresholds, architecture details, main training step budget and pilot acceptance rules must be committed before main training. The pilot budget above is not an approved main-run budget.

## Proposed main classifier conditions

| Condition | Training source | Conventional transformations |
| --- | --- | --- |
| A | Real training images | None |
| B | Real training images | Archived flip/rotation/zoom recipe |
| C | Real + synthetic, target 1:1 presentations | None |
| D | Same real + synthetic pool as C | Same recipe on both sources |
| E, control | Replace synthetic slots in C with real samples of the same requested class | None |

Primary synthetic pool proposal: one generated image per original training image, matching each class count exactly (1,050 synthetic images total). No increase targeted at poorly performing validation classes. Keep all real training images available. Target 50% real and 50% synthetic training presentations by a declared sampler; record realized source/class counts.

For E, draw replacement real images from the training split only, with replacement, matching the synthetic-slot labels in C. This checks whether adding real repeated examples gives similar results. It is a control, not a separate new dataset.

Hold the maximum CNN optimizer steps fixed: 33 batches per epoch, at most 25 epochs, batch size 32 with the same final-batch policy across all new conditions. Freeze CNN architecture, optimizer, preprocessing, real-training-derived class weights, dropout seed and validation-loss checkpoint rule from the paired study. Use patience 5.

A fixed mixed-source budget reduces real-image presentations relative to A/B. Report that explicitly; this is a fixed-compute comparison, not equal real-image exposure. Do not mix it with a larger-dataset experiment that silently doubles steps. Any equal-real-exposure experiment is secondary and must be specified separately.

Because the sampler changes, rerun A/B under the common new implementation and compare new conditions within that block. Keep the existing six-run study as historical evidence, not an interchangeable control for a changed pipeline.

## Seeds and pairing

Proposed three main generator seeds: 142, 143, 144; classifier seeds: 42, 43, 44, paired in that order. Each classifier seed starts all conditions from the same saved initial CNN weights and initial optimizer state. Reuse the corresponding generated pool for C/D. Use independent, recorded streams for generation, data order and augmentation.

Three one-to-one generator/classifier pairs characterize combined variability; they do not separately estimate generator and classifier variance. A crossed design would be more informative but costs more runs. Confirm the design and compute budget before training.

## Quality checks before classifier training

1. Save uncurated, reproducible per-class grids and within-class nearest-neighbor comparisons against training images; include failed examples.
2. Reject exact duplicates using hashes of consistently decoded/resized pixels, both against real training images and within the synthetic pool. Log all exclusions and deterministic replacement attempts.
3. Examine near copies using both pixel distances and a fixed declared feature extractor; compare nearest-neighbor distance distributions with real-to-real training distances. Near-copy thresholds must be defined using training data and committed before main runs. A large distance does not establish safety or novelty.
4. Assess within-class diversity and mode collapse. Do not judge quality solely from GAN losses or a single FID value. Small per-class reference sets make distribution metrics unstable; report counts, extractor and limitations if used.
5. Review class plausibility using a written rubric. Amir/Sandra visual review can flag artifacts but cannot verify diagnoses. Conditional labels are requested labels, not clinically confirmed ground truth.
6. Do not filter by the existing CNN's confidence: that could reproduce its errors and favor easy examples.
7. Define how failed classes/runs stop or amend the experiment in advance. Do not silently remove difficult classes or cherry-pick visually appealing images.

Without qualified clinical review, report conditional-label uncertainty as a limitation. Technical image generation alone is insufficient to conclude diagnostic correctness.

## Evaluation and evidence

Primary endpoint: validation macro F1, with per-seed values, mean, sample SD and paired differences. Secondary: accuracy, balanced accuracy, per-class precision/recall/F1 and confusion matrices. Save learning curves, sample counts, training steps, initialization hashes and all run failures. Reuse all eight labels when computing macro F1.

Validation guides classifier selection; repeated choices on this split create selection optimism. Record every tried ratio/model. Only one primary ratio is proposed here; 25% synthetic presentations may be a separately declared sensitivity experiment if resources permit.

Keep the 356-image test set sealed until the protocol, comparisons and selection rules are locked. Final test evaluation should cover the prespecified comparison, with uncertainty accounting for known lesion groups where possible; seeds are not independent patient cohorts.

Archive GAN checkpoints, discriminator/optimizer state for resume, generated-image manifests (seed, latent ID, requested class, source checkpoint, pixel hash), quality reports and classifier evidence. Check data redistribution rights before mirroring generated-image bundles publicly.

## Decisions to resolve with Sandra/supervisor

- Confirm eight-class scope and terminology.
- Accept or revise the simple conditional GAN pilot and define its pass/fail rubric.
- Agree on main generator budget and compute availability.
- Confirm one-to-one three-seed pairing, five classifier conditions and the fixed-compute estimand.
- Freeze image-quality thresholds and label-review limitations before main training.

## Primary references

- [Keras conditional GAN example](https://keras.io/examples/generative/conditional_gan/): class-conditioned generation demonstration on MNIST; not validation for skin lesions.
- [Karras et al., Training Generative Adversarial Networks with Limited Data](https://arxiv.org/abs/2006.06676), NeurIPS 2020: adaptive discriminator augmentation to address limited-data overfitting.
- [Official StyleGAN2-ADA implementation](https://github.com/NVlabs/stylegan2-ada-pytorch): alternative implementation to assess after feasibility review; published requirements do not establish compatibility with the current Kaggle runtime.
