# StyleGAN2-ADA pilot evaluation plan

Prepared 13 November 2026, before receiving campus pilot results. Sandra plans to run the supplied package on Monday, 16 November 2026. This is a prospective review plan; all outcomes remain pending.

## Purpose and fixed protocol

Assess whether the official implementation completes the fixed pilot on the campus GPU, and inspect early generated samples for obvious failures. This pilot does not establish classifier improvement, diagnostic accuracy, or readiness for augmentation.

The supplied launcher fixes 972 training-only RGB128 images (530 benign / 442 malignant), labels 0 / 1, seed 142, one GPU, batch 16, conditional auto configuration, ADA geometry, mirror off, no pretrained weights, no automatic metrics, and 20 kimg. Here 20 kimg means 20,000 real-image presentations. Use the existing package unchanged. Validation and test images remain excluded.

Source of protocol and output names: [run_pilot.py](../handoff/stylegan_pilot/run_pilot.py) and [campus instructions](../handoff/stylegan_pilot/README.md). NVIDIA source pin: `d72cc7d041b42ec8e806021a205ed9349f87c6a4`.

## 1. Collect evidence privately

Preserve the entire `results/pilot_seed142/` directory. First request the two final class grids, `training.log`, and `pilot_state.json`; retain the checkpoint and remaining files for verification.

| Evidence | Expected location under results/pilot_seed142/ | Review |
| --- | --- | --- |
| Run state | pilot_state.json | Status, launch flag, timestamps, data/split hashes, configuration, final checkpoint hash |
| GPU checks | runtime/stylegan_gpu_runtime_report.json and runtime_check.log | All required checks passed; actual device and software recorded |
| Official configuration | official_dry_run.log | Resolved settings match fixed protocol |
| Training record | training.log and runs/ | Reached budget, errors, numerical failures, statistics over time |
| Final checkpoint | runs/**/network-snapshot-000020.pkl | Exists; SHA256 agrees with state file |
| Benign outputs | grid_class0_final.png and samples_class0/ | All seeds 0–15 retained |
| Malignant outputs | grid_class1_final.png and samples_class1/ | All seeds 0–15 retained |
| Generation record | generate_class0.log and generate_class1.log | Both classes generated successfully |
| Environment and provenance | pip_freeze.txt and runtime/ | Versions, source, patches and configuration retained |

Keep all images, the complete ZIP, and image-containing result folders off GitHub. Share these privately. Publish only text summaries and reviewed aggregate findings without private paths or identifying environment details.

## 2. Separate training completion from export completion

- [ ] Data and source checks passed.
- [ ] CUDA/runtime and production-gradient checks passed.
- [ ] Official dry run matched the fixed protocol.
- [ ] Training log confirms the intended 20-kimg budget was reached.
- [ ] Final 000020 checkpoint exists and its hash matches the state record.
- [ ] Both class sample exports succeeded: 16 images per class, seeds 0–15, truncation 1.
- [ ] State is complete, with the supporting files above.

The launcher's complete status includes final sample export. If training reached 20 kimg but export failed, record “training complete; export failed”; do not label it a training failure or repeat training automatically. If execution stopped early, record the reached budget and exact error. Preserve the original attempt. Any retry uses a separate output directory and documented reason.

## 3. Inspect the full fixed sample set

Review both 4×4 final grids and all 32 individual outputs. Retain every seed, including poor images. Use identical display scale and settings for both classes. Record reviewer and date, counts, seed IDs, and brief observations in the [results template](stylegan2ada-pilot-results-template.md).

| Aspect | What to inspect | How to record |
| --- | --- | --- |
| Gross image plausibility | Skin/lesion-like structure, boundaries, texture and color continuity | Count visibly broken outputs / 16 for each class; list seeds |
| Artifacts | Checkerboards, repeated tiles, bands, extreme saturation or implausible edges | Description, affected count and seeds |
| Within-class diversity | Repeated silhouettes, textures, backgrounds or near-identical outputs | Suspected repeated seed pairs; qualitative severity |
| Between-class behavior | Same latent seeds under class 0 versus class 1 | Describe similarities/differences without assigning clinical correctness |
| Progress during training | Saved snapshot grids using the same fixed inputs, where available | Snapshot names and observed change; avoid comparing different random samples as progress |

Use “not assessed” if evidence is missing. Do not invent a quality score or pass percentage after viewing results. Different class labels need not produce an obvious visual separation: both groups contain heterogeneous diagnoses. A visual reviewer cannot establish correct benign/malignant conditioning from appearance alone.

Losses and ADA probability provide training context. Neither stable losses nor a completed run alone establishes good images. No FID/KID values are produced by this protocol; record them as “not measured,” never zero.

## 4. Screen for obvious copying using training images only

This is an additional analysis after results arrive, not a feature of the supplied launcher.

- [ ] Compare all 32 outputs with the 972 training images using a documented, reproducible nearest-neighbor method.
- [ ] Fix and record preprocessing and similarity method before inspecting ranked matches; report the method's limitations.
- [ ] Check exact decoded-pixel duplicates separately.
- [ ] Inspect the top matches for each output privately, considering lesion structure and background rather than color alone.
- [ ] Record candidate training IDs, seeds, distances and reviewer judgments in private evidence; publish aggregate findings only.

No numeric similarity cutoff is specified here. A nearest neighbor is not proof of memorization; absence of suspicious matches among 32 samples does not rule it out. Do not use validation/test images for this screen. If screening is not performed, state “memorization not assessed.”

## 5. Record the next decision

| Observed outcome | Next action |
| --- | --- |
| Setup/runtime failure or interrupted training | Diagnose the recorded error, preserve evidence, document any environment change before a new attempt |
| Training complete but export incomplete | Investigate export from the saved checkpoint without automatically retraining |
| Completed pilot with clear corruption, severe repetition or suspicious copying | Investigate cause; hold classifier augmentation and review before allocating more training |
| Completed pilot, no obvious failure in the small reviewed set | Discuss a longer, separately specified run with Sandra; early undertraining can still limit quality |
| Missing outputs or uncertain findings | Mark inconclusive and obtain evidence before deciding |

A 20-kimg feasibility run is short. Poor early samples do not establish that StyleGAN2-ADA is unsuitable; apparently plausible samples do not establish that augmentation will help. Any longer-run budget, checkpoint selection, larger sample review, and later classifier comparison require a written protocol before execution. Keep the held-out test set untouched.

## Completion record

Fill the companion template from returned evidence. Distinguish observed facts, reviewer judgments, missing assessments and proposed next actions. No downstream classifier result is attributable to this pilot until a separate controlled experiment is performed.
