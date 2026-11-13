# StyleGAN2-ADA campus pilot: results template

Status: **Pending campus execution**. Prepared 13 November 2026. Planned execution: Monday, 16 November 2026. Actual date: pending.

Use the [prospective evaluation plan](stylegan2ada-pilot-evaluation.md). Replace pending entries only with supported results. Keep private evidence and all images off GitHub.

## Experiment record

| Field | Planned value | Observed value / evidence |
| --- | --- | --- |
| Run/attempt ID | pilot_seed142 | Pending |
| Actual start / finish; timezone | Record timestamps | Pending |
| GPU; PyTorch; CUDA | One compatible NVIDIA GPU; supplied environment | Pending |
| Training input | 972 RGB128; benign 530, malignant 442 | Pending |
| Label mapping | 0 benign; 1 malignant | Pending |
| Dataset SHA256 | 028216f4780719022b0200463c0e411a5cc18b64b4f4460e55c49a227a287f6d | Pending |
| Frozen split fingerprint | 4f3037e5dc1726e5a534e3a6f155df3e0e0be52229b057659e097483fd0800cd | Pending |
| NVIDIA source pin | d72cc7d041b42ec8e806021a205ed9349f87c6a4 | Pending |
| Seed / batch / GPUs | 142 / 16 / 1 | Pending |
| Configuration | auto; conditional; ADA geometry; mirror off | Pending |
| Pretrained weights / automatic metrics | None / none | Pending |
| Budget | 20 kimg, 20,000 real-image presentations | Pending reached kimg |
| Protocol deviations | None planned | Pending; describe any change |
| Final checkpoint / SHA256 | network-snapshot-000020.pkl | Pending; omit private absolute paths |

## Execution and evidence

| Check | Outcome: pass / fail / not assessed | Evidence / explanation |
| --- | --- | --- |
| Data and source verification | Pending | Pending |
| Runtime and gradient checks | Pending | Pending |
| Official dry run | Pending | Pending |
| Training reached 20 kimg | Pending | Pending |
| Final checkpoint and hash verified | Pending | Pending |
| Both class exports: 16 seeds each | Pending | Pending |
| State complete with supporting evidence | Pending | Pending |
| Logs and full results retained privately | Pending | Pending |

Training outcome: pending. Sample-export outcome: pending. Error and reached budget if interrupted: pending. Runtime duration, including scope of timing: pending.

## Sample review

Reviewer(s): pending. Review date: pending. Sample selection: final checkpoint, seeds 0–15 for each class, truncation 1; 32 unfiltered outputs.

| Assessment | Benign: 16 outputs | Malignant: 16 outputs | Evidence / seeds |
| --- | --- | --- | --- |
| Outputs actually reviewed | Pending | Pending | Pending |
| Visibly broken outputs, count / 16 | Pending | Pending | Pending |
| Visible artifacts, count / 16 and description | Pending | Pending | Pending |
| Suspected repeated pairs / diversity observations | Pending | Pending | Pending |
| Change across saved training snapshots | Pending | Pending | Pending |

Matched-seed class comparison: pending. Clinical correctness of conditioning: not established by visual inspection.

## Training-image similarity screen

Status: **Not assessed** until the additional analysis is performed.

| Field | Result |
| --- | --- |
| Preprocessing and nearest-neighbor method | Not assessed |
| Reference set | 972 training images only, if screen is performed |
| Generated images screened / 32 | Not assessed |
| Exact decoded-pixel duplicate count | Not assessed |
| Suspicious matches after private visual review | Not assessed |
| Reviewer/date; private evidence reference | Not assessed |
| Limitations | Small fixed sample; nearest-neighbor findings cannot establish absence of memorization |

## Thesis summary table

| Outcome | Result | Interpretation / limit |
| --- | --- | --- |
| Campus execution feasibility | Pending | Requires logs, state and checkpoint evidence |
| Training budget reached | Pending | 20 kimg is a feasibility budget |
| Final class exports | Pending | Expected 16 outputs per class |
| Image plausibility and artifacts | Pending | Exploratory visual assessment |
| Sample diversity | Pending | Only the fixed reviewed sample |
| Training-image copying screen | Not assessed | Additional analysis required |
| FID / KID | Not measured | Disabled in this pilot |
| Downstream classifier benefit | Not evaluated | Separate controlled comparison required |
| Held-out test evaluation | Not performed in this pilot | Test set remains untouched |

## Decision and thesis paragraph

Decision: pending. Choose: investigate runtime; repair export; investigate image failures; inconclusive / gather evidence; or propose a separately specified longer run.

Reason and supporting evidence: pending.
Proposed next experiment, if any: pending discussion with Sandra.

Draft paragraph — fill brackets before thesis use:

“The conditional StyleGAN2-ADA feasibility pilot used 972 training images at 128×128 resolution, seed 142 and a planned budget of 20 kimg. On [actual GPU/software], training [reached X kimg / did not launch], and final sample export [outcome]. Review of [N] unfiltered outputs found [observations]. A training-only similarity screen [method and findings / was not performed]. These exploratory findings [support investigating a longer run / require further investigation / are inconclusive]. No classifier augmentation benefit or clinical validity was established, and the held-out test set was not evaluated.”
