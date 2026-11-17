# Local GAN classifier comparisons

Prepared 17 November 2026; not executed. These two notebooks train the CNN on an existing accepted synthetic pool. They do not train StyleGAN2-ADA itself.

- [14: real + GAN](../notebooks/14_isic2019_real_plus_gan_local.ipynb)
- [15: real + GAN + conventional augmentation](../notebooks/15_isic2019_real_plus_gan_conventional_local.ipynb)

## Local environment

Open either notebook in Jupyter or VS Code using a compatible local Python 3.11 environment with TensorFlow 2.20.0, NumPy, pandas, Pillow, scikit-learn, JupyterLab and ipykernel. Each notebook contains terminal setup instructions and all implementation code; it does not download scripts, install packages or access Kaggle. CPU is allowed; a compatible TensorFlow GPU is optional. Platform support and GPU installation follow the [official TensorFlow guide](https://www.tensorflow.org/install/pip). Hardware support is not guaranteed simply by opening a notebook.

TensorFlow version and seed-specific initialization hashes are checked against the archived reference. If the local platform cannot use that runtime or initialization does not match, stop and establish a separately documented paired local baseline rather than bypassing checks. These notebooks have not been fitted in TensorFlow here or validated on the user's machine.

## Frozen data compatibility

The current `artifacts/isic2019_binary_split.csv` in the repository differs from the completed study: it contains 1,761 images, maps AK into malignant, and has 1,056/353/352 training/validation/test images. It is preserved unchanged.

The notebooks require a separate local copy named `isic2019_frozen_binary_split_1632.csv`, matching the original verified manifest from commit `18447b4faf5ffda3f5653bf05ac976a7bfe3c143`. The CSV is not added to the public repository by this change; supply the original saved manifest privately. The required cohort is: 1,632 images, AK excluded, 972/329/331 split, fingerprint `4f3037e5dc1726e5a534e3a6f155df3e0e0be52229b057659e097483fd0800cd`. It matches the completed binary comparisons and the campus generator package. Its historical `path` column is ignored; local real images are resolved by image ID under the configured local root. Use the original archived membership, not a newly generated split. Public publication of the metadata copy was blocked by automatic approval review; it remains a local input.

## Proposed first comparison recipe

| Setting | Both conditions |
| --- | --- |
| Real cohort | Frozen 972 training / 329 real validation; no test-image contents read |
| Synthetic pool | Fixed 972 RGB128 PNGs, 530 benign-requested / 442 malignant-requested |
| Per-epoch sample | 486 real + 486 synthetic; 265 benign / 221 malignant per source |
| Sampling | Without replacement within each source/class per epoch; deterministic seed/epoch schedule |
| Pairing | Same pool, source/class schedule, initial weights and seeds 42/43/44 |
| Classifier | Archived four-convolution CNN; Adam 0.0001; training-derived weights |
| Training budget | 972 presentations, 31 batches per epoch; at most 25 epochs |
| Selection | Minimum unweighted validation BCE; patience 5; threshold 0.5 |
| Difference | Notebook 15 adds the recorded flips, rotations and zoom to both sources |

The 1:1 recipe is a proposed, explicitly recorded candidate, not a completed result or an assertion that this is the optimal ratio. Half the real-image presentations are replaced by synthetic presentations to preserve the historical presentation/update ceiling. All source/class memberships and order are identical between notebooks 14 and 15, before conventional transformations. Real-only historical loaders used a different shuffling mechanism; identical initial weights do not imply identical data ordering across all historical conditions.

Early stopping may change actual updates across conditions. Report realized updates and epoch counts. Do not claim exact equal realized compute from a shared maximum budget. Strict fixed-update comparisons would require a separately specified protocol and, where needed, new paired controls.

## Local inputs

Set `PROJECT_ROOT` in each notebook. Defaults are:

| Input | Location relative to project root |
| --- | --- |
| Frozen manifest | artifacts/isic2019_frozen_binary_split_1632.csv |
| Original real images | data/real/ (subdirectories allowed; preserve image IDs) |
| Synthetic image manifest | data/stylegan_pool/pool_manifest.csv |
| Completed human review | data/stylegan_pool/pool_review.json |
| Separate outputs | results/local_comparisons/real_gan/ or real_gan_conventional/ |

Only real training and validation image contents are read. Held-out filenames/manifest metadata may be inventoried for split checks. The two notebooks refuse an existing output folder; keep interrupted evidence and choose a new path deliberately. Their definition cells do not start training, and `EXECUTE_TRAINING` defaults to False.

## Synthetic-pool manifest and acceptance record

The pool must come from a reviewed official StyleGAN2-ADA checkpoint trained only on the frozen real training cohort. The 32 final pilot examples are not a full pool. The old custom conditional-GAN pool is not accepted. Generation of this 972-image pool and its review remain separate future work.

The CSV must contain:

| Column | Contract |
| --- | --- |
| sample_id | Unique synthetic ID, distinct from all real IDs |
| path | PNG path relative to the CSV folder; no absolute paths or parent traversal |
| target | 0 benign-requested or 1 malignant-requested; total counts 530/442 |
| seed | Recorded generation seed; unique within each target |
| pixel_sha256 | SHA256 of decoded RGB uint8 pixel bytes, shape 128×128×3, row-major order |

The notebooks verify image hashes, dimensions, RGB mode, file uniqueness and lack of exact duplicate synthetic pixel hashes. These checks do not establish absence of near-copying or clinical validity.

The review JSON must record the completed human decision and bind it to the exact pool manifest. Schema example below is deliberately NOT an accepted review; replace placeholders with actual evidence only after review.

```json
{
  "generator_type": "official_stylegan2ada",
  "split_fingerprint": "4f3037e5dc1726e5a534e3a6f155df3e0e0be52229b057659e097483fd0800cd",
  "generator_checkpoint_sha256": "REPLACE_WITH_ACTUAL_64_CHARACTER_SHA256",
  "pool_manifest_sha256": "REPLACE_WITH_ACTUAL_CSV_FILE_SHA256",
  "validation_test_images_used": false,
  "approved_for_classifier": false,
  "reviewer": "REPLACE_WITH_REVIEWER",
  "reviewed_at": "REPLACE_WITH_DATE",
  "quality_review": "pending",
  "diversity_review": "pending",
  "training_only_copy_review": "pending"
}
```

A reviewer must document quality, diversity and a training-only copying screen before setting acceptance to true. This human record is not a clinical certification and is not automatically generated by the notebook. Requested synthetic labels remain unverified diagnoses. Follow the [pilot evaluation plan](stylegan2ada-pilot-evaluation.md).

## Outputs and validation performed

Both notebooks preserve all three seeds, initial weights, best Keras checkpoints, histories, per-epoch source/class schedules, real-validation predictions, confusion matrices, metrics, runtime/package metadata and combined result CSVs. No GAN weights are loaded and no test scores are computed.

Checked here: Python syntax for every code cell; identical shared implementation between notebooks; classifier architecture AST equality with the archived lower-rate baseline; all 75 deterministic seed/epoch schedules and exact source/class counts; 31 batches including the partial last batch; missing-input and image-dimension rejection; full fixture input validation; changed synthetic-pixel and unreviewed-pool rejection. Unreadable test placeholders confirmed that test-image contents were not decoded. The current incompatible split was rejected as expected.

Not checked here: TensorFlow fitting, local hardware installation, runtime initialization hashes on the user's machine, checkpoint serialization during fitting, completed classifier results or generator/pool suitability. These remain execution checks. Keep all images, private review records and result directories off GitHub.
