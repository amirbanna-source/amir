# Dataset audit

## Inputs

- Image source: [Skin Cancer ISIC, nine-class Kaggle dataset](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic).
- Reference diagnoses: `ISIC_2019_Training_GroundTruth.csv` from the [ISIC 2019 challenge](https://challenge.isic-archive.com/data/#2019).
- Lesion grouping: `ISIC_2019_Training_Metadata.csv` from the same challenge.

The two official CSVs were added to the Kaggle notebook as separate private dataset inputs. The image dataset was attached as another input; the large archive did not need to be uploaded into ChatGPT.

## Initial audit

| Check | Observed result |
| --- | ---: |
| Image files in Kaggle dataset | 4,714 |
| Unique image IDs/filenames | 2,202 |
| IDs present in more than one file | 2,202 |
| IDs with conflicting Kaggle folder labels | 155 |
| Filenames occurring in original Train and Test folders | 18 |

Repeated file pairs can have different folder labels even when the image bytes are identical. For example, some IDs assigned to `nevus` in one folder appeared as `actinic keratosis` in another. We therefore did not interpret folder location as verified diagnosis, or use the supplied split for evaluation.

## Official matching and checks

| Check | Observed result |
| --- | ---: |
| Unique IDs matched to official ISIC 2019 labels | 1,761 |
| IDs absent from ISIC 2019 ground truth and excluded | 441 |
| Verified images with lesion ID | 1,437 |
| Verified images without lesion ID | 324 |
| Distinct known lesion IDs | 1,062 |
| Known lesions represented by multiple images | 309 |
| Images belonging to those multi-image lesions | 684 |
| Verified IDs whose file copies differed in bytes | 0 |
| Distinct IDs sharing identical image content | 0 |
| Known lesions with multiple official diagnoses | 0 |
| Unreadable retained images | 0 |

One copy per matched image ID was retained. Exact content checks do not rule out visually similar images with different bytes, and absent patient IDs mean patient-level independence cannot be established.

## Class distribution and split

The split was produced using stratified group splitting, with known lesion ID as the group and image ID as the fallback when lesion ID was missing. These are observed counts from the notebook; the split generation code and its random seed should be preserved when exporting the notebook.

| Official class | Train | Validation | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| AK | 78 | 26 | 25 | 129 |
| BCC | 236 | 72 | 84 | 392 |
| BKL | 295 | 94 | 89 | 478 |
| DF | 58 | 22 | 24 | 104 |
| MEL | 96 | 28 | 30 | 154 |
| NV | 99 | 34 | 37 | 170 |
| SCC | 110 | 50 | 32 | 192 |
| VASC | 78 | 29 | 35 | 142 |
| **Total** | **1,050** | **355** | **356** | **1,761** |

The original ISIC Gallery was considered for external validation but has not yet been assembled, checked for overlap, or mapped to compatible verified diagnoses. It is not the held-out test set reported above.
