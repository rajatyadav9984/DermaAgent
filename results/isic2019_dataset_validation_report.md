# ISIC 2019 Dataset Validation & Overlap Report
**Generated at**: Automated System Verification
**Dataset Location**: `C:\DermaAgent\data_raw\ISIC2019`

## 1. File Existence & Integrity
| File | Path | Exists | Size |
|---|---|---|---|
| GroundTruth CSV | `ISIC_2019_Training_GroundTruth.csv` | ✅ Yes | 1.23 MB |
| Metadata CSV | `ISIC_2019_Training_Metadata.csv` | ✅ Yes | 1.16 MB |
| Input ZIP Archive | `ISIC_2019_Training_Input.zip` | ✅ Yes | 9.10 GB |

## 2. Ground Truth Class Distribution
- **Total Samples**: 25331
- **Unique Image IDs**: 25331
- **Duplicate IDs**: 0

| Disease Class | Description / Taxonomy | Sample Count | Percentage |
|---|---|---|---|
| **MEL** | Melanoma (Malignant) | 4522 | 17.85% |
| **NV** | Melanocytic Nevi (Moles) | 12875 | 50.83% |
| **BCC** | Basal Cell Carcinoma (Malignant) | 3323 | 13.12% |
| **AK** | Actinic Keratosis (Pre-cancerous, corresponds to akiec) | 867 | 3.42% |
| **BKL** | Benign Keratosis (Solar lentigo / Seborrheic keratosis) | 2624 | 10.36% |
| **DF** | Dermatofibroma (Benign) | 239 | 0.94% |
| **VASC** | Vascular Lesion (Benign) | 253 | 1.00% |
| **SCC** | Squamous Cell Carcinoma (Malignant - New in ISIC 2019) | 628 | 2.48% |
| **UNK** | None of the others / Out-of-distribution | 0 | 0.00% |

## 3. Metadata Analysis & Patient Grouping
- **Total Metadata Records**: 25331
- **Images with valid `lesion_id`**: 23247 (91.8%)
- **Images without `lesion_id`**: 2084
- **Unique Lesion Clusters**: 11847
> [!IMPORTANT]
> When splitting into Train and Validation sets, lesions with the same `lesion_id` must stay in the same split to prevent **data leakage**.

## 4. Image ID Consistency Verification
- **Images in ZIP**: 25331
- **Images in Ground Truth**: 25331
- **Missing in ZIP**: 0
- **Missing in Metadata**: 0
- **Integrity Status**: ✅ 100% ID Match (25,331 images match exactly)

## 5. HAM10000 & ISIC 2019 Overlap Analysis
- **Total HAM10000 images**: 10015
- **Overlapping images in ISIC 2019**: 10015 (100.0% of HAM10000)
- **New Non-HAM10000 images in ISIC 2019**: 15316
> [!CAUTION]
> HAM10000 is directly incorporated into the ISIC 2019 challenge. Therefore, images must **NOT** be counted twice or merged naively. Models trained on ISIC 2019 and evaluated on HAM10000 test splits must strictly observe dataset boundary rules.

## 6. Class Taxonomy & Mapping Strategy
| HAM10000 (7 Classes) | ISIC 2019 (8 Classes) | Handling in DermaAgent |
|---|---|---|
| `akiec` | `AK` | Mapped as Actinic Keratosis; models retain their respective mappings |
| `bcc` | `BCC` | Exact match |
| `bkl` | `BKL` | Exact match |
| `df` | `DF` | Exact match |
| `mel` | `MEL` | Exact match |
| `nv` | `NV` | Exact match |
| `vasc` | `VASC` | Exact match |
| *N/A* | `SCC` | **Squamous Cell Carcinoma** added as separate 8th class |

