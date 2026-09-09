"""
DermaAgent - Dataset Configuration & Class Mapping
Defines target skin condition classes, descriptions, and dataset directory paths.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# Target Skin Disease Classes (Based on HAM10000 / ISIC standard taxonomy)
DISEASE_CLASSES = {
    "akiec": {
        "full_name": "Actinic Keratosis / Bowen's Disease",
        "category": "Pre-cancerous / Malignant",
        "description": "Pre-cancerous scaly patches caused by sun damage."
    },
    "bcc": {
        "full_name": "Basal Cell Carcinoma",
        "category": "Malignant",
        "description": "Common form of skin cancer arising in basal cells."
    },
    "bkl": {
        "full_name": "Benign Keratosis",
        "category": "Benign",
        "description": "Non-cancerous skin growth (seborrheic keratosis / solar lentigo)."
    },
    "df": {
        "full_name": "Dermatofibroma",
        "category": "Benign",
        "description": "Harmless skin nodule often found on lower legs."
    },
    "mel": {
        "full_name": "Melanoma",
        "category": "Malignant",
        "description": "Serious form of skin cancer developing in melanocytes."
    },
    "nv": {
        "full_name": "Melanocytic Nevi (Moles)",
        "category": "Benign",
        "description": "Common benign moles or birthmarks."
    },
    "vasc": {
        "full_name": "Vascular Lesion",
        "category": "Benign",
        "description": "Blood vessel abnormalities like cherry angiomas or vascular malformations."
    }
}

CLASS_NAMES = sorted(list(DISEASE_CLASSES.keys()))
NUM_CLASSES = len(CLASS_NAMES)
