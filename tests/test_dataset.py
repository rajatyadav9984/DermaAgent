"""
DermaAgent - Dataset Validation & Split Integrity Tests
Validates ISIC 2019 ground truth formats, class dimensions, and split leakage absence.
"""

from pathlib import Path
import pytest
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ISIC_DIR = PROJECT_ROOT / "data_raw" / "ISIC2019"

CLASSES_8 = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

def test_isic2019_files_exist():
    """Verify presence of core ISIC 2019 raw and manifest files."""
    gt_csv = ISIC_DIR / "ISIC_2019_Training_GroundTruth.csv"
    meta_csv = ISIC_DIR / "ISIC_2019_Training_Metadata.csv"
    zip_file = ISIC_DIR / "ISIC_2019_Training_Input.zip"

    assert gt_csv.exists(), "GroundTruth CSV missing"
    assert meta_csv.exists(), "Metadata CSV missing"
    assert zip_file.exists(), "Zip archive missing"

def test_isic2019_groundtruth_classes():
    """Verify ground truth columns match the 8 official ISIC classes."""
    gt_csv = ISIC_DIR / "ISIC_2019_Training_GroundTruth.csv"
    df = pd.read_csv(gt_csv)
    for col in CLASSES_8:
        assert col in df.columns, f"Missing class column {col} in ground truth"
    assert len(df) == 25331, f"Expected 25331 rows, got {len(df)}"

def test_leakage_aware_three_way_split_integrity():
    """Verify strict 0-leakage (zero image and group overlap) across Train, Val, and Test splits."""
    train_manifest = ISIC_DIR / "isic2019_train_split.csv"
    val_manifest = ISIC_DIR / "isic2019_val_split.csv"
    test_manifest = ISIC_DIR / "isic2019_test_split.csv"

    assert train_manifest.exists(), "Train manifest missing"
    assert val_manifest.exists(), "Val manifest missing"
    assert test_manifest.exists(), "Test manifest missing"

    df_train = pd.read_csv(train_manifest)
    df_val = pd.read_csv(val_manifest)
    df_test = pd.read_csv(test_manifest)

    splits = {
        "train": df_train,
        "val": df_val,
        "test": df_test
    }

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]

    for s1, s2 in pairs:
        # Check image overlap
        img_overlap = set(splits[s1]["image"]).intersection(set(splits[s2]["image"]))
        assert len(img_overlap) == 0, f"Duplicate images found between {s1} and {s2}: {img_overlap}"

        # Check lesion group overlap
        grp_overlap = set(splits[s1]["group_id"]).intersection(set(splits[s2]["group_id"]))
        assert len(grp_overlap) == 0, f"Lesion group leakage detected between {s1} and {s2}: {grp_overlap}"

def test_three_way_class_distribution():
    """Verify all 8 classes are present with non-zero counts across Train, Val, and Test splits."""
    for split_name in ["train", "val", "test"]:
        manifest = ISIC_DIR / f"isic2019_{split_name}_split.csv"
        df = pd.read_csv(manifest)
        assert len(df) > 0, f"Empty manifest for {split_name}"
        classes_present = set(df["target_class"].unique())
        for c in CLASSES_8:
            assert c in classes_present, f"Class {c} missing in {split_name} split!"
            count = (df["target_class"] == c).sum()
            assert count > 0, f"Class {c} has 0 samples in {split_name} split!"

def test_isic2019_class_weights():
    """Verify ISIC 2019 class weights exist and have 8 positive values."""
    import torch
    weights_path = PROJECT_ROOT / "model" / "isic2019_class_weights.pt"
    assert weights_path.exists(), "isic2019_class_weights.pt missing"
    data = torch.load(weights_path, weights_only=False)
    assert "classes" in data
    assert data["classes"] == CLASSES_8
    assert "weights" in data
    assert len(data["weights"]) == 8
    assert (data["weights"] > 0).all()
