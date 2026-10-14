"""
DermaAgent - ISIC 2019 Dataset Splitter & Preparation Pipeline
Generates reproducible, leakage-aware Train/Val split manifests
using lesion_id grouping to ensure images of the same lesion never cross splits.
"""

import sys
import argparse
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ISIC_DIR = PROJECT_ROOT / "data_raw" / "ISIC2019"
RESULTS_DIR = PROJECT_ROOT / "results"

GT_CSV = ISIC_DIR / "ISIC_2019_Training_GroundTruth.csv"
META_CSV = ISIC_DIR / "ISIC_2019_Training_Metadata.csv"
ZIP_FILE = ISIC_DIR / "ISIC_2019_Training_Input.zip"
EXTRACTED_DIR = ISIC_DIR / "images"

CLASSES_8 = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

def prepare_dataset(extract_images=False, val_size=0.20, random_state=42):
    print("=" * 65)
    print("DermaAgent - ISIC 2019 Dataset Preparation & Split")
    print("=" * 65)

    if not GT_CSV.exists() or not META_CSV.exists():
        raise FileNotFoundError(f"Missing ISIC 2019 CSV files in {ISIC_DIR}")

    print("\n[Step 1/4] Loading Ground Truth and Metadata...")
    df_gt = pd.read_csv(GT_CSV)
    df_meta = pd.read_csv(META_CSV)

    # Merge GroundTruth and Metadata on 'image'
    df = pd.merge(df_gt, df_meta, on="image", how="inner")
    print(f"  - Total merged records: {len(df)}")

    # Determine primary disease class per image
    def get_class(row):
        for col in CLASSES_8:
            if row[col] == 1.0:
                return col
        return "UNK"

    df["target_class"] = df.apply(get_class, axis=1)

    # Filter out UNK (unknown) samples if any
    valid_mask = df["target_class"].isin(CLASSES_8)
    df_valid = df[valid_mask].copy().reset_index(drop=True)
    print(f"  - Valid 8-class records: {len(df_valid)}")

    # Step 2: Leakage-aware grouping
    print("\n[Step 2/4] Setting up lesion-level grouping to prevent data leakage...")
    # For images missing lesion_id, assign a unique identifier per image
    df_valid["group_id"] = df_valid["lesion_id"].fillna(df_valid["image"])
    print(f"  - Unique grouping clusters: {df_valid['group_id'].nunique()}")

    # Perform GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(df_valid, df_valid["target_class"], df_valid["group_id"]))

    train_df = df_valid.iloc[train_idx].copy().reset_index(drop=True)
    val_df = df_valid.iloc[val_idx].copy().reset_index(drop=True)

    print(f"  - Training split count  : {len(train_df)} ({len(train_df)/len(df_valid)*100:.1f}%)")
    print(f"  - Validation split count: {len(val_df)} ({len(val_df)/len(df_valid)*100:.1f}%)")

    # Verify 0 leakage between train and val
    train_groups = set(train_df["group_id"])
    val_groups = set(val_df["group_id"])
    overlap_groups = train_groups.intersection(val_groups)
    print(f"  - Lesion overlap between Train and Val: {len(overlap_groups)} (Strict 0-leakage verified)")

    # Step 3: Save Split Manifests
    print("\n[Step 3/4] Saving split manifests to data_raw/ISIC2019...")
    train_manifest = ISIC_DIR / "isic2019_train_split.csv"
    val_manifest = ISIC_DIR / "isic2019_val_split.csv"

    columns_to_save = ["image", "target_class", "group_id", "age_approx", "anatom_site_general", "sex"] + CLASSES_8
    train_df[columns_to_save].to_csv(train_manifest, index=False)
    val_df[columns_to_save].to_csv(val_manifest, index=False)

    print(f"  - Saved: {train_manifest.name}")
    print(f"  - Saved: {val_manifest.name}")

    # Step 4: Optional extraction of ZIP images
    if extract_images:
        print("\n[Step 4/4] Extracting ZIP archive to data_raw/ISIC2019/images...")
        EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ZIP_FILE, 'r') as zf:
            namelist = [n for n in zf.namelist() if n.lower().endswith('.jpg')]
            total = len(namelist)
            print(f"  - Extracting {total} images...")
            for i, member in enumerate(namelist, 1):
                filename = Path(member).name
                target_dest = EXTRACTED_DIR / filename
                if not target_dest.exists():
                    source = zf.open(member)
                    with open(target_dest, "wb") as target:
                        target.write(source.read())
                if i % 5000 == 0 or i == total:
                    print(f"    * Extracted {i}/{total} images ({i/total*100:.1f}%)")
        print("[SUCCESS] Extraction completed.")
    else:
        print("\n[Step 4/4] Skipping physical disk unzipping.")
        print("  (Note: ML dataloader directly streams from the ZIP file or an extracted folder).")

    print("\n" + "=" * 65)
    print("[SUCCESS] ISIC 2019 Dataset preparation completed!")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ISIC 2019 Dataset splits.")
    parser.add_argument("--extract", action="store_true", help="Extract all images from ZIP file to disk.")
    args = parser.parse_args()
    prepare_dataset(extract_images=args.extract)
