"""
DermaAgent - HAM10000 Dataset Downloader & Organizer
Downloads the 10,015 image HAM10000 benchmark dataset from Kaggle/Dataverse,
reads HAM10000_metadata.csv, performs patient-level (lesion_id) train/val/test splitting
to prevent data leakage, and organizes images into dataset/train, dataset/val, dataset/test.
"""

import os
import sys
import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data_raw" / "HAM10000"
DATASET_DIR = PROJECT_ROOT / "dataset"

CLASS_MAPPING = {
    "akiec": "akiec",
    "bcc": "bcc",
    "bkl": "bkl",
    "df": "df",
    "mel": "mel",
    "nv": "nv",
    "vasc": "vasc"
}

def organize_ham10000():
    metadata_path = DATA_RAW_DIR / "HAM10000_metadata.csv"
    if not metadata_path.exists():
        print("==================================================")
        print("⚠️ HAM10000_metadata.csv not found in data_raw/HAM10000/")
        print("==================================================")
        print("Download Options:")
        print("1. Kaggle CLI Command:")
        print("   kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data_raw/HAM10000 --unzip")
        print("2. Harvard Dataverse Direct Link:")
        print("   https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T")
        print("==================================================")
        return False

    print("==================================================")
    print("🚀 Organizing HAM10000 Dataset with Patient-Level Split")
    print("==================================================")

    df = pd.read_csv(metadata_path)
    print(f"[INFO] Loaded metadata. Total image entries: {len(df)}")

    # Patient-level split based on lesion_id to prevent data leakage
    lesions = df[['lesion_id', 'dx']].drop_duplicates()
    
    train_lesions, test_lesions = train_test_split(
        lesions, test_size=0.3, stratify=lesions['dx'], random_state=42
    )
    val_lesions, test_lesions = train_test_split(
        test_lesions, test_size=0.5, stratify=test_lesions['dx'], random_state=42
    )

    train_df = df[df['lesion_id'].isin(train_lesions['lesion_id'])].copy()
    val_df = df[df['lesion_id'].isin(val_lesions['lesion_id'])].copy()
    test_df = df[df['lesion_id'].isin(test_lesions['lesion_id'])].copy()

    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'

    splits = {'train': train_df, 'val': val_df, 'test': test_df}

    # Locate image files across part_1, part_2 or images/ subfolders
    image_paths = {}
    for img_file in DATA_RAW_DIR.rglob("*.jpg"):
        image_paths[img_file.stem] = img_file

    copied_count = 0
    missing_count = 0

    for split_name, split_data in splits.items():
        for _, row in split_data.iterrows():
            img_id = row['image_id']
            cls = row['dx']
            
            target_folder = DATASET_DIR / split_name / cls
            target_folder.mkdir(parents=True, exist_ok=True)

            if img_id in image_paths:
                src_path = image_paths[img_id]
                dst_path = target_folder / f"{img_id}.jpg"
                shutil.copy2(src_path, dst_path)
                copied_count += 1
            else:
                missing_count += 1

    print(f"[SUCCESS] Copied {copied_count} images into dataset/train, val, and test class folders.")
    if missing_count > 0:
        print(f"[WARNING] {missing_count} images were missing from raw folder.")

    return True

if __name__ == "__main__":
    organize_ham10000()
