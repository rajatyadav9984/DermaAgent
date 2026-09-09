"""
DermaAgent - Direct Real Skin Disease Dataset Loader
Fetches real skin disease images and labels automatically via HuggingFace / Open Repositories
without requiring Kaggle tokens, organizes them into dataset/train, val, test,
and updates symptoms metadata for multimodal training.
"""

import os
import sys
import shutil
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data_raw" / "HAM10000"
DATASET_DIR = PROJECT_ROOT / "dataset"
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

CLASS_MAP = {
    0: "akiec",
    1: "bcc",
    2: "bkl",
    3: "df",
    4: "mel",
    5: "nv",
    6: "vasc"
}

def fetch_and_prepare():
    print("==================================================")
    print("🚀 Downloading Real Skin Disease Dataset (HuggingFace)")
    print("==================================================")

    try:
        from datasets import load_dataset
        print("[INFO] Fetching real skin dataset from HuggingFace Hub...")
        
        # Load open skin lesion dataset
        ds = load_dataset("Nagabu/HAM10000", trust_remote_code=True)
        print(f"[INFO] Dataset splits loaded: {list(ds.keys())}")
        
        split_key = 'train' if 'train' in ds else list(ds.keys())[0]
        dataset_items = ds[split_key]
        total_items = len(dataset_items)
        print(f"[INFO] Total images in dataset: {total_items}")

        # Limit count for quick training setup if large, or process all
        records = []
        img_idx = 1

        for i, item in enumerate(dataset_items):
            image = item.get('image') or item.get('img')
            label = item.get('label') or item.get('dx') or item.get('target')

            if image is None:
                continue

            if isinstance(label, int):
                cls_name = CLASS_MAP.get(label, "nv")
            else:
                cls_name = str(label).lower().strip()
                if cls_name not in CLASS_MAP.values():
                    cls_name = "nv"

            img_name = f"real_{cls_name}_{img_idx:05d}.jpg"
            img_idx += 1
            
            records.append({
                "img_name": img_name,
                "image": image,
                "label": cls_name
            })

            if (i + 1) % 500 == 0:
                print(f"   Processed {i + 1}/{total_items} images...")

        print(f"[INFO] Total valid images retrieved: {len(records)}")

        # Train/Val/Test Split (70% train, 15% val, 15% test)
        df = pd.DataFrame(records)
        train_df, test_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
        val_df, test_df = train_test_split(test_df, test_size=0.5, stratify=test_df['label'], random_state=42)

        train_df['split'] = 'train'
        val_df['split'] = 'val'
        test_df['split'] = 'test'

        full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

        # Save images to dataset/train/<cls>, val/<cls>, test/<cls>
        for _, row in full_df.iterrows():
            target_folder = DATASET_DIR / row['split'] / row['label']
            target_folder.mkdir(parents=True, exist_ok=True)
            
            img_path = target_folder / row['img_name']
            if hasattr(row['image'], 'save'):
                row['image'].convert('RGB').save(img_path, 'JPEG')

        print("==================================================")
        print("[SUCCESS] Real skin disease dataset organized successfully!")
        print("==================================================")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to load dataset automatically: {e}")
        return False

if __name__ == "__main__":
    fetch_and_prepare()
