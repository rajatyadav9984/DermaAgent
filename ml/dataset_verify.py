"""
DermaAgent - Dataset Verification & Health Check Script
Scans train/val/test folders, checks image integrity, counts per class,
and outputs dataset statistics with visualization plot.
"""

import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from dataset_config import DATASET_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, CLASS_NAMES, DISEASE_CLASSES

def verify_dataset():
    """Verify dataset integrity, check corrupted images, and compute split statistics."""
    print("==================================================")
    print("[INFO] DermaAgent Dataset Verification & Health Check")
    print("==================================================")
    
    splits = {"Train": TRAIN_DIR, "Validation": VAL_DIR, "Test": TEST_DIR}
    records = []
    total_images = 0
    corrupt_count = 0
    
    for split_name, split_path in splits.items():
        if not os.path.exists(split_path):
            print(f"[WARNING] Directory '{split_name}' does not exist at {split_path}")
            continue
            
        for cls in CLASS_NAMES:
            cls_folder = os.path.join(split_path, cls)
            if not os.path.exists(cls_folder):
                records.append({"Split": split_name, "Class": cls, "Count": 0, "Full Name": DISEASE_CLASSES[cls]["full_name"]})
                continue
                
            files = [f for f in os.listdir(cls_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            valid_files = 0
            
            for fname in files:
                fpath = os.path.join(cls_folder, fname)
                try:
                    with Image.open(fpath) as img:
                        img.verify()  # Verify image integrity
                    valid_files += 1
                except Exception as e:
                    print(f"[ERROR] Corrupt image found: {fpath} ({e})")
                    corrupt_count += 1
            
            records.append({
                "Split": split_name,
                "Class": cls,
                "Count": valid_files,
                "Full Name": DISEASE_CLASSES[cls]["full_name"]
            })
            total_images += valid_files
            
    df = pd.DataFrame(records)
    
    if df.empty or df['Count'].sum() == 0:
        print("\n[WARNING] No images found in dataset folders! Run setup_dataset_folders.py to populate sample images.")
        return df

    print("\n[SUMMARY] Dataset Image Count Summary:")
    pivot_df = df.pivot(index="Class", columns="Split", values="Count").fillna(0).astype(int)
    pivot_df["Total"] = pivot_df.sum(axis=1)
    print(pivot_df.to_string())
    
    print("\n--------------------------------------------------")
    print(f" Total Images Verified : {total_images}")
    print(f" Corrupted Images      : {corrupt_count}")
    print(f" Total Classes         : {len(CLASS_NAMES)}")
    print("--------------------------------------------------")
    
    # Plot dataset distribution graph
    plot_distribution(pivot_df)
    
    return pivot_df

def plot_distribution(pivot_df):
    """Generates and saves a bar chart of class distribution across train/val/test splits."""
    plt.figure(figsize=(10, 5))
    plot_data = pivot_df.drop(columns=["Total"], errors="ignore")
    plot_data.plot(kind="bar", stacked=True, color=["#4F46E5", "#10B981", "#F59E0B"], figsize=(10, 5))
    
    plt.title("DermaAgent Skin Disease Dataset Class Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Disease Class", fontsize=12)
    plt.ylabel("Number of Images", fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    
    plot_path = os.path.join(DATASET_DIR, "dataset_distribution.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[INFO] Class distribution graph saved to: {plot_path}")

if __name__ == "__main__":
    verify_dataset()
