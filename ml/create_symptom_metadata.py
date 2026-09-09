"""
DermaAgent - Symptom Metadata Generator
Scans dataset/train, dataset/val, and dataset/test folders,
generates symptom metadata for each image (for demonstration pipeline),
and saves dataset/symptoms_metadata.csv.
"""

import os
import sys
import random
import pandas as pd
from pathlib import Path

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
CSV_PATH = DATASET_DIR / "symptoms_metadata.csv"

# Disease-typical symptom probability profiles for synthetic metadata generation
DISEASE_SYMPTOM_PROBS = {
    "akiec": {"itching": 0.6, "redness": 0.8, "scaling": 0.9, "crusting": 0.5},
    "bcc":   {"redness": 0.7, "bleeding": 0.6, "crusting": 0.4, "change_in_size": 0.7},
    "bkl":   {"itching": 0.4, "scaling": 0.7, "crusting": 0.5, "change_in_color": 0.6},
    "df":    {"pain": 0.5, "swelling": 0.6, "change_in_size": 0.4},
    "mel":   {"itching": 0.5, "change_in_size": 0.9, "change_in_color": 0.95, "bleeding": 0.5},
    "nv":    {"change_in_color": 0.3, "change_in_size": 0.2},
    "vasc":  {"redness": 0.9, "bleeding": 0.7, "swelling": 0.5}
}

SYMPTOM_COLUMNS = [
    "itching", "redness", "pain", "swelling", "bleeding",
    "scaling", "burning", "crusting", "change_in_size", "change_in_color"
]

def generate_symptom_metadata():
    print("==================================================")
    print("[INFO] Generating Multimodal Symptom Metadata CSV")
    print("==================================================")

    splits = ["train", "val", "test"]
    records = []

    for split in splits:
        split_dir = DATASET_DIR / split
        if not split_dir.exists():
            continue

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue

            cls_name = class_dir.name
            prob_profile = DISEASE_SYMPTOM_PROBS.get(cls_name, {})

            for img_file in class_dir.glob("*.[jJ][pP][gG]"):
                rel_path = img_file.relative_to(PROJECT_ROOT).as_posix()
                
                # Sample symptoms based on class probabilities
                symptom_vals = {}
                for sym in SYMPTOM_COLUMNS:
                    prob = prob_profile.get(sym, 0.1)
                    symptom_vals[sym] = 1 if random.random() < prob else 0

                record = {
                    "image_path": rel_path,
                    "split": split,
                    "label": cls_name,
                    **symptom_vals
                }
                records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(CSV_PATH, index=False)

    print(f"[SUCCESS] Metadata created! Total records: {len(df)}")
    print(f"[INFO] Metadata saved to: {CSV_PATH}")
    print("\nSample records:")
    print(df.head(5).to_string())

if __name__ == "__main__":
    generate_symptom_metadata()
