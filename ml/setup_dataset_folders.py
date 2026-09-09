"""
DermaAgent - Dataset Folder Setup & Sample Generator
Creates train, val, test subdirectories for all disease classes.
Generates initial sample images for testing the pipeline if no dataset exists.
"""

import os
import sys
import random
import numpy as np
from PIL import Image, ImageDraw

# Ensure stdout uses UTF-8 or fall back cleanly on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from dataset_config import DATASET_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, CLASS_NAMES, DISEASE_CLASSES

def create_directory_structure():
    """Create train, val, test folders for each disease class."""
    splits = [TRAIN_DIR, VAL_DIR, TEST_DIR]
    
    print("[INFO] Creating dataset folder structure...")
    for split_dir in splits:
        for cls in CLASS_NAMES:
            folder_path = os.path.join(split_dir, cls)
            os.makedirs(folder_path, exist_ok=True)
    print("[SUCCESS] Folder structure created successfully!")

def generate_sample_images(samples_per_class=20):
    """
    Generates synthetic sample skin lesion images for development and pipeline testing.
    Each class gets unique texture/color patterns mimicking dermoscopic image features.
    """
    print(f"\n[INFO] Generating sample demonstration images ({samples_per_class} per class)...")
    
    # Class-specific visual color themes (RGB)
    class_colors = {
        "akiec": (220, 140, 120),  # Rough reddish patch
        "bcc": (200, 100, 110),    # Pearly pink/red nodule
        "bkl": (160, 110, 70),     # Brownish crusty lesion
        "df": (140, 90, 80),       # Firm reddish-brown papule
        "mel": (50, 40, 40),       # Irregular dark brown/black
        "nv": (120, 80, 60),       # Uniform brown mole
        "vasc": (190, 30, 60)      # Deep red/purple vascular
    }

    splits_ratio = {"train": 0.7, "val": 0.15, "test": 0.15}

    for cls in CLASS_NAMES:
        base_color = class_colors.get(cls, (150, 100, 80))
        
        train_count = int(samples_per_class * splits_ratio["train"])
        val_count = int(samples_per_class * splits_ratio["val"])
        test_count = samples_per_class - train_count - val_count

        distributions = [
            (TRAIN_DIR, train_count),
            (VAL_DIR, val_count),
            (TEST_DIR, test_count)
        ]

        img_idx = 1
        for split_dir, count in distributions:
            class_folder = os.path.join(split_dir, cls)
            for _ in range(count):
                # Generate 224x224 RGB image with skin background & lesion center
                img_data = np.full((224, 224, 3), fill_value=[235, 195, 175], dtype=np.uint8)
                # Add subtle noise for realistic skin texture
                noise = np.random.randint(-15, 15, (224, 224, 3), dtype=np.int16)
                skin_img = np.clip(img_data.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                
                img = Image.fromarray(skin_img)
                draw = ImageDraw.Draw(img)
                
                # Draw central lesion blob
                cx, cy = 112 + random.randint(-10, 10), 112 + random.randint(-10, 10)
                rx, ry = random.randint(30, 55), random.randint(30, 55)
                bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
                
                # Lesion color with variation
                l_color = tuple(np.clip(np.array(base_color) + np.random.randint(-20, 20, 3), 0, 255))
                draw.ellipse(bbox, fill=l_color)
                
                filename = f"{cls}_sample_{img_idx:03d}.jpg"
                filepath = os.path.join(class_folder, filename)
                img.save(filepath, "JPEG")
                img_idx += 1

    print("[SUCCESS] Sample images generated for testing!")

if __name__ == "__main__":
    create_directory_structure()
    generate_sample_images(samples_per_class=20)
