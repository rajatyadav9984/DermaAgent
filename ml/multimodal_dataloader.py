"""
DermaAgent - Multimodal PyTorch Dataset & DataLoader
Loads Image Tensors + Symptom Vectors + Class Labels simultaneously.
"""

import sys
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from preprocessing import train_transforms, val_test_transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "dataset" / "symptoms_metadata.csv"

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASS_NAMES)}

SYMPTOM_COLUMNS = [
    "itching", "redness", "pain", "swelling", "bleeding",
    "scaling", "burning", "crusting", "change_in_size", "change_in_color"
]


class MultimodalSkinDataset(Dataset):
    def __init__(self, metadata_df, project_root, transform=None):
        self.df = metadata_df.reset_index(drop=True)
        self.project_root = Path(project_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.project_root / row["image_path"]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        # Extract symptom vector
        symptom_vector = row[SYMPTOM_COLUMNS].values.astype("float32")
        symptom_tensor = torch.tensor(symptom_vector, dtype=torch.float32)

        # Extract class label
        label_idx = CLASS_TO_IDX[row["label"]]
        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        return image, symptom_tensor, label_tensor


def get_multimodal_dataloaders(batch_size=8):
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found at {CSV_PATH}. Run create_symptom_metadata.py first.")

    df = pd.read_csv(CSV_PATH)
    
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    train_dataset = MultimodalSkinDataset(train_df, PROJECT_ROOT, transform=train_transforms)
    val_dataset = MultimodalSkinDataset(val_df, PROJECT_ROOT, transform=val_test_transforms)
    test_dataset = MultimodalSkinDataset(test_df, PROJECT_ROOT, transform=val_test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_multimodal_dataloaders(batch_size=8)
    
    print("[INFO] Multimodal DataLoaders initialized successfully!")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")
    
    images, symptoms, labels = next(iter(train_loader))
    print("\nFirst batch shapes:")
    print("Images shape  :", images.shape)
    print("Symptoms shape:", symptoms.shape)
    print("Labels shape  :", labels.shape)
