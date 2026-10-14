"""
DermaAgent - PyTorch Dataset Loader for ISIC 2019
Streams images seamlessly either from an extracted directory OR directly from the ZIP archive.
Provides class weights computation for handling class imbalance.
"""

from pathlib import Path
import io
import zipfile
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ISIC_DIR = PROJECT_ROOT / "data_raw" / "ISIC2019"
ZIP_PATH = ISIC_DIR / "ISIC_2019_Training_Input.zip"
EXTRACTED_DIR = ISIC_DIR / "images"

CLASSES_8 = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASSES_8)}
IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(CLASSES_8)}

IMAGE_SIZE = 224

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ISIC2019Dataset(Dataset):
    """
    PyTorch Dataset for 8-class ISIC 2019 classification.
    Supports on-the-fly streaming from ZIP or loose files on disk.
    """
    def __init__(self, manifest_csv, transform=None):
        self.manifest_path = Path(manifest_csv)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.df = pd.read_csv(self.manifest_path)
        self.transform = transform
        self.zip_ref = None
        self.use_zip = not (EXTRACTED_DIR.exists() and any(EXTRACTED_DIR.iterdir()))

        # Filter strictly to 8 supported classes
        self.df = self.df[self.df["target_class"].isin(CLASSES_8)].reset_index(drop=True)

    def _get_zip(self):
        # Thread-safe lazy initialization per worker
        if self.zip_ref is None:
            self.zip_ref = zipfile.ZipFile(ZIP_PATH, 'r')
        return self.zip_ref

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image"]
        class_name = row["target_class"]
        label = CLASS_TO_IDX[class_name]

        # Load image either from extracted disk or from ZIP stream
        if not self.use_zip:
            img_path = EXTRACTED_DIR / f"{image_id}.jpg"
            if img_path.exists():
                image = Image.open(img_path).convert("RGB")
            else:
                zf = self._get_zip()
                data = zf.read(f"ISIC_2019_Training_Input/{image_id}.jpg")
                image = Image.open(io.BytesIO(data)).convert("RGB")
        else:
            zf = self._get_zip()
            data = zf.read(f"ISIC_2019_Training_Input/{image_id}.jpg")
            image = Image.open(io.BytesIO(data)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long), image_id

    def get_class_weights(self):
        """Computes inverse class frequencies for weighted cross-entropy."""
        counts = self.df["target_class"].value_counts().to_dict()
        total = len(self.df)
        weights = []
        for cls in CLASSES_8:
            cnt = counts.get(cls, 1)
            weights.append(total / (len(CLASSES_8) * cnt))
        return torch.tensor(weights, dtype=torch.float32)

def get_dataloaders(train_csv=None, val_csv=None, batch_size=32, num_workers=0):
    if train_csv is None:
        train_csv = ISIC_DIR / "isic2019_train_split.csv"
    if val_csv is None:
        val_csv = ISIC_DIR / "isic2019_val_split.csv"

    train_dataset = ISIC2019Dataset(train_csv, transform=TRAIN_TRANSFORMS)
    val_dataset = ISIC2019Dataset(val_csv, transform=VAL_TRANSFORMS)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, train_dataset.get_class_weights()
