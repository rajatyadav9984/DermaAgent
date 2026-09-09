from pathlib import Path
import sys

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from torchvision import datasets
from torch.utils.data import DataLoader

from preprocessing import (
    train_transforms,
    val_test_transforms
)


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset directory
DATASET_DIR = PROJECT_ROOT / "dataset"


# Dataset paths
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"


# Load datasets
train_dataset = datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=train_transforms
)

val_dataset = datasets.ImageFolder(
    root=VAL_DIR,
    transform=val_test_transforms
)

test_dataset = datasets.ImageFolder(
    root=TEST_DIR,
    transform=val_test_transforms
)


# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)


if __name__ == "__main__":

    print("Dataset loaded successfully!")

    print("\nClasses:")
    print(train_dataset.classes)

    print("\nClass-to-index mapping:")
    print(train_dataset.class_to_idx)

    print("\nDataset sizes:")
    print("Train:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Test:", len(test_dataset))

    images, labels = next(iter(train_loader))

    print("\nFirst batch:")
    print("Image shape:", images.shape)
    print("Label shape:", labels.shape)
