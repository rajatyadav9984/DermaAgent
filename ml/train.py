"""
DermaAgent - Model Training Script
Trains ResNet18 transfer learning model on skin disease dataset,
computes loss & accuracy for each epoch, and saves the best model checkpoint.
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from dataloader import train_loader, val_loader
from model import create_model

# Path settings
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_SAVE_DIR = PROJECT_ROOT / "model"
MODEL_SAVE_PATH = MODEL_SAVE_DIR / "skin_disease_model.pth"

# Training Hyperparameters
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4


def train_model():
    # 1. Device configuration (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using compute device: {device}")

    # 2. Instantiate Model and move to device
    model = create_model().to(device)

    # 3. Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    print("\n==================================================")
    print("🚀 Starting DermaAgent Model Training Loop")
    print("==================================================")

    for epoch in range(1, NUM_EPOCHS + 1):
        # ------------------------------------------------
        # A. Training Phase
        # ------------------------------------------------
        model.train()
        running_train_loss = 0.0
        correct_train_preds = 0
        total_train_samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Clear gradients from previous step
            optimizer.zero_grad()

            # Forward pass: compute predicted outputs
            outputs = model(images)

            # Compute batch loss
            loss = criterion(outputs, labels)

            # Backward pass: compute gradients
            loss.backward()

            # Optimizer step: update model parameters
            optimizer.step()

            # Track statistics
            running_train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train_preds += torch.sum(preds == labels.data).item()
            total_train_samples += images.size(0)

        epoch_train_loss = running_train_loss / total_train_samples
        epoch_train_acc = (correct_train_preds / total_train_samples) * 100.0

        # ------------------------------------------------
        # B. Validation Phase
        # ------------------------------------------------
        model.eval()
        running_val_loss = 0.0
        correct_val_preds = 0
        total_val_samples = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val_preds += torch.sum(preds == labels.data).item()
                total_val_samples += images.size(0)

        epoch_val_loss = running_val_loss / total_val_samples
        epoch_val_acc = (correct_val_preds / total_val_samples) * 100.0

        print(
            f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] | "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%"
        )

        # ------------------------------------------------
        # C. Save Best Model Checkpoint
        # ------------------------------------------------
        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': epoch_val_acc,
                'val_loss': epoch_val_loss
            }, MODEL_SAVE_PATH)
            print(f"   --> Best model checkpoint saved to: {MODEL_SAVE_PATH}")

    print("\n==================================================")
    print(f"[SUCCESS] Training completed! Best Validation Accuracy: {best_val_acc:.2f}%")
    print("==================================================")


if __name__ == "__main__":
    train_model()
