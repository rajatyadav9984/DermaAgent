"""
DermaAgent - Real Image-Only ResNet18 Model Evaluation Script
Evaluates real_skin_disease_model.pth on dataset_real/test dataset,
computes Accuracy, Precision, Recall, F1-Score, Classification Report,
and plots/saves the Confusion Matrix.
"""

from pathlib import Path
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "real_skin_disease_model.pth"
TEST_DIR = PROJECT_ROOT / "dataset_real" / "test"
IMAGE_SIZE = 224
BATCH_SIZE = 32

def evaluate_real_model():
    print("=" * 60)
    print("DermaAgent - Real ResNet18 Model Evaluation")
    print("=" * 60)
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Model Path: {MODEL_PATH}")
    print(f"[INFO] Test Directory: {TEST_DIR}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test directory not found at {TEST_DIR}")

    # Transforms
    test_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Dataset & DataLoader
    test_dataset = ImageFolder(TEST_DIR, transform=test_transforms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    classes = test_dataset.classes
    print(f"[INFO] Classes ({len(classes)}): {classes}")
    print(f"[INFO] Total test images: {len(test_dataset)}")

    # Load Model Architecture & Checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    print("[SUCCESS] Trained Real Model loaded successfully!")

    all_predictions = []
    all_labels = []

    print("\n[INFO] Running inference on test dataset...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Compute Metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision_w = precision_score(all_labels, all_predictions, average="weighted", zero_division=0)
    recall_w = recall_score(all_labels, all_predictions, average="weighted", zero_division=0)
    f1_w = f1_score(all_labels, all_predictions, average="weighted", zero_division=0)
    f1_macro = f1_score(all_labels, all_predictions, average="macro", zero_division=0)

    # Display Results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Test Accuracy : {accuracy * 100:.2f}%")
    print(f"Precision (W) : {precision_w * 100:.2f}%")
    print(f"Recall (W)    : {recall_w * 100:.2f}%")
    print(f"F1 Score (W)  : {f1_w * 100:.2f}%")
    print(f"Macro F1      : {f1_macro * 100:.2f}%")

    # Classification Report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(all_labels, all_predictions, target_names=classes, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    print(cm)

    # Save Confusion Matrix Plot
    save_dirs = [PROJECT_ROOT / "model", PROJECT_ROOT / "results"]
    for s_dir in save_dirs:
        s_dir.mkdir(parents=True, exist_ok=True)
        
    plt.figure(figsize=(9, 7))
    plt.imshow(cm, cmap='Blues')
    plt.title("DermaAgent - Real Model Confusion Matrix", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.yticks(range(len(classes)), classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")

    plt.tight_layout()
    
    out_path_1 = PROJECT_ROOT / "model" / "confusion_matrix_real.png"
    out_path_2 = PROJECT_ROOT / "results" / "confusion_matrix_real.png"
    plt.savefig(out_path_1, dpi=200)
    plt.savefig(out_path_2, dpi=200)
    plt.close()

    print("\n" + "=" * 60)
    print("[SUCCESS] Model evaluation complete!")
    print(f"[SUCCESS] Confusion matrix saved to: {out_path_1}")
    print("=" * 60)

if __name__ == "__main__":
    evaluate_real_model()
