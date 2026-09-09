"""
DermaAgent - Multimodal Model Evaluation Script
Evaluates MultimodalDermaModel on the 1,201 test dataset images + paired symptoms,
computes Accuracy, Precision, Recall, F1-Score, Classification Report,
and plots the Confusion Matrix.
"""

from pathlib import Path
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

# Import project modules
ML_DIR = Path(__file__).resolve().parent

if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from multimodal_model import MultimodalDermaModel
from multimodal_dataloader import get_multimodal_dataloaders

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "multimodal_skin_disease_model.pth"
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

def evaluate_multimodal_model():
    print("=" * 60)
    print("DermaAgent - Multimodal Model Evaluation")
    print("=" * 60)
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Model Path: {MODEL_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    # Load Model Architecture & Checkpoint
    model = MultimodalDermaModel()
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    print("[SUCCESS] Trained Multimodal Model loaded successfully!")

    # Get Test DataLoader
    _, _, test_loader = get_multimodal_dataloaders(batch_size=32)
    print(f"[INFO] Test dataset batches: {len(test_loader)}")

    all_predictions = []
    all_labels = []

    print("\n[INFO] Running inference on test dataset...")
    with torch.no_grad():
        for images, symptoms, labels in test_loader:
            images = images.to(DEVICE)
            symptoms = symptoms.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images, symptoms)
            preds = torch.argmax(outputs, dim=1)

            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Compute Metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_predictions, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average="weighted", zero_division=0)

    # Display Results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1 Score  : {f1 * 100:.2f}%")

    # Classification Report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(all_labels, all_predictions, target_names=CLASSES, zero_division=0))

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
    plt.title("DermaAgent - Confusion Matrix", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(range(len(CLASSES)), CLASSES, rotation=45)
    plt.yticks(range(len(CLASSES)), CLASSES)

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")

    plt.tight_layout()
    
    out_path_1 = PROJECT_ROOT / "model" / "confusion_matrix.png"
    out_path_2 = PROJECT_ROOT / "results" / "confusion_matrix.png"
    plt.savefig(out_path_1, dpi=200)
    plt.savefig(out_path_2, dpi=200)
    plt.close()

    print("\n" + "=" * 60)
    print("[SUCCESS] Model evaluation complete!")
    print(f"[SUCCESS] Confusion matrix saved to: {out_path_1}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_multimodal_model()
