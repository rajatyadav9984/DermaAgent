"""
DermaAgent - ISIC 2019 Independent Test Set Evaluation Script
Evaluates isic2019_skin_disease_model.pth on data_raw/ISIC2019/isic2019_test_split.csv,
computes Accuracy, Macro & Weighted Precision, Recall, F1-Score,
generates per-class breakdown, saves Confusion Matrix, and outputs a Markdown Report.
"""

from pathlib import Path
import sys
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torchvision import models, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ML_DIR = PROJECT_ROOT / "ml"
MODEL_DIR = PROJECT_ROOT / "model"
RESULTS_DIR = PROJECT_ROOT / "results"
ISIC_DIR = PROJECT_ROOT / "data_raw" / "ISIC2019"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from dataset_loader import ISIC2019Dataset, CLASSES_8, VAL_TRANSFORMS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = MODEL_DIR / "isic2019_skin_disease_model.pth"
TEST_MANIFEST = ISIC_DIR / "isic2019_test_split.csv"

def evaluate_isic_model(batch_size=32, max_batches=None):
    print("=" * 65)
    print("DermaAgent - ISIC 2019 Independent Test Set Evaluation")
    print("=" * 65)
    print(f"[INFO] Device         : {DEVICE}")
    print(f"[INFO] Model Path     : {MODEL_PATH}")
    print(f"[INFO] Test Manifest  : {TEST_MANIFEST}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    if not TEST_MANIFEST.exists():
        raise FileNotFoundError(f"Test manifest not found at {TEST_MANIFEST}")

    # 1. Dataset & DataLoader
    test_dataset = ISIC2019Dataset(TEST_MANIFEST, transform=VAL_TRANSFORMS)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"[INFO] Total test images: {len(test_dataset)}")
    print(f"[INFO] Target Classes ({len(CLASSES_8)}): {CLASSES_8}")

    # 2. Load Model Architecture & Checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, len(CLASSES_8))
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    print("[SUCCESS] Checkpoint loaded successfully!")

    # 3. Inference Loop
    all_predictions = []
    all_labels = []

    print("\n[INFO] Running inference on independent test set...")
    with torch.no_grad():
        for b_idx, (images, labels, _) in enumerate(test_loader, 1):
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if b_idx % 10 == 0 or b_idx == len(test_loader):
                print(f"  Processed batch {b_idx}/{len(test_loader)} ({len(all_predictions)} images)...")

            if max_batches and b_idx >= max_batches:
                print(f"[INFO] Halting after {max_batches} batches (smoke mode).")
                break

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # 4. Compute Metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision_w = precision_score(all_labels, all_predictions, average="weighted", zero_division=0)
    recall_w = recall_score(all_labels, all_predictions, average="weighted", zero_division=0)
    f1_w = f1_score(all_labels, all_predictions, average="weighted", zero_division=0)
    precision_macro = precision_score(all_labels, all_predictions, average="macro", zero_division=0)
    recall_macro = recall_score(all_labels, all_predictions, average="macro", zero_division=0)
    f1_macro = f1_score(all_labels, all_predictions, average="macro", zero_division=0)

    # 5. Display Results
    print("\n" + "=" * 65)
    print("INDEPENDENT TEST EVALUATION RESULTS")
    print("=" * 65)
    print(f"Test Accuracy    : {accuracy * 100:.2f}%")
    print(f"Weighted F1      : {f1_w * 100:.2f}%")
    print(f"Weighted Recall  : {recall_w * 100:.2f}%")
    print(f"Weighted Prec    : {precision_w * 100:.2f}%")
    print(f"Macro F1         : {f1_macro * 100:.2f}%")
    print(f"Macro Recall     : {recall_macro * 100:.2f}%")
    print(f"Macro Precision  : {precision_macro * 100:.2f}%")

    print("\n" + "=" * 65)
    print("CLASSIFICATION REPORT")
    print("=" * 65)
    report_dict = classification_report(all_labels, all_predictions, labels=list(range(len(CLASSES_8))), target_names=CLASSES_8, output_dict=True, zero_division=0)
    report_str = classification_report(all_labels, all_predictions, labels=list(range(len(CLASSES_8))), target_names=CLASSES_8, zero_division=0)
    print(report_str)

    # 6. Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions, labels=list(range(len(CLASSES_8))))
    print("CONFUSION MATRIX:")
    print(cm)

    # 7. Save Confusion Matrix Plot
    cm_dir = RESULTS_DIR / "confusion_matrix"
    cm_dir.mkdir(parents=True, exist_ok=True)
    cm_path_results = cm_dir / "isic2019_test_confusion_matrix.png"
    cm_path_model = MODEL_DIR / "isic2019_test_confusion_matrix.png"

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=CLASSES_8,
        yticklabels=CLASSES_8,
        title='ISIC 2019 Independent Test Confusion Matrix',
        ylabel='True Label',
        xlabel='Predicted Label'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(cm_path_results, dpi=200, bbox_inches='tight')
    plt.savefig(cm_path_model, dpi=200, bbox_inches='tight')
    plt.close(fig)

    # 8. Save Metrics JSON
    metrics_dir = RESULTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "isic2019_test_evaluation.json"

    evaluation_data = {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision_w),
        "recall_weighted": float(recall_w),
        "f1_weighted": float(f1_w),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "samples_evaluated": len(all_predictions),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "classes": CLASSES_8
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_data, f, indent=2)

    # 9. Save Markdown Report
    report_md_path = RESULTS_DIR / "isic2019_test_evaluation_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# ISIC 2019 Independent Test Set Evaluation Report\n\n")
        f.write(f"- **Evaluated Samples**: {len(all_predictions)}\n")
        f.write(f"- **Accuracy**: {accuracy * 100:.2f}%\n")
        f.write(f"- **Weighted F1**: {f1_w * 100:.2f}%\n")
        f.write(f"- **Macro F1**: {f1_macro * 100:.2f}%\n")
        f.write(f"- **Weighted Precision**: {precision_w * 100:.2f}%\n")
        f.write(f"- **Weighted Recall**: {recall_w * 100:.2f}%\n\n")
        f.write("## Per-Class Performance\n\n")
        f.write("| Disease Class | Precision | Recall | F1-Score | Support |\n|---|---|---|---|---|\n")
        for cls in CLASSES_8:
            cls_data = report_dict.get(cls, {})
            p = cls_data.get("precision", 0) * 100
            r = cls_data.get("recall", 0) * 100
            f1 = cls_data.get("f1-score", 0) * 100
            sup = cls_data.get("support", 0)
            f.write(f"| **{cls}** | {p:.2f}% | {r:.2f}% | {f1:.2f}% | {sup} |\n")
        f.write("\n## Confusion Matrix\n\n")
        f.write(f"![Confusion Matrix](confusion_matrix/isic2019_test_confusion_matrix.png)\n")

    print("\n" + "=" * 65)
    print("[SUCCESS] Test set evaluation completed!")
    print(f"Metrics JSON : {metrics_path}")
    print(f"CM Plot      : {cm_path_results}")
    print(f"Report MD    : {report_md_path}")
    print("=" * 65)
    return evaluation_data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate ISIC 2019 Model on Test Set")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit batches for fast check")
    args = parser.parse_args()
    evaluate_isic_model(batch_size=args.batch_size, max_batches=args.max_batches)
