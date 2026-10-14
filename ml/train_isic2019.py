"""
DermaAgent - ISIC 2019 8-Class Skin Disease Model Training Pipeline
Trains a ResNet18 classifier on ISIC 2019 (MEL, NV, BCC, AK, BKL, DF, VASC, SCC)
without touching the existing HAM10000 real_skin_disease_model.pth checkpoint.
"""

from pathlib import Path
import sys
import time
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ML_DIR = PROJECT_ROOT / "ml"
MODEL_DIR = PROJECT_ROOT / "model"
RESULTS_DIR = PROJECT_ROOT / "results"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from dataset_loader import get_dataloaders, CLASSES_8, CLASS_TO_IDX

TARGET_CHECKPOINT = MODEL_DIR / "isic2019_skin_disease_model.pth"

def build_model(num_classes=8, use_ham_backbone=True):
    """Initializes ResNet18 transfer learning backbone with 8-class head."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    ham_path = MODEL_DIR / "real_skin_disease_model.pth"
    if use_ham_backbone and ham_path.exists():
        try:
            print(f"[INFO] Initializing dermoscopy representations from {ham_path.name}...")
            cp = torch.load(ham_path, map_location="cpu", weights_only=False)
            sd = cp["model_state_dict"] if isinstance(cp, dict) and "model_state_dict" in cp else cp
            # Filter out fc layer to adapt from 7 classes to 8 classes
            sd_filtered = {k: v for k, v in sd.items() if not k.startswith("fc.")}
            model.load_state_dict(sd_filtered, strict=False)
            print("[SUCCESS] Pre-trained dermoscopy backbone weights transferred successfully!")
        except Exception as e:
            print(f"[WARNING] Could not transfer HAM weights ({e}); using ImageNet pretrained weights.")

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

def plot_and_save_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES_8))))
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=CLASSES_8,
        yticklabels=CLASSES_8,
        title='ISIC 2019 Confusion Matrix',
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
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def verify_checkpoint_reload(checkpoint_path, device):
    """Verifies that the saved checkpoint can be reloaded and forward passed cleanly."""
    print(f"\n[VERIFY] Testing checkpoint reloadability from {checkpoint_path.name}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    assert "model_state_dict" in checkpoint, "Missing model_state_dict in checkpoint!"
    assert "classes" in checkpoint and checkpoint["classes"] == CLASSES_8, "Checkpoint classes mismatch!"

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES_8))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Forward dummy tensor to ensure computation graph integrity
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (1, 8), f"Expected shape (1, 8), got {out.shape}"
    print(f"[SUCCESS] Checkpoint reload verified: output shape {tuple(out.shape)}, val_f1={checkpoint.get('val_f1', 0):.4f}")

def train(epochs=10, batch_size=32, lr=1e-4, num_workers=0, smoke_test=False, max_train_batches=None, max_val_batches=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("DermaAgent - ISIC 2019 Training Engine")
    print("=" * 65)
    print(f"[INFO] Compute Device : {device}")
    print(f"[INFO] Target Output  : {TARGET_CHECKPOINT}")
    print(f"[INFO] Batch Size     : {batch_size}")
    print(f"[INFO] Learning Rate  : {lr}")
    print(f"[INFO] Epochs         : {epochs}")
    print(f"[INFO] Smoke Test     : {smoke_test}")

    if smoke_test:
        max_train_batches = max_train_batches or 4
        max_val_batches = max_val_batches or 2
        print(f"[INFO] Smoke test active: limiting to {max_train_batches} train batches, {max_val_batches} val batches.")

    # Build DataLoaders
    print("\n[INFO] Loading training and validation sets...")
    train_loader, val_loader, class_weights = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers
    )

    class_weights = class_weights.to(device)
    print("\n[INFO] Class weights applied to CrossEntropyLoss:")
    for cls_name, w in zip(CLASSES_8, class_weights.cpu().numpy()):
        print(f"  * {cls_name:<6} -> {w:.4f}")

    model = build_model(num_classes=len(CLASSES_8)).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = -1.0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    print("\n[INFO] Commencing training loop...")
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        total_train = 0

        for step, (images, labels, _) in enumerate(train_loader, 1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            total_train += images.size(0)

            if step % 10 == 0 or step == len(train_loader) or (max_train_batches and step >= max_train_batches):
                print(f"  Epoch {epoch:02d}/{epochs:02d} | Step {step:04d} | Batch Loss: {loss.item():.4f}")

            if max_train_batches and step >= max_train_batches:
                break

        scheduler.step()
        epoch_train_loss = running_loss / max(1, total_train)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        val_preds = []
        val_targets = []
        val_total = 0

        with torch.no_grad():
            for v_step, (images, labels, _) in enumerate(val_loader, 1):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)
                val_total += images.size(0)

                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

                if max_val_batches and v_step >= max_val_batches:
                    break

        epoch_val_loss = running_val_loss / max(1, val_total)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        elapsed = time.time() - start_time

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"\n[Epoch {epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_f1:.4f}")

        # Save checkpoint if best F1 achieved (or in smoke test / first epoch)
        if val_f1 >= best_val_f1 or epoch == 1:
            best_val_f1 = val_f1
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_f1": val_f1,
                "classes": CLASSES_8,
                "class_to_idx": CLASS_TO_IDX
            }, TARGET_CHECKPOINT)
            print(f"  🔥 Best checkpoint saved to {TARGET_CHECKPOINT.name} (Val Macro-F1: {val_f1:.4f})")

    # Verify saved checkpoint reloadability
    verify_checkpoint_reload(TARGET_CHECKPOINT, device)

    # Generate final evaluation report & confusion matrix
    cm_path = RESULTS_DIR / "confusion_matrix" / "isic2019_confusion_matrix.png"
    plot_and_save_confusion_matrix(val_targets, val_preds, cm_path)

    metrics_path = RESULTS_DIR / "metrics" / "isic2019_training_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report = classification_report(
        val_targets,
        val_preds,
        labels=list(range(len(CLASSES_8))),
        target_names=CLASSES_8,
        output_dict=True,
        zero_division=0
    )
    report["training_history"] = history
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 65)
    print(f"[SUCCESS] Training finished! Best Macro-F1: {best_val_f1:.4f}")
    print(f"Checkpoint : {TARGET_CHECKPOINT}")
    print(f"Metrics    : {metrics_path}")
    print(f"CM Plot    : {cm_path}")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ISIC 2019 ResNet18.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick smoke test (few batches)")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Max train batches per epoch")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Max val batches per epoch")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.workers,
        smoke_test=args.smoke_test,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches
    )
