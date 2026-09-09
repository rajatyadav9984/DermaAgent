from pathlib import Path
import torch
from torchvision.datasets import ImageFolder
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "dataset_real" / "train"

print("=" * 60)
print("DermaAgent - CLASS IMBALANCE ANALYSIS")
print("=" * 60)

dataset = ImageFolder(TRAIN_DIR)

classes = dataset.classes
labels = np.array(dataset.targets)

print("\nClasses:")
for index, class_name in enumerate(classes):
    count = np.sum(labels == index)
    print(f"{index}: {class_name:<8} -> {count} images")

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(classes)),
    y=labels
)

weights_tensor = torch.tensor(
    weights,
    dtype=torch.float32
)

print("\n===== CLASS WEIGHTS =====")

for class_name, weight in zip(classes, weights):
    print(f"{class_name:<8} -> {weight:.4f}")

OUTPUT_FILE = PROJECT_ROOT / "model" / "class_weights.pt"

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

torch.save(
    {
        "classes": classes,
        "weights": weights_tensor
    },
    OUTPUT_FILE
)

print("\nWeights saved to:")
print(OUTPUT_FILE)

print("\n[SUCCESS] Class imbalance analysis complete!")
