from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "real_skin_disease_model.pth"
TEST_DIR = BASE_DIR / "dataset_real" / "test"
OUTPUT_DIR = BASE_DIR / "model" / "gradcam"

IMAGE_SIZE = 224

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DermaAgent - Real ResNet18 Grad-CAM")
print("=" * 60)

print(f"[INFO] Device: {DEVICE}")
print(f"[INFO] Model: {MODEL_PATH}")
print(f"[INFO] Test directory: {TEST_DIR}")


# ============================================================
# LOAD MODEL
# ============================================================

model = models.resnet18(weights=None)

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    len(CLASS_NAMES)
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    checkpoint = checkpoint["model_state_dict"]

model.load_state_dict(checkpoint)

model = model.to(DEVICE)
model.eval()

print("[SUCCESS] Real ResNet18 model loaded")


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# FIND A REAL TEST IMAGE
# ============================================================

image_files = list(TEST_DIR.rglob("*.jpg"))

if not image_files:
    raise FileNotFoundError(
        f"No JPG images found inside {TEST_DIR}"
    )

test_img = image_files[0]

print(f"[INFO] Selected test image: {test_img}")


# ============================================================
# GRAD-CAM STORAGE
# ============================================================

activations = None
gradients = None


def forward_hook(module, input, output):
    global activations
    activations = output


def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]


# ResNet18 final convolutional block
target_layer = model.layer4[-1]

forward_handle = target_layer.register_forward_hook(
    forward_hook
)

backward_handle = target_layer.register_full_backward_hook(
    backward_hook
)


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

def generate_gradcam(image_path):

    global activations
    global gradients

    original_image = Image.open(
        image_path
    ).convert("RGB")

    input_tensor = transform(
        original_image
    ).unsqueeze(0).to(DEVICE)

    model.zero_grad()

    output = model(input_tensor)

    probabilities = F.softmax(
        output,
        dim=1
    )

    confidence, predicted_class = torch.max(
        probabilities,
        dim=1
    )

    predicted_index = predicted_class.item()

    predicted_name = CLASS_NAMES[
        predicted_index
    ]

    print()
    print("=" * 60)
    print("PREDICTION")
    print("=" * 60)

    print(
        f"Predicted class : {predicted_name}"
    )

    print(
        f"Class index     : {predicted_index}"
    )

    print(
        f"Confidence      : "
        f"{confidence.item() * 100:.2f}%"
    )

    # Backpropagate predicted class
    score = output[
        0,
        predicted_index
    ]

    score.backward()

    # --------------------------------------------------------
    # Grad-CAM calculation
    # --------------------------------------------------------

    weights = gradients.mean(
        dim=(2, 3),
        keepdim=True
    )

    cam = (
        weights * activations
    ).sum(dim=1)

    cam = F.relu(cam)

    cam = cam.squeeze().detach().cpu().numpy()

    # Normalize CAM
    cam -= cam.min()

    if cam.max() != 0:
        cam /= cam.max()

    # Resize CAM to original image size
    cam_image = Image.fromarray(
        np.uint8(cam * 255)
    )

    cam_image = cam_image.resize(
        original_image.size
    )

    cam_array = np.array(
        cam_image
    ) / 255.0

    # --------------------------------------------------------
    # Create heatmap
    # --------------------------------------------------------

    import matplotlib.pyplot as plt

    plt.figure(
        figsize=(8, 8)
    )

    plt.imshow(
        original_image
    )

    plt.imshow(
        cam_array,
        cmap="jet",
        alpha=0.45
    )

    plt.axis("off")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR /
        f"gradcam_{predicted_name}_{image_path.stem}.jpg"
    )

    plt.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()

    print()
    print(
        f"[SUCCESS] Grad-CAM saved:"
    )

    print(
        output_path
    )

    return output_path


# ============================================================
# RUN
# ============================================================

try:

    output = generate_gradcam(
        test_img
    )

finally:

    forward_handle.remove()
    backward_handle.remove()


print()
print("=" * 60)
print("[SUCCESS] Grad-CAM generation complete!")
print("=" * 60)
