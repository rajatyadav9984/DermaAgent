"""
DermaAgent - Real Image Inference Engine

Pipeline:
Image
  ↓
Real HAM10000 ResNet18
  ↓
Prediction + Top-3
  ↓
Confidence
  ↓
Entropy / Uncertainty
  ↓
Agents
  ↓
Grad-CAM availability
"""

from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


# ============================================================
# WINDOWS CONSOLE
# ============================================================

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from uncertainty import (
    calculate_confidence,
    calculate_entropy,
    normalize_entropy,
    get_uncertainty_level
)

from agents.orchestrator import DermaOrchestrator


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "model"
    / "real_skin_disease_model.pth"
)

TEST_DIR = (
    PROJECT_ROOT
    / "dataset_real"
    / "test"
)

GRADCAM_DIR = (
    PROJECT_ROOT
    / "model"
    / "gradcam"
)

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

IMAGE_SIZE = 224

MEAN = [
    0.485,
    0.456,
    0.406
]

STD = [
    0.229,
    0.224,
    0.225
]


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

image_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=MEAN,
        std=STD
    )
])


# ============================================================
# CREATE REAL RESNET18 MODEL
# ============================================================

def create_model():

    model = models.resnet18(
        weights=None
    )

    num_features = model.fc.in_features

    model.fc = nn.Linear(
        num_features,
        len(CLASS_NAMES)
    )

    return model


# ============================================================
# LOAD REAL TRAINED MODEL
# ============================================================

def load_model():

    print(
        f"[INFO] Loading real model:"
    )

    print(
        MODEL_PATH
    )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = create_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    # Support both:
    # 1. Direct state_dict
    # 2. {"model_state_dict": ...}

    if isinstance(checkpoint, dict) and \
       "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        "[SUCCESS] Real ResNet18 model loaded!"
    )

    return model


# ============================================================
# FIND GRAD-CAM
# ============================================================

def find_gradcam(image_path):

    image_name = Path(
        image_path
    ).stem

    # Our current Grad-CAM naming:
    # gradcam_akiec_ISIC_0024511.jpg

    matching_files = list(
        GRADCAM_DIR.glob(
            f"gradcam_*_{image_name}.jpg"
        )
    )

    if matching_files:

        return matching_files[0]

    return None


# ============================================================
# PREDICTION
# ============================================================

def predict(image_path):

    image_path = Path(
        image_path
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    image_tensor = image_transform(
        image
    ).unsqueeze(0)

    image_tensor = image_tensor.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            image_tensor
        )

        confidence_tensor, \
        predicted_class_tensor, \
        probabilities = calculate_confidence(
            logits
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predicted_index = (
        predicted_class_tensor.item()
    )

    predicted_disease = (
        CLASS_NAMES[predicted_index]
    )

    confidence = (
        confidence_tensor.item()
        * 100
    )

    # --------------------------------------------------------
    # Uncertainty
    # --------------------------------------------------------

    entropy = calculate_entropy(
        probabilities
    )

    normalized_entropy = normalize_entropy(
        entropy,
        num_classes=len(CLASS_NAMES)
    )

    uncertainty_level = get_uncertainty_level(
        normalized_entropy.item()
    )

    # --------------------------------------------------------
    # Top-3 predictions
    # --------------------------------------------------------

    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=3,
        dim=1
    )

    top3 = []

    for probability, index in zip(
        top_probabilities[0],
        top_indices[0]
    ):

        top3.append({
            "class": CLASS_NAMES[
                index.item()
            ],

            "confidence": round(
                probability.item() * 100,
                2
            )
        })

    # --------------------------------------------------------
    # Prediction Agent Input
    # --------------------------------------------------------

    prediction_result = {

        "disease": predicted_disease,

        "confidence": round(
            confidence,
            2
        ),

        "top3": top3
    }

    # --------------------------------------------------------
    # Reliability Agent Input
    # --------------------------------------------------------

    uncertainty_result = {

        "entropy": round(
            entropy.item(),
            4
        ),

        "normalized_score": round(
            normalized_entropy.item(),
            4
        ),

        "uncertainty_level":
            uncertainty_level
    }

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    gradcam_path = find_gradcam(
        image_path
    )

    if gradcam_path:

        print(
            f"[INFO] Grad-CAM found:"
        )

        print(
            gradcam_path
        )

    else:

        print(
            "[INFO] Grad-CAM not found for this image."
        )

        # Expected because Grad-CAM must first be
        # generated for the selected image.

        gradcam_path = (
            GRADCAM_DIR
            / f"gradcam_{image_path.stem}.jpg"
        )

    # --------------------------------------------------------
    # Agents
    # --------------------------------------------------------

    orchestrator = DermaOrchestrator()

    agent_response = orchestrator.run(
        prediction_result,
        uncertainty_result,
        str(gradcam_path)
    )

    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    agent_response["image_path"] = str(
        image_path
    )

    return agent_response


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("DermaAgent - Real ResNet18 Inference")
    print("=" * 60)

    print(
        f"[INFO] Device: {DEVICE}"
    )

    # --------------------------------------------------------
    # Select a real test image automatically
    # --------------------------------------------------------

    image_files = list(
        TEST_DIR.rglob("*.jpg")
    )

    if not image_files:

        raise FileNotFoundError(
            f"No test images found in {TEST_DIR}"
        )

    test_image = image_files[0]

    print(
        f"[INFO] Test image: {test_image}"
    )

    # --------------------------------------------------------
    # Run prediction
    # --------------------------------------------------------

    result = predict(
        test_image
    )

    # --------------------------------------------------------
    # Display PredictionAgent
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("[PredictionAgent]")
    print("=" * 60)

    print(
        f"Disease    : "
        f"{result['prediction']['disease']}"
    )

    print(
        f"Confidence : "
        f"{result['prediction']['confidence']}%"
    )

    print(
        "\nTop-3:"
    )

    for rank, item in enumerate(
        result["prediction"]["top3"],
        start=1
    ):

        print(
            f"  {rank}. "
            f"{item['class']} -> "
            f"{item['confidence']}%"
        )

    # --------------------------------------------------------
    # Display ReliabilityAgent
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("[ReliabilityAgent]")
    print("=" * 60)

    print(
        f"Entropy           : "
        f"{result['reliability']['entropy']}"
    )

    print(
        f"Normalized Score  : "
        f"{result['reliability']['normalized_score']}"
    )

    print(
        f"Uncertainty Level : "
        f"{result['reliability']['uncertainty_level']}"
    )

    print(
        f"Advisory          : "
        f"{result['reliability']['advisory']}"
    )

    # --------------------------------------------------------
    # Display ExplainabilityAgent
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("[ExplainabilityAgent]")
    print("=" * 60)

    print(
        f"Grad-CAM Available : "
        f"{result['explainability']['gradcam_available']}"
    )

    print(
        f"Grad-CAM Path      : "
        f"{result['explainability']['gradcam_path']}"
    )

    print()
    print("=" * 60)
    print("[SUCCESS] Real inference completed!")
    print("=" * 60)
