"""
DermaAgent - Prediction Service Layer
Provides persistent model caching, decoupled inference pipeline,
and seamless Grad-CAM and Multi-Agent orchestration.
"""

from pathlib import Path
import sys
import threading
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ML_DIR = PROJECT_ROOT / "ml"
AGENTS_DIR = PROJECT_ROOT / "agents"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from uncertainty import (
    calculate_confidence,
    calculate_entropy,
    normalize_entropy,
    get_uncertainty_level
)
from agents.orchestrator import DermaOrchestrator

HAM_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
ISIC_CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 224
INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class PredictionService:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.device = DEVICE
        self.models = {}
        self.orchestrator = DermaOrchestrator()
        self._load_baseline_model()
        self.load_isic_model_if_available()

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _load_baseline_model(self):
        """Loads and caches the 7-class baseline ResNet18 model."""
        model_path = PROJECT_ROOT / "model" / "real_skin_disease_model.pth"
        if not model_path.exists():
            print(f"[WARNING] Baseline model not found at {model_path}")
            return

        print(f"[INFO] Initializing and caching baseline model ({model_path.name})...")
        model = models.resnet18(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, len(HAM_CLASSES))

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        self.models["ham10000"] = model
        print("[SUCCESS] Baseline ResNet18 model cached in memory.")

    def load_isic_model_if_available(self):
        """Loads the 8-class ISIC 2019 model if checkpoint exists."""
        model_path = PROJECT_ROOT / "model" / "isic2019_skin_disease_model.pth"
        if model_path.exists() and "isic2019" not in self.models:
            print(f"[INFO] Loading 8-class ISIC 2019 model ({model_path.name})...")
            model = models.resnet18(weights=None)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, len(ISIC_CLASSES))

            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()
            self.models["isic2019"] = model
            print("[SUCCESS] ISIC 2019 model cached in memory.")

    def predict_image(self, image_path, model_type="auto", symptoms=None):
        """
        Executes fast in-memory inference, uncertainty quantification,
        and agent orchestration for a given image.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Auto-detect or select model
        if model_type in ["auto", "isic2019"]:
            self.load_isic_model_if_available()
            if "isic2019" in self.models:
                model = self.models["isic2019"]
                class_names = ISIC_CLASSES
                active_type = "isic2019"
            else:
                if "ham10000" not in self.models:
                    self._load_baseline_model()
                model = self.models.get("ham10000")
                class_names = HAM_CLASSES
                active_type = "ham10000"
        else:
            if "ham10000" not in self.models:
                self._load_baseline_model()
            model = self.models.get("ham10000")
            class_names = HAM_CLASSES
            active_type = "ham10000"

        if model is None:
            raise RuntimeError("No skin disease classification model is currently loaded.")

        # Preprocess image
        image = Image.open(image_path).convert("RGB")
        tensor = INFERENCE_TRANSFORMS(image).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            logits = model(tensor)
            confidence_tensor, pred_class_tensor, probabilities = calculate_confidence(logits)

        pred_index = pred_class_tensor.item()
        pred_disease = class_names[pred_index]
        confidence = confidence_tensor.item() * 100

        # Uncertainty quantification
        entropy = calculate_entropy(probabilities)
        normalized_entropy = normalize_entropy(entropy, num_classes=len(class_names))
        uncertainty_level = get_uncertainty_level(normalized_entropy.item())

        # Top-3 predictions
        k = min(3, len(class_names))
        top_probs, top_indices = torch.topk(probabilities, k=k, dim=1)
        top3 = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            top3.append({
                "class": class_names[idx.item()],
                "confidence": round(prob.item() * 100, 2)
            })

        prediction_result = {
            "disease": pred_disease,
            "confidence": round(confidence, 2),
            "top3": top3
        }

        uncertainty_result = {
            "entropy": round(entropy.item(), 4),
            "normalized_score": round(normalized_entropy.item(), 4),
            "uncertainty_level": uncertainty_level
        }

        # Multi-Agent pipeline
        agent_response = self.orchestrator.run(
            prediction_result=prediction_result,
            uncertainty_result=uncertainty_result,
            gradcam_path=str(image_path),
            image_path=str(image_path),
            symptoms=symptoms
        )

        agent_response["image_path"] = str(image_path)
        agent_response["model_type"] = active_type
        agent_response["num_classes"] = len(class_names)
        agent_response["_model_instance"] = model
        agent_response["_class_names"] = class_names

        return agent_response
