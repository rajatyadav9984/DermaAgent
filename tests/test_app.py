"""
DermaAgent - Backend API Integration Tests
Tests /api/health and /api/predict validation, error handling, and response payloads.
"""

import io
from pathlib import Path
import pytest
from PIL import Image

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend.app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """GET /api/health returns 200 and healthy status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert "service" in data
    assert "models" in data

def test_predict_no_image(client):
    """POST /api/predict without an image file returns 400 Bad Request."""
    response = client.post("/api/predict", data={})
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert "No image file provided" in data["error"]

def test_predict_invalid_extension(client):
    """POST /api/predict with unsupported extension returns 400."""
    data = {
        "image": (io.BytesIO(b"dummy text content"), "test_file.txt")
    }
    response = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    res = response.get_json()
    assert res["status"] == "error"
    assert "Unsupported file type" in res["error"]

def test_predict_corrupted_image(client):
    """POST /api/predict with corrupt jpg bytes returns 400."""
    data = {
        "image": (io.BytesIO(b"\xFF\xD8\xFFcorrupted_invalid_data"), "corrupt.jpg")
    }
    response = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    res = response.get_json()
    assert res["status"] == "error"
    assert "Corrupted or invalid image" in res["error"]

def test_predict_valid_image(client):
    """POST /api/predict with a real valid image returns 200 with full multi-agent payload."""
    test_img = PROJECT_ROOT / "dataset_real" / "test" / "akiec" / "ISIC_0024511.jpg"
    if not test_img.exists():
        pytest.skip("Test image not available in dataset_real/test.")

    with open(test_img, "rb") as f:
        data = {
            "image": (f, "ISIC_0024511.jpg"),
            "symptoms": ["itching", "redness"]
        }
        response = client.post("/api/predict", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    res = response.get_json()
    assert res["status"] == "success"
    payload = res["data"]

    # Verify PredictionAgent output
    assert "prediction" in payload
    assert "disease" in payload["prediction"]
    assert "confidence" in payload["prediction"]
    assert 0 <= payload["prediction"]["confidence"] <= 100
    assert len(payload["prediction"]["top3"]) > 0

    # Verify ReliabilityAgent output
    assert "reliability" in payload
    assert "uncertainty_level" in payload["reliability"]
    assert payload["reliability"]["uncertainty_level"] in ["Low", "Moderate", "High"]

    # Verify ExplainabilityAgent output
    assert "explainability" in payload
    assert "gradcam_available" in payload["explainability"]

def test_predict_isic2019_model(client):
    """POST /api/predict with model_type='isic2019' returns 8-class prediction, top3, and Grad-CAM."""
    test_img = PROJECT_ROOT / "dataset_real" / "test" / "akiec" / "ISIC_0024511.jpg"
    if not test_img.exists():
        pytest.skip("Test image not available.")

    with open(test_img, "rb") as f:
        data = {
            "image": (f, "ISIC_0024511.jpg"),
            "model_type": "isic2019",
            "symptoms": ["bleeding", "scaling"]
        }
        response = client.post("/api/predict", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    res = response.get_json()
    assert res["status"] == "success"
    payload = res["data"]

    # Verify 8-class model metadata
    assert payload["model_type"] == "isic2019"
    assert payload["num_classes"] == 8

    # Verify PredictionAgent output
    assert "prediction" in payload
    assert payload["prediction"]["disease"] in ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    assert 0 <= payload["prediction"]["confidence"] <= 100
    assert len(payload["prediction"]["top3"]) == 3
    for item in payload["prediction"]["top3"]:
        assert item["class"] in ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
        assert 0 <= item["confidence"] <= 100

    # Verify ReliabilityAgent output
    assert "reliability" in payload
    assert payload["reliability"]["uncertainty_level"] in ["Low", "Moderate", "High"]

    # Verify ExplainabilityAgent output & Grad-CAM
    assert "explainability" in payload
    assert payload["explainability"]["gradcam_available"] is True
    assert payload["explainability"]["gradcam_url"] is not None

    # Verify ImageQualityAgent output
    assert "image_quality" in payload
    assert payload["image_quality"]["agent"] == "ImageQualityAgent"
    assert "is_acceptable" in payload["image_quality"]

    # Verify RiskAssessmentAgent output
    assert "risk_assessment" in payload
    assert payload["risk_assessment"]["agent"] == "RiskAssessmentAgent"
    assert payload["risk_assessment"]["priority_level"] in ["Low", "Moderate", "High"]
    assert "recommendation" in payload["risk_assessment"]

    # Verify QuestionAgent output
    assert "clinical_questions" in payload
    assert payload["clinical_questions"]["agent"] == "QuestionAgent"
    assert len(payload["clinical_questions"]["questions"]) > 0
