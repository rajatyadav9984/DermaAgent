"""
DermaAgent - ML Inference & Prediction Service Unit Tests
Tests model loading, uncertainty quantification, and agent formatting.
"""

from pathlib import Path
import pytest
import torch

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ML_DIR = PROJECT_ROOT / "ml"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from backend.services.prediction_service import PredictionService
from uncertainty import calculate_confidence, calculate_entropy, normalize_entropy, get_uncertainty_level

def test_uncertainty_math():
    """Validates mathematical correctness of entropy and uncertainty level."""
    # Definite distribution (no uncertainty)
    certain_probs = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    ent = calculate_entropy(certain_probs)
    norm_ent = normalize_entropy(ent, num_classes=7)
    level = get_uncertainty_level(norm_ent.item())
    assert level == "Low"
    assert norm_ent.item() < 0.05

    # Uniform distribution (maximum uncertainty)
    uniform_probs = torch.full((1, 7), 1.0 / 7.0)
    ent_max = calculate_entropy(uniform_probs)
    norm_max = normalize_entropy(ent_max, num_classes=7)
    level_max = get_uncertainty_level(norm_max.item())
    assert level_max == "High"
    assert abs(norm_max.item() - 1.0) < 0.01

def test_prediction_service_baseline():
    """Tests that PredictionService loads the 7-class baseline ResNet18 and runs inference."""
    service = PredictionService.get_instance()
    assert "ham10000" in service.models

    test_img = PROJECT_ROOT / "dataset_real" / "test" / "akiec" / "ISIC_0024511.jpg"
    if not test_img.exists():
        pytest.skip("Test image not available in dataset_real/test.")

    res = service.predict_image(test_img, model_type="ham10000")
    assert "prediction" in res
    assert res["prediction"]["disease"] in ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    assert 0 <= res["prediction"]["confidence"] <= 100
    assert len(res["prediction"]["top3"]) == 3

def test_uncertainty_math_8_classes():
    """Validates mathematical correctness of entropy for 8 ISIC classes."""
    certain_probs = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    ent = calculate_entropy(certain_probs)
    norm_ent = normalize_entropy(ent, num_classes=8)
    level = get_uncertainty_level(norm_ent.item())
    assert level == "Low"
    assert norm_ent.item() < 0.05

    uniform_probs = torch.full((1, 8), 1.0 / 8.0)
    ent_max = calculate_entropy(uniform_probs)
    norm_max = normalize_entropy(ent_max, num_classes=8)
    level_max = get_uncertainty_level(norm_max.item())
    assert level_max == "High"
    assert abs(norm_max.item() - 1.0) < 0.01

def test_prediction_service_isic2019():
    """Tests that PredictionService loads the 8-class ISIC 2019 model and performs inference."""
    isic_model_path = PROJECT_ROOT / "model" / "isic2019_skin_disease_model.pth"
    if not isic_model_path.exists():
        pytest.skip("isic2019_skin_disease_model.pth not yet trained.")

    service = PredictionService.get_instance()
    service.load_isic_model_if_available()
    assert "isic2019" in service.models

    test_img = PROJECT_ROOT / "dataset_real" / "test" / "akiec" / "ISIC_0024511.jpg"
    if not test_img.exists():
        pytest.skip("Test image not available.")

    res = service.predict_image(test_img, model_type="isic2019")
    assert "prediction" in res
    assert res["prediction"]["disease"] in ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    assert 0 <= res["prediction"]["confidence"] <= 100
    assert len(res["prediction"]["top3"]) == 3
    assert res["model_type"] == "isic2019"
    assert res["num_classes"] == 8
    assert res["reliability"]["uncertainty_level"] in ["Low", "Moderate", "High"]
    # Verify new agents integrated into prediction response
    assert "image_quality" in res
    assert "is_acceptable" in res["image_quality"]
    assert "risk_assessment" in res
    assert res["risk_assessment"]["priority_level"] in ["Low", "Moderate", "High"]
    assert "clinical_questions" in res
    assert len(res["clinical_questions"]["questions"]) > 0

def test_image_quality_agent():
    """Validates image quality checks for sharpness, lighting, and dimensions."""
    from agents.image_quality_agent import ImageQualityAgent
    agent = ImageQualityAgent()
    test_img = PROJECT_ROOT / "dataset_real" / "test" / "akiec" / "ISIC_0024511.jpg"
    if not test_img.exists():
        pytest.skip("Test image not available.")

    res = agent.process(test_img)
    assert res["agent"] == "ImageQualityAgent"
    assert "is_acceptable" in res
    assert 0 <= res["quality_score"] <= 100
    assert "metrics" in res
    assert "sharpness" in res["metrics"]
    assert "mean_brightness" in res["metrics"]
    assert "contrast" in res["metrics"]
    assert "advisory" in res

def test_question_agent():
    """Validates tailored clinical question generation."""
    from agents.question_agent import QuestionAgent
    agent = QuestionAgent()
    pred_res = {"disease": "MEL", "confidence": 72.0}
    symptoms = ["itching", "bleeding"]
    res = agent.process(prediction_result=pred_res, symptoms=symptoms)

    assert res["agent"] == "QuestionAgent"
    assert res["total_questions"] > 0
    assert len(res["questions"]) == res["total_questions"]
    for q in res["questions"]:
        assert "question" in q
        assert "category" in q
        assert q["importance"] in ["High", "Medium", "Low"]

def test_risk_assessment_agent():
    """Validates rule-based screening priority triage."""
    from agents.risk_assessment_agent import RiskAssessmentAgent
    agent = RiskAssessmentAgent()

    # Case 1: High priority (Malignant + Bleeding)
    high_res = agent.process(
        prediction_result={"disease": "MEL", "confidence": 60.0, "top3": [{"class": "MEL", "confidence": 60.0}]},
        uncertainty_result={"uncertainty_level": "Low"},
        symptoms=["bleeding"]
    )
    assert high_res["priority_level"] == "High"
    assert len(high_res["reasoning"]) > 0
    assert "recommendation" in high_res

    # Case 2: Low priority (Benign + Low uncertainty + no alarm symptoms)
    low_res = agent.process(
        prediction_result={"disease": "NV", "confidence": 90.0, "top3": [{"class": "NV", "confidence": 90.0}]},
        uncertainty_result={"uncertainty_level": "Low"},
        symptoms=[]
    )
    assert low_res["priority_level"] == "Low"
