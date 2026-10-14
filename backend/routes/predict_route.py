"""
DermaAgent - Flask Prediction Route
Handles image upload, validation, Grad-CAM generation, and multi-agent AI prediction.
"""

from pathlib import Path
import sys
import uuid
from flask import Blueprint, request, jsonify, send_from_directory

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
ML_DIR = PROJECT_ROOT / "ml"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
GRADCAM_DIR = PROJECT_ROOT / "model" / "gradcam"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from backend.utils.image_validator import validate_image
from backend.services.prediction_service import PredictionService
from gradcam import generate_gradcam

predict_bp = Blueprint("predict", __name__)

@predict_bp.route("/api/predict", methods=["POST"])
def handle_prediction():
    # 1. Check if image part is in request
    if "image" not in request.files:
        return jsonify({
            "status": "error",
            "error": "No image file provided in request."
        }), 400

    file = request.files["image"]

    # 2. Strict image validation (integrity, size, format, corrupt check)
    is_valid, error_msg, extension = validate_image(file)
    if not is_valid:
        return jsonify({
            "status": "error",
            "error": error_msg
        }), 400

    # 3. Save uploaded image with unique ID
    unique_filename = f"skin_{uuid.uuid4().hex[:8]}.{extension}"
    saved_image_path = UPLOADS_DIR / unique_filename
    file.save(saved_image_path)

    # 4. Read symptoms (form-data or JSON)
    symptoms = request.form.getlist("symptoms")
    if not symptoms and request.is_json:
        data = request.get_json(silent=True) or {}
        symptoms = data.get("symptoms", [])

    # 5. Execute model prediction via cached PredictionService
    try:
        model_type = request.form.get("model_type", "auto")
        service = PredictionService.get_instance()
        agent_response = service.predict_image(saved_image_path, model_type=model_type, symptoms=symptoms)

        model_instance = agent_response.pop("_model_instance", None)
        class_names = agent_response.pop("_class_names", None)

        # 6. Generate Grad-CAM explainability heatmap with the active model
        gradcam_path = None
        try:
            gradcam_path = generate_gradcam(saved_image_path, model_instance=model_instance, class_names=class_names)
        except Exception as err:
            print(f"[WARNING] Grad-CAM generation failed: {err}")

        host_url = request.host_url.rstrip("/")
        agent_response["image_url"] = f"{host_url}/uploads/{unique_filename}"
        agent_response["symptoms"] = symptoms

        # Attach Grad-CAM results to explainability section
        if gradcam_path and Path(gradcam_path).exists():
            gradcam_name = Path(gradcam_path).name
            agent_response["explainability"]["gradcam_available"] = True
            agent_response["explainability"]["gradcam_path"] = str(gradcam_path)
            agent_response["explainability"]["gradcam_url"] = f"{host_url}/gradcam/{gradcam_name}"
        else:
            agent_response["explainability"]["gradcam_available"] = False
            agent_response["explainability"]["gradcam_url"] = None

        return jsonify({
            "status": "success",
            "data": agent_response
        }), 200

    except Exception as err:
        print(f"[ERROR] Prediction failed: {err}")
        return jsonify({
            "status": "error",
            "error": f"Prediction failed: {str(err)}"
        }), 500


@predict_bp.route("/uploads/<filename>", methods=["GET"])
def serve_upload(filename):
    return send_from_directory(UPLOADS_DIR, filename)


@predict_bp.route("/gradcam/<filename>", methods=["GET"])
def serve_gradcam(filename):
    return send_from_directory(GRADCAM_DIR, filename)
