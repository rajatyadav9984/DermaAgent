"""
DermaAgent - Flask Prediction Route

Handles:
- Image upload
- Optional symptoms
- Real ResNet18 prediction
- Uncertainty calculation
- AI Agents
- Grad-CAM generation
- Image/Grad-CAM serving
"""

import sys
import uuid
from pathlib import Path

from flask import (
    Blueprint,
    request,
    jsonify,
    send_from_directory
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent.parent
)

ML_DIR = PROJECT_ROOT / "ml"

UPLOADS_DIR = PROJECT_ROOT / "uploads"

GRADCAM_DIR = (
    PROJECT_ROOT
    / "model"
    / "gradcam"
)

UPLOADS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

GRADCAM_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# ML PATH
# ============================================================

if str(ML_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(ML_DIR)
    )


# ============================================================
# ML IMPORTS
# ============================================================

from predict import predict

from gradcam import generate_gradcam


# ============================================================
# BLUEPRINT
# ============================================================

predict_bp = Blueprint(
    "predict",
    __name__
)


# ============================================================
# ALLOWED FILE TYPES
# ============================================================

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg"
}


def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# PREDICTION API
# ============================================================

@predict_bp.route(
    "/api/predict",
    methods=["POST"]
)
def handle_prediction():

    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if "image" not in request.files:

        return jsonify({
            "status": "error",
            "error": "No image file provided."
        }), 400

    file = request.files["image"]

    if file.filename == "":

        return jsonify({
            "status": "error",
            "error": "No file selected."
        }), 400

    if not allowed_file(file.filename):

        return jsonify({
            "status": "error",
            "error": (
                "Invalid file type. "
                "Allowed: png, jpg, jpeg"
            )
        }), 400


    # --------------------------------------------------------
    # Save uploaded image
    # --------------------------------------------------------

    extension = (
        file.filename
        .rsplit(".", 1)[1]
        .lower()
    )

    unique_filename = (
        f"skin_{uuid.uuid4().hex[:8]}."
        f"{extension}"
    )

    saved_image_path = (
        UPLOADS_DIR
        / unique_filename
    )

    file.save(
        saved_image_path
    )

    print(
        f"[INFO] Image uploaded: "
        f"{saved_image_path}"
    )


    # --------------------------------------------------------
    # Read symptoms
    # --------------------------------------------------------

    symptoms = request.form.getlist(
        "symptoms"
    )

    if not symptoms and request.is_json:

        data = request.get_json(
            silent=True
        ) or {}

        symptoms = data.get(
            "symptoms",
            []
        )

    print(
        f"[INFO] Symptoms received: "
        f"{symptoms}"
    )


    # --------------------------------------------------------
    # Generate Grad-CAM
    # --------------------------------------------------------

    gradcam_path = None

    try:

        gradcam_path = generate_gradcam(
            saved_image_path
        )

        print(
            f"[SUCCESS] Grad-CAM generated: "
            f"{gradcam_path}"
        )

    except Exception as error:

        print(
            "[WARNING] "
            f"Grad-CAM generation failed: "
            f"{error}"
        )


    # --------------------------------------------------------
    # Real ResNet18 Prediction
    # --------------------------------------------------------

    try:

        # Current real predict.py accepts
        # only image_path.

        agent_response = predict(
            saved_image_path
        )


        # ----------------------------------------------------
        # Base image URL
        # ----------------------------------------------------

        host_url = (
            request.host_url
            .rstrip("/")
        )

        agent_response[
            "image_url"
        ] = (
            f"{host_url}/uploads/"
            f"{unique_filename}"
        )


        # ----------------------------------------------------
        # Symptoms information
        # ----------------------------------------------------

        agent_response[
            "symptoms"
        ] = symptoms


        # ----------------------------------------------------
        # Grad-CAM URL
        # ----------------------------------------------------

        if (
            gradcam_path
            and
            Path(gradcam_path).exists()
        ):

            gradcam_name = (
                Path(gradcam_path).name
            )

            agent_response[
                "explainability"
            ][
                "gradcam_available"
            ] = True

            agent_response[
                "explainability"
            ][
                "gradcam_path"
            ] = str(
                gradcam_path
            )

            agent_response[
                "explainability"
            ][
                "gradcam_url"
            ] = (
                f"{host_url}/gradcam/"
                f"{gradcam_name}"
            )

        else:

            agent_response[
                "explainability"
            ][
                "gradcam_available"
            ] = False

            agent_response[
                "explainability"
            ][
                "gradcam_url"
            ] = None


        # ----------------------------------------------------
        # Final API response
        # ----------------------------------------------------

        return jsonify({

            "status": "success",

            "data": agent_response

        }), 200


    except Exception as error:

        print(
            "[ERROR] Prediction failed:"
        )

        print(error)

        return jsonify({

            "status": "error",

            "error": (
                f"Prediction failed: "
                f"{str(error)}"
            )

        }), 500


# ============================================================
# SERVE UPLOADED IMAGE
# ============================================================

@predict_bp.route(
    "/uploads/<filename>",
    methods=["GET"]
)
def serve_upload(filename):

    return send_from_directory(
        UPLOADS_DIR,
        filename
    )


# ============================================================
# SERVE GRAD-CAM IMAGE
# ============================================================

@predict_bp.route(
    "/gradcam/<filename>",
    methods=["GET"]
)
def serve_gradcam(filename):

    return send_from_directory(
        GRADCAM_DIR,
        filename
    )
