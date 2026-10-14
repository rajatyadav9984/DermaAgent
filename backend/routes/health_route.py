"""
DermaAgent - Health Check Route
Provides system status, API version, and loaded model verification.
"""

from flask import Blueprint, jsonify
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

health_bp = Blueprint("health", __name__)

@health_bp.route("/api/health", methods=["GET"])
def health_check():
    """Health status endpoint."""
    ham_model = PROJECT_ROOT / "model" / "real_skin_disease_model.pth"
    isic_model = PROJECT_ROOT / "model" / "isic2019_skin_disease_model.pth"

    return jsonify({
        "status": "healthy",
        "service": "DermaAgent API",
        "version": "1.0.0",
        "models": {
            "ham10000_baseline": "available" if ham_model.exists() else "not_found",
            "isic2019_classifier": "available" if isic_model.exists() else "not_trained"
        }
    }), 200
