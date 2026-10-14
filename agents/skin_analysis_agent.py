"""
DermaAgent - Skin Analysis Agent
Higher-level clinical research agent that aggregates predictions,
uncertainty levels, and visual explainability into a coherent diagnostic summary.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.orchestrator import DermaOrchestrator

class SkinAnalysisAgent:
    """
    Unified Skin Analysis Agent wrapping the DermaOrchestrator pipeline.
    """
    def __init__(self):
        self.name = "SkinAnalysisAgent"
        self.orchestrator = DermaOrchestrator()

    def analyze(self, prediction_result, uncertainty_result, gradcam_path, image_path=None, symptoms=None, patient_context=None):
        """
        Executes multi-agent synthesis with optional clinical symptoms context and image quality analysis.
        """
        response = self.orchestrator.run(
            prediction_result=prediction_result,
            uncertainty_result=uncertainty_result,
            gradcam_path=gradcam_path,
            image_path=image_path,
            symptoms=symptoms,
            patient_context=patient_context
        )

        response["symptoms_evaluated"] = symptoms or []
        response["educational_disclaimer"] = (
            "DermaAgent is an AI research prototype for educational and screening purposes only. "
            "It does not replace professional consultation with a certified medical doctor or dermatologist."
        )

        return response
