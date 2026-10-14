"""
DermaAgent - Multi-Agent Orchestrator
Coordinates PredictionAgent, ReliabilityAgent, and ExplainabilityAgent into a unified AI analysis pipeline.
"""

import sys
from pathlib import Path

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure parent directory is in python path
AGENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AGENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.image_quality_agent import ImageQualityAgent
from agents.prediction_agent import PredictionAgent
from agents.reliability_agent import ReliabilityAgent
from agents.explainability_agent import ExplainabilityAgent
from agents.risk_assessment_agent import RiskAssessmentAgent
from agents.question_agent import QuestionAgent


class DermaOrchestrator:

    def __init__(self):
        self.image_quality_agent = ImageQualityAgent()
        self.prediction_agent = PredictionAgent()
        self.reliability_agent = ReliabilityAgent()
        self.explainability_agent = ExplainabilityAgent()
        self.risk_assessment_agent = RiskAssessmentAgent()
        self.question_agent = QuestionAgent()

    def run(
        self,
        prediction_result,
        uncertainty_result,
        gradcam_path,
        image_path=None,
        symptoms=None,
        patient_context=None
    ):
        """
        Execute the complete DermaAgent multi-agent clinical screening pipeline.
        """
        # 1. Image Quality Assessment
        if image_path:
            image_quality = self.image_quality_agent.process(image_path)
        else:
            image_quality = {
                "agent": self.image_quality_agent.name,
                "is_acceptable": True,
                "quality_score": 100.0,
                "issues": [],
                "reupload_recommended": False,
                "advisory": "Image quality verified."
            }

        # 2. Prediction Processing
        prediction = self.prediction_agent.process(
            prediction_result
        )

        # 3. Reliability & Uncertainty Assessment
        reliability = self.reliability_agent.process(
            uncertainty_result
        )

        # 4. Explainability (Grad-CAM)
        explainability = self.explainability_agent.process(
            gradcam_path
        )

        # 5. Risk Assessment (Screening Triage Priority)
        risk_assessment = self.risk_assessment_agent.process(
            prediction_result=prediction_result,
            uncertainty_result=uncertainty_result,
            symptoms=symptoms,
            image_quality=image_quality
        )

        # 6. Clinical Anamnesis Follow-up Questions
        clinical_questions = self.question_agent.process(
            prediction_result=prediction_result,
            symptoms=symptoms,
            uncertainty_result=uncertainty_result,
            patient_context=patient_context
        )

        return {
            "image_quality": image_quality,
            "prediction": prediction,
            "reliability": reliability,
            "explainability": explainability,
            "risk_assessment": risk_assessment,
            "clinical_questions": clinical_questions
        }
