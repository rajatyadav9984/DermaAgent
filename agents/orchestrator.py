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

from agents.prediction_agent import PredictionAgent
from agents.reliability_agent import ReliabilityAgent
from agents.explainability_agent import ExplainabilityAgent


class DermaOrchestrator:

    def __init__(self):
        self.prediction_agent = PredictionAgent()
        self.reliability_agent = ReliabilityAgent()
        self.explainability_agent = ExplainabilityAgent()

    def run(
        self,
        prediction_result,
        uncertainty_result,
        gradcam_path
    ):
        """
        Execute the complete DermaAgent pipeline.
        """

        prediction = self.prediction_agent.process(
            prediction_result
        )

        reliability = self.reliability_agent.process(
            uncertainty_result
        )

        explainability = self.explainability_agent.process(
            gradcam_path
        )

        return {
            "prediction": prediction,
            "reliability": reliability,
            "explainability": explainability
        }
