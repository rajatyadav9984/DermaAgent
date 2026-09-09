"""
DermaAgent - Explainability Agent
Packages Grad-CAM visual heatmap results for user output.
"""

from pathlib import Path
import sys

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


class ExplainabilityAgent:

    def __init__(self):
        self.name = "ExplainabilityAgent"

    def process(self, gradcam_path):
        """
        Package the Grad-CAM output for the final response.
        """

        path = Path(gradcam_path)

        return {
            "agent": self.name,
            "gradcam_available": path.exists(),
            "gradcam_path": str(path)
        }
