"""
DermaAgent - Reliability Agent
Interprets predictive uncertainty metrics and provides reliability advisory.
"""

import sys

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


class ReliabilityAgent:

    def __init__(self):
        self.name = "ReliabilityAgent"

    def process(self, uncertainty_result):
        """
        Interpret uncertainty metrics and generate
        an AI reliability advisory.
        """

        entropy = uncertainty_result["entropy"]
        normalized_score = uncertainty_result["normalized_score"]
        uncertainty_level = uncertainty_result["uncertainty_level"]

        if uncertainty_level == "Low":
            advisory = "Model prediction has relatively low uncertainty."

        elif uncertainty_level == "Moderate":
            advisory = "Model prediction has moderate uncertainty. Review recommended."

        else:
            advisory = "Model prediction has high uncertainty. Medical professional review recommended."

        return {
            "agent": self.name,
            "entropy": entropy,
            "normalized_score": normalized_score,
            "uncertainty_level": uncertainty_level,
            "advisory": advisory
        }
