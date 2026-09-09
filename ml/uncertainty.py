"""
DermaAgent - Prediction Uncertainty & Reliability Processor
Computes Softmax confidence, predictive Shannon entropy, normalized entropy score,
and assigns qualitative reliability levels (Low, Moderate, High).
"""

import sys
import math
import torch
import torch.nn.functional as F

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def calculate_confidence(logits):
    """
    Calculate prediction confidence from model logits.
    """

    probabilities = F.softmax(logits, dim=1)

    confidence, predicted_class = torch.max(
        probabilities,
        dim=1
    )

    return confidence, predicted_class, probabilities


def calculate_entropy(probabilities):
    """
    Calculate predictive entropy.
    Higher entropy = higher uncertainty.
    """

    entropy = -torch.sum(
        probabilities * torch.log(probabilities + 1e-8),
        dim=1
    )

    return entropy


def normalize_entropy(entropy, num_classes=7):
    """
    Normalize entropy between 0 and 1.
    """

    max_entropy = math.log(num_classes)

    normalized = entropy / max_entropy

    return normalized


def get_uncertainty_level(normalized_entropy):
    """
    Convert uncertainty score into a human-readable level.
    """

    if normalized_entropy < 0.30:
        return "Low"

    elif normalized_entropy < 0.60:
        return "Moderate"

    else:
        return "High"


if __name__ == "__main__":
    # Test uncertainty calculation with dummy logits
    dummy_logits = torch.tensor([[3.5, 1.2, 0.5, 0.1, -0.2, 0.0, -1.0]])
    conf, pred_cls, probs = calculate_confidence(dummy_logits)
    entropy = calculate_entropy(probs)
    norm_entropy = normalize_entropy(entropy, num_classes=7)
    level = get_uncertainty_level(norm_entropy.item())

    print("[INFO] Uncertainty Processor Self-Test:")
    print(f"Top Confidence     : {conf.item() * 100:.2f}%")
    print(f"Predictive Entropy : {entropy.item():.4f}")
    print(f"Normalized Entropy : {norm_entropy.item():.4f}")
    print(f"Uncertainty Level  : {level}")
