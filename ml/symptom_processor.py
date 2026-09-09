"""
DermaAgent - Symptom Processor
Converts user text/selection symptoms into a binary feature vector.
"""

import sys

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SYMPTOM_VOCAB = [
    "itching",
    "redness",
    "pain",
    "swelling",
    "bleeding",
    "scaling",
    "burning",
    "crusting",
    "change_in_size",
    "change_in_color"
]


def process_symptoms(symptoms):
    """
    Convert user symptoms into a binary feature vector.
    """

    symptoms = [
        symptom.lower().strip()
        for symptom in symptoms
    ]

    feature_vector = []

    for symptom in SYMPTOM_VOCAB:

        if symptom in symptoms:
            feature_vector.append(1)
        else:
            feature_vector.append(0)

    return feature_vector


if __name__ == "__main__":

    user_symptoms = [
        "itching",
        "redness",
        "pain"
    ]

    features = process_symptoms(
        user_symptoms
    )

    print("Symptoms:")
    print(user_symptoms)

    print("\nFeature vector:")
    print(features)
