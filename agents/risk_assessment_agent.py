"""
DermaAgent - Risk Assessment & Clinical Screening Priority Agent
Assigns transparent rule-based screening urgency priority (Low, Moderate, High)
based on lesion classification, symptoms, uncertainty, and image quality.
Provides rationale, physician consultation guidance, and safety disclaimers.
"""

from pathlib import Path
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Malignant and Pre-cancerous Disease Class Identifiers
MALIGNANT_CLASSES = {"MEL", "MELANOMA", "BCC", "SCC"}
PRECANCEROUS_CLASSES = {"AK", "AKIEC"}
BENIGN_CLASSES = {"NV", "BKL", "DF", "VASC"}

# High-Risk Alarm Symptoms
ALARM_SYMPTOMS = {"bleeding", "change_in_size", "change_in_color"}


class RiskAssessmentAgent:
    """
    Evaluates clinical screening priority (Triage Priority: Low, Moderate, High)
    using rule-based logic without making diagnostic medical claims.
    """

    def __init__(self):
        self.name = "RiskAssessmentAgent"

    def process(self, prediction_result=None, uncertainty_result=None, symptoms=None, image_quality=None):
        """
        Computes screening priority, clinical reasoning, and next-step recommendations.
        """
        prediction_result = prediction_result or {}
        uncertainty_result = uncertainty_result or {}
        symptoms = symptoms or []
        image_quality = image_quality or {}

        top_disease = str(prediction_result.get("disease", "")).upper()
        confidence = float(prediction_result.get("confidence", 0.0))
        top3 = prediction_result.get("top3", [])
        uncertainty_level = str(uncertainty_result.get("uncertainty_level", "Moderate"))
        symptom_set = set(str(s).lower() for s in symptoms)

        reasoning = []
        priority = "Low"
        urgency_timeline = "Routine Check"

        # Check for presence of malignancy in top prediction or Top-3 candidates
        is_top_malignant = top_disease in MALIGNANT_CLASSES
        is_top_precancerous = top_disease in PRECANCEROUS_CLASSES

        malignant_in_top3 = any(
            str(item.get("class", "")).upper() in MALIGNANT_CLASSES and float(item.get("confidence", 0)) >= 12.0
            for item in top3
        )

        alarm_symptoms_present = [s for s in ALARM_SYMPTOMS if s in symptom_set]

        # -------------------------------------------------------------
        # 1. High Screening Priority Rules
        # -------------------------------------------------------------
        if is_top_malignant:
            priority = "High"
            urgency_timeline = "Prompt Review Recommended (Within 1–2 Weeks)"
            reasoning.append(
                f"Primary AI model prediction indicates a potential malignant condition ({top_disease}) with {confidence:.1f}% confidence."
            )
            if alarm_symptoms_present:
                reasoning.append(
                    f"Reported alarm symptoms ({', '.join(alarm_symptoms_present).replace('_', ' ')}) are clinically associated with evolving lesions."
                )

        elif malignant_in_top3 and (alarm_symptoms_present or uncertainty_level == "High"):
            priority = "High"
            urgency_timeline = "Prompt Review Recommended (Within 1–2 Weeks)"
            reasoning.append(
                "A malignant condition (e.g., Melanoma/SCC/BCC) ranks prominently in Top-3 differential predictions alongside active symptoms or high uncertainty."
            )

        elif len(alarm_symptoms_present) >= 2:
            priority = "High"
            urgency_timeline = "Prompt Review Recommended"
            reasoning.append(
                f"Multiple rapid lesion change symptoms reported: {', '.join(alarm_symptoms_present).replace('_', ' ')}."
            )

        # -------------------------------------------------------------
        # 2. Moderate Screening Priority Rules
        # -------------------------------------------------------------
        elif is_top_precancerous:
            priority = "Moderate"
            urgency_timeline = "Schedule Evaluation (Within 2–4 Weeks)"
            reasoning.append(
                f"Predicted condition ({top_disease}) is categorized as pre-cancerous (Actinic Keratosis / Bowen's), which warrants dermatological monitoring and prophylactic management."
            )

        elif malignant_in_top3:
            priority = "Moderate"
            urgency_timeline = "Schedule Evaluation (Within 2–4 Weeks)"
            reasoning.append(
                "A potentially serious condition appears in the Top-3 differential predictions, suggesting in-person dermoscopy is prudent."
            )

        elif uncertainty_level == "High":
            priority = "Moderate"
            urgency_timeline = "Professional Review Recommended"
            reasoning.append(
                "Prediction shows elevated Shannon entropy (High Uncertainty), meaning the AI model cannot definitively rule out alternative skin pathologies."
            )

        elif len(symptom_set) >= 2:
            priority = "Moderate"
            urgency_timeline = "Routine Clinical Consultation"
            reasoning.append(
                f"Multiple local symptoms reported ({', '.join(symptoms).replace('_', ' ')}), which could indicate active inflammation or infection."
            )

        # -------------------------------------------------------------
        # 3. Low Screening Priority Rules
        # -------------------------------------------------------------
        else:
            priority = "Low"
            urgency_timeline = "Routine Self-Monitoring / Periodic Skin Exam"
            reasoning.append(
                f"Predicted condition ({top_disease}) is categorized as benign with {uncertainty_level} uncertainty and no reported alarming change symptoms."
            )

        # 4. Image Quality Factor
        if image_quality.get("is_acceptable") is False:
            reasoning.append(
                "Note: Suboptimal image quality was detected. Re-evaluating with a clearer, well-lit dermoscopic image may refine this assessment."
            )

        # 5. Tailored Clinical Recommendation
        if priority == "High":
            recommendation = (
                "Please schedule an in-person clinical consultation with a licensed dermatologist or skin cancer specialist. "
                "Bring note of when the spot appeared, any changes noticed, and avoid scratch trauma or self-treatment."
            )
        elif priority == "Moderate":
            recommendation = (
                "A non-urgent medical appointment with a healthcare professional or primary care physician is recommended "
                "to perform an in-person dermoscopic skin check."
            )
        else:
            recommendation = (
                "Continue routine self-monitoring. Periodically re-check the spot using the ABCDE guidelines (Asymmetry, Border, Color, Diameter, Evolution). "
                "Consult a doctor promptly if you notice new growth, itching, or bleeding."
            )

        return {
            "agent": self.name,
            "priority_level": priority,
            "urgency_timeline": urgency_timeline,
            "reasoning": reasoning,
            "recommendation": recommendation,
            "alarm_symptoms_flagged": alarm_symptoms_present,
            "disclaimer": (
                "DermaAgent is an AI research screening prototype, NOT a diagnostic medical device. "
                "Priority levels reflect triage urgency for professional clinical review, not a validated clinical diagnosis."
            )
        }


if __name__ == "__main__":
    agent = RiskAssessmentAgent()
    pred_sample = {"disease": "MEL", "confidence": 58.2, "top3": [{"class": "MEL", "confidence": 58.2}]}
    uncert_sample = {"uncertainty_level": "Low", "entropy": 0.42}
    symptoms_sample = ["bleeding", "change_in_size"]
    res = agent.process(pred_sample, uncert_sample, symptoms_sample)
    print(f"[INFO] Priority Level: {res['priority_level']} ({res['urgency_timeline']})")
    print("Reasoning:")
    for r in res['reasoning']:
        print(f"  - {r}")
    print(f"Recommendation: {res['recommendation']}")
