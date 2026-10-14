"""
DermaAgent - Clinical Question Agent
Generates contextual clinical follow-up questions based on symptoms,
predicted skin condition, uncertainty, duration, and anatomical body site.
Identifies missing diagnostic information for patient review.
"""

from pathlib import Path
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class QuestionAgent:
    """
    Formulates clinical anamnesis questions tailored to the predicted lesion and symptoms.
    Assists patient in gathering complete history prior to professional consultation.
    """

    def __init__(self):
        self.name = "QuestionAgent"

    def process(self, prediction_result=None, symptoms=None, uncertainty_result=None, patient_context=None):
        """
        Generates targeted follow-up questions.
        """
        symptoms = symptoms or []
        patient_context = patient_context or {}
        disease = (prediction_result or {}).get("disease", "").upper()

        questions = []
        missing_info = []

        # 1. Lesion Duration & Evolution (Essential for all skin screening)
        if "duration" not in patient_context:
            missing_info.append("Lesion Duration")
            questions.append({
                "id": "q_duration",
                "category": "History & Timeline",
                "question": "How long has this skin spot or lesion been present? (e.g., newly appeared within weeks, present for months/years, or present since birth?)",
                "importance": "High",
                "rationale": "Rapidly evolving new lesions have higher diagnostic urgency than stable longstanding birthmarks."
            })

        # 2. ABCDE Evolution (Changes in size, shape, color)
        if not any(s in symptoms for s in ["change_in_size", "change_in_color"]):
            missing_info.append("Evolution / Growth History")
            questions.append({
                "id": "q_evolution",
                "category": "Lesion Changes",
                "question": "Have you noticed any change in the size, border shape, darkness, or color pattern of this lesion over the past 1–3 months?",
                "importance": "High",
                "rationale": "Evolution (E in ABCDE criteria) is the single most sensitive indicator of dysplastic or malignant transformation."
            })

        # 3. Anatomical Body Location
        if "body_area" not in patient_context:
            missing_info.append("Anatomical Body Location")
            questions.append({
                "id": "q_location",
                "category": "Anatomical Site",
                "question": "On which part of your body is this lesion located? (e.g., face, scalp, back, chest, arm, leg, palm, or sole?)",
                "importance": "Medium",
                "rationale": "Sun-exposed areas (face, shoulders) correlate more with keratoses and carcinomas, while acral areas require specific examination."
            })

        # 4. Symptom-Specific Deep Dives
        symptom_set = set(str(s).lower() for s in symptoms)
        if "itching" in symptom_set or "pain" in symptom_set:
            questions.append({
                "id": "q_discomfort_severity",
                "category": "Symptom Characteristics",
                "question": "Is the itching or pain constant or intermittent? Does it wake you up at night or worsen after sun exposure/friction?",
                "importance": "Medium",
                "rationale": "Pruritus severity and pain patterns help differentiate inflamed benign lesions from neuropathic ulcerations."
            })
        elif len(symptoms) == 0:
            missing_info.append("Subjective Sensations")
            questions.append({
                "id": "q_sensations",
                "category": "Symptoms Check",
                "question": "Does the lesion itch, hurt, sting, bleed easily, or produce any clear fluid/crust?",
                "importance": "Medium",
                "rationale": "Asymptomatic lesions can still be clinically significant, but ulceration or bleeding increases urgency."
            })

        if "bleeding" in symptom_set or "crusting" in symptom_set:
            questions.append({
                "id": "q_bleeding_trigger",
                "category": "Ulceration & Bleeding",
                "question": "Does the lesion bleed spontaneously without trauma, or does it only bleed when scratched or rubbed by clothing?",
                "importance": "High",
                "rationale": "Spontaneous bleeding or non-healing crusted sores are characteristic warning signs in malignant lesions like SCC or BCC."
            })

        # 5. Disease-Specific Contextual Inquiry
        if disease in ["MEL", "MELANOMA"]:
            questions.append({
                "id": "q_melanoma_risk",
                "category": "Risk Background",
                "question": "Do you have a personal or family history of melanoma or other skin cancers? Do you have numerous moles on your body?",
                "importance": "High",
                "rationale": "Genetic predisposition and dysplastic nevus syndrome strongly elevate melanoma risk."
            })
        elif disease in ["BCC", "SCC", "AK", "AKIEC"]:
            questions.append({
                "id": "q_sun_history",
                "category": "Sun Exposure History",
                "question": "Have you had significant lifetime sun exposure, frequent outdoor occupational exposure, or severe blistering sunburns in the past?",
                "importance": "Medium",
                "rationale": "Cumulative ultraviolet radiation is the primary etiological driver of keratinocyte cancers and actinic damage."
            })

        # 6. Clinical Summary Prompt
        summary_prompt = (
            f"Based on your screening, {len(questions)} relevant follow-up questions were identified. "
            "Preparing answers to these questions will significantly help your doctor or dermatologist during your clinical evaluation."
        )

        return {
            "agent": self.name,
            "total_questions": len(questions),
            "questions": questions,
            "missing_information": missing_info,
            "summary_prompt": summary_prompt
        }


if __name__ == "__main__":
    agent = QuestionAgent()
    pred_sample = {"disease": "SCC", "confidence": 35.0}
    symptom_sample = ["bleeding", "itching"]
    res = agent.process(pred_sample, symptom_sample)
    print(f"[INFO] QuestionAgent generated {res['total_questions']} questions:")
    for idx, q in enumerate(res['questions'], 1):
        print(f"  {idx}. [{q['category']}] ({q['importance']}) {q['question']}")
