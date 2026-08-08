import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


CLASS_DESCRIPTIONS = {
    'nv':    'melanocytic nevi (a common benign mole)',
    'mel':   'melanoma (a malignant skin cancer requiring urgent attention)',
    'bkl':   'benign keratosis (a non-cancerous growth)',
    'bcc':   'basal cell carcinoma (a type of skin cancer)',
    'akiec': 'actinic keratoses (a pre-cancerous condition)',
    'vasc':  'vascular lesion (a benign blood vessel growth)',
    'df':    'dermatofibroma (a benign skin growth)'
}

NARRATOR_SYSTEM_PROMPT = """You are a medical AI assistant explaining a skin lesion classification result to a clinician.

You will receive:
- The model prediction and confidence score
- What image features contributed to the prediction
- Which region of the image the model focused on
- A critic review noting any agreements or contradictions

Write a plain English explanation in exactly 3 sentences:
1. What the model predicted and how confident it was
2. What the visual evidence shows combining both analysis findings
3. Whether the prediction should be trusted or reviewed by a human

Keep language simple. Do not use jargon. Do not mention SHAP or Grad-CAM by name.
End with either "This prediction appears reliable." or "Human review is recommended." """


def run_narrator(state):
    load_dotenv(override=True)

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )

    print("  [Narrator node] writing plain English explanation...")

    prediction     = state.get('prediction', 'unknown')
    confidence     = state.get('confidence', 0.0)
    shap_result    = state.get('shap_result', {})
    gradcam_result = state.get('gradcam_result', {})
    critique       = state.get('critique', '')
    contradictions = state.get('contradictions', [])

    class_desc = CLASS_DESCRIPTIONS.get(prediction, prediction)

    shap_interp = (
        shap_result.get('interpretation', 'not available')
        if shap_result.get('status') == 'success'
        else 'feature attribution analysis was not available'
    )

    gradcam_interp = (
        gradcam_result.get('interpretation', 'not available')
        if gradcam_result.get('status') == 'success'
        else 'visual attention analysis was not available'
    )

    gradcam_region = gradcam_result.get('attention_region', 'unknown region')

    human_message = f"""Prediction: {prediction} ({class_desc})
Confidence: {confidence*100:.1f}%

Feature analysis: {shap_interp}

Visual attention: model focused on {gradcam_region}. {gradcam_interp}

Critic review: {critique}

Contradictions found: {len(contradictions)} — {', '.join(contradictions) if contradictions else 'none'}

Please write the 3 sentence explanation."""

    try:
        response = llm.invoke([
            SystemMessage(content=NARRATOR_SYSTEM_PROMPT),
            HumanMessage(content=human_message)
        ])

        explanation = response.content.strip()

        confidence_note = (
            "High confidence — explainability methods agree, prediction is reliable"
            if not contradictions
            else "Low confidence — contradictions detected, human review recommended"
        )

        print(f"  [Narrator node] done. explanation length={len(explanation)} chars")
        return {
            "explanation":     explanation,
            "confidence_note": confidence_note
        }

    except Exception as e:
        print(f"  [Narrator node] LLM call failed: {e}")

        explanation = (
            f"The model predicted {prediction} ({class_desc}) "
            f"with {confidence*100:.1f}% confidence. "
            f"Visual analysis shows the model focused on the {gradcam_region}. "
            f"{'Human review is recommended.' if contradictions else 'This prediction appears reliable.'}"
        )

        return {
            "explanation":     explanation,
            "confidence_note": f"narrator fallback used: {str(e)}"
        }