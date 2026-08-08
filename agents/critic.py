import os
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq



CRITIC_SYSTEM_PROMPT = """You are an AI model auditor reviewing explainability results for a skin lesion classifier.

You will receive:
- The model's prediction and confidence score
- SHAP analysis results (which pixels contributed most to the prediction)
- Grad-CAM results (which region of the image the model focused on)

Your job is to find agreements and contradictions between the two methods.

You must respond ONLY with a valid JSON object in this exact format, no other text:
{
  "agreement": "one sentence describing what both methods agree on",
  "contradictions": ["contradiction 1 if any", "contradiction 2 if any"],
  "confidence_assessment": "one sentence assessing if the model confidence matches the evidence",
  "next_focus": "one sentence on what to investigate if running again"
}

If there are no contradictions, return an empty list for contradictions.
Be specific and technical but keep each field to one sentence.
Important: if the model confidence is above 70%, SHAP analysis may not have been run by design — this is expected behaviour and should NOT be listed as a contradiction.
 Only flag SHAP absence as a contradiction if confidence is below 70%."""


def run_critic(state):
    load_dotenv(override=True)
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    print("  [Critic node] reviewing SHAP and Grad-CAM with LLM...")

    shap_result    = state.get('shap_result', {})
    gradcam_result = state.get('gradcam_result', {})
    prediction     = state.get('prediction', 'unknown')
    confidence     = state.get('confidence', 0.0)
    loop_count     = state.get('loop_count', 0)

    shap_ok    = shap_result.get('status') == 'success'
    gradcam_ok = gradcam_result.get('status') == 'success'

    if not shap_ok and not gradcam_ok:
        print("  [Critic node] both tools failed, skipping LLM")
        return {
            "contradictions": ["both SHAP and Grad-CAM failed to run"],
            "critique": "unable to audit — both explainability tools returned errors",
            "loop_count": loop_count + 1
        }

    shap_summary = (
        f"status={shap_result.get('status')}, "
        f"mean_shap={shap_result.get('mean_shap', 0):.6f}, "
        f"max_shap={shap_result.get('max_shap', 0):.6f}, "
        f"interpretation={shap_result.get('interpretation', 'not available')}"
        if shap_ok else "SHAP did not run successfully"
    )

    gradcam_summary = (
        f"status={gradcam_result.get('status')}, "
        f"attention_region={gradcam_result.get('attention_region', 'unknown')}, "
        f"max_attention={gradcam_result.get('max_attention', 0):.4f}, "
        f"interpretation={gradcam_result.get('interpretation', 'not available')}"
        if gradcam_ok else "Grad-CAM did not run successfully"
    )

    human_message = f"""Model prediction: {prediction}
Confidence: {confidence*100:.1f}%

SHAP results: {shap_summary}

Grad-CAM results: {gradcam_summary}

Please audit these results and return your JSON assessment."""

    try:
        response = llm.invoke([
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=human_message)
        ])

        raw = response.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)

        contradictions      = parsed.get("contradictions", [])
        agreement           = parsed.get("agreement", "")
        conf_assessment     = parsed.get("confidence_assessment", "")
        next_focus          = parsed.get("next_focus", "")

        critique = (
            f"Agreement: {agreement} "
            f"Confidence assessment: {conf_assessment} "
            f"Next focus: {next_focus}"
        )

        print(f"  [Critic node] done. contradictions={len(contradictions)}, loop={loop_count}")
        for c in contradictions:
            print(f"    - {c}")

        return {
            "contradictions": contradictions,
            "critique":       critique,
            "loop_count":     loop_count + 1
        }

    except Exception as e:
        print(f"  [Critic node] LLM call failed: {e}")
        return {
            "contradictions": [],
            "critique": f"Critic LLM call failed: {str(e)}",
            "loop_count": loop_count + 1
        }


def route_from_critic(state):
    contradictions = state.get('contradictions', [])
    loop_count     = state.get('loop_count', 0)

    if contradictions and loop_count < 2:
        print("  [Critic router] contradiction found, looping back to planner")
        return "planner"
    else:
        print("  [Critic router] going to narrator")
        return "narrator"