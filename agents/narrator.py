def run_narrator(state):
    print("  [Narrator node] writing explanation...")

    prediction     = state.get('prediction', 'unknown')
    confidence     = state.get('confidence', 0.0)
    shap_result    = state.get('shap_result', {})
    gradcam_result = state.get('gradcam_result', {})
    critique       = state.get('critique', '')

    shap_interp    = shap_result.get('interpretation', 'not available')
    gradcam_region = gradcam_result.get('attention_region', 'not available')
    gradcam_interp = gradcam_result.get('interpretation', 'not available')

    # stub explanation - plain string assembly
    # on Day 12 this becomes a real LLM call
    explanation = (
        f"The model predicted '{prediction}' with {confidence*100:.1f}% confidence. "
        f"SHAP analysis shows: {shap_interp}. "
        f"Grad-CAM shows the model focused on the {gradcam_region}. "
        f"{gradcam_interp} "
        f"Critic review: {critique}."
    )

    confidence_note = (
        "High confidence - both tools agree, prediction is reliable"
        if not state.get('contradictions')
        else "Low confidence - contradictions found, human review recommended"
    )

    print(f"  [Narrator node] explanation written ({len(explanation)} chars)")
    return {
        "explanation":     explanation,
        "confidence_note": confidence_note
    }