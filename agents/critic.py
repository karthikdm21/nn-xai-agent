def run_critic(state):
    print("  [Critic node] reviewing SHAP and Grad-CAM results...")

    shap_result    = state.get('shap_result', {})
    gradcam_result = state.get('gradcam_result', {})
    loop_count     = state.get('loop_count', 0)

    shap_ok    = shap_result.get('status') == 'success'
    gradcam_ok = gradcam_result.get('status') == 'success'

    contradictions = []

    # check 1: did both tools succeed
    if not shap_ok:
        contradictions.append("SHAP analysis failed or returned error")
    if not gradcam_ok:
        contradictions.append("Grad-CAM analysis failed or returned error")

    # check 2: basic sanity - if SHAP mean is near zero but model is confident
    # this is a stub check, the real LLM check comes on Day 11
    confidence = state.get('confidence', 1.0)
    mean_shap  = shap_result.get('mean_shap', 0.0)

    if confidence > 0.90 and mean_shap < 0.0001 and shap_ok:
        contradictions.append(
            "Model is very confident but SHAP shows near-zero feature attribution - suspicious"
        )

    has_contradiction = len(contradictions) > 0 and loop_count < 2

    print(f"  [Critic node] contradictions={len(contradictions)}, loop_count={loop_count}")
    for c in contradictions:
        print(f"    - {c}")

    return {
        "contradictions": contradictions,
        "critique": (
            "Issues found: " + "; ".join(contradictions)
            if contradictions
            else "SHAP and Grad-CAM results look consistent"
        ),
        "loop_count": loop_count + 1
    }


def route_from_critic(state):
    contradictions = state.get('contradictions', [])
    loop_count     = state.get('loop_count', 0)

    if contradictions and loop_count < 2:
        print("  [Critic router] contradiction found, looping back to planner")
        return "planner"
    else:
        print("  [Critic router] no contradiction or max loops, going to narrator")
        return "narrator"