def run_planner(state):
    confidence = state.get('confidence', 0.0)
    loop_count = state.get('loop_count', 0)

    print(f"  [Planner] confidence={confidence:.2f}, loop={loop_count}")

    if confidence >= 0.70:
        next_action = "gradcam_only"
    else:
        next_action = "shap_and_gradcam"

    return {"next_action": next_action}


def route_from_planner(state):
    action = state.get('next_action', 'shap_and_gradcam')

    if action == "gradcam_only":
        print("  [Planner router] high confidence, going gradcam only")
        return "gradcam"
    else:
        print("  [Planner router] low confidence, going shap first")
        return "shap"