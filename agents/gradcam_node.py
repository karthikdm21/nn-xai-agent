import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.gradcam_tool import explain_gradcam


def run_gradcam(state):
    print("  [GradCAM node] running Grad-CAM analysis...")

    image_tensor = state.get('image_tensor')
    pred_class   = state.get('pred_class_idx', 0)

    if image_tensor is None:
        return {
            "gradcam_result": {
                "status": "error",
                "error": "no image tensor in state",
                "attention_region": "unknown",
                "interpretation": "Grad-CAM skipped - no image",
                "plot_b64": ""
            }
        }

    result = explain_gradcam(image_tensor, pred_class)

    print(f"  [GradCAM node] done. status={result['status']}, region={result.get('attention_region', 'unknown')}")
    return {"gradcam_result": result}