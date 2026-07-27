import torch
from torchvision import transforms
from PIL import Image
import glob
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.shap_tool import explain_shap

# load 10 background images once when this module is first imported
# this saves reloading them on every agent call

_background = None

def get_background():
    global _background
    if _background is not None:
        return _background

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    part1 = glob.glob(os.path.join(base, "data/HAM10000_images_part_1/*.jpg"))
    part2 = glob.glob(os.path.join(base, "data/HAM10000_images_part_2/*.jpg"))
    all_files = (part1 + part2)[:10]

    tensors = []
    for f in all_files:
        img = Image.open(f).convert("RGB")
        tensors.append(transform(img))

    _background = torch.stack(tensors)
    print(f"  [SHAP node] background loaded: {_background.shape}")
    return _background


def run_shap(state):
    print("  [SHAP node] running SHAP analysis...")

    image_tensor = state.get('image_tensor')
    pred_class   = state.get('pred_class_idx', 0)

    if image_tensor is None:
        return {
            "shap_result": {
                "status": "error",
                "error": "no image tensor in state",
                "mean_shap": 0.0,
                "max_shap": 0.0,
                "interpretation": "SHAP skipped - no image",
                "plot_b64": ""
            }
        }

    background = get_background()
    result = explain_shap(image_tensor, background, pred_class)

    print(f"  [SHAP node] done. status={result['status']}, mean={result.get('mean_shap', 0):.6f}")
    return {"shap_result": result}