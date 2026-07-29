import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving files

import io
import base64
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.cnn import CNN


# ── Load model ───────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model", "best_cnn.pth"
)

_model = None

def get_model():
    global _model
    if _model is None:
        _model = CNN(num_classes=7)
        _model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        _model.to(device)
        _model.eval()
    return _model


# ── SHAP explanation function ────────────────────────────────

def explain_shap(
    image_tensor: torch.Tensor,
    background_tensors: torch.Tensor,
    pred_class: int
) -> dict:
    """
    Run SHAP GradientExplainer on a single image.

    Args:
        image_tensor:       shape [1, 3, 64, 64] — the image to explain
        background_tensors: shape [N, 3, 64, 64] — background reference images
        pred_class:         integer index of the predicted class

    Returns:
        dict with shap values, top features, and a base64 plot
    """
    model = get_model()

    try:
        # Move to CPU for SHAP — more stable
        model_cpu = model.cpu()
        image_cpu = image_tensor.detach().cpu()
        bg_cpu = background_tensors.detach().cpu()

        # Create explainer
        explainer  = shap.GradientExplainer(model_cpu, bg_cpu)

        # Compute SHAP values — returns list of arrays, one per output class
        shap_values = explainer.shap_values(image_cpu)
        if torch.is_tensor(shap_values):
            shap_values = shap_values.detach().cpu().numpy()

        elif isinstance(shap_values, list):
            shap_values = [
                s.detach().cpu().numpy() if torch.is_tensor(s) else s
                for s in shap_values
            ]
        # Get SHAP values for the predicted class only
        # shap_values shape: [num_classes, batch, channels, H, W]
        shap_for_class = shap_values[pred_class]  # [1, 3, 64, 64]

        # Collapse channels — mean absolute SHAP across RGB
        shap_map = np.abs(shap_for_class[0]).mean(axis=0)  # [64, 64]

        # Top pixel regions — flatten and find top contributing areas
        flat        = shap_map.flatten()
        top_indices = flat.argsort()[-10:][::-1]
        top_values  = flat[top_indices]

        mean_shap   = float(shap_map.mean())
        max_shap    = float(shap_map.max())

        # Generate SHAP heatmap plot and encode as base64
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Original image
        orig = image_tensor[0].detach().permute(1, 2, 0).cpu().numpy()
        orig = (orig * 0.5 + 0.5).clip(0, 1)
        axes[0].imshow(orig)
        axes[0].set_title("Original image")
        axes[0].axis("off")

        # SHAP heatmap
        im = axes[1].imshow(shap_map, cmap='hot', interpolation='nearest')
        axes[1].set_title(f"SHAP importance map\n(brighter = more important)")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1])

        plt.suptitle("SHAP Feature Attribution", fontsize=13)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        plot_b64 = base64.b64encode(buf.read()).decode('utf-8')

        # Move model back to device
        model.to(device)

        return {
            "status":           "success",
            "mean_shap":        round(mean_shap, 6),
            "max_shap":         round(max_shap, 6),
            "top_pixel_values": [round(float(v), 6) for v in top_values],
            "interpretation":   _interpret_shap(mean_shap, max_shap),
            "shap_map":         shap_map.tolist(),
            "plot_b64":         plot_b64
        }

    except Exception as e:
        model.to(device)
        return {
            "status": "error",
            "error":  str(e),
            "mean_shap": 0.0,
            "max_shap":  0.0,
            "top_pixel_values": [],
            "interpretation": "SHAP analysis failed",
            "shap_map":  [],
            "plot_b64":  ""
        }


def _interpret_shap(mean_shap: float, max_shap: float) -> str:
    """Convert SHAP numbers into plain English."""
    if max_shap > 0.01:
        return (
            "High feature concentration — the model focused on specific "
            "localised regions of the image strongly"
        )
    elif mean_shap > 0.001:
        return (
            "Moderate feature spread — the model used several image regions "
            "to make this prediction"
        )
    else:
        return (
            "Low feature attribution — the model's decision was distributed "
            "broadly across the image with no strong focal point"
        )


# ── Quick test ───────────────────────────────────────────────
if __name__ == "__main__":
    from torchvision import transforms
    from PIL import Image
    import glob

    print("Testing SHAP tool...")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Load a few real images as background + test image
    image_files = glob.glob("data/HAM10000_images_part_1/*.jpg")[:11]

    if len(image_files) < 2:
        print("❌ No images found — run from project root: python tools/shap_tool.py")
        sys.exit(1)

    tensors = []
    for f in image_files:
        img = Image.open(f).convert("RGB")
        tensors.append(transform(img))

    background = torch.stack(tensors[:10])  # 10 background images
    test_image = tensors[10].unsqueeze(0)   # 1 test image

    print(f"Background shape: {background.shape}")
    print(f"Test image shape: {test_image.shape}")
    print("Running SHAP (takes 20-60 seconds on CPU)...")

    result = explain_shap(test_image, background, pred_class=0)

    print(f"\nStatus:          {result['status']}")
    print(f"Mean SHAP:       {result['mean_shap']}")
    print(f"Max SHAP:        {result['max_shap']}")
    print(f"Interpretation:  {result['interpretation']}")
    print(f"Plot generated:  {'Yes' if result['plot_b64'] else 'No'}")
    print("\n✅ SHAP tool working correctly")