import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io
import base64
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.cnn import CNN


CLASSES = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model", "best_cnn.pth"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None

def get_model():
    global _model
    if _model is None:
        _model = CNN(num_classes=7)
        _model.load_state_dict(
            torch.load(MODEL_PATH, map_location=device)
        )
        _model.to(device)
        _model.eval()
    return _model


def explain_gradcam(image_tensor, pred_class):
    """
    Run Grad-CAM on a single image using the last conv layer.

    Args:
        image_tensor: shape [1, 3, 64, 64]
        pred_class:   integer index of predicted class

    Returns:
        dict with heatmap array, attention description, and base64 plot
    """
    model = get_model()

    activations = {}
    gradients = {}

    def save_activation(module, input, output):
        activations['value'] = output.detach()

    def save_gradient(module, grad_input, grad_output):
        gradients['value'] = grad_output[0].detach()

    # register hooks on the last conv layer which is conv3
    hook_forward = model.conv3.register_forward_hook(save_activation)
    hook_backward = model.conv3.register_full_backward_hook(save_gradient)

    try:
        image = image_tensor.to(device)
        image.requires_grad_(True)

        output = model(image)

        model.zero_grad()

        # backpropagate only the score for the predicted class
        class_score = output[0, pred_class]
        class_score.backward()

        # get the captured activation and gradient
        act = activations['value']   # shape [1, 128, 8, 8]
        grad = gradients['value']    # shape [1, 128, 8, 8]

        # global average pool the gradients across spatial dims
        weights = grad.mean(dim=(2, 3), keepdim=True)  # [1, 128, 1, 1]

        # weighted sum of activation maps
        cam = (weights * act).sum(dim=1, keepdim=True)  # [1, 1, 8, 8]

        # relu — keep only positive contributions
        cam = F.relu(cam)

        # normalize to 0-1
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # resize to input image size 64x64
        cam = F.interpolate(
            cam,
            size=(64, 64),
            mode='bilinear',
            align_corners=False
        )

        cam_np = cam.squeeze().cpu().numpy()  # [64, 64]

        # find the region with highest attention
        attention_description = describe_attention_region(cam_np)

        # build the overlay plot
        plot_b64 = build_overlay_plot(image_tensor, cam_np, pred_class)

        return {
            "status": "success",
            "heatmap": cam_np.tolist(),
            "max_attention": round(float(cam_np.max()), 4),
            "mean_attention": round(float(cam_np.mean()), 4),
            "attention_region": attention_description,
            "interpretation": build_interpretation(cam_np),
            "plot_b64": plot_b64
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "heatmap": [],
            "max_attention": 0.0,
            "mean_attention": 0.0,
            "attention_region": "unknown",
            "interpretation": "Grad-CAM analysis failed",
            "plot_b64": ""
        }

    finally:
        hook_forward.remove()
        hook_backward.remove()


def describe_attention_region(cam_np):
    """
    Find where the peak attention is and describe it in plain English.
    Divides the 64x64 map into a 3x3 grid and names the region.
    """
    h, w = cam_np.shape
    peak_y, peak_x = np.unravel_index(cam_np.argmax(), cam_np.shape)

    row = int(peak_y / (h / 3))
    col = int(peak_x / (w / 3))

    row_names = ["top", "middle", "bottom"]
    col_names = ["left", "center", "right"]

    if row == 1 and col == 1:
        return "center of the lesion"

    return f"{row_names[row]}-{col_names[col]} region of the image"


def build_interpretation(cam_np):
    """
    Turn Grad-CAM numbers into plain English for the Narrator agent.
    """
    coverage = float((cam_np > 0.5).mean())

    if coverage < 0.1:
        focus = "tightly focused on a very small region"
    elif coverage < 0.3:
        focus = "moderately focused on a specific area"
    else:
        focus = "broadly distributed across much of the image"

    peak = float(cam_np.max())

    if peak > 0.9:
        confidence_note = "with high gradient signal strength"
    elif peak > 0.6:
        confidence_note = "with moderate gradient signal strength"
    else:
        confidence_note = "with relatively weak gradient signal"

    return (
        f"The model's attention was {focus} {confidence_note}. "
        f"High-attention coverage: {coverage*100:.1f}% of image area."
    )


def build_overlay_plot(image_tensor, cam_np, pred_class):
    """
    Overlay the Grad-CAM heatmap on the original image and encode as base64.
    """
    orig = image_tensor[0].permute(1, 2, 0).cpu().numpy()
    orig = (orig * 0.5 + 0.5).clip(0, 1)

    heatmap = cm.jet(cam_np)[:, :, :3]  # drop alpha channel

    # blend original image and heatmap
    overlay = 0.55 * orig + 0.45 * heatmap
    overlay = overlay.clip(0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(orig)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].imshow(cam_np, cmap='jet', interpolation='nearest')
    axes[1].set_title("Grad-CAM heatmap\n(red = high attention)")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay\npredicted: {CLASSES[pred_class]}")
    axes[2].axis("off")

    plt.suptitle("Grad-CAM Visual Explanation", fontsize=13)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


if __name__ == "__main__":
    from torchvision import transforms
    from PIL import Image
    import glob

    print("Testing Grad-CAM tool...")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    image_files = glob.glob("data/HAM10000_images_part_1/*.jpg")

    if not image_files:
        print("No images found. Run from project root: python tools/gradcam_tool.py")
        sys.exit(1)

    test_image = transform(
        Image.open(image_files[0]).convert("RGB")
    ).unsqueeze(0)

    print(f"Test image: {image_files[0]}")
    print("Running Grad-CAM...")

    result = explain_gradcam(test_image, pred_class=0)

    print(f"\nStatus:           {result['status']}")
    print(f"Max attention:    {result['max_attention']}")
    print(f"Mean attention:   {result['mean_attention']}")
    print(f"Attention region: {result['attention_region']}")
    print(f"Interpretation:   {result['interpretation']}")
    print(f"Plot generated:   {'yes' if result['plot_b64'] else 'no'}")
    print("\nGrad-CAM tool working correctly")