import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from model.cnn import CNN

# ── App setup ──────────────────────────────────────────────
app = FastAPI(
    title="Neural Network Explainability Agent — Model Server",
    description="Upload a skin lesion image and get a diagnosis prediction",
    version="1.0.0"
)

# ── Constants ───────────────────────────────────────────────
CLASSES = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']

CLASS_DESCRIPTIONS = {
    'nv':    'Melanocytic nevi (moles) — benign',
    'mel':   'Melanoma — malignant, requires urgent attention',
    'bkl':   'Benign keratosis — benign',
    'bcc':   'Basal cell carcinoma — malignant',
    'akiec': 'Actinic keratoses — pre-cancerous',
    'vasc':  'Vascular lesions — benign',
    'df':    'Dermatofibroma — benign'
}

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model", "best_cnn.pth"
)

# ── Load model once at startup ──────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN(num_classes=7)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print(f"✅ Model loaded from {MODEL_PATH}")
print(f"✅ Running on {device}")

# ── Image transform ─────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


# ── Routes ──────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "Neural Network Explainability Agent — Model Server",
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "health":  "/health",
            "docs":    "/docs"
        }
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model":  "CNN — HAM10000 skin lesion classifier",
        "device": str(device),
        "classes": CLASSES
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # ── Validate file type ──────────────────────────────────
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (jpg, png, etc.)"
        )

    try:
        # ── Read and preprocess image ───────────────────────
        contents = await file.read()
        image    = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor   = transform(image).unsqueeze(0).to(device)

        # ── Run inference ───────────────────────────────────
        with torch.no_grad():
            output      = model(tensor)
            probs       = F.softmax(output, dim=1)
            confidence  = probs.max().item()
            pred_idx    = probs.argmax().item()
            pred_class  = CLASSES[pred_idx]

        # ── Build top-3 predictions ─────────────────────────
        top3_probs, top3_idx = torch.topk(probs, 3, dim=1)
        top3 = [
            {
                "class":       CLASSES[i],
                "description": CLASS_DESCRIPTIONS[CLASSES[i]],
                "confidence":  round(top3_probs[0][j].item(), 4)
            }
            for j, i in enumerate(top3_idx[0].tolist())
        ]

        return JSONResponse({
            "prediction":   pred_class,
            "description":  CLASS_DESCRIPTIONS[pred_class],
            "confidence":   round(confidence, 4),
            "confidence_pct": f"{confidence*100:.1f}%",
            "top_3":        top3,
            "needs_agent":  confidence < 0.70,
            "message": (
                "Low confidence — explainability agent will run full analysis"
                if confidence < 0.70 else
                "High confidence prediction"
            )
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))