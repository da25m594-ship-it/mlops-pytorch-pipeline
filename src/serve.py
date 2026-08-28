import io
import os

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from src.model import get_model


MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/app/checkpoints/classifier.pt",
)

NUM_CLASSES = 10

app = FastAPI(title="CIFAR-10 Model Serving API")

model = None

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    ),
])


def load_model():
    global model

    model = get_model(
        architecture="small_cnn",
        num_classes=NUM_CLASSES,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()


@app.on_event("startup")
def startup_event():
    load_model()


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    return {
        "status": "ok",
        "model": "small_cnn",
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    try:
        image_bytes = await image.read()
        pil_image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {exc}",
        )

    tensor = transform(pil_image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    predicted_class = int(
        torch.argmax(probabilities).item()
    )

    return {
        "predicted_class": predicted_class,
        "class_probabilities": probabilities.tolist(),
    }
