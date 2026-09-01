from pathlib import Path
import io
import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="MobileNetV2 Image Classifier", version="1.0.0")
model = MobileNetV2(weights="imagenet")

app.mount("/css", StaticFiles(directory=str(BASE_DIR / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(BASE_DIR / "js")), name="js")

@app.get("/")
def home():
    return FileResponse(str(BASE_DIR / "index.html"))

@app.get("/health")
def health():
    return {"status": "healthy", "model": "MobileNetV2-ImageNet"}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Maximum image size is 10 MB.")

    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")

    img = img.resize((224, 224))
    arr = np.asarray(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    prediction = model.predict(arr, verbose=0)
    decoded = decode_predictions(prediction, top=3)[0]

    predictions = [
        {"class": label.replace("_", " "), "confidence": round(float(score) * 100, 2)}
        for _, label, score in decoded
    ]

    return {
        "success": True,
        "filename": file.filename,
        "model": "MobileNetV2-ImageNet",
        "predictions": predictions,
    }
