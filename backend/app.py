# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import io
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn
from typing import List

# --- logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("skin-disease-api")

# --- app setup ---
app = FastAPI(title="Skin Disease Detection API")

# For development/testing we allow all origins. For production, restrict this to your frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # change to ["https://your-vercel-app.vercel.app"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- simple disease info DB ---
DISEASE_INFO = {
    "Eczema": {
        "description": "Patches of skin become inflamed, itchy, red, cracked, and rough.",
        "precautions": [
            "Moisturize your skin frequently.",
            "Avoid harsh soaps and known irritants.",
            "Apply anti-itch cream to affected areas.",
            "Use a humidifier in dry environments."
        ],
    },
    "Psoriasis": {
        "description": "Red, itchy scaly patches on skin (knees, elbows, scalp, trunk).",
        "precautions": [
            "Use topical treatments as prescribed.",
            "Get regular, small doses of sunlight.",
            "Manage stress and avoid skin injury.",
            "Avoid alcohol and smoking."
        ],
    },
    "Ringworm": {
        "description": "A common fungal infection that causes a circular rash shaped like a ring.",
        "precautions": [
            "Keep the affected area clean and dry.",
            "Use antifungal creams as recommended.",
            "Avoid sharing personal items.",
            "Wash clothes and bedding regularly."
        ],
    },
    "Benign Mole": {
        "description": "A typically harmless skin growth; evaluate any new or changing mole professionally.",
        "precautions": [
            "Monitor for changes (Asymmetry, Border, Color, Diameter, Evolving).",
            "Use sunscreen to protect your skin.",
            "Schedule regular skin checks with a dermatologist.",
            "Avoid excessive sun exposure."
        ],
    },
    "Acne Vulgaris": {
        "description": "Occurs when hair follicles become clogged with oil and dead skin cells, causing pimples.",
        "precautions": [
            "Keep your face clean.",
            "Use non-comedogenic products.",
            "Avoid touching your face frequently.",
            "Don't squeeze or pop pimples."
        ],
    },
    "Impetigo": {
        "description": "A highly contagious bacterial skin infection causing red sores (often around nose & mouth).",
        "precautions": [
            "Keep the affected area clean and covered.",
            "Avoid scratching the sores.",
            "Use prescribed antibiotic ointments.",
            "Wash hands frequently to prevent spread."
        ],
    },
    # Fallback
    "Unknown": {
        "description": "The condition could not be confidently identified. Consult a professional.",
        "precautions": [
            "Consult a healthcare professional for an accurate diagnosis.",
            "Monitor for any changes.",
            "Avoid self-treatment without proper diagnosis."
        ],
    },
}

# --- load model & processor (Hugging Face) ---
try:
    logger.info("Loading processor and model from Hugging Face...")
    processor = AutoImageProcessor.from_pretrained("Jayanth2002/dinov2-base-finetuned-SkinDisease")
    model = AutoModelForImageClassification.from_pretrained("Jayanth2002/dinov2-base-finetuned-SkinDisease")
    model.eval()
    logger.info("Model and processor loaded successfully.")
except Exception as e:
    logger.exception("Failed to load model/processor.")
    raise RuntimeError("Failed to load model/processor") from e


# --- POST /predict (actual prediction endpoint) ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image file (multipart/form-data, field name 'file') and returns:
    { prediction, confidence, description, precautions }
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (Content-Type starts with image/)")

    try:
        # Read image bytes
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess
        inputs = processor(images=image, return_tensors="pt")
        # Move inputs to same device as model if needed (CPU by default)
        # If using GPU, you can uncomment the following:
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model.to(device)
        # inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # shape: (1, num_labels)
            predicted_idx = int(logits.argmax(-1).item())
            scores = torch.softmax(logits, dim=-1)[0]
            confidence = float(scores[predicted_idx].item())

        predicted_label = model.config.id2label.get(predicted_idx, str(predicted_idx))
        disease_info = DISEASE_INFO.get(predicted_label, DISEASE_INFO["Unknown"])

        return {
            "prediction": predicted_label,
            "confidence": round(confidence * 100, 2),  # percentage with 2 decimals
            "description": disease_info["description"],
            "precautions": disease_info["precautions"],
        }

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Error processing image")


# --- GET /predict (informational / quick browser check) ---
@app.get("/predict")
async def predict_info():
    return {"message": "Send a POST request with multipart/form-data (file field name 'file') to /predict to get inference."}


# --- health & metadata endpoints ---
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/diseases")
async def get_diseases():
    """Return list of disease labels available in the small backend DB"""
    return list(DISEASE_INFO.keys())


# --- optional root ---
@app.get("/")
async def root():
    return {"message": "Skin Disease Detection API. Use /docs for Swagger UI."}


# --- run server (optional) ---
if __name__ == "__main__":
    # Use this to run locally: `python main.py`
    uvicorn.run(app, host="0.0.0.0", port=8000)
