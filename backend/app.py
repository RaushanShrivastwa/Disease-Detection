from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import io
from fastapi.middleware.cors import CORSMiddleware
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Disease information database - This should be on the backend
DISEASE_INFO = {
    "Eczema": {
        "description": "A condition where patches of skin become inflamed, itchy, red, cracked, and rough. Blisters may sometimes occur.",
        "precautions": [
            "Moisturize your skin frequently.",
            "Avoid harsh soaps and known irritants.",
            "Apply anti-itch cream to affected areas.",
            "Use a humidifier in dry environments."
        ]
    },
    "Psoriasis": {
        "description": "A skin disease that causes red, itchy scaly patches, most commonly on the knees, elbows, trunk, and scalp.",
        "precautions": [
            "Use topical treatments as prescribed.",
            "Get regular, small doses of sunlight.",
            "Manage stress and avoid skin injury.",
            "Avoid alcohol and smoking."
        ]
    },
    "Ringworm": {
        "description": "A common fungal infection that causes a circular rash shaped like a ring. It is contagious and can be spread through contact.",
        "precautions": [
            "Keep the affected area clean and dry.",
            "Use over-the-counter antifungal creams.",
            "Avoid sharing personal items like towels or clothing.",
            "Wash clothes and bedding regularly."
        ]
    },
    "Benign Mole": {
        "description": "A common, typically harmless skin growth. While this appears benign, any new or changing mole should be professionally evaluated.",
        "precautions": [
            "Monitor for changes (Asymmetry, Border, Color, Diameter, Evolving).",
            "Use sunscreen to protect your skin.",
            "Schedule regular skin checks with a dermatologist.",
            "Avoid excessive sun exposure."
        ]
    },
    "Acne Vulgaris": {
        "description": "A common skin condition that occurs when hair follicles become clogged with oil and dead skin cells, causing pimples, blackheads, or whiteheads.",
        "precautions": [
            "Keep your face clean.",
            "Use non-comedogenic makeup and skincare products.",
            "Avoid touching your face frequently.",
            "Don't squeeze or pop pimples."
        ]
    },
    "Impetigo": {
        "description": "A highly contagious bacterial skin infection that causes red sores, mainly on the face, especially around the nose and mouth.",
        "precautions": [
            "Keep the affected area clean and covered.",
            "Avoid scratching the sores.",
            "Use prescribed antibiotic ointments.",
            "Wash hands frequently to prevent spread."
        ]
    },
    # Add default for unknown predictions
    "Unknown": {
        "description": "The condition could not be confidently identified. This may be a rare condition or the image quality may be insufficient.",
        "precautions": [
            "Consult a healthcare professional for accurate diagnosis.",
            "Monitor for any changes in the condition.",
            "Avoid self-treatment without proper diagnosis."
        ]
    }
}

# Load Hugging Face model + processor
try:
    processor = AutoImageProcessor.from_pretrained("Jayanth2002/dinov2-base-finetuned-SkinDisease")
    model = AutoModelForImageClassification.from_pretrained("Jayanth2002/dinov2-base-finetuned-SkinDisease")
    logger.info("Model and processor loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise RuntimeError("Failed to load model") from e

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Preprocess
        inputs = processor(images=image, return_tensors="pt")
        
        # Run model
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            confidence = torch.softmax(logits, dim=-1)[0][predicted_class_idx].item()
        
        # Get label
        predicted_class = model.config.id2label[predicted_class_idx]
        
        # Get disease information
        disease_info = DISEASE_INFO.get(predicted_class, DISEASE_INFO["Unknown"])
        
        return {
            "prediction": predicted_class,
            "confidence": round(confidence * 100, 2),
            "description": disease_info["description"],
            "precautions": disease_info["precautions"]
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing image")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/diseases")
async def get_diseases():
    """Endpoint to get list of diseases the model can detect"""
    return list(DISEASE_INFO.keys())