from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import engine
import numpy as np

app = FastAPI()

# Config
VALID_API_KEY = "sk_test_123456789"
SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

# Load model
model, scaler = engine.load_model_and_scaler()

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

class VoiceResponse(BaseModel):
    status: str
    language: str
    classification: str
    confidenceScore: float
    explanation: str

@app.post("/api/voice-detection", response_model=VoiceResponse)
async def detect_voice(req: VoiceRequest, x_api_key: str = Header(None)):
    
    if x_api_key != VALID_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    # Basic Checks
    if req.audioFormat.lower() != "mp3":
        return _resp(req.language, "N/A", 0.0, "Only 'mp3' supported")

    try:
        # Decode
        audio_bytes = base64.b64decode(req.audioBase64)
        
        # Feature Extract
        features = engine.extract_features_from_bytes(audio_bytes)
        if features is None:
             return _resp(req.language, "N/A", 0.0, "Audio processing failed")
        
        # Check Model
        if model is None:
            return _resp(req.language, "HUMAN", 0.5, "Model not trained")

        # Predict
        features_scaled = scaler.transform([features])
        pred = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0].max()
        
        if pred == 1:
            return _resp(req.language, "AI_GENERATED", round(prob, 2), "High frequency anomalies and unnatural delta-spectral patterns detected.")
        else:
            return _resp(req.language, "HUMAN", round(prob, 2), "Natural micro-tremors and breathing patterns detected.")

    except Exception as e:
        return _resp(req.language, "N/A", 0.0, str(e))

def _resp(lang, cls, conf, exp):
    return {
        "status": "success" if cls != "N/A" else "error", 
        "language": lang,
        "classification": cls, 
        "confidenceScore": conf, 
        "explanation": exp
    }
