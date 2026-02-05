from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from joblib import load
import base64
import numpy as np
import librosa
import io

app = FastAPI(
    title="AI-Generated Voice Detection API",
    version="1.0"
)

# ---------------- CONFIG ----------------
API_KEY = "my_secret_key"
SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
CONFIDENCE_THRESHOLD = 0.6

# Load trained ML model
model = load("model/voice_model.pkl")

# ---------------- REQUEST MODEL ----------------
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

# ---------------- API KEY VALIDATION ----------------
def validate_api_key(x_api_key: str):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# ---------------- MAIN ENDPOINT ----------------
@app.post("/api/voice-detection")
async def detect_voice(
    request: VoiceRequest,
    x_api_key: str = Header(...)
):
    # Validate API key
    validate_api_key(x_api_key)

    # Validate language
    if request.language not in SUPPORTED_LANGUAGES:
        return {
            "status": "error",
            "message": "Unsupported language"
        }

    # Validate audio format
    if request.audioFormat.lower() != "mp3":
        return {
            "status": "error",
            "message": "Only MP3 audio format is supported"
        }

    # Decode Base64 audio
    try:
        audio_bytes = base64.b64decode(request.audioBase64)
        audio_stream = io.BytesIO(audio_bytes)
        audio, sr = librosa.load(audio_stream, sr=None)
    except Exception:
        return {
            "status": "error",
            "message": "Invalid Base64 audio input"
        }

    # ---------------- FEATURE EXTRACTION ----------------
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(audio)
    rms = librosa.feature.rms(y=audio)

    # Feature means (CRITICAL FIX)
    mfcc_mean = float(mfccs.mean())
    spectral_centroid_mean = float(spectral_centroid.mean())
    zero_crossing_rate = float(zcr.mean())
    rms_energy = float(rms.mean())

    # Model input
    features_array = np.array([[
        mfcc_mean,
        spectral_centroid_mean,
        zero_crossing_rate,
        rms_energy
    ]])

    # ---------------- PREDICTION ----------------
    prediction = model.predict(features_array)[0]
    confidence = float(model.predict_proba(features_array).max())

    if confidence < CONFIDENCE_THRESHOLD:
        classification = "UNCERTAIN"
        explanation = "Low confidence prediction"
    else:
        classification = "HUMAN" if prediction == 1 else "AI_GENERATED"
        explanation = "Classification based on spectral and temporal audio features"

    duration = len(audio) / sr

    # ---------------- RESPONSE ----------------
    return {
        "status": "success",
        "language": request.language,
        "classification": classification,
        "confidenceScore": round(confidence, 2),
        "explanation": explanation
    }
