import librosa
import numpy as np
import joblib
import os
import io
import soundfile as sf
import warnings

warnings.filterwarnings("ignore")

# --- Configuration ---
MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"
SAMPLE_RATE = 22050

def extract_features_from_file(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        return _process_audio_array(y, sr)
    except Exception as e:
        print(f"⚠️ Error reading {file_path}: {e}")
        return None

def extract_features_from_bytes(audio_bytes):
    try:
        with io.BytesIO(audio_bytes) as audio_io:
            y, sr = sf.read(audio_io)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        return _process_audio_array(y, sr)
    except Exception as e:
        print(f"⚠️ Error processing bytes: {e}")
        return None

def _process_audio_array(y, sr):
    if len(y) == 0: return None
    
    # 1. MFCC (Texture) - Standard 13 is often enough, but 20 captures more detail
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    
    # 2. Delta MFCC (Rate of change) - Critical for detecting robotic artifacts
    delta_mfccs = librosa.feature.delta(mfccs)
    delta_mean = np.mean(delta_mfccs.T, axis=0)

    # 3. Chroma (Pitch)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    # 4. Spectral Contrast (Tone/Brightness)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast.T, axis=0)
    
    # 5. Zero Crossing Rate (Noisiness)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr.T, axis=0)

    # Combine all features into one vector
    return np.hstack([mfccs_mean, delta_mean, chroma_mean, contrast_mean, zcr_mean])

def load_model_and_scaler():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        return joblib.load(MODEL_FILE), joblib.load(SCALER_FILE)
    return None, None

def save_model_and_scaler(model, scaler):
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"✅ Model saved to {MODEL_FILE}")
    print(f"✅ Scaler saved to {SCALER_FILE}")
