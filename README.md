# 🎙️ AI-Generated Voice Detection System

A secure, high-accuracy Flask REST API that detects whether a voice sample is **AI-generated (synthetic)** or **spoken by a real human**. The system is optimized for multilingual voice analysis, supporting **Tamil, English, Hindi, Malayalam, and Telugu**.

---

## 🚀 Features

- 🤖 Detects AI-generated vs Human voice
- 🌍 Multilingual support
  - English
  - Hindi
  - Tamil
  - Malayalam
  - Telugu
- 🔐 API Key Authentication
- 🎵 Audio Feature Extraction
  - MFCC (Mel-Frequency Cepstral Coefficients)
  - Spectral Centroid
  - Zero Crossing Rate (ZCR)
- 📊 Machine Learning Based Classification
- ⚡ Fast Flask REST API
- 📦 Easy Deployment
- 🛡️ Secure API Endpoints

---

# 📂 Repository Structure

```text
AI-Generated-Voice-Detection/
│
├── engine.py              # ML engine and feature extraction
├── main.py                # Flask REST API server
├── model.pkl              # Trained ML model
├── scaler.pkl             # Feature scaler
├── requirements.txt       # Project dependencies
├── sample voice 1.mp3     # Sample audio file
└── README.md
```

---

# 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Flask | REST API Framework |
| Scikit-learn | Machine Learning |
| Librosa | Audio Feature Extraction |
| NumPy | Numerical Computing |
| Joblib | Model Serialization |

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/AI-Generated-Voice-Detection.git

cd AI-Generated-Voice-Detection
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. (Optional) Train the Model

If you have your own dataset, retrain the model using:

```bash
python engine.py --train
```

This regenerates:

- `model.pkl`
- `scaler.pkl`

---

## 4. Run the API

```bash
python main.py
```

The server will start at:

```
http://127.0.0.1:5000
```

---

# 📡 API Documentation

## Endpoint

```
POST /api/voice-detection
```

---

## Headers

| Header | Value |
|---------|-------|
| Content-Type | application/json |
| x-api-key | YOUR_SECRET_API_KEY |

---

## Request Body

```json
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "BASE64_AUDIO_STRING"
}
```

---

## Success Response

```json
{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.91,
  "explanation": "Unnatural pitch consistency and robotic speech patterns detected."
}
```

---

## Error Response

```json
{
  "status": "error",
  "message": "Invalid API key or malformed request."
}
```

---

# 🧪 Test with cURL

```bash
curl -X POST http://127.0.0.1:5000/api/voice-detection \
-H "Content-Type: application/json" \
-H "x-api-key: sk_test_123456789" \
-d '{
  "language":"English",
  "audioFormat":"mp3",
  "audioBase64":"BASE64_AUDIO_STRING"
}'
```

---

# 🎵 Supported Languages

- 🇮🇳 English
- 🇮🇳 Hindi
- 🇮🇳 Tamil
- 🇮🇳 Malayalam
- 🇮🇳 Telugu

---

# 📊 Audio Features Used

The model extracts multiple acoustic features for accurate classification:

- Mel-Frequency Cepstral Coefficients (MFCC)
- Spectral Centroid
- Zero Crossing Rate (ZCR)
- Audio Energy
- Spectral Characteristics

These features help distinguish natural human speech from AI-generated voices.

---

# 🔒 Security

- API Key Authentication
- Structured JSON Responses
- Input Validation
- Base64 Audio Processing
- Secure Request Handling

---

# 📦 Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

Example libraries include:

- Flask
- NumPy
- Librosa
- Scikit-learn
- Joblib

---

# 📁 Sample Audio

The repository includes:

```
sample voice 1.mp3
```

Use it for testing or verifying your deployment.

---

# 🚀 Future Improvements

- Deep Learning Models (CNN/LSTM)
- Transformer-based Audio Classification
- WAV & FLAC Support
- Batch Audio Processing
- Docker Deployment
- Swagger/OpenAPI Documentation
- Real-time Streaming Detection
- Web Dashboard

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Add new feature"
```

4. Push to GitHub

```bash
git push origin feature-name
```

5. Open a Pull Request

---

# 👨‍💻 Author

**Mratunjay Pandey**

- 🎓 B.Tech Computer Science Student
- 💻 AI & Machine Learning Enthusiast
- 🌐 Passionate about Audio Intelligence & Cybersecurity

---

# 📄 License

This project is licensed under the **MIT License**.

---

<div align="center">

### ⭐ If you found this project useful, don't forget to Star the repository!

**Made with ❤️ by Mratunjay Pandey**

</div>
