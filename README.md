\# AI-Generated Voice Detection API



This project detects whether a given voice sample is \*\*AI-generated\*\* or \*\*Human\*\* using audio signal analysis.



\## 🚀 Features

\- Supports 5 languages:

&nbsp; - Tamil

&nbsp; - English

&nbsp; - Hindi

&nbsp; - Malayalam

&nbsp; - Telugu

\- Accepts \*\*Base64 encoded MP3 audio\*\*

\- Returns classification with confidence score

\- API Key protected

\- FastAPI + ML-based backend



\## 🧠 Approach

The system extracts audio features such as:

\- MFCC (Mel-Frequency Cepstral Coefficients)

\- Spectral Centroid

\- Zero Crossing Rate

\- RMS Energy



These features are passed to a trained ML model to classify the voice as:

\- `AI\_GENERATED`

\- `HUMAN`



\## 🔐 Authentication

All requests must include:



x-api-key: my\_secret\_key





\## 📡 API Endpoint



\### POST `/api/voice-detection`



\#### Request Body

```json

{

&nbsp; "language": "Tamil",

&nbsp; "audioFormat": "mp3",

&nbsp; "audioBase64": "<BASE64\_AUDIO>"

}

Response

{

&nbsp; "status": "success",

&nbsp; "language": "Tamil",

&nbsp; "classification": "HUMAN",

&nbsp; "confidenceScore": 0.65,

&nbsp; "explanation": "Classification based on spectral and temporal audio features"

}

