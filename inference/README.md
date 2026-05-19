# Inference API

FastAPI-based dual-model inference server, deployed on [Hugging Face Spaces](https://huggingface.co/spaces/theashish03/medical-assistant-api).

## Live API

**Base URL:** `https://theashish03-medical-assistant-api.hf.space`

### Health Check

```bash
curl https://theashish03-medical-assistant-api.hf.space/
```

### Predict Disease

```bash
curl -X POST https://theashish03-medical-assistant-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"symptoms": "headache, fever, nausea, vomiting"}'
```

**Response:**
```json
{
  "predicted_disease": "Dengue",
  "confidence": 95.23,
  "winner": "SYMPTOM MODEL",
  "symptoms": ["headache", "fever", "nausea", "vomiting"],
  "symptom_model": { "disease": "Dengue", "confidence": 95.23, "matched": ["..."], "unmatched": [] },
  "nlp_model": { "disease": "Dengue", "confidence": 82.14 }
}
```

## How It Works

1. User sends symptoms as comma-separated text
2. **Symptom Model**: Maps known symptoms to binary feature vector → DenseNet+BiLSTM prediction
3. **NLP Model**: Tokenizes raw text → CNN+BiLSTM prediction
4. **Winner-takes-all**: Returns the model with higher confidence

## Trained Model Weights

Download from Hugging Face:

🧠 **[Browse Models →](https://huggingface.co/spaces/theashish03/medical-assistant-api/tree/main/models)**

📦 **[Full HF Space →](https://huggingface.co/spaces/theashish03/medical-assistant-api/tree/main)**

## Run Locally

```bash
pip install -r requirements.txt

# Download models from HF Space and place in a models/ directory
# See the HF Space for the exact model structure

uvicorn app:app --host 0.0.0.0 --port 7860
```

## Deploy to Hugging Face Spaces

The `Dockerfile` is configured for HF Spaces Docker deployment. Push to a new Space with Docker SDK selected.
