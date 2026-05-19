<div align="center">

# A Multimodal Deep Learning Framework for Symptom-Based Disease Prediction and Clinical Decision Support

[![Published in NCAA](https://img.shields.io/badge/Published_in-Neural_Computing_and_Applications-blue?style=for-the-badge)](https://doi.org/PLACEHOLDER)
[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-ayuseva.vercel.app-brightgreen?style=for-the-badge)](https://ayuseva.vercel.app)
[![API](https://img.shields.io/badge/🤗_API-Hugging_Face_Spaces-yellow?style=for-the-badge)](https://huggingface.co/spaces/theashish03/medical-assistant-api)
[![Models](https://img.shields.io/badge/🧠_Models-Hugging_Face-orange?style=for-the-badge)](https://huggingface.co/spaces/theashish03/medical-assistant-api/tree/main/models)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

**Official implementation of the paper accepted at *Neural Computing and Applications* (Springer).**

A dual-model medical diagnosis system combining structured symptom analysis with natural language understanding, featuring calibrated predictions, cost-sensitive evaluation, and a full-stack deployment pipeline.

[Live Demo](https://ayuseva.vercel.app) · [API Docs](https://huggingface.co/spaces/theashish03/medical-assistant-api) · [Trained Models](https://huggingface.co/spaces/theashish03/medical-assistant-api/tree/main/models) · [Paper](#citation)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Results](#key-results)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Trained Models](#trained-models)
- [Live Applications](#live-applications)
- [Datasets](#datasets)
- [Citation](#citation)
- [License](#license)

---

## Overview

This repository contains the official code for **"A Multimodal Deep Learning Framework for Symptom-Based Disease Prediction and Clinical Decision Support"**, accepted at *Neural Computing and Applications* (Springer).

We propose a dual-model inference system (**AyuSeva**) that uses two complementary deep learning architectures:

1. **Symptom Model** — A DenseNet + Bidirectional LSTM + Attention architecture for structured symptom-based diagnosis across 41 disease classes
2. **NLP Model** — A CNN + Bidirectional LSTM + Attention architecture for free-text symptom description diagnosis across 55 disease classes

The system runs both models simultaneously and selects the prediction with higher confidence (dual-model winner-takes-all inference). It includes calibrated probability estimates, cost-sensitive evaluation for acute conditions, and a complete deployment pipeline.

### Key Contributions

- 🔬 **Dual-Model Inference** — Structured symptoms + free-text NLP, best-of-both approach
- 📊 **5-Fold Stratified Cross-Validation** — Rigorous evaluation with per-fold metrics
- ⚖️ **Calibrated Predictions** — Temperature scaling + reliability diagrams + ECE metrics
- 🏥 **Cost-Sensitive Evaluation** — Higher penalty for misclassifying acute/critical conditions
- 🔍 **Confidence Thresholding** — Configurable rejection for low-confidence predictions
- 🧪 **Comprehensive Benchmarks** — Compared against GRU, TCN, MLP, Random Forest, XGBoost, TextCNN
- 🚀 **Full-Stack Deployment** — FastAPI backend on HF Spaces + Next.js frontend on Vercel

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INPUT                              │
│           "I have headache, fever, and nausea"              │
└──────────────────────┬──────────────────────────────────────┘
                       │
           ┌───────────┴───────────┐
           ▼                       ▼
┌─────────────────────┐  ┌─────────────────────┐
│   SYMPTOM MODEL     │  │     NLP MODEL        │
│ ─────────────────── │  │ ─────────────────── │
│ DenseNet Blocks (3) │  │ Embedding (256d)     │
│ Conv1D + BN + Drop  │  │ Conv1D (256,128)     │
│ BiLSTM (200 units)  │  │ MaxPool + BN + Drop  │
│ Self-Attention (2x) │  │ BiLSTM (128 units)   │
│ Dense (512→256→128) │  │ Self-Attention       │
│ Softmax (41 classes)│  │ Dense (256→128)      │
│                     │  │ Softmax (55 classes) │
└────────┬────────────┘  └────────┬────────────┘
         │                        │
         └──────────┬─────────────┘
                    ▼
         ┌──────────────────┐
         │ WINNER-TAKES-ALL │
         │ Higher Confidence│
         │ → Final Diagnosis│
         └────────┬─────────┘
                  ▼
         ┌──────────────────┐
         │  Gemini API      │
         │  → Explanation   │
         │  → Precautions   │
         │  → Recommendations│
         └──────────────────┘
```

---

## Repository Structure

```
multimodal-disease-prediction/
├── README.md                      ← You are here
├── LICENSE                        ← MIT License
├── CITATION.cff                   ← Citation for the paper
│
├── training/                      ← ML Training Pipeline
│   ├── 01_symptom_model.py        ← Symptom model (DenseNet+BiLSTM+Attn)
│   ├── 02_symptom_ordering.py     ← Clinical vs shuffled ordering analysis
│   ├── 03_symptom_benchmarks.py   ← Baselines (GRU, TCN, MLP, RF)
│   ├── 04_nlp_model.py           ← NLP model (CNN+BiLSTM+Attn)
│   ├── 05_nlp_benchmarks.py      ← Baselines (CNN, XGBoost, GRU+Attn)
│   ├── 06_generative_eval.py     ← Clinical evaluation framework
│   └── requirements.txt
│
├── data/                          ← Datasets
│   ├── symbipredict_2022.csv      ← Structured symptoms (4961 samples, 41 classes)
│   └── bert_train.csv             ← Free-text NLP data (6833 samples, 55 classes)
│
└── inference/                     ← FastAPI Inference API (source code)
    ├── app.py                     ← Dual-model inference server
    ├── Dockerfile                 ← HF Spaces deployment
    └── requirements.txt
```

---

## Quick Start

### 1. Clone & Reproduce Training

```bash
git clone https://github.com/nameisashish/multimodal-disease-prediction.git
cd multimodal-disease-prediction
```

Upload the training scripts to Google Colab with a T4 GPU:

| Script | Runtime | What It Does |
|--------|---------|--------------|
| `01_symptom_model.py` | ~3-4 hrs | Train symptom model, 5-fold CV, calibration |
| `02_symptom_ordering.py` | ~2-3 hrs | Clinical vs shuffled ordering analysis |
| `03_symptom_benchmarks.py` | ~2-3 hrs | GRU, TCN, MLP, RF baselines |
| `04_nlp_model.py` | ~3.5 hrs | Train NLP model, 5-fold CV, calibration |
| `05_nlp_benchmarks.py` | ~3 hrs | CNN, XGBoost, GRU+Attn baselines |
| `06_generative_eval.py` | ~5 min | Clinical evaluation framework (CPU) |

### 2. Test the Live API

```bash
curl -X POST https://theashish03-medical-assistant-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"symptoms": "headache, fever, nausea, vomiting"}'
```

---

## Trained Models

The trained model weights are hosted on Hugging Face Spaces:

🧠 **[Browse Models on Hugging Face →](https://huggingface.co/spaces/theashish03/medical-assistant-api/tree/main/models)**

| Model | File | Description |
|-------|------|-------------|
| Symptom Model | `final_symptom_model.keras` | DenseNet+BiLSTM+Attention (41 classes) |
| Symptom Scaler | `symptom_scaler.pkl` | StandardScaler for symptom features |
| Symptom Encoder | `symptom_label_encoder.pkl` | Label encoder (41 disease classes) |
| Temperature | `symptom_temperature.pkl` | Calibration temperature parameter |
| NLP Model | `best_nlp_model.keras` | CNN+BiLSTM+Attention (55 classes) |
| NLP Tokenizer | `nlp_tokenizer.pkl` | Keras tokenizer (15,000 vocab) |
| NLP Encoder | `nlp_label_encoder.pkl` | Label encoder (55 disease classes) |

**Full HF Space:** [huggingface.co/spaces/theashish03/medical-assistant-api](https://huggingface.co/spaces/theashish03/medical-assistant-api/tree/main)

---

## Live Applications

### 🌐 AyuSeva — Web Application

**Live Demo:** [ayuseva.vercel.app](https://ayuseva.vercel.app)

Interactive chatbot interface with symptom input (structured + free-text), real-time disease prediction, and AI-generated explanations via Gemini API.

**Source Code & Details:** [github.com/nameisashish/ayuseva-frontend](https://github.com/nameisashish/ayuseva-frontend)

---

### 🟣 Health & Fitness Planner

AI-powered health and fitness planner built with Streamlit.

**Source Code & Details:** [github.com/nameisashish/ayuseva-planner](https://github.com/nameisashish/ayuseva-planner)

---

### 🤗 Inference API

FastAPI dual-model inference server deployed on Hugging Face Spaces.

**Live API:** [huggingface.co/spaces/theashish03/medical-assistant-api](https://huggingface.co/spaces/theashish03/medical-assistant-api)

**Endpoints:**
- `GET /` — Health check
- `POST /predict` — Disease prediction (accepts `{"symptoms": "..."}`)

---

## Datasets

| Dataset | Samples | Features | Classes | Description |
|---------|---------|----------|---------|-------------|
| `symbipredict_2022.csv` | 4,961 | 132 binary symptoms | 41 | Pre-balanced structured symptom data (121/class) |
| `bert_train.csv` | 6,833 | Free-text descriptions | 55 | Natural language symptom descriptions |

> **Note:** `symbipredict_2022.csv` contains trailing spaces in `"Diabetes "` and `"Hypertension "` — this is intentional and handled in the training code.

---

## Citation

If you use this code or models in your research, please cite our paper:

```bibtex
@article{kishore2026multimodal,
  title     = {A Multimodal Deep Learning Framework for Symptom-Based Disease Prediction and Clinical Decision Support},
  author    = {Kishore, Ashish},
  journal   = {Neural Computing and Applications},
  publisher = {Springer},
  year      = {2026},
  url       = {https://github.com/nameisashish/multimodal-disease-prediction},
  license   = {MIT}
}
```

> 📄 **Paper DOI:** *Will be updated after publication.*

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Related Repositories

| Repository | Description |
|------------|-------------|
| [ayuseva-frontend](https://github.com/nameisashish/ayuseva-frontend) | Next.js web interface (deployed at [ayuseva.vercel.app](https://ayuseva.vercel.app)) |
| [ayuseva-planner](https://github.com/nameisashish/ayuseva-planner) | Streamlit health & fitness planner |
| [medical-assistant-api](https://huggingface.co/spaces/theashish03/medical-assistant-api) | HF Spaces inference API + trained models |

---

## Acknowledgements

- Datasets sourced from public medical symptom repositories
- Trained on Google Colab free tier (T4 GPU)
- Deployed on Hugging Face Spaces and Vercel

---

<div align="center">

**Built with ❤️ by [Ashish Kishore](https://github.com/nameisashish)**

</div>
