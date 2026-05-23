# Training Pipeline

Reproduce the full training pipeline on **free Google Colab T4 GPU**.

## Prerequisites

- Google Colab with GPU runtime (T4 recommended)
- Upload the datasets from `../data/` to your Colab `/content/` directory

## Run Order

> **Important:** The dataset `symbipredict_2022.csv` is pre-balanced at 121 samples/class. **No SMOTE is needed.**

| # | File Name | Google Colab | Runtime | Produces |
|---|-----------|--------------|---------|----------|
| 1 | `01_symptom_model` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nameisashish/multimodal-disease-prediction/blob/main/training/01_symptom_model.ipynb) | ~3-4 hrs | `final_symptom_model.keras`, `best_symptom_model.keras`, scaler, encoder |
| 2 | `02_symptom_benchmarks` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nameisashish/multimodal-disease-prediction/blob/main/training/02_symptom_benchmarks.ipynb) | ~2-3 hrs | GRU, TCN, MLP, RF baselines (values only) |
| 3 | `03_symptom_deployment_prep` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nameisashish/multimodal-disease-prediction/blob/main/training/03_symptom_deployment_prep.ipynb) | ~2-3 hrs | Production-ready `final_symptom_model.keras` (single run) |
| 4 | `04_nlp_model` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nameisashish/multimodal-disease-prediction/blob/main/training/04_nlp_model.ipynb) | ~3.5 hrs | `final_nlp_model.keras`, `best_nlp_model.keras`, tokenizer, encoder |
| 5 | `05_nlp_benchmarks` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nameisashish/multimodal-disease-prediction/blob/main/training/05_nlp_benchmarks.ipynb) | ~3 hrs | CNN, XGBoost, GRU+Attn baselines (values only) |

## Symptom Model (01, 02, 03)

- **01**: DenseNet + BiLSTM + Self-Attention, 5-fold stratified CV with calibration, confusion matrices, ROC, cost-sensitive evaluation, and confidence thresholding
- **02**: Benchmarks against GRU (no attention), TCN, MLP+Residual, TCN-MLP Hybrid, Random Forest
- **03**: Single training run (no cross-validation) to produce the deployment-ready model with synthetic data expansion, augmentation, MixUp, and temperature scaling

## NLP Model (04, 05)

- **04**: CNN + BiLSTM + Self-Attention with synonym augmentation, 5-fold stratified CV, plus final retrain on all train+val data for deployment
- **05**: Benchmarks against CNN (no attention), XGBoost, GRU+Attention, TextCNN+Attention

## Dataset Notes

- `symbipredict_2022.csv`: 4961 samples, 132 features, 41 classes, 121 samples/class (pre-balanced)
  - `"Diabetes "` and `"Hypertension "` have trailing spaces — this is intentional
- `bert_train.csv`: Free-text symptom descriptions, 55 classes after filtering singletons
  - Synonym augmentation applied to classes with < 20 samples
