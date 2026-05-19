# Training Pipeline

Reproduce the full training pipeline on **free Google Colab T4 GPU**.

## Prerequisites

- Google Colab with GPU runtime (T4 recommended)
- Upload the datasets from `../data/` to your Colab `/content/` directory

## Run Order

> **Important:** The dataset `symbipredict_2022.csv` is pre-balanced at 121 samples/class. **No SMOTE is needed.**

| # | Script | Runtime | Produces |
|---|--------|---------|----------|
| 1 | `01_symptom_model.py` | ~3-4 hrs | `final_symptom_model.keras`, `best_symptom_model.keras`, scaler, encoder |
| 2 | `03_symptom_benchmarks.py` | ~2-3 hrs | GRU, TCN, MLP, RF baselines (values only) |
| 3 | `04_nlp_model.py` | ~3.5 hrs | `final_nlp_model.keras`, `best_nlp_model.keras`, tokenizer, encoder |
| 4 | `05_nlp_benchmarks.py` | ~3 hrs | CNN, XGBoost, GRU+Attn baselines (values only) |
| 5 | `06_generative_eval.py` | ~5 min | Clinical evaluation template (CPU ok) |
| 6 | `07_symptom_deployment_prep.ipynb` | ~2-3 hrs | Production-ready `final_symptom_model.keras` (single run, no folds) |

## Model Training (01, 04)

These produce the saved models used in the inference API:

- **Symptom Model** (`01`): DenseNet + BiLSTM + Self-Attention, 5-fold stratified CV
- **NLP Model** (`04`): CNN + BiLSTM + Self-Attention with synonym augmentation, 5-fold stratified CV

Both scripts include:
- 5-fold cross-validation with per-fold training curves
- Aggregate confusion matrix and ROC analysis
- Calibration analysis (reliability diagrams, ECE, Brier scores)
- Confidence threshold sweep
- Cost-sensitive evaluation (higher penalties for acute conditions)

## Benchmarks (03, 05)

Comparison against baseline architectures:
- **Symptom** (`03`): GRU (no attention), TCN, MLP+Residual, TCN-MLP Hybrid, Random Forest
- **NLP** (`05`): CNN (no attention), XGBoost, GRU+Attention, TextCNN+Attention

## Generative Evaluation (06)

Framework for clinical evaluation of generative responses:
- `InterceptionLogger` — tracks prediction confidence and hallucination rates
- `generate_evaluation_template()` — creates CSV forms for doctor evaluation
- `analyze_ratings()` — processes filled evaluation forms with inter-rater reliability

## Deployment Model Prep (07)

Single training run (no cross-validation) of the symptom model to produce the final deployment-ready model. Same DenseNet+BiLSTM+Attention architecture as `01`, with synthetic data expansion, augmentation, MixUp, and temperature scaling.

## Dataset Notes

- `symbipredict_2022.csv`: 4961 samples, 132 features, 41 classes, 121 samples/class (pre-balanced)
  - `"Diabetes "` and `"Hypertension "` have trailing spaces — this is intentional
- `bert_train.csv`: Free-text symptom descriptions, 55 classes after filtering singletons
  - Synonym augmentation applied to classes with < 20 samples
