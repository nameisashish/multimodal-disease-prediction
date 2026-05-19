# Training Pipeline

Reproduce the full training pipeline on **free Google Colab T4 GPU**.

## Prerequisites

- Google Colab with GPU runtime (T4 recommended)
- Upload the datasets from `../data/` to your Colab `/content/` directory

## Run Order

> **Important:** The dataset `symbipredict_2022.csv` is pre-balanced at 121 samples/class. **No SMOTE is needed.**

| Session | Script | Runtime | Produces |
|---------|--------|---------|----------|
| 1 | `01_symptom_model.py` | ~3-4 hrs | `final_symptom_model.keras`, `best_symptom_model.keras`, scaler, encoder |
| 2 | `04_nlp_model.py` | ~3.5 hrs | `final_nlp_model.keras`, `best_nlp_model.keras`, tokenizer, encoder |
| 3 | `02_symptom_ordering.py` | ~2-3 hrs | Ordering comparison (analysis only) |
| 4 | `03_symptom_benchmarks.py` | ~2-3 hrs | GRU, TCN, MLP, RF baselines (values only) |
| 5 | `05_nlp_benchmarks.py` | ~3 hrs | CNN, XGBoost, GRU+Attn baselines (values only) |
| 6 | `06_generative_eval.py` | ~5 min | Clinical evaluation template (CPU ok) |

## Scripts 1 & 2 (Model Training)

These produce the saved models used in the inference API:

- **Symptom Model**: DenseNet + BiLSTM + Self-Attention, 5-fold stratified CV
- **NLP Model**: CNN + BiLSTM + Self-Attention with synonym augmentation, 5-fold stratified CV

Both scripts include:
- 5-fold cross-validation with per-fold training curves
- Aggregate confusion matrix and ROC analysis
- Calibration analysis (reliability diagrams, ECE, Brier scores)
- Confidence threshold sweep
- Cost-sensitive evaluation (higher penalties for acute conditions)

## Scripts 3, 4, 5 (Benchmarks)

Comparison against baseline architectures:
- **Symptom**: GRU (no attention), TCN, MLP+Residual, TCN-MLP Hybrid, Random Forest
- **NLP**: CNN (no attention), XGBoost, GRU+Attention, TextCNN+Attention

## Script 6 (Generative Evaluation)

Framework for clinical evaluation of generative responses:
- `InterceptionLogger` — tracks prediction confidence and Gemini API hallucination rates
- `generate_evaluation_template()` — creates CSV forms for doctor evaluation
- `analyze_ratings()` — processes filled evaluation forms with inter-rater reliability

## Dataset Notes

- `symbipredict_2022.csv`: 4961 samples, 132 features, 41 classes, 121 samples/class (pre-balanced)
  - `"Diabetes "` and `"Hypertension "` have trailing spaces — this is intentional
- `bert_train.csv`: Free-text symptom descriptions, 55 classes after filtering singletons
  - Synonym augmentation applied to classes with < 20 samples
