# Datasets

| File | Samples | Features | Classes | Size |
|------|---------|----------|---------|------|
| `symbipredict_2022.csv` | 4,961 | 132 binary symptom columns | 41 diseases | 1.4 MB |
| `bert_train.csv` | 6,833 | Free-text `text` column | 55 diseases | 1.9 MB |

## symbipredict_2022.csv

Structured symptom-disease dataset with binary features (0/1) for 132 symptoms.

- **Pre-balanced**: 121 samples per class — no SMOTE needed
- **Target column**: `prognosis`
- **Known quirks**: `"Diabetes "` and `"Hypertension "` have trailing spaces in the class labels

Used by: `01_symptom_model` (.py/.ipynb), `02_symptom_benchmarks` (.py/.ipynb), `03_symptom_deployment_prep.ipynb`

## bert_train.csv

Natural language symptom descriptions for NLP-based diagnosis.

- **Columns**: `#`, `Label`, `text`
- **Preprocessing**: Singletons (classes with 1 sample) are filtered out
- **Augmentation**: Classes with < 20 samples are augmented via WordNet synonym replacement

Used by: `04_nlp_model` (.py/.ipynb), `05_nlp_benchmarks` (.py/.ipynb)
