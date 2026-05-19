# Datasets

## Files

| File | Samples | Features | Classes | Size |
|------|---------|----------|---------|------|
| `symbipredict_2022.csv` | 4,961 | 132 binary symptom columns | 41 diseases | 1.4 MB |
| `bert_train.csv` | 6,833 | Free-text `text` column | 55 diseases | 1.9 MB |
| `Symptom2Disease.csv` | 1,200 | `label`, `text` columns | — | 230 KB |
| `dataset.csv` | 4,920 | 17 symptom columns per row | — | 632 KB |

## symbipredict_2022.csv

Structured symptom-disease dataset with binary features (0/1) for 132 symptoms.

- **Pre-balanced**: 121 samples per class — no SMOTE needed
- **Target column**: `prognosis`
- **Known quirks**: `"Diabetes "` and `"Hypertension "` have trailing spaces in the class labels

Used by: `01_symptom_model.py`, `02_symptom_ordering.py`, `03_symptom_benchmarks.py`

## bert_train.csv

Natural language symptom descriptions for NLP-based diagnosis.

- **Columns**: `#`, `Label`, `text`
- **Preprocessing**: Singletons (classes with 1 sample) are filtered out
- **Augmentation**: Classes with < 20 samples are augmented via WordNet synonym replacement

Used by: `04_nlp_model.py`, `05_nlp_benchmarks.py`

## Symptom2Disease.csv & dataset.csv

Supplementary datasets for reference and additional analysis.

## Source

These datasets are derived from publicly available medical symptom datasets. If you use them in your research, please cite this repository.
