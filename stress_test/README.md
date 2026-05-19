# Stress Tests

Robustness validation of trained models before deployment.

## Symptom Model

12 clinical stress tests designed the way a doctor validates a diagnostic AI:

| Test | Name | What It Does |
|------|------|-------------|
| T2 | Per-Class Accuracy | Accuracy breakdown per disease class |
| T3 | Confusable Pairs | Tests most-confused disease pairs |
| T9 | Misclassification Analysis | Identifies systematic prediction errors |
| T10 | Symptom Dropout | Per-symptom importance (remove one at a time) |
| T11 | Gaussian Noise | Robustness to noisy symptom vectors |
| T12 | Calibration | Confidence vs actual accuracy alignment |
| T13 | Comorbidity | Multi-disease symptom overlap handling |
| T14 | Gradual Onset | Partial symptom presentation (early-stage) |
| T15 | Worst-Case | Adversarial edge cases |

### Results Summary

Both **best-fold** and **final** (retrained on all train+val) models are compared across all tests.

**Recommendation:** The **FINAL model** is recommended for deployment.

| Metric | Best-Fold | Final |
|--------|-----------|-------|
| Composite Score | 0.7450 | **0.8211** |
| Avg Accuracy | 0.9175 | **0.9368** |
| Avg F1 | 0.9080 | **0.9308** |
| Wins | 2 | **10** (15 ties) |

Full results: [`symptom_model/results/`](symptom_model/results/)
