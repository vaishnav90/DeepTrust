# Polish Language Model Evaluation Report

## Dataset Information

**Language Filter:** Polish ('pl')

- **Total entries before filtering:** 304,000
- **Total entries after filtering:** 16,040

### Dataset Split Distribution

| Split | Real Samples | Fake Samples | Total |
|-------|--------------|--------------|-------|
| TRAIN | 0 | 0 | 0 |
| VAL   | 0 | 0 | 0 |
| TEST  | 8,040 | 8,000 | 16,040 |

**Class Weights:**
- Real: 1.9950
- Fake: 2.0050

## Evaluation Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **EER (Equal Error Rate)** | 44.55% |
| **Accuracy** | 0.5784 (57.84%) |
| **F1 Score** | 0.6553 |
| **Precision** | 0.5533 |
| **Recall** | 0.8035 |
| **ROC AUC** | 0.5790 |
| **Optimal Threshold** | 1.0000 |

### Confusion Matrix

```
                Predicted
              Real    Fake
Actual Real   2,850   5,190
       Fake   1,572   6,428
```

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Real  | 0.64      | 0.35   | 0.46     | 8,040   |
| Fake  | 0.55      | 0.80   | 0.66     | 8,000   |
| **Macro Avg** | 0.60 | 0.58 | 0.56 | 16,040 |
| **Weighted Avg** | 0.60 | 0.58 | 0.56 | 16,040 |

## Summary

The model was evaluated on 16,040 Polish language samples from the test set. The model demonstrates lower performance with an overall accuracy of 57.84%, which is close to random chance. The EER of 44.55% indicates significant challenges in distinguishing between real and fake audio samples, similar to Spanish. Notably, this dataset has a balanced class distribution (8,040 real vs 8,000 fake samples), which is more balanced than other languages. The model shows moderate precision for real samples (0.64) but very low recall (0.35), meaning it misses most real samples. For fake samples, the model has high recall (0.80) but lower precision (0.55), indicating it captures most fake samples but also misclassifies many real samples as fake. The ROC AUC of 0.58 suggests limited discriminative ability, barely better than random. This performance suggests the model may need additional training data or fine-tuning specifically for Polish language audio.
