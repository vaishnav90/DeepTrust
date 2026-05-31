# French Language Model Evaluation Report

## Dataset Information

**Language Filter:** French ('fr')

- **Total entries before filtering:** 304,000
- **Total entries after filtering:** 46,895

### Dataset Split Distribution

| Split | Real Samples | Fake Samples | Total |
|-------|--------------|--------------|-------|
| TRAIN | 0 | 0 | 0 |
| VAL   | 0 | 0 | 0 |
| TEST  | 27,895 | 19,000 | 46,895 |

**Class Weights:**
- Real: 1.6811
- Fake: 2.4682

## Evaluation Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **EER (Equal Error Rate)** | 1.93% |
| **Accuracy** | 0.9845 (98.45%) |
| **F1 Score** | 0.9810 |
| **Precision** | 0.9719 |
| **Recall** | 0.9903 |
| **ROC AUC** | 0.9854 |
| **Optimal Threshold** | 1.0000 |

### Confusion Matrix

```
                Predicted
              Real    Fake
Actual Real  27,352    543
       Fake     185  18,815
```

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Real  | 0.99      | 0.98   | 0.99     | 27,895  |
| Fake  | 0.97      | 0.99   | 0.98     | 19,000  |
| **Macro Avg** | 0.98 | 0.99 | 0.98 | 46,895 |
| **Weighted Avg** | 0.98 | 0.98 | 0.98 | 46,895 |

## Summary

The model was evaluated on 46,895 French language samples from the test set. The model demonstrates excellent performance in detecting fake audio samples (F1-score: 0.98) with an overall accuracy of 98.45%. The EER of 1.93% indicates the point where false acceptance and false rejection rates are equal. The model shows high precision and recall for both real and fake samples, suggesting outstanding performance across both classes.
