# German Language Model Evaluation Report

## Dataset Information

**Language Filter:** German ('de')

- **Total entries before filtering:** 304,000
- **Total entries after filtering:** 53,468

### Dataset Split Distribution

| Split | Real Samples | Fake Samples | Total |
|-------|--------------|--------------|-------|
| TRAIN | 0 | 0 | 0 |
| VAL   | 0 | 0 | 0 |
| TEST  | 36,468 | 17,000 | 53,468 |

**Class Weights:**
- Real: 1.4662
- Fake: 3.1452

## Evaluation Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **EER (Equal Error Rate)** | 11.92% |
| **Accuracy** | 0.8972 (89.72%) |
| **F1 Score** | 0.8551 |
| **Precision** | 0.7750 |
| **Recall** | 0.9537 |
| **ROC AUC** | 0.9123 |
| **Optimal Threshold** | 1.0000 |

### Confusion Matrix

```
                Predicted
              Real    Fake
Actual Real  31,761  4,707
       Fake     787  16,213
```

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Real  | 0.98      | 0.87   | 0.92     | 36,468  |
| Fake  | 0.78      | 0.95   | 0.86     | 17,000  |
| **Macro Avg** | 0.88 | 0.91 | 0.89 | 53,468 |
| **Weighted Avg** | 0.91 | 0.90 | 0.90 | 53,468 |

## Summary

The model was evaluated on 53,468 German language samples from the test set. The model demonstrates good performance in detecting fake audio samples (F1-score: 0.86) with an overall accuracy of 89.72%. The EER of 11.92% indicates the point where false acceptance and false rejection rates are equal. The model shows high precision for real samples (0.98) and high recall for fake samples (0.95), suggesting good performance in identifying both classes, though with some room for improvement in precision for fake samples.
