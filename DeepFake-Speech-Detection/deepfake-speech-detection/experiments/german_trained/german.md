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
| **EER (Equal Error Rate)** | 5.05% |
| **Accuracy** | 0.9592 (95.92%) |
| **F1 Score** | 0.9387 |
| **Precision** | 0.8976 |
| **Recall** | 0.9839 |
| **ROC AUC** | 0.9658 |
| **Optimal Threshold** | 1.0000 |

### Confusion Matrix

```
                Predicted
              Real    Fake
Actual Real  34,559  1,909
       Fake     274  16,726
```

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Real  | 0.99      | 0.95   | 0.97     | 36,468  |
| Fake  | 0.90      | 0.98   | 0.94     | 17,000  |
| **Macro Avg** | 0.94 | 0.97 | 0.95 | 53,468 |
| **Weighted Avg** | 0.96 | 0.96 | 0.96 | 53,468 |

## Summary

The model was evaluated on 53,468 German language samples from the test set. The model demonstrates strong performance in detecting fake audio samples (F1-score: 0.94) with an overall accuracy of 95.92%. The EER of 5.05% indicates the point where false acceptance and false rejection rates are equal. The model shows high precision for real samples (0.99) and high recall for fake samples (0.98), suggesting excellent performance in identifying both classes.
