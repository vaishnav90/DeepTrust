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
| **EER (Equal Error Rate)** | 9.24% |
| **Accuracy** | 0.9461 (94.61%) |
| **F1 Score** | 0.9433 |
| **Precision** | 0.9924 |
| **Recall** | 0.8989 |
| **ROC AUC** | 0.9460 |
| **Optimal Threshold** | 1.0000 |

### Confusion Matrix

```
                Predicted
              Real    Fake
Actual Real   7,985     55
       Fake     809  7,191
```

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Real  | 0.91      | 0.99   | 0.95     | 8,040   |
| Fake  | 0.99      | 0.90   | 0.94     | 8,000   |
| **Macro Avg** | 0.95 | 0.95 | 0.95 | 16,040 |
| **Weighted Avg** | 0.95 | 0.95 | 0.95 | 16,040 |

## Summary

The model was evaluated on 16,040 Polish language samples from the test set. The model demonstrates strong performance in detecting fake audio samples (F1-score: 0.94) with an overall accuracy of 94.61%. The EER of 9.24% indicates the point where false acceptance and false rejection rates are equal. The model shows high precision for fake samples (0.99) and high recall for real samples (0.99), suggesting good performance in identifying both classes.
