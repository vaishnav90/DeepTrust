# English Language Model Evaluation Report

## Dataset Information

**Language Filter:** English ('en')

- **Total entries before filtering:** 304,000
- **Total entries after filtering:** 89,475

### Dataset Split Distribution

| Split | Real Samples | Fake Samples | Total |
|-------|--------------|--------------|-------|
| TRAIN | 0 | 0 | 0 |
| VAL   | 0 | 0 | 0 |
| TEST  | 21,475 | 68,000 | 89,475 |

**Class Weights:**
- Real: 4.1665
- Fake: 1.3158

## Evaluation Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **EER (Equal Error Rate)** | 4.11% |
| **Accuracy** | 0.9646 (96.46%) |
| **F1 Score** | 0.9762 |
| **Precision** | 0.9954 |
| **Recall** | 0.9577 |
| **ROC AUC** | 0.9719 |
| **Optimal Threshold** | 1.0000 |

### Confusion Matrix

```
                Predicted
              Real    Fake
Actual Real  21,177    298
       Fake   2,873  65,127
```

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Real  | 0.88      | 0.99   | 0.93     | 21,475  |
| Fake  | 1.00      | 0.96   | 0.98     | 68,000  |
| **Macro Avg** | 0.94 | 0.97 | 0.95 | 89,475 |
| **Weighted Avg** | 0.97 | 0.96 | 0.97 | 89,475 |

## Summary

The model was evaluated on 89,475 English language samples from the test set. The model demonstrates excellent performance in detecting fake audio samples (F1-score: 0.98) with an overall accuracy of 96.46%. The EER of 4.11% indicates the point where false acceptance and false rejection rates are equal. The model shows high precision and recall for both real and fake samples, suggesting strong performance across both classes.
