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
| **EER (Equal Error Rate)** | 5.98% |
| **Accuracy** | 0.9455 (94.55%) |
| **F1 Score** | 0.9635 |
| **Precision** | 0.9803 |
| **Recall** | 0.9473 |
| **ROC AUC** | 0.9436 |
| **Optimal Threshold** | 1.0000 |

### Confusion Matrix

```
                Predicted
              Real    Fake
Actual Real  20,182  1,293
       Fake   3,582  64,418
```

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Real  | 0.85      | 0.94   | 0.89     | 21,475  |
| Fake  | 0.98      | 0.95   | 0.96     | 68,000  |
| **Macro Avg** | 0.91 | 0.94 | 0.93 | 89,475 |
| **Weighted Avg** | 0.95 | 0.95 | 0.95 | 89,475 |

## Summary

The model was evaluated on 89,475 English language samples from the test set. The model demonstrates excellent performance in detecting fake audio samples (F1-score: 0.96) with an overall accuracy of 94.55%. The EER of 5.98% indicates the point where false acceptance and false rejection rates are equal. The model shows high precision and recall for fake samples, suggesting strong performance in identifying synthetic audio.
