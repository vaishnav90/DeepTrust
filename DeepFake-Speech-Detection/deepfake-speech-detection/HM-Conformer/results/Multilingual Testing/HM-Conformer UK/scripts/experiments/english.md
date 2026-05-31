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
| **EER (Equal Error Rate)** | 29.33% |
| **Accuracy** | 0.8375 (83.75%) |
| **F1 Score** | 0.8943 |
| **Precision** | 0.8841 |
| **Recall** | 0.9047 |
| **ROC AUC** | 0.7646 |
| **Optimal Threshold** | 1.0000 |

### Confusion Matrix

```
                Predicted
              Real    Fake
Actual Real  13,410  8,065
       Fake   6,479  61,521
```

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Real  | 0.67      | 0.62   | 0.65     | 21,475  |
| Fake  | 0.88      | 0.90   | 0.89     | 68,000  |
| **Macro Avg** | 0.78 | 0.76 | 0.77 | 89,475 |
| **Weighted Avg** | 0.83 | 0.84 | 0.84 | 89,475 |

## Summary

The model was evaluated on 89,475 English language samples from the test set. The model demonstrates strong performance in detecting fake audio samples (F1-score: 0.89) with an overall accuracy of 83.75%. The EER of 29.33% indicates the point where false acceptance and false rejection rates are equal. The model shows higher precision and recall for fake samples compared to real samples, suggesting better performance in identifying synthetic audio.
