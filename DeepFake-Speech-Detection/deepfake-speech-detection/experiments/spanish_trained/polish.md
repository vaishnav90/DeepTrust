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
| **EER (Equal Error Rate)** | 20.16% |
| **Accuracy** | 0.8297 (82.97%) |
| **F1 Score** | 0.8379 |
| **Precision** | 0.7976 |
| **Recall** | 0.8824 |
| **ROC AUC** | 0.8298 |
| **Optimal Threshold** | 1.0000 |

### Confusion Matrix

```
                Predicted
              Real    Fake
Actual Real   6,249  1,791
       Fake     941  7,059
```

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Real  | 0.87      | 0.78   | 0.82     | 8,040   |
| Fake  | 0.80      | 0.88   | 0.84     | 8,000   |
| **Macro Avg** | 0.83 | 0.83 | 0.83 | 16,040 |
| **Weighted Avg** | 0.83 | 0.83 | 0.83 | 16,040 |

## Summary

The model was evaluated on 16,040 Polish language samples from the test set. The model demonstrates moderate performance in detecting fake audio samples (F1-score: 0.84) with an overall accuracy of 82.97%. The EER of 20.16% indicates the point where false acceptance and false rejection rates are equal. The model shows balanced precision and recall for both classes, with slightly better performance for fake samples, suggesting reasonable performance but with room for improvement.
