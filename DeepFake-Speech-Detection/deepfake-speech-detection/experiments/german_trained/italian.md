# Italian Language Model Evaluation Report

## Dataset Information

**Language Filter:** Italian ('it')

- **Total entries before filtering:** 304,000
- **Total entries after filtering:** 37,624

### Dataset Split Distribution

| Split | Real Samples | Fake Samples | Total |
|-------|--------------|--------------|-------|
| TRAIN | 0 | 0 | 0 |
| VAL   | 0 | 0 | 0 |
| TEST  | 22,624 | 15,000 | 37,624 |

**Class Weights:**
- Real: 1.6630
- Fake: 2.5083

## Evaluation Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **EER (Equal Error Rate)** | 2.92% |
| **Accuracy** | 0.9715 (97.15%) |
| **F1 Score** | 0.9646 |
| **Precision** | 0.9566 |
| **Recall** | 0.9727 |
| **ROC AUC** | 0.9717 |
| **Optimal Threshold** | 1.0000 |

### Confusion Matrix

```
                Predicted
              Real    Fake
Actual Real  21,962    662
       Fake     410  14,590
```

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Real  | 0.98      | 0.97   | 0.98     | 22,624  |
| Fake  | 0.96      | 0.97   | 0.96     | 15,000  |
| **Macro Avg** | 0.97 | 0.97 | 0.97 | 37,624 |
| **Weighted Avg** | 0.97 | 0.97 | 0.97 | 37,624 |

## Summary

The model was evaluated on 37,624 Italian language samples from the test set. The model demonstrates excellent performance in detecting fake audio samples (F1-score: 0.96) with an overall accuracy of 97.15%. The EER of 2.92% indicates the point where false acceptance and false rejection rates are equal. The model shows high precision and recall for both real and fake samples, suggesting strong performance across both classes.
