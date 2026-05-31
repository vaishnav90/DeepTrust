# Ukrainian Language Model Evaluation Report

## Dataset Information

**Language Filter:** Ukrainian ('uk')

- **Total entries before filtering:** 304,000
- **Total entries after filtering:** 16,817

### Dataset Split Distribution

| Split | Real Samples | Fake Samples | Total |
|-------|--------------|--------------|-------|
| TRAIN | 0 | 0 | 0 |
| VAL   | 0 | 0 | 0 |
| TEST  | 10,817 | 6,000 | 16,817 |

**Class Weights:**
- Real: 1.5547
- Fake: 2.8028

## Evaluation Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **EER (Equal Error Rate)** | 11.80% |
| **Accuracy** | 0.9115 (91.15%) |
| **F1 Score** | 0.8887 |
| **Precision** | 0.8057 |
| **Recall** | 0.9907 |
| **ROC AUC** | 0.9291 |
| **Optimal Threshold** | 1.0000 |

### Confusion Matrix

```
                Predicted
              Real    Fake
Actual Real   9,384  1,433
       Fake      56  5,944
```

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Real  | 0.99      | 0.87   | 0.93     | 10,817  |
| Fake  | 0.81      | 0.99   | 0.89     | 6,000   |
| **Macro Avg** | 0.90 | 0.93 | 0.91 | 16,817 |
| **Weighted Avg** | 0.93 | 0.91 | 0.91 | 16,817 |

## Summary

The model was evaluated on 16,817 Ukrainian language samples from the test set. The model demonstrates strong performance in detecting fake audio samples (F1-score: 0.89) with an overall accuracy of 91.15%. The EER of 11.80% indicates the point where false acceptance and false rejection rates are equal. The model shows high precision for real samples (0.99) and high recall for fake samples (0.99), suggesting good performance in identifying both classes, though with some room for improvement in precision for fake samples.
