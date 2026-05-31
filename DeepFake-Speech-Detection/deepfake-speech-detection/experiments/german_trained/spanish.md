# Spanish Language Model Evaluation Report

## Dataset Information

**Language Filter:** Spanish ('es')

- **Total entries before filtering:** 304,000
- **Total entries after filtering:** 30,498

### Dataset Split Distribution

| Split | Real Samples | Fake Samples | Total |
|-------|--------------|--------------|-------|
| TRAIN | 0 | 0 | 0 |
| VAL   | 0 | 0 | 0 |
| TEST  | 18,498 | 12,000 | 30,498 |

**Class Weights:**
- Real: 1.6487
- Fake: 2.5415

## Evaluation Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **EER (Equal Error Rate)** | 3.23% |
| **Accuracy** | 0.9721 (97.21%) |
| **F1 Score** | 0.9650 |
| **Precision** | 0.9510 |
| **Recall** | 0.9794 |
| **ROC AUC** | 0.9734 |
| **Optimal Threshold** | 1.0000 |

### Confusion Matrix

```
                Predicted
              Real    Fake
Actual Real  17,893    605
       Fake     247  11,753
```

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Real  | 0.99      | 0.97   | 0.98     | 18,498  |
| Fake  | 0.95      | 0.98   | 0.97     | 12,000  |
| **Macro Avg** | 0.97 | 0.97 | 0.97 | 30,498 |
| **Weighted Avg** | 0.97 | 0.97 | 0.97 | 30,498 |

## Summary

The model was evaluated on 30,498 Spanish language samples from the test set. The model demonstrates excellent performance in detecting fake audio samples (F1-score: 0.97) with an overall accuracy of 97.21%. The EER of 3.23% indicates the point where false acceptance and false rejection rates are equal. The model shows high precision and recall for both real and fake samples, suggesting strong performance across both classes.
