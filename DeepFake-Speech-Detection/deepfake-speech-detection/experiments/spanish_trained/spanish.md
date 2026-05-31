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
| **EER (Equal Error Rate)** | 1.97% |
| **Accuracy** | 0.9853 (98.53%) |
| **F1 Score** | 0.9813 |
| **Precision** | 0.9824 |
| **Recall** | 0.9802 |
| **ROC AUC** | 0.9844 |
| **Optimal Threshold** | 1.0000 |

### Confusion Matrix

```
                Predicted
              Real    Fake
Actual Real  18,287    211
       Fake     238  11,762
```

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Real  | 0.99      | 0.99   | 0.99     | 18,498  |
| Fake  | 0.98      | 0.98   | 0.98     | 12,000  |
| **Macro Avg** | 0.98 | 0.98 | 0.98 | 30,498 |
| **Weighted Avg** | 0.99 | 0.99 | 0.99 | 30,498 |

## Summary

The model was evaluated on 30,498 Spanish language samples from the test set. The model demonstrates outstanding performance in detecting fake audio samples (F1-score: 0.98) with an overall accuracy of 98.53%. The EER of 1.97% indicates the point where false acceptance and false rejection rates are equal. The model shows exceptionally high precision and recall for both real and fake samples, suggesting excellent performance across both classes, which is expected given the model was trained on Spanish data.
