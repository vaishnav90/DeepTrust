# French Language Model Evaluation Report

## Dataset Information

**Language Filter:** French ('fr')

- **Total entries before filtering:** 304,000
- **Total entries after filtering:** 46,895

### Dataset Split Distribution

| Split | Real Samples | Fake Samples | Total |
|-------|--------------|--------------|-------|
| TRAIN | 0 | 0 | 0 |
| VAL   | 0 | 0 | 0 |
| TEST  | 27,895 | 19,000 | 46,895 |

**Class Weights:**
- Real: 1.6811
- Fake: 2.4682

## Evaluation Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **EER (Equal Error Rate)** | 8.18% |
| **Accuracy** | 0.9413 (94.13%) |
| **F1 Score** | 0.9315 |
| **Precision** | 0.8844 |
| **Recall** | 0.9838 |
| **ROC AUC** | 0.9481 |
| **Optimal Threshold** | 1.0000 |

### Confusion Matrix

```
                Predicted
              Real    Fake
Actual Real  25,451  2,444
       Fake     307  18,693
```

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Real  | 0.99      | 0.91   | 0.95     | 27,895  |
| Fake  | 0.88      | 0.98   | 0.93     | 19,000  |
| **Macro Avg** | 0.94 | 0.95 | 0.94 | 46,895 |
| **Weighted Avg** | 0.95 | 0.94 | 0.94 | 46,895 |

## Summary

The model was evaluated on 46,895 French language samples from the test set. The model demonstrates strong performance in detecting fake audio samples (F1-score: 0.93) with an overall accuracy of 94.13%. The EER of 8.18% indicates the point where false acceptance and false rejection rates are equal. The model shows high precision for real samples (0.99) and high recall for fake samples (0.98), suggesting strong performance in identifying both classes.
