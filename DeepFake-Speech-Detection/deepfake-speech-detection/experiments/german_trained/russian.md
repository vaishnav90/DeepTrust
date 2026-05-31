# Russian Language Model Evaluation Report

## Dataset Information

**Language Filter:** Russian ('ru')

- **Total entries before filtering:** 304,000
- **Total entries after filtering:** 13,183

### Dataset Split Distribution

| Split | Real Samples | Fake Samples | Total |
|-------|--------------|--------------|-------|
| TRAIN | 0 | 0 | 0 |
| VAL   | 0 | 0 | 0 |
| TEST  | 6,183 | 7,000 | 13,183 |

**Class Weights:**
- Real: 2.1321
- Fake: 1.8833

## Evaluation Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **EER (Equal Error Rate)** | 27.28% |
| **Accuracy** | 0.8080 (80.80%) |
| **F1 Score** | 0.8408 |
| **Precision** | 0.7512 |
| **Recall** | 0.9547 |
| **ROC AUC** | 0.7983 |
| **Optimal Threshold** | 1.0000 |

### Confusion Matrix

```
                Predicted
              Real    Fake
Actual Real   3,969  2,214
       Fake     317  6,683
```

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Real  | 0.93      | 0.64   | 0.76     | 6,183   |
| Fake  | 0.75      | 0.95   | 0.84     | 7,000   |
| **Macro Avg** | 0.84 | 0.80 | 0.80 | 13,183 |
| **Weighted Avg** | 0.83 | 0.81 | 0.80 | 13,183 |

## Summary

The model was evaluated on 13,183 Russian language samples from the test set. The model demonstrates moderate performance in detecting fake audio samples (F1-score: 0.84) with an overall accuracy of 80.80%. The EER of 27.28% indicates the point where false acceptance and false rejection rates are equal. The model shows higher recall for fake samples (0.95) compared to precision (0.75), and lower recall for real samples (0.64) despite high precision (0.93), suggesting better performance in identifying synthetic audio but with some challenges in correctly classifying real samples.
