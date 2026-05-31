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
| **EER (Equal Error Rate)** | 22.98% |
| **Accuracy** | 0.8557 (85.57%) |
| **F1 Score** | 0.8792 |
| **Precision** | 0.7914 |
| **Recall** | 0.9890 |
| **ROC AUC** | 0.8469 |
| **Optimal Threshold** | 1.0000 |

### Confusion Matrix

```
                Predicted
              Real    Fake
Actual Real   4,358  1,825
       Fake      77  6,923
```

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Real  | 0.98      | 0.70   | 0.82     | 6,183   |
| Fake  | 0.79      | 0.99   | 0.88     | 7,000   |
| **Macro Avg** | 0.89 | 0.85 | 0.85 | 13,183 |
| **Weighted Avg** | 0.88 | 0.86 | 0.85 | 13,183 |

## Summary

The model was evaluated on 13,183 Russian language samples from the test set. The model demonstrates moderate performance in detecting fake audio samples (F1-score: 0.88) with an overall accuracy of 85.57%. The EER of 22.98% indicates the point where false acceptance and false rejection rates are equal. The model shows high recall for fake samples (0.99) but lower precision (0.79), and lower recall for real samples (0.70) despite high precision (0.98), suggesting better performance in identifying synthetic audio but with challenges in correctly classifying real samples.
