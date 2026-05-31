# German Language Model Evaluation Report

## Dataset Information

**Language Filter:** German ('de')

- **Total entries before filtering:** 304,000
- **Total entries after filtering:** 53,468

### Dataset Split Distribution

| Split | Real Samples | Fake Samples | Total |
|-------|--------------|--------------|-------|
| TRAIN | 0 | 0 | 0 |
| VAL   | 0 | 0 | 0 |
| TEST  | 36,468 | 17,000 | 53,468 |

**Class Weights:**
- Real: 1.4662
- Fake: 3.1452

## Evaluation Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **EER (Equal Error Rate)** | 35.55% |
| **Accuracy** | 0.6312 (63.12%) |
| **F1 Score** | 0.6008 |
| **Precision** | 0.4581 |
| **Recall** | 0.8729 |
| **ROC AUC** | 0.6957 |
| **Optimal Threshold** | 1.0000 |

### Confusion Matrix

```
                Predicted
              Real    Fake
Actual Real  18,911  17,557
       Fake   2,161  14,839
```

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Real  | 0.90      | 0.52   | 0.66     | 36,468  |
| Fake  | 0.46      | 0.87   | 0.60     | 17,000  |
| **Macro Avg** | 0.68 | 0.70 | 0.63 | 53,468 |
| **Weighted Avg** | 0.76 | 0.63 | 0.64 | 53,468 |

## Summary

The model was evaluated on 53,468 German language samples from the test set. The model demonstrates moderate performance with an overall accuracy of 63.12%. The EER of 35.55% indicates some challenges in distinguishing between real and fake audio samples, though performance is better than Spanish. The model shows high precision for real samples (0.90) but relatively low recall (0.52), meaning when it predicts a sample as real, it's usually correct, but it misses many real samples. For fake samples, the model has high recall (0.87) but lower precision (0.46), indicating it captures most fake samples but also misclassifies many real samples as fake. The ROC AUC of 0.70 suggests moderate discriminative ability. The class imbalance (more real than fake samples) may contribute to the performance characteristics.
