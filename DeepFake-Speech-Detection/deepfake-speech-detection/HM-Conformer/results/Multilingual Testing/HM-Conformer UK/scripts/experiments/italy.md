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
| **EER (Equal Error Rate)** | 37.05% |
| **Accuracy** | 0.6418 (64.18%) |
| **F1 Score** | 0.6680 |
| **Precision** | 0.5297 |
| **Recall** | 0.9039 |
| **ROC AUC** | 0.6859 |
| **Optimal Threshold** | 1.0000 |

### Confusion Matrix

```
                Predicted
              Real    Fake
Actual Real  10,587  12,037
       Fake   1,441  13,559
```

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Real  | 0.88      | 0.47   | 0.61     | 22,624  |
| Fake  | 0.53      | 0.90   | 0.67     | 15,000  |
| **Macro Avg** | 0.70 | 0.69 | 0.64 | 37,624 |
| **Weighted Avg** | 0.74 | 0.64 | 0.63 | 37,624 |

## Summary

The model was evaluated on 37,624 Italian language samples from the test set. The model demonstrates moderate performance with an overall accuracy of 64.18%. The EER of 37.05% indicates some challenges in distinguishing between real and fake audio samples, similar to German. The model shows high precision for real samples (0.88) but relatively low recall (0.47), meaning when it predicts a sample as real, it's usually correct, but it misses over half of the real samples. For fake samples, the model has very high recall (0.90) but lower precision (0.53), indicating it captures most fake samples effectively but also misclassifies many real samples as fake. The ROC AUC of 0.69 suggests moderate discriminative ability. The performance pattern is similar to German, with the model being more conservative in predicting real samples while being more aggressive in detecting fake samples.
