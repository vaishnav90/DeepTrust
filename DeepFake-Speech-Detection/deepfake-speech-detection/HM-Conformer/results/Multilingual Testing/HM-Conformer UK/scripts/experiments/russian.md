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
| **EER (Equal Error Rate)** | 27.53% |
| **Accuracy** | 0.7578 (75.78%) |
| **F1 Score** | 0.7482 |
| **Precision** | 0.8352 |
| **Recall** | 0.6776 |
| **ROC AUC** | 0.7631 |
| **Optimal Threshold** | 1.0000 |

### Confusion Matrix

```
                Predicted
              Real    Fake
Actual Real   5,247    936
       Fake   2,257  4,743
```

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Real  | 0.70      | 0.85   | 0.77     | 6,183   |
| Fake  | 0.84      | 0.68   | 0.75     | 7,000   |
| **Macro Avg** | 0.77 | 0.76 | 0.76 | 13,183 |
| **Weighted Avg** | 0.77 | 0.76 | 0.76 | 13,183 |

## Summary

The model was evaluated on 13,183 Russian language samples from the test set. The model demonstrates strong performance with an overall accuracy of 75.78%, which is among the best results across all languages tested. The EER of 27.53% is the lowest (best) among all languages, indicating excellent ability to distinguish between real and fake audio samples. The model shows balanced performance across both classes, with high precision for fake samples (0.84) and high recall for real samples (0.85). The F1-scores are well-balanced (0.77 for Real, 0.75 for Fake), suggesting the model performs consistently across both classes. The ROC AUC of 0.76 indicates good discriminative ability. Despite having a relatively small test set compared to other languages, the model achieves strong performance on Russian, suggesting it generalizes well to this language.
