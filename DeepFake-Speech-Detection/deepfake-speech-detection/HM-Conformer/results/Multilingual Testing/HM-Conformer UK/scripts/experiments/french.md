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
| **EER (Equal Error Rate)** | 30.50% |
| **Accuracy** | 0.7200 (72.00%) |
| **F1 Score** | 0.7155 |
| **Precision** | 0.6081 |
| **Recall** | 0.8688 |
| **ROC AUC** | 0.7437 |
| **Optimal Threshold** | 1.0000 |

### Confusion Matrix

```
                Predicted
              Real    Fake
Actual Real  17,257  10,638
       Fake   2,492  16,508
```

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Real  | 0.87      | 0.62   | 0.72     | 27,895  |
| Fake  | 0.61      | 0.87   | 0.72     | 19,000  |
| **Macro Avg** | 0.74 | 0.74 | 0.72 | 46,895 |
| **Weighted Avg** | 0.77 | 0.72 | 0.72 | 46,895 |

## Summary

The model was evaluated on 46,895 French language samples from the test set. The model demonstrates moderate performance with an overall accuracy of 72.00%. The EER of 30.50% is comparable to the English results, indicating reasonable performance in distinguishing between real and fake audio samples. The model shows balanced F1-scores for both classes (0.72 for both Real and Fake), with high precision for real samples (0.87) and high recall for fake samples (0.87). This suggests the model is more conservative in predicting real samples (high precision) but captures most fake samples effectively (high recall). The ROC AUC of 0.74 indicates decent discriminative ability.
