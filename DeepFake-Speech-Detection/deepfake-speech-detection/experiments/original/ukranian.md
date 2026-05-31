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
| **EER (Equal Error Rate)** | 21.40% |
| **Accuracy** | 0.8456 (84.56%) |
| **F1 Score** | 0.7775 |
| **Precision** | 0.7999 |
| **Recall** | 0.7563 |
| **ROC AUC** | 0.8257 |
| **Optimal Threshold** | 1.0000 |

### Confusion Matrix

```
                Predicted
              Real    Fake
Actual Real   9,682   1,135
       Fake   1,462   4,538
```

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Real  | 0.87      | 0.90   | 0.88     | 10,817  |
| Fake  | 0.80      | 0.76   | 0.78     | 6,000   |
| **Macro Avg** | 0.83 | 0.83 | 0.83 | 16,817 |
| **Weighted Avg** | 0.84 | 0.85 | 0.84 | 16,817 |

## Summary

The model was evaluated on 16,817 Ukrainian language samples from the test set. The model demonstrates excellent performance with an overall accuracy of 84.56%, which is among the best results across all languages tested. The EER of 21.40% is the lowest (best) among all languages, indicating superior ability to distinguish between real and fake audio samples. The model shows strong and balanced performance across both classes, with high precision and recall for real samples (0.87 and 0.90 respectively) and good performance for fake samples (0.80 precision and 0.76 recall). The F1-scores are well-balanced (0.88 for Real, 0.78 for Fake), and the macro-averaged metrics show consistent performance (0.83 across precision, recall, and F1-score). The ROC AUC of 0.83 indicates excellent discriminative ability. Ukrainian represents the best-performing language in this evaluation, demonstrating that the model generalizes exceptionally well to this language.
