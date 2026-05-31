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
| **EER (Equal Error Rate)** | 46.19% |
| **Accuracy** | 0.4971 (49.71%) |
| **F1 Score** | 0.5738 |
| **Precision** | 0.4304 |
| **Recall** | 0.8604 |
| **ROC AUC** | 0.5609 |
| **Optimal Threshold** | 1.0000 |

### Confusion Matrix

```
                Predicted
              Real    Fake
Actual Real   4,835  13,663
       Fake   1,675  10,325
```

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Real  | 0.74      | 0.26   | 0.39     | 18,498  |
| Fake  | 0.43      | 0.86   | 0.57     | 12,000  |
| **Macro Avg** | 0.59 | 0.56 | 0.48 | 30,498 |
| **Weighted Avg** | 0.62 | 0.50 | 0.46 | 30,498 |

## Summary

The model was evaluated on 30,498 Spanish language samples from the test set. The model shows lower performance compared to English, with an overall accuracy of 49.71%, which is close to random chance. The EER of 46.19% indicates significant challenges in distinguishing between real and fake audio samples. The model demonstrates high recall for fake samples (0.86) but low precision (0.43), suggesting a tendency to classify many real samples as fake. Conversely, real samples show high precision (0.74) but very low recall (0.26), indicating that when the model predicts a sample as real, it's usually correct, but it misses most real samples. This imbalance suggests the model may need further training or tuning specifically for Spanish language audio.
