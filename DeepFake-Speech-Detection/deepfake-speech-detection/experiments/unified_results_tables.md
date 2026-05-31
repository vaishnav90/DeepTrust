# Unified Results Tables (Experiments)


## Index

- [Original (baseline) model](#original-baseline-model)
- [English-trained model](#english-trained-model)
- [Spanish-trained model](#spanish-trained-model)
- [German-trained model](#german-trained-model)

---

## Original (baseline) model

- **Source**: `experiments/original/summary_report.md`

| Language | Test Samples | EER (%) | Accuracy (%) | F1 | Precision | Recall | ROC AUC | Best Metric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ukrainian** | 16,817 | **21.40** | **84.56** | 0.7775 | — | — | **0.8257** | Best overall |
| **English** | 89,475 | 29.33 | 83.75 | **0.8943** | — | — | 0.7646 | Best F1 score |
| **Russian** | 13,183 | **27.53** | 75.78 | 0.7482 | — | — | 0.7631 | 2nd best EER |
| **French** | 46,895 | 30.50 | 72.00 | 0.7155 | — | — | 0.7437 | Moderate |
| **Italian** | 37,624 | 37.05 | 64.18 | 0.6680 | — | — | 0.6859 | Moderate |
| **German** | 53,468 | 35.55 | 63.12 | 0.6008 | — | — | 0.6957 | Moderate |
| **Polish** | 16,040 | 44.55 | 57.84 | 0.6553 | — | — | 0.5790 | Lower performance |
| **Spanish** | 30,498 | **46.19** | **49.71** | 0.5738 | — | — | **0.5609** | Worst overall |

---

## English-trained model

- **Source**: `experiments/english_trained/summary_report.md`

| Language | Test Samples | EER (%) | Accuracy (%) | F1 | Precision | Recall | ROC AUC | Best Metric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **French** | 46,895 | 1.45 | 98.87 | 0.9861 | 0.9788 | 0.9935 | 0.9894 | Best overall |
| **English** | 89,475 | 1.76 | 98.60 | 0.9907 | 0.9995 | 0.9821 | 0.9903 | Best F1 |
| **Polish** | 16,040 | 1.47 | 98.55 | 0.9855 | 0.9857 | 0.9852 | 0.9855 | 2nd best EER |
| **Italian** | 37,624 | 2.27 | 98.36 | 0.9794 | 0.9816 | 0.9771 | 0.9825 | Strong performance |
| **Ukrainian** | 16,817 | 5.25 | 96.12 | 0.9480 | 0.9091 | 0.9903 | 0.9677 | Moderate |
| **German** | 53,468 | 8.94 | 92.97 | 0.8993 | 0.8260 | 0.9868 | 0.9450 | Moderate |
| **Spanish** | 30,498 | 14.69 | 89.24 | 0.8786 | 0.7902 | 0.9892 | 0.9094 | Lower performance |
| **Russian** | 13,183 | 28.48 | 80.19 | 0.8382 | 0.7398 | 0.9669 | 0.7910 | Worst overall |

---

## Spanish-trained model

- **Source**: `experiments/spanish_trained/summary_report.md`

| Language | Test Samples | EER (%) | Accuracy (%) | F1 | Precision | Recall | ROC AUC | Best Metric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Spanish** | 30,498 | **1.97** | **98.53** | **0.9813** | — | — | **0.9844** | Best overall (training language) |
| **Italian** | 37,624 | **4.37** | 96.10 | 0.9519 | — | — | 0.9623 | 2nd best EER |
| **English** | 89,475 | 5.98 | 94.55 | 0.9635 | — | — | 0.9436 | Best F1 score |
| **French** | 46,895 | 8.18 | 94.13 | 0.9315 | — | — | 0.9481 | Strong performance |
| **Ukrainian** | 16,817 | 11.80 | 91.15 | 0.8887 | — | — | 0.9291 | Moderate |
| **German** | 53,468 | 11.92 | 89.72 | 0.8551 | — | — | 0.9123 | Moderate |
| **Russian** | 13,183 | **22.98** | 85.57 | 0.8792 | — | — | 0.8469 | Worst EER |
| **Polish** | 16,040 | 20.16 | 82.97 | 0.8379 | — | — | 0.8298 | Lower performance |

---

## German-trained model

- **Source**: `experiments/german_trained/summary_report.md`

| Language | Test Samples | EER (%) | Accuracy (%) | F1 | Precision | Recall | ROC AUC | Best Metric |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **French** | 46,895 | **1.93** | **98.45** | **0.9810** | — | — | **0.9854** | Best overall (best cross-lingual) |
| **Spanish** | 30,498 | 3.23 | 97.21 | 0.9650 | — | — | 0.9734 | Strong performance |
| **Italian** | 37,624 | **2.92** | 97.15 | 0.9646 | — | — | 0.9717 | 2nd best EER |
| **English** | 89,475 | 4.11 | 96.46 | 0.9762 | — | — | 0.9719 | Best F1 score |
| **German** | 53,468 | 5.05 | 95.92 | 0.9387 | — | — | 0.9658 | Training language |
| **Polish** | 16,040 | 9.24 | 94.61 | 0.9433 | — | — | 0.9460 | Moderate |
| **Ukrainian** | 16,817 | 7.81 | 93.85 | 0.9189 | — | — | 0.9470 | Moderate |
| **Russian** | 13,183 | **27.28** | 80.80 | 0.8408 | — | — | 0.7983 | Worst EER |

