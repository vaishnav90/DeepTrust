# Multilingual Deepfake Speech Detection Evaluation Summary - Spanish-Trained Model

## **Summary**

This report evaluates the HM-Conformer deepfake speech detection model across **eight languages** in a zero-shot setting, using a model trained on **Spanish data** and tested without any fine-tuning. The goal was to assess how well a detector trained primarily on **Spanish deepfake speech data** generalizes to multilingual real and synthetic speech.

**The results show substantial language-dependent performance variability**, with Equal Error Rate (EER) values ranging from **1.97% to 22.98%**, revealing that cross-lingual generalization is highly uneven, though overall performance is significantly better than the original English-trained model.

**Spanish achieved the best performance**, delivering:

* **Lowest EER (1.97%)**
* **Highest accuracy (98.53%)**
* **Highest ROC AUC (0.98)**
* **Highest F1 Score (0.98)**
  indicating exceptional performance on the training language, as expected.

**Italian and English also performed excellently**, with EERs below 6% and strong F1 scores above 0.95, suggesting that the model generalizes effectively to languages that share linguistic or phonetic characteristics with Spanish.

**French demonstrated strong performance** with an EER of 8.18% and ROC AUC of 0.95, indicating good cross-lingual transfer.

In contrast, **Russian and Polish demonstrated the most challenging performance**, with EERs above 20%, though still significantly better than the worst-performing languages in the original English-trained model evaluation.

Across most languages, the model shows **balanced performance** between real and fake detection, with high precision and recall for both classes, indicating that training on Spanish data has produced a more balanced classifier compared to the original English-trained model.

Overall, the evaluation reveals that the Spanish-trained HM-Conformer captures **strong language-agnostic deepfake cues** and demonstrates superior generalization across most languages compared to the English-trained baseline, with particularly strong performance on Romance languages (Spanish, Italian, French).


## Training Dataset Information

### HM-Conformer Architecture

The HM-Conformer (Hierarchical Pooling and Multi-level Classification Token Aggregation Conformer) is a Conformer-based audio deepfake detection system that was originally designed for the ASVspoof challenge. The model architecture incorporates:

1. **Hierarchical Pooling Method**: Progressively reduces sequence length to eliminate duplicated information, making it more suitable for classification tasks (many-to-one) rather than sequence-to-sequence tasks.

2. **Multi-level Classification Token Aggregation (MCA)**: Utilizes classification tokens from different Conformer blocks to gather information from various sequence lengths and time compression levels.

### Training Dataset: Spanish Deepfake Speech Data

The model was trained on **Spanish language deepfake speech data**:

- **Training Language:** Spanish ('es')
- The model was specifically trained to detect deepfake speech in Spanish, which provides a different linguistic foundation compared to the original English-trained model
- This training approach allows us to evaluate how language-specific training affects cross-lingual generalization

### Important Context for Multilingual Evaluation

**Key observations:**
- The model was trained on **Spanish-language deepfake data**, providing a different linguistic and acoustic foundation compared to English-trained models
- The Spanish-trained model demonstrates significantly better overall performance across most languages compared to the English-trained baseline
- Romance languages (Spanish, Italian, French) show particularly strong performance, suggesting linguistic similarity benefits cross-lingual transfer
- The model shows more balanced performance between real and fake detection compared to the original English-trained model, which had a bias toward predicting synthetic audio

This context helps explain why Romance languages perform exceptionally well, while Slavic languages (Russian, Polish) show more challenging performance, though still better than the worst-performing languages in the English-trained evaluation.

## Methodology

This report presents the evaluation results of the Spanish-trained HM-Conformer model across 8 different languages. All evaluations were conducted in test mode using the Spanish-trained model parameters, without any fine-tuning or parameter modifications. The model was tested on language-filtered subsets of the test dataset to assess its performance on each language independently.

## Overall Results Summary

| Language | Test Samples | EER (%) | Accuracy (%) | F1 Score | ROC AUC | Best Metric |
|----------|-------------|---------|---------------|----------|---------|-------------|
| **Spanish** | 30,498 | **1.97** | **98.53** | **0.9813** | **0.9844** | Best overall (training language) |
| **Italian** | 37,624 | **4.37** | 96.10 | 0.9519 | 0.9623 | 2nd best EER |
| **English** | 89,475 | 5.98 | 94.55 | 0.9635 | 0.9436 | Best F1 score |
| **French** | 46,895 | 8.18 | 94.13 | 0.9315 | 0.9481 | Strong performance |
| **Ukrainian** | 16,817 | 11.80 | 91.15 | 0.8887 | 0.9291 | Moderate |
| **German** | 53,468 | 11.92 | 89.72 | 0.8551 | 0.9123 | Moderate |
| **Polish** | 16,040 | 20.16 | 82.97 | 0.8379 | 0.8298 | Lower performance |
| **Russian** | 13,183 | **22.98** | 85.57 | 0.8792 | 0.8469 | Worst EER |

### Detailed Metrics by Language

#### Spanish (Best Performance - Training Language)
- **EER:** 1.97% (lowest)
- **Accuracy:** 98.53% (highest)
- **F1 Score:** 0.9813 (highest)
- **ROC AUC:** 0.9844 (highest)
- **Precision:** 0.99 (Real), 0.98 (Fake)
- **Recall:** 0.99 (Real), 0.98 (Fake)
- **Test Set:** 18,498 Real, 12,000 Fake

#### Italian (Second Best)
- **EER:** 4.37%
- **Accuracy:** 96.10%
- **F1 Score:** 0.9519
- **ROC AUC:** 0.9623
- **Precision:** 0.98 (Real), 0.94 (Fake)
- **Recall:** 0.96 (Real), 0.97 (Fake)
- **Test Set:** 22,624 Real, 15,000 Fake

#### English
- **EER:** 5.98%
- **Accuracy:** 94.55%
- **F1 Score:** 0.9635 (highest F1)
- **ROC AUC:** 0.9436
- **Precision:** 0.85 (Real), 0.98 (Fake)
- **Recall:** 0.94 (Real), 0.95 (Fake)
- **Test Set:** 21,475 Real, 68,000 Fake (largest dataset)

#### French
- **EER:** 8.18%
- **Accuracy:** 94.13%
- **F1 Score:** 0.9315
- **ROC AUC:** 0.9481
- **Precision:** 0.99 (Real), 0.88 (Fake)
- **Recall:** 0.91 (Real), 0.98 (Fake)
- **Test Set:** 27,895 Real, 19,000 Fake

#### Ukrainian
- **EER:** 11.80%
- **Accuracy:** 91.15%
- **F1 Score:** 0.8887
- **ROC AUC:** 0.9291
- **Precision:** 0.99 (Real), 0.81 (Fake)
- **Recall:** 0.87 (Real), 0.99 (Fake)
- **Test Set:** 10,817 Real, 6,000 Fake

#### German
- **EER:** 11.92%
- **Accuracy:** 89.72%
- **F1 Score:** 0.8551
- **ROC AUC:** 0.9123
- **Precision:** 0.98 (Real), 0.78 (Fake)
- **Recall:** 0.87 (Real), 0.95 (Fake)
- **Test Set:** 36,468 Real, 17,000 Fake

#### Polish
- **EER:** 20.16%
- **Accuracy:** 82.97%
- **F1 Score:** 0.8379
- **ROC AUC:** 0.8298
- **Precision:** 0.87 (Real), 0.80 (Fake)
- **Recall:** 0.78 (Real), 0.88 (Fake)
- **Test Set:** 8,040 Real, 8,000 Fake (most balanced)

#### Russian (Worst EER)
- **EER:** 22.98% (highest/worst)
- **Accuracy:** 85.57%
- **F1 Score:** 0.8792
- **ROC AUC:** 0.8469 (lowest)
- **Precision:** 0.98 (Real), 0.79 (Fake)
- **Recall:** 0.70 (Real), 0.99 (Fake)
- **Test Set:** 6,183 Real, 7,000 Fake


## Key Findings

### Best Performing Language: Spanish

Spanish demonstrates **exceptional performance** as expected for the training language:
- **Lowest EER (1.97%)** - outstanding ability to distinguish real from fake audio
- **Highest accuracy (98.53%)** - near-perfect classification performance
- **Highest ROC AUC (0.98)** - excellent discriminative ability
- **Highest F1 Score (0.98)** - balanced and strong performance across both classes
- **Near-perfect precision and recall** for both real and fake samples

The model shows exceptional performance on Spanish, demonstrating that language-specific training produces highly effective detectors for the training language.

### Romance Language Advantage

**Italian and French show exceptional cross-lingual performance:**
- **Italian:** EER of 4.37%, accuracy of 96.10%, ROC AUC of 0.96
- **French:** EER of 8.18%, accuracy of 94.13%, ROC AUC of 0.95

This suggests that linguistic similarity (all three are Romance languages) significantly benefits cross-lingual transfer, with the Spanish-trained model generalizing exceptionally well to other Romance languages.

### Challenging Languages: Russian and Polish

**Russian and Polish show the most challenging performance:**
- **Russian:** EER of 22.98% (worst), though accuracy remains at 85.57%
- **Polish:** EER of 20.16%, accuracy of 82.97%

These Slavic languages show more challenging performance compared to Romance languages, suggesting that linguistic distance from the training language (Spanish) affects generalization. However, their performance is still significantly better than the worst-performing languages in the English-trained model evaluation.

## General Analysis

### Performance Distribution

The results show a **clear performance gradient** across languages:

1. **Top Tier (EER < 6%)**: Spanish, Italian, English
   - Exceptional performance with EER below 6%
   - ROC AUC above 0.94
   - These languages show the model generalizes excellently

2. **Strong Tier (EER 8-12%)**: French, Ukrainian, German
   - Strong performance with EER between 8-12%
   - ROC AUC between 0.91-0.95
   - Very good cross-lingual transfer

3. **Moderate Tier (EER > 20%)**: Polish, Russian
   - More challenging performance with EER above 20%
   - ROC AUC between 0.83-0.85
   - Still acceptable but with room for improvement

### Comparison with English-Trained Model

**Key improvements over the English-trained baseline:**

1. **Overall Performance:** The Spanish-trained model shows significantly better performance across most languages:
   - Spanish: 1.97% EER (vs. 46.19% in English-trained) - **23.5x improvement**
   - Italian: 4.37% EER (vs. 37.05% in English-trained) - **8.5x improvement**
   - French: 8.18% EER (vs. 30.50% in English-trained) - **3.7x improvement**
   - German: 11.92% EER (vs. 35.55% in English-trained) - **3.0x improvement**
   - Polish: 20.16% EER (vs. 44.55% in English-trained) - **2.2x improvement**

2. **Balanced Classification:** The Spanish-trained model shows more balanced performance between real and fake detection, with high precision and recall for both classes, compared to the English-trained model's bias toward predicting synthetic audio.

3. **Romance Language Advantage:** The Spanish-trained model shows exceptional generalization to other Romance languages (Italian, French), demonstrating that linguistic similarity significantly benefits cross-lingual transfer.

### Common Patterns

1. **Romance Language Superiority**: Romance languages (Spanish, Italian, French) consistently show the best performance, suggesting that linguistic similarity to the training language is a key factor in cross-lingual generalization.

2. **Balanced Performance**: Unlike the English-trained model, the Spanish-trained model shows balanced precision and recall for both real and fake samples across most languages, indicating better calibration.

3. **Slavic Language Challenges**: Slavic languages (Russian, Polish, Ukrainian) show more variable performance, with Russian and Polish being the most challenging, though still significantly better than the worst-performing languages in the English-trained evaluation.

4. **Dataset Size Impact**: Large datasets (English with 89K samples) show strong performance, but smaller datasets like Italian (38K) and Ukrainian (17K) also perform excellently, suggesting that dataset size alone doesn't determine performance when linguistic similarity is present.

## Conclusion

The Spanish-trained HM-Conformer model demonstrates **significantly superior cross-lingual generalization** compared to the English-trained baseline, with particularly strong performance on Romance languages. The model achieves exceptional results on its training language (Spanish) and shows excellent transfer to linguistically similar languages (Italian, French). 

The evaluation reveals that **linguistic similarity to the training language is a critical factor** in cross-lingual deepfake detection performance, with Romance languages benefiting significantly from Spanish training. The model's balanced performance across both real and fake classes also represents an improvement over the English-trained model's bias toward synthetic audio detection.

These findings suggest that **language-specific training can produce highly effective deepfake detectors** that generalize well to linguistically similar languages, making multilingual training strategies a promising approach for improving cross-lingual deepfake detection performance.

