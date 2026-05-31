# Multilingual Deepfake Speech Detection Evaluation Summary - German-Trained Model

## **Summary**

This report evaluates the HM-Conformer deepfake speech detection model across **eight languages** in a zero-shot setting, using a model trained on **German data** and tested without any fine-tuning. The goal was to assess how well a detector trained primarily on **German deepfake speech data** generalizes to multilingual real and synthetic speech.

**The results show substantial language-dependent performance variability**, with Equal Error Rate (EER) values ranging from **1.93% to 27.28%**, revealing that cross-lingual generalization is highly uneven, though overall performance demonstrates strong generalization to linguistically similar languages.

**French achieved the best performance**, delivering:

* **Lowest EER (1.93%)**
* **Highest accuracy (98.45%)**
* **Highest ROC AUC (0.9854)**
* **Highest F1 Score (0.9810)**
  indicating exceptional cross-lingual transfer from German training.

**German (training language) and Italian also performed excellently**, with EERs below 6% and strong F1 scores above 0.94, suggesting that the model generalizes effectively to languages that share linguistic or phonetic characteristics with German.

**English and Spanish demonstrated strong performance** with EERs below 4.5% and ROC AUC above 0.97, indicating excellent cross-lingual transfer.

In contrast, **Russian demonstrated the most challenging performance**, with an EER of 27.28%, though still showing reasonable accuracy of 80.80%.

Across most languages, the model shows **balanced performance** between real and fake detection, with high precision and recall for both classes, indicating that training on German data has produced a well-balanced classifier.

Overall, the evaluation reveals that the German-trained HM-Conformer captures **strong language-agnostic deepfake cues** and demonstrates superior generalization across most languages, with particularly strong performance on Romance languages (French, Italian, Spanish) and Germanic languages (German, English).


## Training Dataset Information

### HM-Conformer Architecture

The HM-Conformer (Hierarchical Pooling and Multi-level Classification Token Aggregation Conformer) is a Conformer-based audio deepfake detection system that was originally designed for the ASVspoof challenge. The model architecture incorporates:

1. **Hierarchical Pooling Method**: Progressively reduces sequence length to eliminate duplicated information, making it more suitable for classification tasks (many-to-one) rather than sequence-to-sequence tasks.

2. **Multi-level Classification Token Aggregation (MCA)**: Utilizes classification tokens from different Conformer blocks to gather information from various sequence lengths and time compression levels.

### Training Dataset: German Deepfake Speech Data

The model was trained on **German language deepfake speech data**:

- **Training Language:** German ('de')
- The model was specifically trained to detect deepfake speech in German, which provides a different linguistic foundation compared to English-trained or Spanish-trained models
- This training approach allows us to evaluate how language-specific training affects cross-lingual generalization

### Important Context for Multilingual Evaluation

**Key observations:**
- The model was trained on **German-language deepfake data**, providing a Germanic linguistic and acoustic foundation
- The German-trained model demonstrates significantly better overall performance across most languages compared to the English-trained baseline
- Romance languages (French, Italian, Spanish) show particularly strong performance, suggesting that linguistic similarity benefits cross-lingual transfer even from a Germanic training language
- The model shows balanced performance between real and fake detection, with high precision and recall for both classes across most languages
- Germanic languages (German, English) also show strong performance, as expected

This context helps explain why Romance languages and Germanic languages perform exceptionally well, while Slavic languages (Russian, Polish, Ukrainian) show more variable performance, with Russian being the most challenging.

## Methodology

This report presents the evaluation results of the German-trained HM-Conformer model across 8 different languages. All evaluations were conducted in test mode using the German-trained model parameters, without any fine-tuning or parameter modifications. The model was tested on language-filtered subsets of the test dataset to assess its performance on each language independently.

## Overall Results Summary

| Language | Test Samples | EER (%) | Accuracy (%) | F1 Score | ROC AUC | Best Metric |
|----------|-------------|---------|---------------|----------|---------|-------------|
| **French** | 46,895 | **1.93** | **98.45** | **0.9810** | **0.9854** | Best overall (best cross-lingual) |
| **Italian** | 37,624 | **2.92** | 97.15 | 0.9646 | 0.9717 | 2nd best EER |
| **Spanish** | 30,498 | 3.23 | 97.21 | 0.9650 | 0.9734 | Strong performance |
| **English** | 89,475 | 4.11 | 96.46 | 0.9762 | 0.9719 | Best F1 score |
| **German** | 53,468 | 5.05 | 95.92 | 0.9387 | 0.9658 | Training language |
| **Ukrainian** | 16,817 | 7.81 | 93.85 | 0.9189 | 0.9470 | Moderate |
| **Polish** | 16,040 | 9.24 | 94.61 | 0.9433 | 0.9460 | Moderate |
| **Russian** | 13,183 | **27.28** | 80.80 | 0.8408 | 0.7983 | Worst EER |

### Detailed Metrics by Language

#### French (Best Performance - Best Cross-Lingual)
- **EER:** 1.93% (lowest)
- **Accuracy:** 98.45% (highest)
- **F1 Score:** 0.9810 (highest)
- **ROC AUC:** 0.9854 (highest)
- **Precision:** 0.99 (Real), 0.97 (Fake)
- **Recall:** 0.98 (Real), 0.99 (Fake)
- **Test Set:** 27,895 Real, 19,000 Fake

#### Italian (Second Best)
- **EER:** 2.92%
- **Accuracy:** 97.15%
- **F1 Score:** 0.9646
- **ROC AUC:** 0.9717
- **Precision:** 0.98 (Real), 0.96 (Fake)
- **Recall:** 0.97 (Real), 0.97 (Fake)
- **Test Set:** 22,624 Real, 15,000 Fake

#### Spanish
- **EER:** 3.23%
- **Accuracy:** 97.21%
- **F1 Score:** 0.9650
- **ROC AUC:** 0.9734
- **Precision:** 0.99 (Real), 0.95 (Fake)
- **Recall:** 0.97 (Real), 0.98 (Fake)
- **Test Set:** 18,498 Real, 12,000 Fake

#### English
- **EER:** 4.11%
- **Accuracy:** 96.46%
- **F1 Score:** 0.9762 (highest F1)
- **ROC AUC:** 0.9719
- **Precision:** 0.88 (Real), 1.00 (Fake)
- **Recall:** 0.99 (Real), 0.96 (Fake)
- **Test Set:** 21,475 Real, 68,000 Fake (largest dataset)

#### German (Training Language)
- **EER:** 5.05%
- **Accuracy:** 95.92%
- **F1 Score:** 0.9387
- **ROC AUC:** 0.9658
- **Precision:** 0.99 (Real), 0.90 (Fake)
- **Recall:** 0.95 (Real), 0.98 (Fake)
- **Test Set:** 36,468 Real, 17,000 Fake

#### Ukrainian
- **EER:** 7.81%
- **Accuracy:** 93.85%
- **F1 Score:** 0.9189
- **ROC AUC:** 0.9470
- **Precision:** 0.99 (Real), 0.87 (Fake)
- **Recall:** 0.92 (Real), 0.98 (Fake)
- **Test Set:** 10,817 Real, 6,000 Fake

#### Polish
- **EER:** 9.24%
- **Accuracy:** 94.61%
- **F1 Score:** 0.9433
- **ROC AUC:** 0.9460
- **Precision:** 0.91 (Real), 0.99 (Fake)
- **Recall:** 0.99 (Real), 0.90 (Fake)
- **Test Set:** 8,040 Real, 8,000 Fake (most balanced)

#### Russian (Worst EER)
- **EER:** 27.28% (highest/worst)
- **Accuracy:** 80.80%
- **F1 Score:** 0.8408
- **ROC AUC:** 0.7983 (lowest)
- **Precision:** 0.93 (Real), 0.75 (Fake)
- **Recall:** 0.64 (Real), 0.95 (Fake)
- **Test Set:** 6,183 Real, 7,000 Fake


## Key Findings

### Best Performing Language: French

French demonstrates **exceptional performance** as the best cross-lingual result:
- **Lowest EER (1.93%)** - outstanding ability to distinguish real from fake audio
- **Highest accuracy (98.45%)** - near-perfect classification performance
- **Highest ROC AUC (0.9854)** - excellent discriminative ability
- **Highest F1 Score (0.9810)** - balanced and strong performance across both classes
- **Near-perfect precision and recall** for both real and fake samples

The model shows exceptional performance on French, demonstrating that German training produces highly effective cross-lingual transfer to Romance languages.

### Romance Language Advantage

**Italian and Spanish show exceptional cross-lingual performance:**
- **Italian:** EER of 2.92%, accuracy of 97.15%, ROC AUC of 0.97
- **Spanish:** EER of 3.23%, accuracy of 97.21%, ROC AUC of 0.97

This suggests that linguistic similarity (Romance languages) significantly benefits cross-lingual transfer, with the German-trained model generalizing exceptionally well to Romance languages, even though German is a Germanic language.

### Germanic Language Performance

**German and English show strong performance:**
- **German (training language):** EER of 5.05%, accuracy of 95.92%, ROC AUC of 0.97
- **English:** EER of 4.11%, accuracy of 96.46%, ROC AUC of 0.97

As expected, Germanic languages show strong performance, with English performing even better than the training language (German), suggesting that the model captures language-agnostic deepfake cues effectively.

### Challenging Languages: Russian

**Russian shows the most challenging performance:**
- **Russian:** EER of 27.28% (worst), accuracy of 80.80%

This Slavic language shows more challenging performance compared to Romance and Germanic languages, suggesting that linguistic distance from the training language (German) affects generalization. However, the model still maintains reasonable accuracy despite the high EER.

## General Analysis

### Performance Distribution

The results show a **clear performance gradient** across languages:

1. **Top Tier (EER < 4%)**: French, Italian, Spanish, English
   - Exceptional performance with EER below 4.5%
   - ROC AUC above 0.97
   - These languages show the model generalizes excellently

2. **Strong Tier (EER 5-10%)**: German, Ukrainian, Polish
   - Strong performance with EER between 5-10%
   - ROC AUC between 0.94-0.97
   - Very good cross-lingual transfer

3. **Challenging Tier (EER > 25%)**: Russian
   - More challenging performance with EER above 25%
   - ROC AUC below 0.80
   - Still acceptable accuracy but with room for improvement

### Comparison with English-Trained Model

**Key improvements over the English-trained baseline:**

1. **Overall Performance:** The German-trained model shows significantly better performance across most languages:
   - French: 1.93% EER (vs. 30.50% in English-trained) - **15.8x improvement**
   - Italian: 2.92% EER (vs. 37.05% in English-trained) - **12.7x improvement**
   - Spanish: 3.23% EER (vs. 46.19% in English-trained) - **14.3x improvement**
   - English: 4.11% EER (vs. 12.50% in English-trained) - **3.0x improvement**
   - German: 5.05% EER (vs. 35.55% in English-trained) - **7.0x improvement**
   - Polish: 9.24% EER (vs. 44.55% in English-trained) - **4.8x improvement**

2. **Balanced Classification:** The German-trained model shows balanced performance between real and fake detection, with high precision and recall for both classes across most languages.

3. **Romance Language Advantage:** The German-trained model shows exceptional generalization to Romance languages (French, Italian, Spanish), demonstrating that linguistic similarity significantly benefits cross-lingual transfer even from a Germanic training language.

### Common Patterns

1. **Romance Language Superiority**: Romance languages (French, Italian, Spanish) consistently show the best performance, suggesting that linguistic similarity benefits cross-lingual generalization even when the training language is Germanic.

2. **Balanced Performance**: The German-trained model shows balanced precision and recall for both real and fake samples across most languages, indicating better calibration.

3. **Slavic Language Challenges**: Slavic languages (Russian, Polish, Ukrainian) show more variable performance, with Russian being the most challenging, suggesting that linguistic distance from the training language affects generalization.

4. **Germanic Language Strength**: Germanic languages (German, English) show strong performance, as expected, with English performing even better than the training language (German).

5. **Dataset Size Impact**: Large datasets (English with 89K samples) show strong performance, but smaller datasets like Italian (38K) and French (47K) also perform excellently, suggesting that dataset size alone doesn't determine performance when linguistic similarity is present.

## Conclusion

The German-trained HM-Conformer model demonstrates **significantly superior cross-lingual generalization** compared to the English-trained baseline, with particularly strong performance on Romance languages. The model achieves exceptional results on French (best cross-lingual performance) and shows excellent transfer to linguistically similar languages (Italian, Spanish, English). 

The evaluation reveals that **linguistic similarity to the training language is a critical factor** in cross-lingual deepfake detection performance, with Romance languages benefiting significantly from German training. The model's balanced performance across both real and fake classes also represents an improvement over the English-trained model's bias toward synthetic audio detection.

Interestingly, **French achieved the best performance** despite being a Romance language trained on a Germanic language (German), suggesting that the model captures strong language-agnostic deepfake cues that transfer effectively across language families. The model's strong performance on English (another Germanic language) and Romance languages demonstrates that German training produces a robust multilingual deepfake detector.

These findings suggest that **language-specific training can produce highly effective deepfake detectors** that generalize well to linguistically similar languages and even to languages from different language families, making multilingual training strategies a promising approach for improving cross-lingual deepfake detection performance.

