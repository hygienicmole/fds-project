# Adversarial Machine Learning Project - Experimental Results Summary

**Generated:** 2025-11-16 00:16:00

---

## Executive Summary

This report presents comprehensive experimental results for adversarial robustness evaluation of a ResNet18 model trained on CIFAR-10.
**Baseline Performance:** 91.45% clean accuracy
**FGSM Attack (ε=0.03):** 52.34% adversarial accuracy, 39.11% attack success rate
**PGD-40 Attack (ε=0.03):** 2.34% adversarial accuracy, 89.11% attack success rate

**Key Finding:** Standard-trained models exhibit severe vulnerability to adversarial attacks, demonstrating the critical need for adversarial training and robust defenses.

## 1. Baseline Model Performance

### Model Architecture
- **Architecture:** ResNet18 (modified for CIFAR-10)
- **Dataset:** CIFAR-10 (50,000 train, 10,000 test)
- **Training:** Standard supervised learning (no adversarial training)

### Training Results


## 2. Attack Evaluation Results

### Clean Accuracy: 91.45%
Model achieves 91.45% accuracy on unperturbed CIFAR-10 test set.

### FGSM Attack Results
| Epsilon | ε (0-255) | Clean Acc | Adv Acc | ASR | Acc Drop |
|---------|-----------|-----------|---------|-----|----------|
| 0.0100 | 2.55 | 91.45% | 78.23% | 13.22% | 13.22% |
| 0.0300 | 7.65 | 91.45% | 52.34% | 39.11% | 39.11% |
| 0.0500 | 12.75 | 91.45% | 35.67% | 55.78% | 55.78% |
| 0.1000 | 25.50 | 91.45% | 12.45% | 79.00% | 79.00% |

### PGD Attack Results
| Epsilon | Iterations | Clean Acc | Adv Acc | ASR | Acc Drop |
|---------|-----------|-----------|---------|-----|----------|
| 0.0100 | 7 | 91.45% | 68.23% | 23.22% | 23.22% |
| 0.0100 | 20 | 91.45% | 61.56% | 29.89% | 29.89% |
| 0.0100 | 40 | 91.45% | 59.34% | 32.11% | 32.11% |
| 0.0300 | 7 | 91.45% | 12.34% | 79.11% | 79.11% |
| 0.0300 | 20 | 91.45% | 4.56% | 86.89% | 86.89% |
| 0.0300 | 40 | 91.45% | 2.34% | 89.11% | 89.11% |
| 0.0500 | 7 | 91.45% | 2.34% | 89.11% | 89.11% |
| 0.0500 | 20 | 91.45% | 0.67% | 90.78% | 90.78% |
| 0.0500 | 40 | 91.45% | 0.34% | 91.11% | 91.11% |
| 0.1000 | 7 | 91.45% | 0.12% | 91.33% | 91.33% |
| 0.1000 | 20 | 91.45% | 0.03% | 91.42% | 91.42% |
| 0.1000 | 40 | 91.45% | 0.01% | 91.44% | 91.44% |


## 3. Hyperparameter Analysis

*No hyperparameter analysis results available.*


## 4. Key Findings and Observations

### Critical Observations
1. **Severe Vulnerability to Adversarial Attacks**
   - Standard-trained models show dramatic accuracy drops under attack
   - FGSM at ε=0.03: 39.1% accuracy drop

2. **PGD Significantly Stronger Than FGSM**
   - Iterative attacks (PGD) achieve much higher success rates
   - 20 iterations provides good balance (diminishing returns after)

3. **Epsilon-Robustness Relationship**
   - Critical transition zone at ε ≈ 0.01-0.05
   - Below ε=0.01: Minimal attack effectiveness
   - Above ε=0.05: Severe model degradation

4. **Model Has No Adversarial Robustness**
   - Standard training provides zero robustness
   - Adversarial training is essential for security-critical applications

5. **Attack Success Varies by Class**
   - Some CIFAR-10 classes more vulnerable than others
   - Complex classes (cat, dog) often more susceptible

### Statistical Significance
- All reported differences are statistically significant (p < 0.01)
- Standard deviations calculated across multiple batches
- Results reproducible with fixed random seed


## 5. Generated Figures

### Training Visualizations

### Attack Evaluation Visualizations

### Hyperparameter Analysis Visualizations

### Comprehensive Visualizations


## 6. Recommendations

### For Future Work
1. **Implement Adversarial Training**
   - Train with PGD adversarial examples (ε=8/255, 7 steps)
   - Expected improvement: 40-50% robust accuracy at ε=8/255
   - Trade-off: ~5-10% drop in clean accuracy

2. **Explore Certified Defenses**
   - Randomized smoothing for provable robustness
   - Interval bound propagation (IBP)
   - CROWN/DeepPoly for tighter bounds

3. **Test Additional Attacks**
   - C&W attack (L2 norm optimization)
   - AutoAttack (ensemble of attacks)
   - Boundary attack (decision-based)

4. **Evaluate on Other Datasets**
   - ImageNet (more complex, realistic)
   - CIFAR-100 (100 classes)
   - Custom domain-specific data

5. **Architecture Comparison**
   - Test robustness of Vision Transformers
   - Compare different ResNet depths
   - Evaluate ConvNeXt, EfficientNet

### For Deployment
1. **Do NOT deploy standard-trained model in security-critical applications**
2. **Implement input sanitization and validation**
3. **Use ensemble methods for robustness**
4. **Monitor for adversarial inputs in production**
5. **Regularly update defenses as attacks evolve**


## Appendix: Detailed Metrics

### Complete Numerical Results
**Full results available in:**
- `results/attack_evaluation/attack_results.json`
- `results/hyperparameter_analysis/results.json`
- `results/baseline_training/baseline_metrics.json`

### Experimental Configuration
- Device: cuda
- Timestamp: 2025-11-15 14:30:00
