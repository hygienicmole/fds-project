# Comprehensive Visualization Guide

This guide explains all the advanced visualizations available for analyzing adversarial attacks.

## Overview

The project includes comprehensive visualization tools for deep analysis of adversarial attacks, providing insights into:
- Attack effectiveness and failure modes
- Perturbation characteristics
- Model behavior under attack
- Class-specific vulnerabilities
- Confidence degradation

## Quick Start

```bash
# Generate all visualizations for FGSM attack
python generate_visualizations.py --model-path ./models/baseline_best_model.pth

# Generate visualizations for PGD attack
python generate_visualizations.py --attack pgd --epsilon 0.03

# Custom configuration
python generate_visualizations.py \
    --attack fgsm \
    --epsilon 0.05 \
    --num-samples 2000 \
    --save-dir ./results/my_analysis
```

## Generated Visualizations

### 1. Comprehensive Comparison (20 samples)

**File**: `{attack}_comparison.png`

**What it shows**:
A 4×5 grid (20 samples) where each sample displays:
- **Left**: Original image with true label, prediction, and confidence
- **Middle**: Adversarial image with prediction and confidence
- **Right**: Perturbation heatmap showing attack magnitude

**Format**: Each row contains 4 samples, each sample has 3 sub-images

**Color coding**:
- Green title = Correct prediction
- Red title = Incorrect prediction (successful attack!)

**Example interpretation**:
```
Sample 1:
  Original: "cat" (predicted: cat, confidence: 0.95) [GREEN]
  Adversarial: (predicted: dog, confidence: 0.78) [RED]
  Heatmap: Shows where perturbations were applied (hot spots in red)
```

**Key insights**:
- Visual similarity between original and adversarial images
- How predictions change despite imperceptible differences
- Spatial distribution of perturbations
- Confidence scores before and after attack

**Expected observations**:
- Adversarial images look nearly identical to originals
- Predictions change drastically (successful attacks shown in red)
- Perturbations are small but strategic
- Confidence often remains high even for wrong predictions

---

### 2. Perturbation Heatmaps (10 samples)

**File**: `{attack}_perturbation_heatmaps.png`

**What it shows**:
Detailed 4-column visualization for 10 samples:
- **Column 1**: Original image
- **Column 2**: Adversarial image
- **Column 3**: Perturbation heatmap (L2 norm, with colorbar)
- **Column 4**: Amplified perturbation (10× magnification for visibility)

**Purpose**: Deep dive into perturbation characteristics

**Heatmap interpretation**:
- **Hot colors (red/yellow)**: Large perturbations
- **Cool colors (blue/black)**: Small/no perturbations
- **Colorbar**: Shows actual perturbation magnitude
- **Max value**: Largest single-pixel perturbation

**Amplified perturbation shows**:
- RGB perturbation pattern (which channels are perturbed)
- Spatial structure of the attack
- Whether perturbations are uniform or localized

**Expected observations**:
- FGSM: Perturbations more uniform across image
- PGD: Perturbations may be more concentrated in critical regions
- Max perturbation ≈ epsilon value (L∞ constraint)
- Most pixels have small perturbations, few have large ones

---

### 3. Confusion Matrix - Clean Predictions

**File**: `confusion_matrix_clean.png`

**What it shows**:
10×10 normalized confusion matrix for model predictions on clean images

**Interpretation**:
- **Rows**: True labels
- **Columns**: Predicted labels
- **Diagonal**: Correct predictions (should be high, ~0.9 for good model)
- **Off-diagonal**: Misclassifications (should be low)

**Color scale**:
- **Dark blue**: High proportion (good for diagonal)
- **Light blue/white**: Low proportion (good for off-diagonal)

**Purpose**: Baseline model performance without attacks

**Expected pattern**:
- Strong diagonal (dark blue stripe)
- Light off-diagonal elements
- Some classes may confuse each other naturally (e.g., cat/dog)

**Key metrics**:
- Row sums = 1.0 (normalized)
- Diagonal values ≈ 0.85-0.95 for well-trained model
- Total accuracy = average of diagonal

---

### 4. Confusion Matrix - Adversarial Predictions

**File**: `confusion_matrix_{attack}.png`

**What it shows**:
10×10 normalized confusion matrix for predictions on adversarial examples

**Interpretation**:
- Shows where adversarial examples are misclassified
- Reveals attack patterns and failure modes
- Identifies which classes are confused under attack

**Expected pattern**:
- **Weak diagonal** (attacks succeed)
- **Spread out off-diagonal** (predictions scattered)
- May show preferential misclassification patterns

**Comparison with clean matrix**:
- Diagonal values drop significantly (accuracy loss)
- Off-diagonal becomes more uniform (random-like errors)
- Some class pairs may still be more confused

**Example insights**:
```
If cat (row 3) has high values in dog (col 5) and bird (col 2):
→ Cat images are often misclassified as dogs or birds under attack
→ These are the nearest classes in feature space
```

---

### 5. Class-wise Attack Success Rates

**File**: `{attack}_class_wise_success.png`

**What it shows**:
Two-panel visualization showing per-class vulnerability:

**Top panel**: Grouped bar chart
- Green bars: Clean accuracy per class
- Red bars: Adversarial accuracy per class
- X-axis: 10 CIFAR-10 classes
- Y-axis: Accuracy (0-100%)
- Value labels on each bar

**Bottom panel**: Single bar chart
- Orange bars: Attack success rate per class
- Red dashed line: Mean ASR across all classes
- Shows which classes are most vulnerable

**Purpose**: Identify class-specific vulnerabilities

**Expected patterns**:
- Some classes are more vulnerable than others
- Complex classes (e.g., "dog", "cat") may have lower adversarial accuracy
- Simpler classes (e.g., "airplane", "ship") may be more robust
- ASR varies significantly across classes (20-90% range)

**Example interpretation**:
```
Class "cat":
  Clean accuracy: 92%
  Adversarial accuracy: 15%
  Attack success rate: 77%
  → Cats are highly vulnerable to this attack

Class "airplane":
  Clean accuracy: 95%
  Adversarial accuracy: 60%
  Attack success rate: 35%
  → Airplanes are more robust
```

**Key insights**:
- Which classes need better defense
- Whether attack exploits specific class characteristics
- Correlation between clean accuracy and robustness

---

### 6. Confidence Score Distributions

**File**: `{attack}_confidence_distributions.png`

**What it shows**:
Four-panel analysis of prediction confidence:

**Top-left**: Histogram of maximum confidence
- Green: Clean predictions
- Red: Adversarial predictions
- Shows overall confidence distribution
- Dashed lines: Mean confidence for each

**Top-right**: Box plot comparison
- Side-by-side comparison of confidence distributions
- Shows median, quartiles, outliers
- Green vs Red: Clean vs Adversarial

**Bottom-left**: Histogram of confidence for true class
- How confident the model is in the correct class
- Green: Clean (should be high)
- Red: Adversarial (expected to drop)

**Bottom-right**: Correct vs Incorrect predictions
- Four box plots comparing:
  1. Clean (Correct): High confidence expected
  2. Clean (Incorrect): Lower confidence
  3. Adv (Correct): Models that survived attack
  4. Adv (Incorrect): Successful attacks

**Purpose**: Understanding confidence calibration under attack

**Expected observations**:

**Clean predictions**:
- High confidence (mean ≈ 0.85-0.95)
- Peak near 1.0 (very confident correct predictions)
- Narrow distribution (consistent confidence)

**Adversarial predictions**:
- Lower average confidence (mean ≈ 0.60-0.80)
- Broader distribution (more uncertainty)
- But often still confident even when wrong!

**Key insights**:
- **Confidence paradox**: Model is confident in wrong predictions
- **Calibration failure**: High confidence doesn't guarantee correctness
- **Attack effect**: Confidence for true class drops dramatically
- **Successful attacks**: Often have moderate-high confidence in wrong class

**Example numbers** (FGSM ε=0.03):
```
Clean predictions:
  Mean confidence: 0.892
  Std: 0.125

Adversarial predictions:
  Mean confidence: 0.714
  Std: 0.234
  Confidence drop: 0.178

Confidence for true class:
  Clean: 0.876
  Adversarial: 0.234  (drops 73%!)
```

---

## Usage Examples

### Basic Usage

```bash
# FGSM with default settings (ε=0.03, 1000 samples)
python generate_visualizations.py

# PGD with custom epsilon
python generate_visualizations.py --attack pgd --epsilon 0.05

# More samples for better statistics
python generate_visualizations.py --num-samples 5000
```

### Advanced Usage

```bash
# PGD with custom parameters
python generate_visualizations.py \
    --attack pgd \
    --epsilon 0.03 \
    --pgd-alpha 0.0075 \
    --pgd-iterations 40 \
    --num-samples 2000

# FGSM with very small epsilon
python generate_visualizations.py \
    --attack fgsm \
    --epsilon 0.01 \
    --num-samples 10000 \
    --save-dir ./results/fgsm_small_epsilon
```

### Using in Python Code

```python
from utils.visualization import generate_all_visualizations

# Generate all visualizations programmatically
results = generate_all_visualizations(
    model=model,
    original_images=clean_images,
    adversarial_images=adv_images,
    true_labels=labels,
    attack_name='FGSM',
    epsilon=0.03,
    save_dir='./my_results'
)

# Access individual visualization paths
comparison_path = results['paths']['comparison']
heatmap_path = results['paths']['heatmaps']

# Access statistics
class_stats = results['class_statistics']
most_vulnerable_class = class_stats['attack_success_rate'].argmax()
print(f"Most vulnerable: {class_stats['class_names'][most_vulnerable_class]}")
```

### Individual Visualizations

```python
from utils.visualization import (
    visualize_comprehensive_comparison,
    visualize_perturbation_heatmaps,
    plot_confusion_matrix,
    plot_class_wise_attack_success,
    plot_confidence_distributions
)

# Generate only comparison
visualize_comprehensive_comparison(
    model, original_images, adversarial_images, labels,
    num_samples=20,
    save_path='./comparison.png'
)

# Generate only confusion matrix
plot_confusion_matrix(
    model, adversarial_images, labels,
    adversarial=True,
    save_path='./confusion_adv.png'
)

# Generate only class-wise analysis
stats = plot_class_wise_attack_success(
    model, original_images, adversarial_images, labels,
    save_path='./class_wise.png'
)

# Print statistics
print(f"Mean ASR: {stats['attack_success_rate'].mean():.1f}%")
```

## Output Files

After running, you'll find in the specified directory (default: `./results/visualizations/`):

```
results/visualizations/
├── fgsm_comparison.png                    # 20-sample comparison grid
├── fgsm_perturbation_heatmaps.png        # Detailed heatmaps
├── confusion_matrix_clean.png             # Clean predictions
├── confusion_matrix_fgsm.png             # Adversarial predictions
├── fgsm_class_wise_success.png           # Per-class vulnerability
└── fgsm_confidence_distributions.png     # Confidence analysis
```

For PGD:
```
results/visualizations/
├── pgd-20_comparison.png
├── pgd-20_perturbation_heatmaps.png
├── confusion_matrix_pgd-20.png
├── pgd-20_class_wise_success.png
└── pgd-20_confidence_distributions.png
```

## Interpretation Guide

### What Makes a Good Visualization?

**Comparison Grid**:
- Clear labeling (easy to read predictions)
- Good contrast in heatmaps
- Visible but subtle perturbations

**Confusion Matrices**:
- Clean: Strong diagonal, weak off-diagonal
- Adversarial: Weakened diagonal, scattered predictions
- Clear difference between clean and adversarial

**Class-wise Success**:
- Varied ASR across classes (reveals vulnerabilities)
- Correlation patterns (which classes are similar)
- Actionable insights for defense

**Confidence Distributions**:
- Clear separation between clean and adversarial
- Shows confidence-correctness mismatch
- Reveals calibration issues

### Common Patterns

**Successful FGSM Attack (ε=0.03)**:
- ~40-60% attack success rate
- Confidence drops by ~0.15-0.20
- Confusion matrix shows scattered predictions
- Some classes much more vulnerable than others

**Successful PGD Attack (ε=0.03, 20 iterations)**:
- ~80-90% attack success rate
- Confidence drops by ~0.60-0.70 for true class
- Confusion matrix nearly uniform (random errors)
- Almost all classes highly vulnerable

**Very Small Epsilon (ε=0.001)**:
- Low attack success (~5-10%)
- Minimal confidence change
- Confusion matrix still shows strong diagonal
- Only most vulnerable classes affected

**Very Large Epsilon (ε=0.1)**:
- Near-complete attack success (>95%)
- Large confidence drop
- Completely scattered predictions
- Perturbations may be visible to humans

## Troubleshooting

### Visualizations Look Strange

**Problem**: Heatmaps are all black or all white

**Cause**: Perturbation magnitude too small/large for colormap

**Solution**: Adjust amplification factor in code or check epsilon value

---

**Problem**: All predictions shown as incorrect (all red)

**Cause**: Model not properly loaded or very strong attack

**Solution**: Verify model checkpoint exists and loads correctly

---

**Problem**: Confusion matrix has no clear pattern

**Cause**: Untrained model or completely successful attack

**Solution**: Check model training status or reduce epsilon

### Generation Errors

**Problem**: Out of memory

**Solution**: Reduce `--num-samples` or use smaller batch size

---

**Problem**: Slow generation

**Solution**: PGD is inherently slow. Use `--attack fgsm` for faster generation, or reduce `--pgd-iterations`

---

**Problem**: File not found errors

**Solution**: Ensure model checkpoint exists at specified path

## Scientific Value

These visualizations enable:

1. **Quantitative Analysis**: Exact metrics for attack effectiveness
2. **Qualitative Understanding**: Visual inspection of failure modes
3. **Comparison**: Side-by-side attack strength comparison
4. **Class-specific Insights**: Identify vulnerable classes
5. **Confidence Calibration**: Understand prediction reliability
6. **Defense Evaluation**: Measure improvement after defenses
7. **Publication Quality**: High-resolution images for papers

## Best Practices

### For Analysis
- Generate visualizations with multiple epsilon values
- Compare FGSM and PGD side-by-side
- Use sufficient samples (≥1000) for reliable statistics
- Document all parameters used

### For Presentations
- Use high DPI (300) for publication
- Include comparison grids to show attack effect
- Use confusion matrices to quantify degradation
- Show confidence distributions to highlight calibration issues

### For Defense Evaluation
- Generate visualizations before and after defense
- Compare attack success rates
- Show improvements in confusion matrices
- Demonstrate confidence calibration improvements

## References

- Goodfellow et al. (2014): "Explaining and Harnessing Adversarial Examples"
- Madry et al. (2017): "Towards Deep Learning Models Resistant to Adversarial Attacks"
- Carlini & Wagner (2017): "Towards Evaluating the Robustness of Neural Networks"

---

**Ready to visualize?** Run `python generate_visualizations.py` and explore your model's vulnerabilities!
