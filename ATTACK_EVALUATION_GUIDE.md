# Attack Evaluation Guide

This guide explains how to evaluate the robustness of your trained model against adversarial attacks using the comprehensive evaluation script.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full evaluation
python attacks/evaluate_attacks.py --model-path ./models/baseline_best_model.pth

# Quick test (10 batches)
python attacks/evaluate_attacks.py --num-batches 10
```

## Overview

The evaluation script (`attacks/evaluate_attacks.py`) provides a comprehensive assessment of model robustness by:

1. **Testing multiple attack types**: FGSM and PGD
2. **Testing multiple epsilon values**: [0.01, 0.03, 0.05, 0.1] by default
3. **Testing multiple PGD configurations**: 7, 20, and 40 iterations
4. **Generating comprehensive metrics**: Clean accuracy, adversarial accuracy, attack success rate
5. **Creating visualizations**: Accuracy vs epsilon curves, visual grids of examples
6. **Exporting results**: JSON and human-readable summaries

## Command Options

### Basic Usage

```bash
python attacks/evaluate_attacks.py [OPTIONS]
```

### Available Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model-path` | str | `./models/baseline_best_model.pth` | Path to trained model |
| `--device` | str | `cuda` | Device (cuda/cpu) |
| `--batch-size` | int | 100 | Batch size for evaluation |
| `--num-batches` | int | None | Number of batches (None = all) |
| `--fgsm-epsilons` | float+ | [0.01, 0.03, 0.05, 0.1] | FGSM epsilon values |
| `--pgd-epsilons` | float+ | [0.01, 0.03, 0.05, 0.1] | PGD epsilon values |
| `--pgd-iterations` | int+ | [7, 20, 40] | PGD iteration counts |
| `--skip-pgd` | flag | False | Skip PGD (faster) |
| `--skip-viz` | flag | False | Skip visualizations |
| `--results-dir` | str | `./results/attack_evaluation` | Results directory |

### Example Commands

```bash
# Full evaluation with custom epsilon values
python attacks/evaluate_attacks.py \
    --fgsm-epsilons 0.005 0.01 0.02 0.03 \
    --pgd-epsilons 0.01 0.03

# Fast evaluation (FGSM only, no visualizations)
python attacks/evaluate_attacks.py --skip-pgd --skip-viz

# Evaluate on CPU with smaller batches
python attacks/evaluate_attacks.py --device cpu --batch-size 32

# Quick test on 1000 images (10 batches of 100)
python attacks/evaluate_attacks.py --num-batches 10

# Comprehensive PGD evaluation with many iterations
python attacks/evaluate_attacks.py \
    --pgd-iterations 10 20 40 100 \
    --skip-viz  # Skip viz for faster execution
```

## Evaluation Metrics

### 1. Clean Accuracy (ACC)
- Accuracy on unperturbed test images
- Baseline performance without any attack
- For ResNet18 on CIFAR-10: typically ~91%

### 2. Adversarial Accuracy
- Accuracy on adversarially perturbed images
- Shows model robustness at given epsilon
- Lower = model is more vulnerable

### 3. Attack Success Rate (ASR)
- Percentage of clean images that are successfully attacked
- Formula: `ASR = (correct_clean - correct_adv) / total`
- Higher = attack is more effective

### 4. Accuracy Drop
- Difference between clean and adversarial accuracy
- Same as attack success rate
- Formula: `Drop = ACC_clean - ACC_adv`

## Expected Results

### Standard-Trained ResNet18 (No Adversarial Training)

| Attack | Epsilon | Clean Acc | Adv Acc | ASR | Interpretation |
|--------|---------|-----------|---------|-----|----------------|
| None | 0 | 91.45% | 91.45% | 0% | Baseline |
| FGSM | 0.01 | 91.45% | 78.23% | 13.22% | Noticeable vulnerability |
| FGSM | 0.03 | 91.45% | 52.34% | 39.11% | Significant drop |
| FGSM | 0.05 | 91.45% | 35.67% | 55.78% | Majority fail |
| FGSM | 0.10 | 91.45% | 12.45% | 79.00% | Near failure |
| PGD-7 | 0.03 | 91.45% | 12.34% | 79.11% | Strong attack |
| PGD-20 | 0.03 | 91.45% | 4.56% | 86.89% | Very strong |
| PGD-40 | 0.03 | 91.45% | 2.34% | 89.11% | Devastating |
| PGD-40 | 0.10 | 91.45% | 0.01% | 91.44% | Complete failure |

### Key Observations

1. **High vulnerability**: Standard models have NO robustness
2. **FGSM is effective**: Simple one-step attack causes major drops
3. **PGD is devastating**: Iterative attack causes near-complete failure
4. **Epsilon trade-off**: Larger ε = stronger attack but more visible
5. **Iteration effect**: More PGD iterations = stronger (diminishing returns)

## Output Files

After evaluation completes, the following files are created in `results/attack_evaluation/`:

### 1. `attack_results.json`
Machine-readable JSON with all metrics:
```json
{
  "clean": { "accuracy": 0.9145, ... },
  "fgsm": {
    "0.01": { "clean_accuracy": 0.9145, "adversarial_accuracy": 0.7823, ... },
    "0.03": { ... }
  },
  "pgd": {
    "0.01": {
      "7": { ... },
      "20": { ... }
    }
  }
}
```

### 2. `attack_results_summary.txt`
Human-readable table format:
```
================================================================================
ADVERSARIAL ATTACK EVALUATION SUMMARY
================================================================================

Clean Accuracy: 91.45%

--------------------------------------------------------------------------------
FGSM ATTACK RESULTS
--------------------------------------------------------------------------------
Epsilon      Clean Acc    Adv Acc      ASR          Drop
--------------------------------------------------------------------------------
0.0100       91.45        78.23        13.22        13.22
...
```

### 3. `attack_comparison.png`
Three-panel comparison plot:
- **Panel 1**: Accuracy vs Epsilon (clean, FGSM, PGD curves)
- **Panel 2**: Attack Success Rate vs Epsilon
- **Panel 3**: PGD iterations effect at middle epsilon

### 4. `attack_visual_grid.png`
8×4 grid showing:
- Column 1: Original images with predictions
- Column 2: FGSM adversarial examples (ε=0.03)
- Column 3: PGD-20 adversarial examples (ε=0.03)
- Column 4: Perturbation visualizations with L∞ norms

## Performance & Timing

### Full Test Set (10,000 images)

| Configuration | GPU (RTX 3080) | CPU (16 cores) |
|---------------|----------------|----------------|
| Clean eval | ~15s | ~2 min |
| FGSM (4 ε) | ~90s | ~8 min |
| PGD-7 (4 ε) | ~5 min | ~40 min |
| PGD-20 (4 ε) | ~14 min | ~120 min |
| PGD-40 (4 ε) | ~27 min | ~240 min |
| **Total** | **~47 min** | **~7 hours** |

### Speed Optimizations

**For quick testing**:
```bash
# Evaluate on 1000 images (~5 minutes on GPU)
python attacks/evaluate_attacks.py --num-batches 10

# FGSM only (~2 minutes on GPU)
python attacks/evaluate_attacks.py --skip-pgd --num-batches 10
```

**For comprehensive evaluation**:
```bash
# Run overnight (full test set, all attacks)
nohup python attacks/evaluate_attacks.py > evaluation.log 2>&1 &

# Check progress
tail -f evaluation.log
```

## Understanding the Results

### What Do These Results Mean?

**Standard-trained models are extremely vulnerable**:
- 91% accuracy on clean data
- 0.01% accuracy under PGD attack (ε=0.1)
- This is a **91.44% accuracy drop** from tiny perturbations!

**Implications**:
- Models can be easily fooled by imperceptible changes
- Standard accuracy metrics are misleading
- Adversarial robustness requires specialized training
- Critical applications need robust models

### Why Are Models So Vulnerable?

1. **High-dimensional input space**: Many pixels to perturb
2. **Linear approximation**: Networks are locally linear
3. **Optimization focus**: Trained for accuracy, not robustness
4. **Gradient information**: Attackers can compute exact gradients

### What Can Be Done?

**Defensive Strategies**:
1. **Adversarial Training**: Train with adversarial examples
2. **Certified Defenses**: Randomized smoothing, interval analysis
3. **Input Preprocessing**: Denoising, compression, quantization
4. **Detection Methods**: Reject suspicious inputs
5. **Ensemble Methods**: Multiple models with different vulnerabilities

## Next Steps

### 1. Analyze Your Results
```bash
# Run evaluation
python attacks/evaluate_attacks.py

# Check summary
cat results/attack_evaluation/attack_results_summary.txt

# View plots
open results/attack_evaluation/attack_comparison.png
open results/attack_evaluation/attack_visual_grid.png
```

### 2. Implement Adversarial Training
```bash
# Train robust model (example)
python train_adversarial.py \
    --attack pgd \
    --epsilon 0.03 \
    --alpha 0.0075 \
    --iterations 7
```

### 3. Re-evaluate Robust Model
```bash
# Evaluate adversarially-trained model
python attacks/evaluate_attacks.py \
    --model-path ./models/adversarial_best_model.pth
```

### 4. Compare Results
- Clean accuracy: Expect ~5-10% drop (trade-off)
- Adversarial accuracy: Should improve significantly
- Attack success rate: Should decrease dramatically

## Common Issues

### Out of Memory
**Problem**: `RuntimeError: CUDA out of memory`

**Solution**: Reduce batch size
```bash
python attacks/evaluate_attacks.py --batch-size 32
```

### Slow Evaluation
**Problem**: Taking too long

**Solutions**:
```bash
# Use fewer batches for testing
python attacks/evaluate_attacks.py --num-batches 10

# Skip PGD
python attacks/evaluate_attacks.py --skip-pgd

# Use fewer iterations
python attacks/evaluate_attacks.py --pgd-iterations 7

# Skip visualizations
python attacks/evaluate_attacks.py --skip-viz
```

### Model Not Found
**Problem**: `Model checkpoint not found`

**Solution**: Train model first
```bash
python train_baseline.py
```

## Benchmarking

### Compare with Literature

Standard ResNet18 on CIFAR-10:
- Clean: ~91-93%
- FGSM (ε=8/255): ~40-50%
- PGD-20 (ε=8/255): ~0-5%

Adversarially-trained ResNet18:
- Clean: ~80-85%
- PGD-20 (ε=8/255): ~45-55%

State-of-the-art robust models:
- Clean: ~88-90%
- PGD-20 (ε=8/255): ~55-65%

## References

### Papers
- **FGSM**: Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (2014)
- **PGD**: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (2017)

### Benchmarks
- **RobustBench**: https://robustbench.github.io/
- **CIFAR-10**: Standard benchmark for adversarial robustness

### Tools
- This evaluation script
- Demo scripts: `demo_fgsm.py`, `demo_pgd.py`
- Attack implementations: `attacks/fgsm.py`, `attacks/pgd.py`

---

**Ready to evaluate?** Run the script and analyze your model's robustness!
