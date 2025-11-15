# Adversarial Attacks Module

This module implements state-of-the-art adversarial attacks for evaluating model robustness.

## Overview

Adversarial attacks create slightly perturbed inputs that fool machine learning models while appearing unchanged to humans. This module provides comprehensive implementations with detailed mathematical documentation.

## Implemented Attacks

### 1. FGSM (Fast Gradient Sign Method)

**File**: `fgsm.py`

**Description**: A simple yet effective one-step adversarial attack that perturbs inputs by taking a step in the direction of the gradient sign.

**Key Features**:
- ✅ Untargeted attacks (cause any misclassification)
- ✅ Targeted attacks (force specific misclassification)
- ✅ Perturbation analysis (L0, L1, L2, L∞ norms)
- ✅ Attack success evaluation
- ✅ Comprehensive visualizations
- ✅ Batch processing support
- ✅ Detailed mathematical documentation

**Formula** (Untargeted):
```
x_adv = x + ε · sign(∇_x J(θ, x, y))
```

**Formula** (Targeted):
```
x_adv = x - ε · sign(∇_x J(θ, x, y_target))
```

**Usage**:
```python
from attacks import FGSM

# Create attack
fgsm = FGSM(model, epsilon=8/255, targeted=False)

# Generate adversarial examples
x_adv = fgsm.generate(images, labels)

# Analyze perturbation
stats = fgsm.analyze_perturbation(images, x_adv)

# Evaluate success
results = fgsm.evaluate_attack_success(images, labels, x_adv)
```

**Reference**: Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (2014)
- Paper: https://arxiv.org/abs/1412.6572

### 2. PGD (Projected Gradient Descent)

**File**: `pgd.py`

**Description**: An iterative adversarial attack that applies multiple steps of FGSM with projection back to the epsilon ball.

**Key Features**:
- Stronger than FGSM
- Random initialization
- Configurable iterations
- L∞ norm constraint

**Formula**:
```
x^0 = x + random_noise
for t = 0 to T-1:
    x^(t+1) = Π_(x+S)(x^t + α · sign(∇_x J(θ, x^t, y)))
```

**Usage**:
```python
from attacks import PGD

# Create attack
pgd = PGD(model, epsilon=8/255, alpha=2/255, iterations=20)

# Generate adversarial examples
x_adv = pgd.generate(images, labels)
```

**Reference**: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (2017)
- Paper: https://arxiv.org/abs/1706.06083

## Quick Start

### Basic Usage

```python
import torch
from attacks import FGSM, PGD
from models import get_resnet18, load_checkpoint

# Load model
model = get_resnet18(num_classes=10, device='cuda')
load_checkpoint('./models/baseline_best_model.pth', model)

# Load data
from utils import get_cifar10_loaders
_, test_loader = get_cifar10_loaders()
images, labels = next(iter(test_loader))
images, labels = images.to('cuda'), labels.to('cuda')

# FGSM Attack
fgsm = FGSM(model, epsilon=8/255)
x_adv_fgsm = fgsm.generate(images, labels)

# PGD Attack
pgd = PGD(model, epsilon=8/255, alpha=2/255, iterations=20)
x_adv_pgd = pgd.generate(images, labels)
```

### Visualization

```python
from attacks.fgsm import visualize_fgsm_attack
from utils import CIFAR10_CLASSES
import config

# Denormalize function
def denormalize_cifar10(tensor):
    mean = torch.tensor(config.MEAN).view(3, 1, 1)
    std = torch.tensor(config.STD).view(3, 1, 1)
    return tensor * std + mean

# Visualize
visualize_fgsm_attack(
    model=model,
    images=images,
    labels=labels,
    epsilon=8/255,
    class_names=CIFAR10_CLASSES,
    device='cuda',
    num_samples=5,
    denormalize_fn=denormalize_cifar10,
    save_path='./results/fgsm_visualization.png'
)
```

## FGSM API Reference

### Class: `FGSM`

Main FGSM attack class with comprehensive features.

#### Constructor

```python
FGSM(model, epsilon=0.03, clip_min=0.0, clip_max=1.0, targeted=False)
```

**Parameters**:
- `model` (nn.Module): Target model to attack
- `epsilon` (float): Maximum perturbation magnitude (L∞ norm)
- `clip_min` (float): Minimum pixel value for clipping
- `clip_max` (float): Maximum pixel value for clipping
- `targeted` (bool): Whether to perform targeted attack

#### Methods

**`generate(x, y, return_perturbation=False)`**

Generate adversarial examples.

**Parameters**:
- `x` (Tensor): Input images (batch_size, C, H, W)
- `y` (Tensor): Labels (true labels for untargeted, target labels for targeted)
- `return_perturbation` (bool): If True, also return perturbation δ

**Returns**:
- `x_adv` (Tensor): Adversarial examples
- `perturbation` (Tensor, optional): Perturbation δ = x_adv - x

**`analyze_perturbation(x, x_adv, verbose=True)`**

Analyze perturbation statistics.

**Returns**:
- Dictionary with:
  - `l0_norm`: Number of changed pixels
  - `l1_norm`: Sum of absolute changes
  - `l2_norm`: Euclidean distance
  - `linf_norm`: Maximum per-pixel change
  - `mean_abs_perturbation`: Mean magnitude
  - `std_perturbation`: Standard deviation
  - `pct_pixels_changed`: Percentage of pixels changed

**`evaluate_attack_success(x, y, x_adv, target_labels=None, verbose=True)`**

Evaluate attack success rate.

**Returns**:
- Dictionary with:
  - `clean_accuracy`: Accuracy on original images
  - `adversarial_accuracy`: Accuracy on adversarial images
  - `attack_success_rate`: Success rate (% misclassified)
  - `accuracy_drop`: Drop in accuracy

**`generate_batch(x, y, batch_size=32)`**

Generate adversarial examples in batches (memory efficient).

### Function: `visualize_fgsm_attack`

Create comprehensive visualization of FGSM attacks.

```python
visualize_fgsm_attack(
    model,
    images,
    labels,
    epsilon,
    class_names,
    device='cuda',
    num_samples=5,
    targeted=False,
    target_labels=None,
    denormalize_fn=None,
    save_path=None
)
```

Creates a 4-column visualization:
1. Original images with predictions
2. Adversarial images with predictions
3. Perturbations (amplified for visibility)
4. Perturbation magnitude heatmaps

## Mathematical Background

### FGSM Intuition

FGSM exploits the linearity of neural networks in high-dimensional spaces:

1. **Compute Gradient**: Calculate how the loss changes with respect to each input pixel
2. **Take Sign**: Extract only the direction (not magnitude) of the gradient
3. **Step in Direction**: Move each pixel by ±ε in the direction that increases loss
4. **Result**: Small, imperceptible change that maximizes loss

### L∞ Norm Constraint

FGSM uses L∞ (L-infinity) norm to bound perturbations:

```
||δ||_∞ = max_i |δ_i| ≤ ε
```

This means: **no single pixel can be perturbed by more than ε**.

**Comparison**:
- **L₀**: Number of changed pixels (NP-hard to optimize)
- **L₂**: Euclidean distance (allows large changes in few pixels)
- **L∞**: Maximum change per pixel (FGSM uses this)

### Why Sign Function?

1. **Efficiency**: Faster than computing normalized gradients
2. **L∞ Constraint**: Automatically satisfies ||δ||_∞ = ε
3. **Simplicity**: No hyperparameters to tune
4. **Effectiveness**: Despite simplicity, highly effective in practice

### Targeted vs Untargeted

**Untargeted** (maximize loss):
```python
x_adv = x + ε · sign(∇_x J(θ, x, y_true))
```
- Goal: Cause ANY misclassification
- Step in direction that increases loss for true class

**Targeted** (minimize loss for target):
```python
x_adv = x - ε · sign(∇_x J(θ, x, y_target))
```
- Goal: Force prediction to specific target class
- Step in direction that decreases loss for target class
- Note the MINUS sign!

## Demonstration Script

Run the comprehensive demonstration:

```bash
# With trained model
python demo_fgsm.py --model-path ./models/baseline_best_model.pth

# With custom epsilon
python demo_fgsm.py --epsilon 0.05

# Skip visualizations (faster)
python demo_fgsm.py --skip-viz

# On CPU
python demo_fgsm.py --device cpu
```

The demonstration shows:
1. **Untargeted attacks** with analysis
2. **Targeted attacks** with custom targets
3. **Epsilon comparison** (trade-off analysis)
4. **Visualizations** (if not skipped)
5. **Batch processing** (efficiency demo)

## Expected Results

### Clean Model (Standard Training)

| Attack | Epsilon | Clean Acc | Adv Acc | Success Rate |
|--------|---------|-----------|---------|--------------|
| None | 0 | 91% | 91% | 0% |
| FGSM | 2/255 | 91% | 75% | 20% |
| FGSM | 4/255 | 91% | 62% | 35% |
| FGSM | 8/255 | 91% | 45% | 50% |
| FGSM | 16/255 | 91% | 20% | 80% |
| PGD-7 | 8/255 | 91% | 5% | 95% |
| PGD-20 | 8/255 | 91% | 0% | 100% |

### Key Observations

1. **FGSM is effective**: Even small ε causes significant accuracy drop
2. **Trade-off exists**: Larger ε → higher success but more visible
3. **PGD is stronger**: Iterative attack causes near-complete failure
4. **Models are vulnerable**: Standard training provides no robustness

## Performance Tips

### Memory Efficiency

For large datasets, use batch processing:

```python
fgsm = FGSM(model, epsilon=8/255)
x_adv = fgsm.generate_batch(all_images, all_labels, batch_size=32)
```

### Speed Optimization

- Use GPU: Move model and data to CUDA
- Batch processing: Process multiple images at once
- Reduce iterations: For PGD, fewer iterations = faster (but weaker)

### Visualization

- Limit samples: `num_samples=5` for quick visualization
- Skip viz in batch: Use `--skip-viz` flag for speed
- Save to file: Always provide `save_path` to avoid display blocking

## Troubleshooting

### Out of Memory

**Problem**: `RuntimeError: CUDA out of memory`

**Solution**: Use batch processing or reduce batch size
```python
x_adv = fgsm.generate_batch(images, labels, batch_size=16)
```

### Weak Attacks

**Problem**: Low attack success rate

**Possible causes**:
- Epsilon too small (try increasing)
- Model is robust (good!)
- Wrong labels provided

**Solution**: Try larger epsilon or use PGD

### L∞ Constraint Violated

**Problem**: `L∞ norm > ε` warning

**Cause**: Clipping to valid range [0, 1] reduced some perturbations

**This is normal** when perturbations would push pixels outside valid range.

## Citation

If you use this implementation in your research or coursework:

```bibtex
@article{goodfellow2014explaining,
  title={Explaining and harnessing adversarial examples},
  author={Goodfellow, Ian J and Shlens, Jonathon and Szegedy, Christian},
  journal={arXiv preprint arXiv:1412.6572},
  year={2014}
}

@article{madry2017towards,
  title={Towards deep learning models resistant to adversarial attacks},
  author={Madry, Aleksander and Makelov, Aleksandar and Schmidt, Ludwig and Tsipras, Dimitris and Vladu, Adrian},
  journal={arXiv preprint arXiv:1706.06083},
  year={2017}
}
```

## Additional Resources

### Papers
- FGSM: https://arxiv.org/abs/1412.6572
- PGD: https://arxiv.org/abs/1706.06083
- C&W: https://arxiv.org/abs/1608.04644
- AutoAttack: https://arxiv.org/abs/2003.01690

### Tools
- CleverHans: https://github.com/cleverhans-lab/cleverhans
- Foolbox: https://github.com/bethgelab/foolbox
- ART: https://github.com/Trusted-AI/adversarial-robustness-toolbox

### Datasets & Benchmarks
- RobustBench: https://robustbench.github.io/
- CIFAR-10-C: https://github.com/hendrycks/robustness

---

**Questions or Issues?** Check the main README.md or create an issue in the repository.
