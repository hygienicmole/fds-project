# Adversarial Machine Learning: FGSM and PGD Attacks on CIFAR-10

A comprehensive implementation of adversarial attacks (FGSM and PGD) on a ResNet18 model trained on CIFAR-10 dataset.

**Course**: CS685 - Advanced Topics in Machine Learning

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Evaluating Adversarial Robustness](#evaluating-adversarial-robustness)
  - [Using the Jupyter Notebook](#using-the-jupyter-notebook)
- [Adversarial Attacks](#adversarial-attacks)
  - [FGSM (Fast Gradient Sign Method)](#fgsm-fast-gradient-sign-method)
  - [PGD (Projected Gradient Descent)](#pgd-projected-gradient-descent)
- [Configuration](#configuration)
- [Results](#results)
- [References](#references)
- [License](#license)

---

## Overview

This project demonstrates the vulnerability of deep neural networks to adversarial attacks. We implement two popular attack methods:

1. **FGSM (Fast Gradient Sign Method)**: A simple one-step attack that adds perturbations in the direction of the gradient sign.
2. **PGD (Projected Gradient Descent)**: A stronger iterative attack that applies multiple steps of FGSM with projection.

The attacks are tested on a **ResNet18** model trained on the **CIFAR-10** dataset.

## Features

- Clean implementation of FGSM and PGD attacks
- ResNet18 model optimized for CIFAR-10 (32x32 images)
- Comprehensive evaluation framework
- Visualization of adversarial examples
- Support for adversarial training
- Interactive Jupyter notebook for experimentation
- Configurable hyperparameters
- TensorBoard logging support

## Project Structure

```
fds-project/
├── attacks/                    # Adversarial attack implementations
│   ├── __init__.py
│   ├── fgsm.py                # FGSM attack
│   └── pgd.py                 # PGD attack
├── data/                      # CIFAR-10 dataset (auto-downloaded)
├── models/                    # Model implementations and checkpoints
│   ├── __init__.py           # Model module exports
│   ├── resnet.py             # ResNet18 implementation with utilities
│   ├── train.py              # Advanced training script with progress tracking
│   └── checkpoints/          # Training checkpoints
├── notebooks/                 # Jupyter notebooks
│   └── adversarial_attacks_demo.ipynb
├── results/                   # Evaluation results and visualizations
│   └── visualizations/       # Generated plots
├── utils/                     # Utility functions
│   ├── __init__.py
│   ├── data_loader.py        # Data loading utilities
│   ├── model_utils.py        # Model utilities
│   └── visualization.py      # Visualization functions
├── config.py                  # Configuration and hyperparameters
├── train.py                   # Simple training script
├── evaluate.py               # Evaluation script
├── test_model.py             # Model testing script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd fds-project
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

The CIFAR-10 dataset will be automatically downloaded when you first run the training or evaluation scripts.

4. **Test the installation** (optional):
   ```bash
   python test_model.py
   ```

## Quick Start

### Complete Experimental Pipeline (Recommended)

Run the full pipeline from training to report generation:

```bash
# Step 1: Train baseline model (~60-90 min on GPU)
python train_baseline.py

# Step 2: Evaluate attacks (~5-10 min)
python attacks/evaluate_attacks.py

# Step 3: Hyperparameter analysis (~60 min on GPU)
python experiments/hyperparameter_analysis.py

# Step 4: Generate comprehensive visualizations (~2-5 min)
python generate_visualizations.py --attack fgsm
python generate_visualizations.py --attack pgd

# Step 5: Generate publication-ready reports (~1 sec)
python utils/report_generator.py
```

After completion, check:
- `reports/latex_tables.tex` - LaTeX tables for your paper
- `reports/EXPERIMENTAL_SUMMARY.md` - Complete markdown summary
- `results/` - All experimental results and visualizations

### Quick Demo (Without Training)

Test attacks on pre-trained model:

```bash
# Demo FGSM attack
python demo_fgsm.py

# Demo PGD attack
python demo_pgd.py

# Interactive notebook
jupyter notebook notebooks/adversarial_attacks_demo.ipynb
```

### Individual Components

**Option A - Baseline training (recommended)**:
```bash
python train_baseline.py
```
Complete 200-epoch training with metrics tracking and visualization.
See [RUN_TRAINING.md](RUN_TRAINING.md) for details.

**Option B - Simple training**:
```bash
python train.py --epochs 100
```

**Option C - Advanced training**:
```bash
python models/train.py --epochs 100 --optimizer sgd --scheduler multistep
```

**Adversarial training** (more robust but slower):
```bash
python train.py --epochs 100 --adv-training
```

## Usage

### Training the Model

**Basic training**:
```bash
python train.py
```

**Training options**:
```bash
python train.py --epochs 100 --adv-training --save-interval 10
```

**Arguments**:
- `--epochs`: Number of training epochs (default: 100)
- `--adv-training`: Use adversarial training for robustness
- `--save-interval`: Save checkpoint every N epochs (default: 10)

**Expected training time**:
- GPU (NVIDIA RTX 3090): ~15 minutes for 100 epochs
- CPU: ~2-3 hours for 100 epochs

### Advanced Training (models/train.py)

The advanced training script provides additional features:

**Full training with all options**:
```bash
python models/train.py \
    --epochs 100 \
    --batch-size 128 \
    --lr 0.1 \
    --optimizer sgd \
    --scheduler multistep \
    --save-every 10 \
    --early-stopping 20
```

**Resume training from checkpoint**:
```bash
python models/train.py --resume ./models/checkpoints/checkpoint_epoch_50.pth --epochs 100
```

**Arguments**:
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Training batch size (default: 128)
- `--lr`: Initial learning rate (default: 0.1)
- `--optimizer`: Optimizer choice: sgd, adam, adamw (default: sgd)
- `--scheduler`: LR scheduler: multistep, cosine, plateau (default: multistep)
- `--pretrained`: Use ImageNet pretrained weights
- `--dropout`: Dropout probability (default: 0.0)
- `--save-every`: Save checkpoint every N epochs (default: 10)
- `--early-stopping`: Early stopping patience (epochs without improvement)
- `--resume`: Path to checkpoint to resume from
- `--no-tensorboard`: Disable TensorBoard logging

**Features**:
- Progress bars with tqdm
- TensorBoard logging
- Automatic best model saving
- Training metrics tracking (JSON export)
- Resume training capability
- Early stopping support
- Multiple optimizer options (SGD, Adam, AdamW)
- Multiple scheduler options (MultiStep, Cosine, ReduceLROnPlateau)

### Attack Evaluation

**Comprehensive attack evaluation** (recommended):
```bash
python attacks/evaluate_attacks.py
```

Evaluates FGSM and PGD with multiple configurations:
- FGSM: ε ∈ {0.01, 0.03, 0.05, 0.1}
- PGD: 12 configurations (4 epsilon × 3 iteration counts)
- Generates comparison plots and visual grids
- Saves results to `results/attack_evaluation/attack_results.json`

**Custom configuration**:
```bash
python attacks/evaluate_attacks.py \
    --model-path ./models/baseline_best_model.pth \
    --fgsm-epsilons 0.01 0.02 0.03 \
    --pgd-iterations 10 20 40 \
    --batch-size 100 \
    --save-dir ./results/custom_eval
```

**Quick evaluation** (simple script):
```bash
python evaluate.py --model-path ./models/best_model.pth --eval-fgsm --eval-pgd --visualize
```

See [ATTACK_EVALUATION_GUIDE.md](ATTACK_EVALUATION_GUIDE.md) for complete documentation.

### Hyperparameter Analysis

**Full analysis** (~60 minutes on GPU):
```bash
python experiments/hyperparameter_analysis.py
```

Performs comprehensive sweep:
- Epsilon range: 0.001 to 0.3 (15 logarithmically-spaced values)
- PGD iterations: [5, 10, 20, 40]
- Statistical analysis with mean and std
- 4 comprehensive visualization plots
- Results saved to `results/hyperparameter_analysis/`

**Quick test** (~6 minutes):
```bash
python experiments/hyperparameter_analysis.py --num-batches 10
```

**FGSM only** (skip PGD for speed):
```bash
python experiments/hyperparameter_analysis.py --skip-pgd-epsilon --skip-iteration-analysis
```

**Custom epsilon range**:
```bash
python experiments/hyperparameter_analysis.py \
    --epsilon-min 0.001 \
    --epsilon-max 0.1 \
    --num-epsilons 20
```

See [experiments/README.md](experiments/README.md) for detailed documentation.

### Comprehensive Visualizations

**Generate all visualizations** for an attack:
```bash
# FGSM visualizations
python generate_visualizations.py --attack fgsm --epsilon 0.03

# PGD visualizations
python generate_visualizations.py --attack pgd --epsilon 0.03 --pgd-iterations 20
```

Generates 6 types of visualizations:
1. **Comprehensive comparison** - 20 samples side-by-side
2. **Perturbation heatmaps** - Detailed 4-column analysis
3. **Confusion matrix (clean)** - Baseline predictions
4. **Confusion matrix (adversarial)** - Attack predictions
5. **Class-wise attack success** - Per-class vulnerability
6. **Confidence distributions** - Statistical analysis

**Custom configuration**:
```bash
python generate_visualizations.py \
    --attack pgd \
    --epsilon 0.05 \
    --pgd-iterations 40 \
    --num-samples 2000 \
    --save-dir ./results/custom_viz
```

All visualizations saved to `results/visualizations/` with detailed statistics.

See [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md) for interpretation guidelines.

### Report Generation

**Generate publication-ready reports**:
```bash
python utils/report_generator.py
```

Creates:
- **`reports/latex_tables.tex`** - 5 LaTeX tables ready for papers
- **`reports/EXPERIMENTAL_SUMMARY.md`** - Complete markdown summary

**What's included:**

**LaTeX Tables:**
1. Attack Evaluation Results (all configurations)
2. FGSM Epsilon Analysis
3. PGD Iteration Effect
4. Baseline Training Results
5. Hyperparameter Analysis Summary

**Markdown Summary:**
- Executive summary
- Baseline model performance
- Complete attack results
- Hyperparameter analysis findings
- Key observations and recommendations
- Figure catalog with paths
- Statistical significance notes

**Usage in LaTeX**:
```latex
\usepackage{booktabs}
\input{reports/latex_tables.tex}

% Reference tables
As shown in Table~\ref{tab:attack_eval}, the model exhibits...
```

See [REPORT_GENERATION_GUIDE.md](REPORT_GENERATION_GUIDE.md) for complete usage.

### Using the Jupyter Notebook

The notebook provides an interactive environment to:
- Load pre-trained models
- Generate adversarial examples
- Visualize attacks
- Compare FGSM vs PGD
- Experiment with different attack parameters

To use:
1. Start Jupyter: `jupyter notebook`
2. Navigate to `notebooks/adversarial_attacks_demo.ipynb`
3. Run cells sequentially

## Adversarial Attacks

### FGSM (Fast Gradient Sign Method)

**Reference**: Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (2014)

**Formula**:
```
x_adv = x + ε · sign(∇_x J(θ, x, y))
```

Where:
- `x`: Original input image
- `ε`: Perturbation magnitude (epsilon)
- `J`: Loss function
- `y`: True label

**Usage in code**:
```python
from attacks import FGSM

fgsm = FGSM(model=model, epsilon=8/255)
adversarial_images = fgsm.generate(clean_images, true_labels)
```

**Characteristics**:
- Fast (single gradient computation)
- One-step attack
- Less powerful than iterative methods
- Good for adversarial training

### PGD (Projected Gradient Descent)

**Reference**: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (2017)

**Algorithm**:
```
x^0 = x + random_noise
For t = 0 to T-1:
    x^(t+1) = Π_(x+S) (x^t + α · sign(∇_x J(θ, x^t, y)))
```

Where:
- `α`: Step size
- `T`: Number of iterations
- `Π`: Projection onto allowed perturbation set

**Usage in code**:
```python
from attacks import PGD

pgd = PGD(
    model=model,
    epsilon=8/255,
    alpha=2/255,
    iterations=20,
    random_start=True
)
adversarial_images = pgd.generate(clean_images, true_labels)
```

**Characteristics**:
- Stronger than FGSM
- Iterative (multiple steps)
- Random initialization helps escape local minima
- Considered one of the strongest first-order attacks

## ResNet18 Model API

The project includes a comprehensive ResNet18 implementation optimized for CIFAR-10.

### Model Creation

```python
from models import get_resnet18

# Create model
model = get_resnet18(
    num_classes=10,        # Number of classes
    pretrained=False,      # Use ImageNet pretrained weights
    dropout=0.0,           # Dropout probability
    device='cuda'          # Device to load on
)
```

### Model Summary

```python
from models import model_summary, count_parameters

# Print detailed model summary
model_summary(model, input_size=(3, 32, 32), batch_size=1, device='cuda')

# Count parameters
total_params, trainable_params = count_parameters(model)
print(f"Total: {total_params:,}, Trainable: {trainable_params:,}")
```

### Checkpoint Management

```python
from models import save_checkpoint, load_checkpoint

# Save checkpoint
save_checkpoint(
    model, optimizer, epoch=10, accuracy=85.5, loss=0.45,
    filepath='./models/checkpoint.pth',
    scheduler=scheduler,
    best_acc=87.0
)

# Load checkpoint
info = load_checkpoint(
    './models/checkpoint.pth',
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    device='cuda'
)

print(f"Loaded epoch {info['epoch']}, accuracy {info['accuracy']:.2f}%")
```

### Feature Extraction

```python
# Extract features before classification layer
features = model.get_features(images)  # Returns (batch_size, 512) tensor
```

## Configuration

All hyperparameters are centralized in `config.py`:

### Key Parameters

**Model Settings**:
```python
MODEL_NAME = 'ResNet18'
NUM_CLASSES = 10
```

**Training Settings**:
```python
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.1
LR_MILESTONES = [50, 75]
```

**FGSM Attack**:
```python
FGSM_EPSILON = 8/255
FGSM_EPSILONS = [0, 2/255, 4/255, 8/255, 16/255, 32/255]
```

**PGD Attack**:
```python
PGD_EPSILON = 8/255
PGD_ALPHA = 2/255
PGD_ITERATIONS = 7
PGD_RANDOM_START = True
```

**Adversarial Training**:
```python
ADV_TRAINING = False
ADV_TRAIN_EPSILON = 8/255
ADV_TRAIN_ALPHA = 2/255
ADV_TRAIN_ITERATIONS = 7
```

Modify these values to experiment with different configurations.

## Results

### Expected Performance (Standard Training)

| Metric | Accuracy |
|--------|----------|
| Clean Test Accuracy | ~93-95% |
| FGSM (ε=8/255) | ~40-50% |
| PGD-7 (ε=8/255) | ~0-10% |
| PGD-20 (ε=8/255) | ~0-5% |

### Key Findings

1. **Standard models are highly vulnerable** to adversarial attacks
2. **PGD is stronger than FGSM**, especially with more iterations
3. **Adversarial training** improves robustness but reduces clean accuracy
4. **Perturbations are imperceptible** to humans even at ε=8/255

### Visualization Examples

The evaluation script generates:
- Adversarial examples side-by-side with originals
- Perturbation visualizations (amplified for visibility)
- Accuracy vs epsilon plots
- Attack comparison charts

All visualizations are saved to `results/visualizations/`.

## References

### Papers

1. **FGSM**: Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). [Explaining and harnessing adversarial examples](https://arxiv.org/abs/1412.6572). arXiv preprint arXiv:1412.6572.

2. **PGD**: Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2017). [Towards deep learning models resistant to adversarial attacks](https://arxiv.org/abs/1706.06083). arXiv preprint arXiv:1706.06083.

3. **ResNet**: He, K., Zhang, X., Ren, S., & Sun, J. (2016). [Deep residual learning for image recognition](https://arxiv.org/abs/1512.03385). In CVPR.

### Additional Resources

- [CleverHans Library](https://github.com/cleverhans-lab/cleverhans)
- [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
- [RobustBench Leaderboard](https://robustbench.github.io/)

## Project Information

- **Course**: CS685 - Advanced Topics in Machine Learning
- **Topic**: Adversarial Machine Learning
- **Model**: ResNet18
- **Dataset**: CIFAR-10
- **Framework**: PyTorch

## Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   - Reduce batch size in `config.py`
   - Use CPU by setting `DEVICE = torch.device('cpu')`

2. **Model file not found**:
   - Train a model first: `python train.py`
   - Check the path in `--model-path` argument

3. **Slow training**:
   - Ensure you're using GPU
   - Reduce number of workers if CPU is bottleneck
   - Use smaller model or fewer epochs

4. **Import errors**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

## Documentation

This project includes comprehensive documentation for all components:

### Core Documentation

- **[docs/methodology.md](docs/methodology.md)** - Complete mathematical explanation of FGSM and PGD
  - Mathematical foundations and formulations
  - Algorithm details and pseudocode
  - Perturbation constraints and $L_p$ norms
  - Implementation details and best practices
  - Evaluation metrics definitions
  - References to foundational papers

- **[docs/experiments.md](docs/experiments.md)** - Complete experimental design and results
  - All experimental configurations
  - Expected results and benchmarks
  - Statistical analysis methodology
  - Computational performance benchmarks
  - Reproducibility guidelines
  - Summary of key findings

### Component-Specific Guides

- **[RUN_TRAINING.md](RUN_TRAINING.md)** - Baseline training guide
- **[EXPECTED_RESULTS.md](EXPECTED_RESULTS.md)** - Expected training outcomes
- **[ATTACK_EVALUATION_GUIDE.md](ATTACK_EVALUATION_GUIDE.md)** - Attack evaluation documentation
- **[experiments/README.md](experiments/README.md)** - Hyperparameter analysis guide
- **[VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md)** - Visualization interpretation
- **[REPORT_GENERATION_GUIDE.md](REPORT_GENERATION_GUIDE.md)** - Report generation usage
- **[attacks/README.md](attacks/README.md)** - FGSM and PGD API documentation

### Quick Reference

**Want to understand the math behind attacks?**
→ Read [docs/methodology.md](docs/methodology.md)

**Want to know what experiments were run?**
→ Read [docs/experiments.md](docs/experiments.md)

**Want to train the model?**
→ Follow [RUN_TRAINING.md](RUN_TRAINING.md)

**Want to evaluate attacks?**
→ Use [ATTACK_EVALUATION_GUIDE.md](ATTACK_EVALUATION_GUIDE.md)

**Want to generate visualizations?**
→ Check [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md)

**Want to create a report?**
→ See [REPORT_GENERATION_GUIDE.md](REPORT_GENERATION_GUIDE.md)

## Project Structure (Updated)

```
fds-project/
├── docs/                               # Comprehensive documentation
│   ├── methodology.md                  # Mathematical details of attacks
│   └── experiments.md                  # Experimental design and results
├── attacks/                            # Attack implementations
│   ├── __init__.py
│   ├── fgsm.py                        # FGSM implementation
│   ├── pgd.py                         # PGD implementation
│   ├── evaluate_attacks.py            # Comprehensive evaluation script
│   └── README.md                      # Attack API documentation
├── experiments/                        # Experimental analysis scripts
│   ├── hyperparameter_analysis.py     # Epsilon/iteration sweep
│   └── README.md                      # Experiment documentation
├── models/                            # Model implementations
│   ├── __init__.py
│   ├── resnet.py                      # ResNet18 for CIFAR-10
│   └── train.py                       # Advanced training script
├── utils/                             # Utilities
│   ├── __init__.py
│   ├── data_loader.py                 # CIFAR-10 data loading
│   ├── model_utils.py                 # Model utilities
│   ├── visualization.py               # Comprehensive visualizations
│   └── report_generator.py            # Publication report generator
├── results/                           # Experimental results (gitignored)
│   ├── attack_evaluation/             # Attack evaluation results
│   ├── hyperparameter_analysis/       # Parameter sweep results
│   ├── baseline_training/             # Training metrics
│   └── visualizations/                # Generated visualizations
├── reports/                           # Generated reports
│   ├── latex_tables.tex               # LaTeX tables for papers
│   └── EXPERIMENTAL_SUMMARY.md        # Complete markdown summary
├── notebooks/                         # Jupyter notebooks
│   └── adversarial_attacks_demo.ipynb # Interactive demo
├── train_baseline.py                  # Baseline training (200 epochs)
├── generate_visualizations.py         # Visualization generator
├── demo_fgsm.py                       # FGSM demo
├── demo_pgd.py                        # PGD demo
├── config.py                          # Configuration
├── requirements.txt                   # Dependencies
├── RUN_TRAINING.md                    # Training guide
├── EXPECTED_RESULTS.md                # Expected outcomes
├── ATTACK_EVALUATION_GUIDE.md         # Evaluation guide
├── VISUALIZATION_GUIDE.md             # Visualization guide
├── REPORT_GENERATION_GUIDE.md         # Report guide
└── README.md                          # This file
```

## Future Enhancements

Potential improvements to this project:

- [ ] Implement C&W attack (L2-norm optimization)
- [ ] Add AutoAttack evaluation (ensemble of attacks)
- [ ] Implement certified defenses (randomized smoothing, IBP)
- [ ] Add more datasets (ImageNet, CIFAR-100, MNIST)
- [ ] Implement adversarial training variants (TRADES, MART, AWP)
- [ ] Add transferability experiments across models
- [ ] Implement defensive distillation
- [ ] Add gradient masking detection
- [ ] Implement adaptive attacks
- [ ] Add model interpretability tools

## License

This project is for educational purposes as part of CS685 coursework.

---

**Author**: CS685 Student
**Date**: 2025
**Course**: Advanced Topics in Machine Learning

For questions or issues, please create an issue in the repository or contact the course instructor.
