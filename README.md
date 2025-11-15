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
├── models/                    # Saved model checkpoints
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
├── train.py                   # Training script
├── evaluate.py               # Evaluation script
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

## Quick Start

### 1. Train a Model

Train a ResNet18 model on CIFAR-10:

```bash
python train.py --epochs 100
```

For adversarial training (more robust but slower):

```bash
python train.py --epochs 100 --adv-training
```

### 2. Evaluate Adversarial Robustness

Evaluate the trained model against adversarial attacks:

```bash
python evaluate.py --model-path ./models/best_model.pth --eval-fgsm --eval-pgd --visualize
```

### 3. Interactive Notebook

Launch the Jupyter notebook for interactive experimentation:

```bash
jupyter notebook notebooks/adversarial_attacks_demo.ipynb
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

### Evaluating Adversarial Robustness

**Full evaluation**:
```bash
python evaluate.py --model-path ./models/best_model.pth
```

**Evaluation options**:
```bash
python evaluate.py \
    --model-path ./models/best_model.pth \
    --batch-size 100 \
    --eval-fgsm \
    --eval-pgd \
    --visualize \
    --num-vis 10
```

**Arguments**:
- `--model-path`: Path to the trained model checkpoint
- `--batch-size`: Batch size for evaluation (default: 100)
- `--eval-fgsm`: Evaluate FGSM attack
- `--eval-pgd`: Evaluate PGD attack
- `--visualize`: Generate visualizations of adversarial examples
- `--num-vis`: Number of examples to visualize (default: 10)

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

## Future Enhancements

Potential improvements to this project:

- [ ] Implement C&W attack
- [ ] Add AutoAttack evaluation
- [ ] Implement certified defenses
- [ ] Add more datasets (ImageNet, MNIST)
- [ ] Implement adversarial training variants (TRADES, MART)
- [ ] Add transferability experiments
- [ ] Implement defensive distillation

## License

This project is for educational purposes as part of CS685 coursework.

---

**Author**: CS685 Student
**Date**: 2025
**Course**: Advanced Topics in Machine Learning

For questions or issues, please create an issue in the repository or contact the course instructor.
