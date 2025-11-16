# Adversarial Attack Methodology

This document provides a comprehensive mathematical and technical explanation of the adversarial attack methods implemented in this project.

## Table of Contents

1. [Introduction to Adversarial Examples](#introduction)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Fast Gradient Sign Method (FGSM)](#fgsm)
4. [Projected Gradient Descent (PGD)](#pgd)
5. [Perturbation Constraints](#perturbation-constraints)
6. [Implementation Details](#implementation-details)
7. [Evaluation Metrics](#evaluation-metrics)
8. [References](#references)

---

## Introduction to Adversarial Examples

### Definition

An **adversarial example** is an input to a machine learning model that has been intentionally perturbed to cause the model to make a mistake, while the perturbation is often imperceptible to humans.

**Formal Definition:**
Given:
- A classifier $f: \mathbb{R}^d \to \mathbb{R}^k$ that maps inputs to class scores
- A clean input $\mathbf{x} \in \mathbb{R}^d$ with true label $y$
- A perturbation bound $\epsilon > 0$

An adversarial example $\mathbf{x}_{adv}$ satisfies:
1. $\|\mathbf{x}_{adv} - \mathbf{x}\|_p \leq \epsilon$ (imperceptibility constraint)
2. $\arg\max f(\mathbf{x}_{adv}) \neq y$ (misclassification)

### Threat Model

**Untargeted Attacks**: Goal is to cause any misclassification
- Adversarial objective: Find $\mathbf{x}_{adv}$ such that $f(\mathbf{x}_{adv}) \neq y$

**Targeted Attacks**: Goal is to cause classification to a specific target class $y_t$
- Adversarial objective: Find $\mathbf{x}_{adv}$ such that $f(\mathbf{x}_{adv}) = y_t$

**White-box Setting**: Attacker has full access to:
- Model architecture
- Model parameters (weights)
- Training data distribution
- Gradient information

This project implements **white-box untargeted attacks** using gradient-based methods.

---

## Mathematical Foundations

### Loss Function

For a neural network classifier, let:
- $f_\theta(\mathbf{x})$ be the model with parameters $\theta$
- $\mathcal{L}(f_\theta(\mathbf{x}), y)$ be the loss function (e.g., cross-entropy)

**Cross-Entropy Loss:**
$$\mathcal{L}(f_\theta(\mathbf{x}), y) = -\log\left(\frac{e^{f_\theta(\mathbf{x})_y}}{\sum_{j=1}^k e^{f_\theta(\mathbf{x})_j}}\right)$$

where $f_\theta(\mathbf{x})_y$ is the logit for the true class $y$.

### Adversarial Optimization Problem

**General Formulation:**

For untargeted attacks, we maximize the loss:
$$\max_{\mathbf{x}_{adv}} \mathcal{L}(f_\theta(\mathbf{x}_{adv}), y) \quad \text{subject to} \quad \|\mathbf{x}_{adv} - \mathbf{x}\|_\infty \leq \epsilon$$

Equivalently:
$$\mathbf{x}_{adv}^* = \arg\max_{\|\delta\|_\infty \leq \epsilon} \mathcal{L}(f_\theta(\mathbf{x} + \delta), y)$$

where $\delta = \mathbf{x}_{adv} - \mathbf{x}$ is the perturbation.

**Challenge:** This is a non-convex constrained optimization problem that is generally NP-hard to solve optimally.

**Solution:** Use iterative gradient-based methods to find approximate solutions.

---

## Fast Gradient Sign Method (FGSM)

### Overview

FGSM (Goodfellow et al., 2014) is a **one-step** attack that computes adversarial perturbations using the sign of the gradient.

### Mathematical Formulation

**One-Step Update:**
$$\mathbf{x}_{adv} = \mathbf{x} + \epsilon \cdot \text{sign}(\nabla_\mathbf{x} \mathcal{L}(f_\theta(\mathbf{x}), y))$$

where:
- $\nabla_\mathbf{x} \mathcal{L}$ is the gradient of the loss with respect to the input
- $\text{sign}(\cdot)$ extracts the sign of each element: $\{-1, 0, +1\}$
- $\epsilon$ controls the perturbation magnitude

### Intuition

1. **Gradient Direction**: $\nabla_\mathbf{x} \mathcal{L}$ points in the direction that increases the loss most rapidly
2. **Sign Extraction**: Taking the sign creates a perturbation in the same direction but with uniform magnitude
3. **Maximum Step**: Using $\epsilon$ as the step size takes the maximum allowed step in the $L_\infty$ ball

### Why It Works

**Linear Approximation:**
Using first-order Taylor expansion:
$$\mathcal{L}(f_\theta(\mathbf{x} + \delta), y) \approx \mathcal{L}(f_\theta(\mathbf{x}), y) + \delta^T \nabla_\mathbf{x} \mathcal{L}$$

To maximize this, we want:
$$\delta = \arg\max_{\|\delta\|_\infty \leq \epsilon} \delta^T \nabla_\mathbf{x} \mathcal{L}$$

The solution is:
$$\delta^* = \epsilon \cdot \text{sign}(\nabla_\mathbf{x} \mathcal{L})$$

This gives $\delta^T \nabla_\mathbf{x} \mathcal{L} = \epsilon \sum_i |\nabla_\mathbf{x} \mathcal{L}_i|$, which is the maximum possible value.

### Algorithm

```
Algorithm: FGSM Attack (Untargeted)
Input: Model f_θ, input x, label y, epsilon ε
Output: Adversarial example x_adv

1. Compute model output: logits = f_θ(x)
2. Compute loss: L = CrossEntropy(logits, y)
3. Compute gradient: g = ∇_x L
4. Create perturbation: δ = ε · sign(g)
5. Create adversarial example: x_adv = x + δ
6. Clip to valid range: x_adv = clip(x_adv, 0, 1)
7. Return x_adv
```

### Advantages and Limitations

**Advantages:**
- **Fast**: Single gradient computation, O(1) iterations
- **Simple**: Easy to implement and understand
- **Transferable**: Often transfers across models

**Limitations:**
- **Suboptimal**: Single-step may not find strongest perturbation
- **Linear assumption**: Doesn't account for non-linearity
- **Fixed step**: Cannot adapt step size

### Targeted FGSM

For targeted attacks towards class $y_t$:
$$\mathbf{x}_{adv} = \mathbf{x} - \epsilon \cdot \text{sign}(\nabla_\mathbf{x} \mathcal{L}(f_\theta(\mathbf{x}), y_t))$$

Note the minus sign: we minimize loss for the target class.

---

## Projected Gradient Descent (PGD)

### Overview

PGD (Madry et al., 2017) is an **iterative** attack that repeatedly applies small gradient steps followed by projection to the constraint set. It is currently considered the **strongest first-order adversary** for $L_\infty$ perturbations.

### Mathematical Formulation

**Iterative Update:**
$$\mathbf{x}^{(t+1)} = \Pi_{\mathbf{x} + \mathcal{S}} \left( \mathbf{x}^{(t)} + \alpha \cdot \text{sign}(\nabla_{\mathbf{x}^{(t)}} \mathcal{L}(f_\theta(\mathbf{x}^{(t)}), y)) \right)$$

where:
- $\mathbf{x}^{(t)}$ is the adversarial example at iteration $t$
- $\alpha$ is the step size (typically $\alpha = \epsilon / k$ for $k$ iterations)
- $\mathcal{S} = \{\delta : \|\delta\|_\infty \leq \epsilon\}$ is the $L_\infty$ ball
- $\Pi_{\mathbf{x} + \mathcal{S}}(\cdot)$ is the projection operator onto $\mathbf{x} + \mathcal{S}$

**Initialization:**
- **Random start**: $\mathbf{x}^{(0)} = \mathbf{x} + \text{Uniform}(-\epsilon, \epsilon)$ (recommended)
- **Clean start**: $\mathbf{x}^{(0)} = \mathbf{x}$

### Projection Operation

**$L_\infty$ Projection:**

$$\Pi_{\mathbf{x} + \mathcal{S}}(\mathbf{z}) = \mathbf{x} + \text{clip}(\mathbf{z} - \mathbf{x}, -\epsilon, \epsilon)$$

This ensures $\|\mathbf{z} - \mathbf{x}\|_\infty \leq \epsilon$ by clipping each dimension:

$$[\Pi_{\mathbf{x} + \mathcal{S}}(\mathbf{z})]_i = \begin{cases}
\mathbf{x}_i + \epsilon & \text{if } \mathbf{z}_i - \mathbf{x}_i > \epsilon \\
\mathbf{x}_i - \epsilon & \text{if } \mathbf{z}_i - \mathbf{x}_i < -\epsilon \\
\mathbf{z}_i & \text{otherwise}
\end{cases}$$

**Pixel Range Projection:**

Additionally, we must ensure valid pixel values $[0, 1]$:
$$\mathbf{x}_{adv} = \text{clip}(\mathbf{x}_{adv}, 0, 1)$$

### Algorithm

```
Algorithm: PGD Attack (Untargeted)
Input: Model f_θ, input x, label y, epsilon ε, alpha α, iterations K
Output: Adversarial example x_adv

1. Random initialization: x_adv = x + Uniform(-ε, ε)
2. Clip to valid range: x_adv = clip(x_adv, 0, 1)

3. For t = 1 to K:
   a. Compute model output: logits = f_θ(x_adv)
   b. Compute loss: L = CrossEntropy(logits, y)
   c. Compute gradient: g = ∇_{x_adv} L
   d. Update: x_adv = x_adv + α · sign(g)
   e. Project to L∞ ball: x_adv = x + clip(x_adv - x, -ε, ε)
   f. Clip to valid range: x_adv = clip(x_adv, 0, 1)

4. Return x_adv
```

### Why Random Start?

**Motivation:**
- The loss landscape is highly non-convex with many local maxima
- Starting from the clean input may get stuck in a suboptimal local maximum
- Random initialization explores different starting points

**Empirical Evidence:**
- Random start significantly improves attack success rate
- Provides better approximation to the optimal attack
- Recommended by Madry et al. (2017) as best practice

### Hyperparameter Selection

**Number of Iterations ($K$):**
- More iterations → stronger attack but slower
- Typical values: 7-100 iterations
- Diminishing returns after ~20-40 iterations
- Our experiments: 20 iterations provides good balance

**Step Size ($\alpha$):**
- Common choice: $\alpha = \epsilon / 4$ or $\alpha = \epsilon / \text{num\_iters}$
- Smaller $\alpha$ → finer search but needs more iterations
- Larger $\alpha$ → faster convergence but may overshoot
- Our experiments: $\alpha = \epsilon \times 0.25$

**Epsilon ($\epsilon$):**
- Standard benchmark: $\epsilon = 8/255 \approx 0.0314$ (CIFAR-10)
- ImageNet: $\epsilon = 4/255 \approx 0.0157$
- Trade-off between imperceptibility and attack strength

### Advantages and Limitations

**Advantages:**
- **Strong**: Most effective first-order attack
- **Reliable**: Consistently finds adversarial examples
- **Standard**: De-facto evaluation method for robustness
- **Adaptive**: Can escape poor local maxima with random start

**Limitations:**
- **Slow**: Requires $K$ forward and backward passes
- **Hyperparameters**: Sensitive to $\alpha$ and $K$ selection
- **First-order**: Still bounded by gradient information (can be improved with second-order methods)

### Relationship to FGSM

**FGSM as PGD-1:**
FGSM is equivalent to PGD with:
- $K = 1$ (one iteration)
- $\alpha = \epsilon$ (full step size)
- No random initialization

**PGD as Iterative FGSM:**
PGD can be viewed as:
- Multiple small FGSM steps
- With projection to ensure constraint satisfaction
- Starting from a random point

### Targeted PGD

For targeted attacks towards class $y_t$:
$$\mathbf{x}^{(t+1)} = \Pi_{\mathbf{x} + \mathcal{S}} \left( \mathbf{x}^{(t)} - \alpha \cdot \text{sign}(\nabla_{\mathbf{x}^{(t)}} \mathcal{L}(f_\theta(\mathbf{x}^{(t)}), y_t)) \right)$$

Note the minus sign: we minimize loss for the target class.

---

## Perturbation Constraints

### $L_p$ Norms

Different norms measure perturbation magnitude differently:

**$L_0$ Norm (Sparsity):**
$$\|\delta\|_0 = |\{i : \delta_i \neq 0\}|$$
- Counts number of changed pixels
- Hard to optimize (combinatorial)

**$L_2$ Norm (Euclidean Distance):**
$$\|\delta\|_2 = \sqrt{\sum_i \delta_i^2}$$
- Total energy of perturbation
- Used by C&W attack

**$L_\infty$ Norm (Maximum Change):**
$$\|\delta\|_\infty = \max_i |\delta_i|$$
- Maximum per-pixel change
- **Used by FGSM and PGD**
- Intuitive: bounds largest change to any pixel

### Why $L_\infty$ for FGSM/PGD?

1. **Imperceptibility**: Limits maximum change to any pixel
2. **Computational Efficiency**: Easy to project (element-wise clipping)
3. **Theoretical Properties**: Matches linear approximation well
4. **Empirical Success**: Very effective in practice

### Perturbation Visualization

For CIFAR-10 images (32×32×3 = 3,072 pixels):

**Epsilon = 0.03 (7.65/255):**
- $L_\infty$: Max change per pixel = 0.03
- $L_2$: Typical magnitude ≈ 1.5-2.0
- $L_1$: Typical magnitude ≈ 50-80
- $L_0$: Typically 2,000-3,000 pixels changed

**Relationship:**
$$\|\delta\|_2 \leq \sqrt{d} \|\delta\|_\infty$$
$$\|\delta\|_1 \leq d \|\delta\|_\infty$$

where $d$ is the input dimension (3,072 for CIFAR-10).

### Pixel Value Clipping

**Normalized Images** (our implementation):
- Pixels in range $[0, 1]$
- Epsilon in normalized scale: 0.0-1.0
- Standard: $\epsilon = 8/255 = 0.0314$

**Unnormalized Images:**
- Pixels in range $[0, 255]$
- Epsilon in pixel scale: 0-255
- Standard: $\epsilon = 8$

**Conversion:**
$$\epsilon_{\text{normalized}} = \frac{\epsilon_{\text{pixel}}}{255}$$

---

## Implementation Details

### PyTorch Implementation Notes

**Gradient Computation:**
```python
# Enable gradient tracking for input
x.requires_grad = True

# Forward pass
outputs = model(x)
loss = criterion(outputs, y)

# Backward pass
model.zero_grad()
loss.backward()

# Get gradient with respect to input
grad = x.grad.data
```

**Sign Gradient:**
```python
# Extract sign of gradient
sign_grad = grad.sign()

# Create perturbation
perturbation = epsilon * sign_grad
```

**Clipping Operations:**
```python
# Clip to L∞ ball
delta = torch.clamp(x_adv - x, -epsilon, epsilon)
x_adv = x + delta

# Clip to valid pixel range [0, 1]
x_adv = torch.clamp(x_adv, 0, 1)
```

### Normalization Handling

**CIFAR-10 Normalization:**
- Mean: [0.4914, 0.4822, 0.4465]
- Std: [0.2470, 0.2435, 0.2616]

**Attack in Normalized Space:**
Perturbations are applied AFTER normalization:
1. Load image in [0, 1] range
2. Normalize: $\mathbf{x}_{\text{norm}} = \frac{\mathbf{x} - \mu}{\sigma}$
3. Apply attack to $\mathbf{x}_{\text{norm}}$
4. Denormalize for visualization

**Alternative:** Attack in unnormalized space
- More intuitive epsilon interpretation
- Our implementation uses normalized space

### Batch Processing

**Vectorized Implementation:**
- Process multiple images simultaneously
- Gradient computation across batch
- Memory-efficient for large datasets

**Batch-wise Attacks:**
```python
batch_size = 100
for images, labels in dataloader:
    # Generate adversarial batch
    adv_images = attack.generate(images, labels)
```

### Computational Complexity

**FGSM:**
- Time: $O(1)$ forward + $O(1)$ backward = $O(1)$ per image
- Memory: Same as standard forward pass

**PGD:**
- Time: $O(K)$ forward + $O(K)$ backward = $O(K)$ per image
- Memory: Same as standard forward pass (no accumulation)

**Scaling:**
- Linear in batch size
- Linear in number of iterations (PGD)
- Independent of model size (fixed per architecture)

---

## Evaluation Metrics

### Attack Success Rate (ASR)

**Definition:**
Percentage of correctly classified clean examples that are misclassified after attack.

$$\text{ASR} = \frac{|\{i : f(\mathbf{x}_i) = y_i \text{ and } f(\mathbf{x}_i^{adv}) \neq y_i\}|}{|\{i : f(\mathbf{x}_i) = y_i\}|}$$

**Interpretation:**
- ASR = 0%: Attack completely fails
- ASR = 50%: Attack succeeds half the time
- ASR = 100%: Attack always succeeds

### Adversarial Accuracy

**Definition:**
Accuracy on adversarial examples.

$$\text{Adv Acc} = \frac{1}{N} \sum_{i=1}^N \mathbb{1}[f(\mathbf{x}_i^{adv}) = y_i]$$

**Relationship to ASR:**
$$\text{ASR} = 1 - \frac{\text{Adv Acc}}{\text{Clean Acc}}$$

### Accuracy Drop

**Definition:**
Difference between clean and adversarial accuracy.

$$\text{Acc Drop} = \text{Clean Acc} - \text{Adv Acc}$$

**Interpretation:**
- Directly measures robustness degradation
- Independent of baseline performance
- Easy to interpret percentage point change

### Perturbation Metrics

**$L_\infty$ Perturbation:**
$$\max_i |\mathbf{x}_i^{adv} - \mathbf{x}_i|$$
- Should equal $\epsilon$ for successful attacks
- Measures worst-case pixel change

**$L_2$ Perturbation:**
$$\sqrt{\sum_i (\mathbf{x}_i^{adv} - \mathbf{x}_i)^2}$$
- Measures total perturbation energy
- Typically $\sqrt{d} \times \epsilon$ for $L_\infty$ attacks

**Average Perturbation:**
- Mean perturbation across all images
- Useful for comparing attack strength

### Robustness Metrics

**Robust Accuracy:**
Accuracy under strongest attack:
$$\text{Robust Acc}_\epsilon = \min_{\|\delta\|_\infty \leq \epsilon} \text{Accuracy}(f, \mathbf{x} + \delta)$$

In practice, use PGD as approximation.

**Certified Robustness:**
- Provable lower bound on robust accuracy
- Requires certified defense methods
- Beyond scope of this project

---

## References

### Foundational Papers

1. **FGSM:**
   - Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). Explaining and harnessing adversarial examples. *ICLR 2015*.
   - arXiv: https://arxiv.org/abs/1412.6572

2. **PGD:**
   - Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2017). Towards deep learning models resistant to adversarial attacks. *ICLR 2018*.
   - arXiv: https://arxiv.org/abs/1706.06083

3. **Original Adversarial Examples:**
   - Szegedy, C., et al. (2013). Intriguing properties of neural networks. *ICLR 2014*.
   - arXiv: https://arxiv.org/abs/1312.6199

### Additional Reading

4. **C&W Attack:**
   - Carlini, N., & Wagner, D. (2017). Towards evaluating the robustness of neural networks. *IEEE S&P*.

5. **AutoAttack:**
   - Croce, F., & Hein, M. (2020). Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks. *ICML 2020*.

6. **Adversarial Training:**
   - Madry, A., et al. (2017). [Same as PGD paper]
   - Zhang, H., et al. (2019). Theoretically principled trade-off between robustness and accuracy. *ICML 2019*.

7. **RobustBench Benchmark:**
   - Croce, F., et al. (2020). RobustBench: a standardized adversarial robustness benchmark.
   - Website: https://robustbench.github.io/

### Surveys and Tutorials

8. **Survey Papers:**
   - Akhtar, N., & Mian, A. (2018). Threat of adversarial attacks on deep learning in computer vision: A survey. *IEEE Access*.
   - Chakraborty, A., et al. (2018). Adversarial attacks and defences: A survey. *arXiv*.

9. **Tutorials:**
   - Goodfellow, I. (2018). Tutorial on adversarial machine learning. *CVPR*.
   - Papernot, N., & McDaniel, P. (2018). Deep learning for security. *IEEE Security & Privacy*.

---

## Appendix: Notation Summary

| Symbol | Meaning |
|--------|---------|
| $\mathbf{x}$ | Clean input image |
| $\mathbf{x}_{adv}$ | Adversarial example |
| $y$ | True class label |
| $y_t$ | Target class (for targeted attacks) |
| $f_\theta$ | Neural network classifier with parameters $\theta$ |
| $\mathcal{L}$ | Loss function (typically cross-entropy) |
| $\epsilon$ | Perturbation bound ($L_\infty$ constraint) |
| $\delta$ | Perturbation: $\delta = \mathbf{x}_{adv} - \mathbf{x}$ |
| $\alpha$ | Step size for iterative attacks |
| $K$ | Number of iterations |
| $\nabla_\mathbf{x} \mathcal{L}$ | Gradient of loss w.r.t. input |
| $\Pi_{\mathcal{S}}$ | Projection operator onto set $\mathcal{S}$ |
| $\|\cdot\|_p$ | $L_p$ norm |

---

**Last Updated:** 2025-11-16
**Version:** 1.0
**Authors:** CS685 Adversarial ML Project
