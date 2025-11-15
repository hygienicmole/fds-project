"""
Fast Gradient Sign Method (FGSM) Attack Implementation

This module implements the FGSM attack with comprehensive features including:
- Untargeted and targeted attacks
- Perturbation analysis
- Visualization utilities
- Detailed mathematical explanations

Reference:
Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014).
Explaining and harnessing adversarial examples.
arXiv preprint arXiv:1412.6572.

Paper: https://arxiv.org/abs/1412.6572
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FGSM:
    """
    Fast Gradient Sign Method (FGSM) Attack

    ═══════════════════════════════════════════════════════════════════════════
    MATHEMATICAL BACKGROUND
    ═══════════════════════════════════════════════════════════════════════════

    FGSM is a simple yet effective one-step adversarial attack that exploits
    the linearity of neural networks in high-dimensional spaces.

    UNTARGETED ATTACK:
    ──────────────────
    Goal: Maximize the loss to cause misclassification

    Formula:
        x_adv = x + ε · sign(∇_x J(θ, x, y))

    Where:
        x       = Original input image
        x_adv   = Adversarial example
        ε       = Perturbation magnitude (epsilon)
        θ       = Model parameters
        J       = Loss function (e.g., Cross-Entropy)
        y       = True label
        ∇_x     = Gradient with respect to input x
        sign()  = Sign function: {-1, 0, +1}

    Intuition:
        - Compute gradient of loss with respect to input
        - Take the SIGN of the gradient (direction that increases loss)
        - Step in that direction by distance ε
        - This maximizes loss, causing misclassification

    TARGETED ATTACK:
    ────────────────
    Goal: Minimize loss for a target class to cause specific misclassification

    Formula:
        x_adv = x - ε · sign(∇_x J(θ, x, y_target))

    Where:
        y_target = Desired target class (different from true class)

    Note the MINUS sign: we want to MINIMIZE loss for the target class,
    so we step in the OPPOSITE direction of the gradient.

    Intuition:
        - Compute gradient of loss with respect to target class
        - Step in OPPOSITE direction (minimize loss for target)
        - This forces model to predict the target class

    WHY SIGN FUNCTION?
    ──────────────────
    1. EFFICIENCY: Using sign() instead of normalized gradient is faster
    2. L∞ CONSTRAINT: Guarantees perturbation ≤ ε in each dimension
    3. SIMPLICITY: No need to normalize gradients
    4. EFFECTIVENESS: Despite simplicity, highly effective in practice

    L∞ NORM CONSTRAINT:
    ───────────────────
    FGSM uses L∞ (L-infinity) norm to bound perturbations:

        ||δ||_∞ = max_i |δ_i| ≤ ε

    This means: no single pixel can be perturbed by more than ε

    Comparison with other norms:
        - L₀: Number of changed pixels (NP-hard to optimize)
        - L₂: Euclidean distance (allows large changes in few pixels)
        - L∞: Maximum change per pixel (FGSM uses this)

    CLIPPING:
    ─────────
    After perturbation, we clip to valid pixel range [clip_min, clip_max]
    to ensure adversarial examples are valid images.

    For images normalized to [0, 1]: clip to [0, 1]
    For images in [-1, 1]: clip to [-1, 1]

    GRADIENT COMPUTATION:
    ─────────────────────
    PyTorch autograd computes:
        ∇_x J(θ, x, y) = ∂J/∂x

    This tells us how to change x to increase (or decrease) the loss.

    ═══════════════════════════════════════════════════════════════════════════

    Args:
        model: The target model to attack
        epsilon: Maximum perturbation magnitude (L-infinity norm)
        clip_min: Minimum pixel value for clipping (default: 0.0)
        clip_max: Maximum pixel value for clipping (default: 1.0)
        targeted: Whether to perform targeted attack (default: False)

    Example Usage:
        >>> # Untargeted attack
        >>> fgsm = FGSM(model, epsilon=0.03, targeted=False)
        >>> x_adv = fgsm.generate(images, true_labels)

        >>> # Targeted attack (make cat look like dog)
        >>> fgsm_targeted = FGSM(model, epsilon=0.03, targeted=True)
        >>> target_labels = torch.tensor([5, 5, 5, 5])  # All target class 5
        >>> x_adv = fgsm_targeted.generate(images, target_labels)
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.03,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        targeted: bool = False
    ):
        self.model = model
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.criterion = nn.CrossEntropyLoss()

        # Statistics for analysis
        self.last_perturbation = None
        self.last_success_rate = None

    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        return_perturbation: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate adversarial examples using FGSM

        Algorithm:
        ──────────
        1. Set model to evaluation mode (disable dropout, batchnorm updates)
        2. Clone input and enable gradient tracking
        3. Forward pass: compute model predictions
        4. Compute loss with respect to labels (true or target)
        5. Backward pass: compute ∇_x J(θ, x, y)
        6. Extract gradient sign: sign(∇_x J)
        7. If targeted: x_adv = x - ε · sign(∇_x J)  [minimize loss]
           If untargeted: x_adv = x + ε · sign(∇_x J)  [maximize loss]
        8. Clip to valid range [clip_min, clip_max]
        9. Return adversarial examples

        Args:
            x: Input images (batch_size, channels, height, width)
            y: Labels - true labels for untargeted, target labels for targeted
            return_perturbation: If True, also return the perturbation δ

        Returns:
            x_adv: Adversarial examples
            perturbation: (optional) The perturbation δ = x_adv - x
        """
        # Set model to evaluation mode
        # This is crucial: we don't want to update model parameters
        # or use training-specific behaviors (dropout, batchnorm)
        self.model.eval()

        # Clone input and enable gradient computation
        # We need gradients w.r.t. input, not model parameters
        x_adv = x.clone().detach()
        x_adv.requires_grad = True

        # ═══════════════════════════════════════════════════════════════════
        # STEP 1: FORWARD PASS
        # ═══════════════════════════════════════════════════════════════════
        # Compute model predictions: f_θ(x)
        outputs = self.model(x_adv)

        # ═══════════════════════════════════════════════════════════════════
        # STEP 2: COMPUTE LOSS
        # ═══════════════════════════════════════════════════════════════════
        # For untargeted: maximize loss for TRUE label
        # For targeted: minimize loss for TARGET label
        loss = self.criterion(outputs, y)

        # ═══════════════════════════════════════════════════════════════════
        # STEP 3: BACKWARD PASS - COMPUTE GRADIENT
        # ═══════════════════════════════════════════════════════════════════
        # Clear any existing gradients on model parameters
        self.model.zero_grad()

        # Compute gradient of loss with respect to input: ∇_x J(θ, x, y)
        # This is the KEY step of FGSM!
        loss.backward()

        # Extract the gradient from the computational graph
        # x_adv.grad contains ∂J/∂x for each pixel
        gradient = x_adv.grad.data

        # ═══════════════════════════════════════════════════════════════════
        # STEP 4: COMPUTE ADVERSARIAL PERTURBATION
        # ═══════════════════════════════════════════════════════════════════
        # Take the SIGN of the gradient
        # sign(x) = +1 if x > 0, -1 if x < 0, 0 if x = 0
        # This gives us the DIRECTION to perturb each pixel
        grad_sign = gradient.sign()

        # Create perturbation: δ = ε · sign(∇_x J)
        # All perturbed pixels have magnitude exactly ε
        # This satisfies the L∞ constraint: ||δ||_∞ = ε
        perturbation = self.epsilon * grad_sign

        # ═══════════════════════════════════════════════════════════════════
        # STEP 5: APPLY PERTURBATION
        # ═══════════════════════════════════════════════════════════════════
        if self.targeted:
            # TARGETED ATTACK: Minimize loss for target class
            # Step in OPPOSITE direction of gradient (subtract perturbation)
            # This makes the model more confident in the target class
            x_adv = x_adv.detach() - perturbation
        else:
            # UNTARGETED ATTACK: Maximize loss for true class
            # Step in SAME direction as gradient (add perturbation)
            # This makes the model less confident in the true class
            x_adv = x_adv.detach() + perturbation

        # ═══════════════════════════════════════════════════════════════════
        # STEP 6: CLIP TO VALID RANGE
        # ═══════════════════════════════════════════════════════════════════
        # Ensure adversarial examples are valid images
        # For pixel values in [0, 1]: clip to [0, 1]
        # This is necessary because x + δ might exceed valid range
        x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)

        # Store perturbation for analysis
        self.last_perturbation = (x_adv - x).detach()

        if return_perturbation:
            return x_adv, self.last_perturbation
        return x_adv

    def generate_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Generate adversarial examples in batches (memory efficient)

        Useful for large datasets that don't fit in GPU memory.

        Args:
            x: Input images
            y: Labels
            batch_size: Number of images to process at once

        Returns:
            Adversarial examples
        """
        x_adv_list = []
        num_batches = (len(x) + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(x))

            x_batch = x[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]

            x_adv_batch = self.generate(x_batch, y_batch)
            x_adv_list.append(x_adv_batch)

        return torch.cat(x_adv_list, dim=0)

    def analyze_perturbation(
        self,
        x: torch.Tensor,
        x_adv: torch.Tensor,
        verbose: bool = True
    ) -> dict:
        """
        Analyze the perturbation between original and adversarial examples

        Computes various metrics to understand the perturbation:
        - L0 norm: Number of changed pixels
        - L1 norm: Sum of absolute changes
        - L2 norm: Euclidean distance
        - L∞ norm: Maximum per-pixel change
        - Mean/std of perturbation magnitude

        Args:
            x: Original images
            x_adv: Adversarial images
            verbose: If True, print analysis

        Returns:
            Dictionary with perturbation statistics
        """
        # Compute perturbation: δ = x_adv - x
        perturbation = (x_adv - x).detach()

        # Flatten for easier computation
        pert_flat = perturbation.view(perturbation.size(0), -1)

        # ═══════════════════════════════════════════════════════════════════
        # COMPUTE DIFFERENT NORMS
        # ═══════════════════════════════════════════════════════════════════

        # L0 norm: Number of pixels changed (non-zero perturbations)
        l0_norm = (pert_flat != 0).sum(dim=1).float().mean().item()

        # L1 norm: Sum of absolute perturbations
        l1_norm = pert_flat.abs().sum(dim=1).mean().item()

        # L2 norm: Euclidean distance (√(Σ δᵢ²))
        l2_norm = pert_flat.norm(p=2, dim=1).mean().item()

        # L∞ norm: Maximum per-pixel perturbation (max|δᵢ|)
        linf_norm = pert_flat.abs().max(dim=1)[0].mean().item()

        # ═══════════════════════════════════════════════════════════════════
        # STATISTICAL ANALYSIS
        # ═══════════════════════════════════════════════════════════════════

        # Mean absolute perturbation
        mean_pert = pert_flat.abs().mean().item()

        # Standard deviation of perturbation
        std_pert = pert_flat.std().item()

        # Percentage of pixels changed
        total_pixels = pert_flat.size(1)
        pct_changed = (l0_norm / total_pixels) * 100

        stats = {
            'l0_norm': l0_norm,
            'l1_norm': l1_norm,
            'l2_norm': l2_norm,
            'linf_norm': linf_norm,
            'mean_abs_perturbation': mean_pert,
            'std_perturbation': std_pert,
            'pct_pixels_changed': pct_changed,
            'epsilon_used': self.epsilon,
        }

        if verbose:
            print("=" * 70)
            print("PERTURBATION ANALYSIS")
            print("=" * 70)
            print(f"Epsilon (ε):              {self.epsilon:.6f}")
            print(f"L₀ norm (changed pixels): {l0_norm:.2f}")
            print(f"L₁ norm:                  {l1_norm:.6f}")
            print(f"L₂ norm:                  {l2_norm:.6f}")
            print(f"L∞ norm:                  {linf_norm:.6f}")
            print(f"Mean |perturbation|:      {mean_pert:.6f}")
            print(f"Std perturbation:         {std_pert:.6f}")
            print(f"Pixels changed:           {pct_changed:.2f}%")
            print("=" * 70)

            # Verify L∞ constraint
            if abs(linf_norm - self.epsilon) < 1e-5:
                print("✓ L∞ constraint satisfied: ||δ||_∞ = ε")
            elif linf_norm < self.epsilon:
                print(f"⚠ L∞ norm ({linf_norm:.6f}) < ε ({self.epsilon:.6f})")
                print("  (Some perturbations were clipped to valid range)")
            else:
                print(f"✗ L∞ constraint violated! {linf_norm:.6f} > {self.epsilon:.6f}")
            print("=" * 70)

        return stats

    def evaluate_attack_success(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_adv: torch.Tensor,
        target_labels: Optional[torch.Tensor] = None,
        verbose: bool = True
    ) -> dict:
        """
        Evaluate the success rate of the adversarial attack

        Success criteria:
        - Untargeted: Prediction changes from true label
        - Targeted: Prediction becomes the target label

        Args:
            x: Original images
            y: True labels
            x_adv: Adversarial images
            target_labels: Target labels (for targeted attacks)
            verbose: If True, print results

        Returns:
            Dictionary with success metrics
        """
        self.model.eval()

        with torch.no_grad():
            # Predictions on original images
            outputs_clean = self.model(x)
            preds_clean = outputs_clean.argmax(dim=1)

            # Predictions on adversarial images
            outputs_adv = self.model(x_adv)
            preds_adv = outputs_adv.argmax(dim=1)

        # Compute metrics
        batch_size = x.size(0)

        # Original accuracy (should be high for clean images)
        clean_correct = (preds_clean == y).sum().item()
        clean_accuracy = clean_correct / batch_size

        # Adversarial accuracy
        adv_correct = (preds_adv == y).sum().item()
        adv_accuracy = adv_correct / batch_size

        if self.targeted and target_labels is not None:
            # For targeted: success = prediction matches target
            targeted_success = (preds_adv == target_labels).sum().item()
            success_rate = targeted_success / batch_size
        else:
            # For untargeted: success = prediction changed
            misclassified = (preds_adv != y).sum().item()
            success_rate = misclassified / batch_size

        results = {
            'clean_accuracy': clean_accuracy,
            'adversarial_accuracy': adv_accuracy,
            'attack_success_rate': success_rate,
            'accuracy_drop': clean_accuracy - adv_accuracy,
        }

        self.last_success_rate = success_rate

        if verbose:
            print("=" * 70)
            print("ATTACK EVALUATION")
            print("=" * 70)
            print(f"Attack type:              {'Targeted' if self.targeted else 'Untargeted'}")
            print(f"Epsilon (ε):              {self.epsilon:.6f}")
            print(f"Clean accuracy:           {clean_accuracy*100:.2f}%")
            print(f"Adversarial accuracy:     {adv_accuracy*100:.2f}%")
            print(f"Accuracy drop:            {(clean_accuracy-adv_accuracy)*100:.2f}%")
            print(f"Attack success rate:      {success_rate*100:.2f}%")
            print("=" * 70)

        return results

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Allow calling the object as a function"""
        return self.generate(x, y)


def visualize_fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    class_names: list,
    device: str = 'cuda',
    num_samples: int = 5,
    targeted: bool = False,
    target_labels: Optional[torch.Tensor] = None,
    denormalize_fn=None,
    save_path: Optional[str] = None
):
    """
    Visualize FGSM attack with original vs adversarial images

    Creates a comprehensive visualization showing:
    - Original images with predictions
    - Adversarial images with predictions
    - Perturbations (amplified for visibility)
    - Perturbation magnitudes

    Args:
        model: Target model
        images: Input images (should be normalized if model expects it)
        labels: True labels
        epsilon: FGSM epsilon parameter
        class_names: List of class names for display
        device: Device to run on
        num_samples: Number of samples to visualize
        targeted: Whether to use targeted attack
        target_labels: Target labels (for targeted attack)
        denormalize_fn: Function to denormalize images for display
        save_path: Path to save the figure (optional)
    """
    # Create FGSM attack
    fgsm = FGSM(model, epsilon=epsilon, targeted=targeted)

    # Select samples
    num_samples = min(num_samples, len(images))
    images = images[:num_samples].to(device)
    labels = labels[:num_samples].to(device)

    if targeted and target_labels is not None:
        target_labels = target_labels[:num_samples].to(device)
        attack_labels = target_labels
    else:
        attack_labels = labels

    # Generate adversarial examples
    x_adv, perturbation = fgsm.generate(images, attack_labels, return_perturbation=True)

    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs_clean = model(images)
        preds_clean = outputs_clean.argmax(dim=1)

        outputs_adv = model(x_adv)
        preds_adv = outputs_adv.argmax(dim=1)

    # Move to CPU for visualization
    images = images.cpu()
    x_adv = x_adv.cpu()
    perturbation = perturbation.cpu()
    preds_clean = preds_clean.cpu()
    preds_adv = preds_adv.cpu()
    labels = labels.cpu()

    # Denormalize if function provided
    if denormalize_fn is not None:
        images_display = denormalize_fn(images)
        x_adv_display = denormalize_fn(x_adv)
    else:
        images_display = images
        x_adv_display = x_adv

    # Create figure
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Convert to numpy and transpose for matplotlib (CHW -> HWC)
        img_clean = images_display[i].permute(1, 2, 0).numpy()
        img_adv = x_adv_display[i].permute(1, 2, 0).numpy()
        pert = perturbation[i].permute(1, 2, 0).numpy()

        # Clip to valid range for display
        img_clean = np.clip(img_clean, 0, 1)
        img_adv = np.clip(img_adv, 0, 1)

        # Compute perturbation magnitude for each pixel (RGB -> grayscale)
        pert_magnitude = np.abs(pert).mean(axis=2)

        # Column 1: Original image
        axes[i, 0].imshow(img_clean)
        pred_class = class_names[preds_clean[i]]
        true_class = class_names[labels[i]]
        color = 'green' if preds_clean[i] == labels[i] else 'red'
        axes[i, 0].set_title(
            f'Original\nTrue: {true_class}\nPred: {pred_class}',
            fontsize=10,
            color=color
        )
        axes[i, 0].axis('off')

        # Column 2: Adversarial image
        axes[i, 1].imshow(img_adv)
        pred_adv_class = class_names[preds_adv[i]]
        color = 'green' if preds_adv[i] == labels[i] else 'red'
        if targeted and target_labels is not None:
            target_class = class_names[target_labels[i]]
            success = '✓' if preds_adv[i] == target_labels[i] else '✗'
            axes[i, 1].set_title(
                f'Adversarial {success}\nTarget: {target_class}\nPred: {pred_adv_class}',
                fontsize=10,
                color='green' if preds_adv[i] == target_labels[i] else 'red'
            )
        else:
            success = '✓' if preds_adv[i] != labels[i] else '✗'
            axes[i, 1].set_title(
                f'Adversarial {success}\nPred: {pred_adv_class}',
                fontsize=10,
                color=color
            )
        axes[i, 1].axis('off')

        # Column 3: Perturbation (amplified)
        # Amplify for visibility
        pert_vis = (pert - pert.min()) / (pert.max() - pert.min() + 1e-8)
        axes[i, 2].imshow(pert_vis)
        axes[i, 2].set_title(
            f'Perturbation\n(Amplified)',
            fontsize=10
        )
        axes[i, 2].axis('off')

        # Column 4: Perturbation magnitude heatmap
        im = axes[i, 3].imshow(pert_magnitude, cmap='hot', vmin=0, vmax=epsilon)
        axes[i, 3].set_title(
            f'Magnitude\nMax: {pert_magnitude.max():.4f}',
            fontsize=10
        )
        axes[i, 3].axis('off')

        # Add colorbar for magnitude
        plt.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)

    # Overall title
    attack_type = 'Targeted' if targeted else 'Untargeted'
    fig.suptitle(
        f'FGSM Attack Visualization ({attack_type}, ε={epsilon:.4f})',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

    plt.show()


def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    device: str,
    targeted: bool = False
) -> torch.Tensor:
    """
    Standalone function for FGSM attack (backward compatible)

    Args:
        model: Target model
        x: Input images
        y: True labels (or target labels if targeted=True)
        epsilon: Perturbation magnitude
        device: Device to run on (cpu/cuda)
        targeted: Whether to perform targeted attack

    Returns:
        Adversarial examples
    """
    fgsm = FGSM(model, epsilon=epsilon, targeted=targeted)
    return fgsm.generate(x.to(device), y.to(device))


# Example usage and demonstration
if __name__ == '__main__':
    print("=" * 80)
    print("FGSM Attack Implementation - Demonstration")
    print("=" * 80)
    print()
    print("This module implements the Fast Gradient Sign Method (FGSM) attack.")
    print()
    print("Key Features:")
    print("  ✓ Untargeted attacks (cause any misclassification)")
    print("  ✓ Targeted attacks (force specific misclassification)")
    print("  ✓ Perturbation analysis (L0, L1, L2, L∞ norms)")
    print("  ✓ Attack success evaluation")
    print("  ✓ Visualization utilities")
    print("  ✓ Comprehensive mathematical documentation")
    print()
    print("Usage Example:")
    print()
    print("  from attacks import FGSM")
    print("  from models import get_resnet18")
    print()
    print("  # Load model")
    print("  model = get_resnet18(num_classes=10, device='cuda')")
    print()
    print("  # Create FGSM attack")
    print("  fgsm = FGSM(model, epsilon=8/255, targeted=False)")
    print()
    print("  # Generate adversarial examples")
    print("  x_adv = fgsm.generate(images, labels)")
    print()
    print("  # Analyze perturbation")
    print("  stats = fgsm.analyze_perturbation(images, x_adv)")
    print()
    print("  # Evaluate attack success")
    print("  results = fgsm.evaluate_attack_success(images, labels, x_adv)")
    print()
    print("=" * 80)
