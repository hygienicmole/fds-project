"""
Projected Gradient Descent (PGD) Attack Implementation

This module implements the PGD attack with comprehensive features including:
- Untargeted and targeted attacks
- Attack progression tracking and visualization
- L-infinity norm constraint enforcement
- Random initialization
- Perturbation analysis
- Detailed mathematical explanations

Reference:
Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2017).
Towards deep learning models resistant to adversarial attacks.
arXiv preprint arXiv:1706.06083.

Paper: https://arxiv.org/abs/1706.06083
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union, List, Dict
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PGD:
    """
    Projected Gradient Descent (PGD) Attack

    ═══════════════════════════════════════════════════════════════════════════
    MATHEMATICAL BACKGROUND
    ═══════════════════════════════════════════════════════════════════════════

    PGD is a powerful iterative adversarial attack that extends FGSM by
    performing multiple gradient steps with projection back to the epsilon ball.

    It is considered one of the strongest first-order attacks and is used as
    the standard for evaluating adversarial robustness.

    ALGORITHM:
    ──────────
    1. Initialize: x^0 = x + random_noise (if random_start=True)
                   or x^0 = x (if random_start=False)

    2. For t = 0 to T-1:
        a. Compute gradient: g = ∇_x J(θ, x^t, y)
        b. Take gradient step: x^(t+1) = x^t + α · sign(g)
        c. Project to epsilon ball: x^(t+1) = Π_(x+S)(x^(t+1))
        d. Clip to valid range: x^(t+1) = clip(x^(t+1), [0, 1])

    3. Return x^T

    Where:
        x       = Original input image
        x^t     = Adversarial example at iteration t
        T       = Number of iterations
        α       = Step size (alpha)
        ε       = Maximum perturbation (epsilon)
        S       = Allowed perturbation set: {δ : ||δ||_∞ ≤ ε}
        Π_S(·)  = Projection onto set S
        sign(·) = Sign function
        J       = Loss function

    UNTARGETED ATTACK:
    ──────────────────
    Goal: Maximize loss for true label

    x^(t+1) = x^t + α · sign(∇_x J(θ, x^t, y_true))

    TARGETED ATTACK:
    ────────────────
    Goal: Minimize loss for target label

    x^(t+1) = x^t - α · sign(∇_x J(θ, x^t, y_target))

    PROJECTION OPERATION:
    ─────────────────────
    The key difference from FGSM is the PROJECTION step after each iteration.

    Projection onto L∞ ball:
        δ = clip(x^t - x, -ε, ε)
        x^t = x + δ

    This ensures: ||x^t - x||_∞ ≤ ε

    Why projection?
    - Keeps perturbation bounded
    - Allows multiple gradient steps without leaving epsilon ball
    - Enables stronger attacks than single-step FGSM

    RANDOM INITIALIZATION:
    ──────────────────────
    Starting from a random point in the epsilon ball helps:
    1. Escape local minima
    2. Find stronger adversarial examples
    3. Increase attack diversity

    x^0 = x + Uniform(-ε, ε)

    STEP SIZE (ALPHA):
    ──────────────────
    Common choices:
    - α = ε / (0.25 * T)    [from original paper]
    - α = ε / 4             [common practice]
    - α = 2ε / T            [alternative]

    Typical: α = 2/255 for ε = 8/255

    ITERATION COUNT:
    ────────────────
    More iterations = stronger attack, but:
    - Diminishing returns after ~20-40 iterations
    - Computational cost increases linearly
    - May overfit to specific images

    Common values: 7, 10, 20, 40, 100

    WHY PGD IS STRONGER THAN FGSM:
    ───────────────────────────────
    1. Multiple gradient steps find better adversarial directions
    2. Random start helps escape local minima
    3. Projection allows cumulative perturbations
    4. Can navigate around defenses that block single-step attacks

    UNTARGETED vs TARGETED:
    ───────────────────────
    Same as FGSM:
    - Untargeted: ADD gradient (maximize loss)
    - Targeted: SUBTRACT gradient (minimize loss for target)

    ═══════════════════════════════════════════════════════════════════════════

    Args:
        model: The target model to attack
        epsilon: Maximum perturbation magnitude (L∞ norm)
        alpha: Step size for each iteration
        iterations: Number of attack iterations
        random_start: Whether to start from random point in epsilon ball
        clip_min: Minimum pixel value for clipping (default: 0.0)
        clip_max: Maximum pixel value for clipping (default: 1.0)
        targeted: Whether to perform targeted attack (default: False)

    Example Usage:
        >>> # Untargeted PGD
        >>> pgd = PGD(model, epsilon=8/255, alpha=2/255, iterations=20)
        >>> x_adv = pgd.generate(images, labels)

        >>> # Targeted PGD with progress tracking
        >>> pgd_targeted = PGD(model, epsilon=8/255, alpha=2/255, iterations=20, targeted=True)
        >>> x_adv, history = pgd_targeted.generate(images, targets, track_progress=True)
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.03,
        alpha: float = 0.01,
        iterations: int = 40,
        random_start: bool = True,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        targeted: bool = False
    ):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.iterations = iterations
        self.random_start = random_start
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.criterion = nn.CrossEntropyLoss()

        # For tracking attack progress
        self.last_perturbation = None
        self.last_success_rate = None
        self.attack_history = None

    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        track_progress: bool = False,
        return_perturbation: bool = False,
        return_intermediate_images: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, Dict], List[Dict]]:
        """
        Generate adversarial examples using PGD

        Algorithm Steps:
        ────────────────
        1. INITIALIZATION:
           - If random_start: x^0 = x + Uniform(-ε, ε)
           - Else: x^0 = x
           - Clip to valid range

        2. ITERATIVE ATTACK (t = 0 to T-1):
           a. Enable gradient tracking for x^t
           b. FORWARD PASS: compute predictions f_θ(x^t)
           c. COMPUTE LOSS: J(θ, x^t, y)
           d. BACKWARD PASS: compute ∇_x J
           e. GRADIENT STEP:
              - If targeted: x^(t+1) = x^t - α · sign(∇_x J)
              - If untargeted: x^(t+1) = x^t + α · sign(∇_x J)
           f. PROJECTION: Π_(x+S)(x^(t+1))
              - δ = clip(x^(t+1) - x, -ε, ε)
              - x^(t+1) = x + δ
           g. CLIPPING: x^(t+1) = clip(x^(t+1), [0, 1])

        3. RETURN: Final adversarial example x^T

        Args:
            x: Input images (batch_size, channels, height, width)
            y: Labels (true labels for untargeted, target labels for targeted)
            track_progress: If True, track metrics at each iteration
            return_perturbation: If True, also return the perturbation
            return_intermediate_images: If True, return list of intermediate results

        Returns:
            x_adv: Adversarial examples
            perturbation: (optional) The final perturbation δ
            history: (optional) Attack progression history
            intermediate_results: (optional) List of dictionaries with intermediate images and metrics
        """
        # Set model to evaluation mode
        self.model.eval()

        # Store original images for projection
        x_original = x.clone().detach()

        # ═══════════════════════════════════════════════════════════════════
        # STEP 1: INITIALIZATION
        # ═══════════════════════════════════════════════════════════════════

        x_adv = x.clone().detach()

        if self.random_start:
            # Random initialization within epsilon ball
            # x^0 = x + Uniform(-ε, ε)
            # This helps escape local minima and find stronger attacks
            random_noise = torch.empty_like(x_adv).uniform_(-self.epsilon, self.epsilon)
            x_adv = x_adv + random_noise

            # Ensure we stay within valid pixel range
            x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)

        # Initialize progress tracking if requested
        if track_progress:
            history = {
                'iterations': [],
                'losses': [],
                'accuracies': [],
                'success_rates': [],
                'linf_norms': [],
                'l2_norms': []
            }

        # Initialize intermediate results if requested
        intermediate_results = []
        if return_intermediate_images:
            # Add initial state (iteration 0)
            with torch.no_grad():
                outputs = self.model(x_adv)
                probs = torch.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, dim=1)
                
                # Calculate perturbation
                pert = x_adv - x_original
                linf = pert.abs().max().item()
                
                intermediate_results.append({
                    'iteration': 0,
                    'image': x_adv.clone().detach().cpu(),
                    'prediction': pred.item(),
                    'confidence': conf.item(),
                    'is_adversarial': (pred != y).item() if not self.targeted else (pred == y).item(),
                    'perturbation_linf': linf
                })

        # ═══════════════════════════════════════════════════════════════════
        # STEP 2: ITERATIVE ATTACK
        # ═══════════════════════════════════════════════════════════════════

        for iteration in range(self.iterations):
            # Enable gradient computation for current adversarial example
            x_adv.requires_grad = True

            # ───────────────────────────────────────────────────────────────
            # 2a. FORWARD PASS
            # ───────────────────────────────────────────────────────────────
            # Compute model predictions: f_θ(x^t)
            outputs = self.model(x_adv)

            # ───────────────────────────────────────────────────────────────
            # 2b. COMPUTE LOSS
            # ───────────────────────────────────────────────────────────────
            # For untargeted: compute loss with true labels
            # For targeted: compute loss with target labels
            loss = self.criterion(outputs, y)

            # ───────────────────────────────────────────────────────────────
            # 2c. BACKWARD PASS
            # ───────────────────────────────────────────────────────────────
            # Compute gradient of loss with respect to input: ∇_x J(θ, x^t, y)
            self.model.zero_grad()
            loss.backward()

            # Extract gradient
            gradient = x_adv.grad.data

            # ───────────────────────────────────────────────────────────────
            # 2d. GRADIENT STEP
            # ───────────────────────────────────────────────────────────────
            with torch.no_grad():
                # Take sign of gradient
                # sign(x) = +1 if x > 0, -1 if x < 0, 0 if x = 0
                grad_sign = gradient.sign()

                if self.targeted:
                    # TARGETED ATTACK: Minimize loss for target class
                    # Step in OPPOSITE direction (subtract gradient)
                    # x^(t+1) = x^t - α · sign(∇_x J)
                    x_adv = x_adv.detach() - self.alpha * grad_sign
                else:
                    # UNTARGETED ATTACK: Maximize loss for true class
                    # Step in SAME direction (add gradient)
                    # x^(t+1) = x^t + α · sign(∇_x J)
                    x_adv = x_adv.detach() + self.alpha * grad_sign

                # ───────────────────────────────────────────────────────────
                # 2e. PROJECTION TO EPSILON BALL
                # ───────────────────────────────────────────────────────────
                # This is the KEY difference from FGSM!
                #
                # After the gradient step, x^(t+1) might be outside the
                # epsilon ball around the original image. We need to project
                # it back to ensure ||x^(t+1) - x||_∞ ≤ ε
                #
                # Projection for L∞ norm:
                # 1. Compute perturbation: δ = x^(t+1) - x
                # 2. Clip perturbation: δ_clipped = clip(δ, -ε, ε)
                # 3. Apply clipped perturbation: x^(t+1) = x + δ_clipped

                perturbation = x_adv - x_original
                perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
                x_adv = x_original + perturbation

                # ───────────────────────────────────────────────────────────
                # 2f. CLIP TO VALID RANGE
                # ───────────────────────────────────────────────────────────
                # Ensure adversarial examples are valid images
                # For images in [0, 1]: clip to [0, 1]
                x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)

            # Track progress if requested
            if track_progress:
                with torch.no_grad():
                    # Compute current perturbation norms
                    current_pert = x_adv - x_original
                    linf_norm = current_pert.abs().max().item()
                    l2_norm = current_pert.norm(p=2).item() / current_pert.numel()**0.5

                    # Compute current predictions
                    current_outputs = self.model(x_adv)
                    current_preds = current_outputs.argmax(dim=1)

                    # Compute accuracy
                    accuracy = (current_preds == y).float().mean().item()

                    # Compute success rate
                    if self.targeted:
                        success_rate = (current_preds == y).float().mean().item()
                    else:
                        success_rate = (current_preds != y).float().mean().item()

                    # Store metrics
                    history['iterations'].append(iteration)
                    history['losses'].append(loss.item())
                    history['accuracies'].append(accuracy)
                    history['success_rates'].append(success_rate)
                    history['linf_norms'].append(linf_norm)
                    history['l2_norms'].append(l2_norm)

            # Store intermediate results if requested
            if return_intermediate_images:
                with torch.no_grad():
                    # We need to re-evaluate because x_adv changed
                    outputs = self.model(x_adv)
                    probs = torch.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs, dim=1)
                    
                    pert = x_adv - x_original
                    linf = pert.abs().max().item()
                    
                    intermediate_results.append({
                        'iteration': iteration + 1,
                        'image': x_adv.clone().detach().cpu(),
                        'prediction': pred.item(),
                        'confidence': conf.item(),
                        'is_adversarial': (pred != y).item() if not self.targeted else (pred == y).item(),
                        'perturbation_linf': linf
                    })

        # ═══════════════════════════════════════════════════════════════════
        # STEP 3: RETURN RESULTS
        # ═══════════════════════════════════════════════════════════════════

        # Store final perturbation for analysis
        self.last_perturbation = (x_adv - x_original).detach()

        # Store history
        if track_progress:
            self.attack_history = history

        # Return based on what was requested
        # Return based on what was requested
        if return_intermediate_images:
            return intermediate_results
        elif track_progress and return_perturbation:
            return x_adv.detach(), self.last_perturbation, history
        elif track_progress:
            return x_adv.detach(), history
        elif return_perturbation:
            return x_adv.detach(), self.last_perturbation
        else:
            return x_adv.detach()

    def generate_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Generate adversarial examples in batches (memory efficient)

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

        Same as FGSM but for PGD results.

        Args:
            x: Original images
            x_adv: Adversarial images
            verbose: If True, print analysis

        Returns:
            Dictionary with perturbation statistics
        """
        perturbation = (x_adv - x).detach()
        pert_flat = perturbation.view(perturbation.size(0), -1)

        # Compute different norms
        l0_norm = (pert_flat != 0).sum(dim=1).float().mean().item()
        l1_norm = pert_flat.abs().sum(dim=1).mean().item()
        l2_norm = pert_flat.norm(p=2, dim=1).mean().item()
        linf_norm = pert_flat.abs().max(dim=1)[0].mean().item()

        # Statistical analysis
        mean_pert = pert_flat.abs().mean().item()
        std_pert = pert_flat.std().item()

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
            print("PGD PERTURBATION ANALYSIS")
            print("=" * 70)
            print(f"Epsilon (ε):              {self.epsilon:.6f}")
            print(f"Alpha (α):                {self.alpha:.6f}")
            print(f"Iterations:               {self.iterations}")
            print(f"Random start:             {self.random_start}")
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
                print("[OK] L∞ constraint satisfied: ||δ||_∞ = ε")
            elif linf_norm < self.epsilon:
                print(f"[WARNING] L∞ norm ({linf_norm:.6f}) < ε ({self.epsilon:.6f})")
                print("  (Some perturbations were clipped to valid range)")
            else:
                print(f"[X] L∞ constraint violated! {linf_norm:.6f} > {self.epsilon:.6f}")
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
        Evaluate the success rate of the PGD attack

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
            outputs_clean = self.model(x)
            preds_clean = outputs_clean.argmax(dim=1)

            outputs_adv = self.model(x_adv)
            preds_adv = outputs_adv.argmax(dim=1)

        batch_size = x.size(0)

        clean_correct = (preds_clean == y).sum().item()
        clean_accuracy = clean_correct / batch_size

        adv_correct = (preds_adv == y).sum().item()
        adv_accuracy = adv_correct / batch_size

        if self.targeted and target_labels is not None:
            targeted_success = (preds_adv == target_labels).sum().item()
            success_rate = targeted_success / batch_size
        else:
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
            print("PGD ATTACK EVALUATION")
            print("=" * 70)
            print(f"Attack type:              {'Targeted' if self.targeted else 'Untargeted'}")
            print(f"Epsilon (ε):              {self.epsilon:.6f}")
            print(f"Alpha (α):                {self.alpha:.6f}")
            print(f"Iterations:               {self.iterations}")
            print(f"Random start:             {self.random_start}")
            print(f"Clean accuracy:           {clean_accuracy*100:.2f}%")
            print(f"Adversarial accuracy:     {adv_accuracy*100:.2f}%")
            print(f"Accuracy drop:            {(clean_accuracy-adv_accuracy)*100:.2f}%")
            print(f"Attack success rate:      {success_rate*100:.2f}%")
            print("=" * 70)

        return results

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Allow calling the object as a function"""
        return self.generate(x, y)


def visualize_pgd_progression(
    history: Dict,
    save_path: Optional[str] = None
):
    """
    Visualize PGD attack progression over iterations

    Creates plots showing how the attack evolves:
    - Loss over iterations
    - Accuracy over iterations
    - Success rate over iterations
    - Perturbation norms (L2 and L∞) over iterations

    Args:
        history: Attack history dictionary from PGD.generate(track_progress=True)
        save_path: Path to save the figure (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    iterations = history['iterations']

    # Plot 1: Loss
    axes[0, 0].plot(iterations, history['losses'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Iteration', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Loss Over PGD Iterations', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Accuracy
    axes[0, 1].plot(iterations, [a*100 for a in history['accuracies']], 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Iteration', fontsize=11)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=11)
    axes[0, 1].set_title('Accuracy Over PGD Iterations', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Success Rate
    axes[1, 0].plot(iterations, [s*100 for s in history['success_rates']], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Iteration', fontsize=11)
    axes[1, 0].set_ylabel('Success Rate (%)', fontsize=11)
    axes[1, 0].set_title('Attack Success Rate Over Iterations', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Perturbation Norms
    axes[1, 1].plot(iterations, history['linf_norms'], 'purple', linewidth=2, label='L∞ Norm')
    axes[1, 1].plot(iterations, history['l2_norms'], 'orange', linewidth=2, label='L2 Norm (avg)')
    axes[1, 1].set_xlabel('Iteration', fontsize=11)
    axes[1, 1].set_ylabel('Norm Value', fontsize=11)
    axes[1, 1].set_title('Perturbation Norms Over Iterations', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Progression visualization saved to: {save_path}")

    plt.show()


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    alpha: float,
    iterations: int,
    device: str,
    random_start: bool = True,
    targeted: bool = False
) -> torch.Tensor:
    """
    Standalone function for PGD attack (backward compatible)

    Args:
        model: Target model
        x: Input images
        y: True labels (or target labels if targeted=True)
        epsilon: Maximum perturbation magnitude
        alpha: Step size
        iterations: Number of iterations
        device: Device to run on (cpu/cuda)
        random_start: Whether to use random initialization
        targeted: Whether to perform targeted attack

    Returns:
        Adversarial examples
    """
    pgd = PGD(
        model,
        epsilon=epsilon,
        alpha=alpha,
        iterations=iterations,
        random_start=random_start,
        targeted=targeted
    )
    return pgd.generate(x.to(device), y.to(device))


# Example usage and demonstration
if __name__ == '__main__':
    print("=" * 80)
    print("PGD Attack Implementation - Demonstration")
    print("=" * 80)
    print()
    print("This module implements the Projected Gradient Descent (PGD) attack.")
    print()
    print("Key Features:")
    print("  [OK] Iterative attack (stronger than FGSM)")
    print("  [OK] Untargeted and targeted attacks")
    print("  [OK] Random initialization")
    print("  [OK] L∞ norm constraint with projection")
    print("  [OK] Attack progression tracking")
    print("  [OK] Perturbation analysis")
    print("  [OK] Visualization of attack evolution")
    print()
    print("Usage Example:")
    print()
    print("  from attacks import PGD")
    print("  from models import get_resnet18")
    print()
    print("  # Load model")
    print("  model = get_resnet18(num_classes=10, device='cuda')")
    print()
    print("  # Create PGD attack")
    print("  pgd = PGD(model, epsilon=8/255, alpha=2/255, iterations=20)")
    print()
    print("  # Generate adversarial examples with progress tracking")
    print("  x_adv, history = pgd.generate(images, labels, track_progress=True)")
    print()
    print("  # Visualize attack progression")
    print("  from attacks.pgd import visualize_pgd_progression")
    print("  visualize_pgd_progression(history)")
    print()
    print("=" * 80)
