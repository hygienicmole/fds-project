"""
Visualization utilities for adversarial examples
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import config
from .data_loader import CIFAR10_CLASSES


def denormalize(tensor, mean=None, std=None):
    """
    Denormalize a tensor image

    Args:
        tensor: Normalized image tensor
        mean: Mean used for normalization
        std: Standard deviation used for normalization

    Returns:
        Denormalized tensor
    """
    if mean is None:
        mean = config.MEAN
    if std is None:
        std = config.STD

    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)

    tensor = tensor * std + mean
    return tensor


def visualize_adversarial_examples(
    original_images,
    adversarial_images,
    original_labels,
    adversarial_predictions,
    num_samples=10,
    save_path=None
):
    """
    Visualize original and adversarial images side by side

    Args:
        original_images: Original clean images
        adversarial_images: Adversarial images
        original_labels: True labels
        adversarial_predictions: Predictions on adversarial examples
        num_samples: Number of samples to visualize
        save_path: Path to save the figure (optional)
    """
    num_samples = min(num_samples, len(original_images))

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Denormalize images
        orig_img = denormalize(original_images[i]).cpu()
        adv_img = denormalize(adversarial_images[i]).cpu()

        # Calculate perturbation
        perturbation = adv_img - orig_img

        # Convert to numpy and transpose for matplotlib
        orig_img = orig_img.permute(1, 2, 0).numpy()
        adv_img = adv_img.permute(1, 2, 0).numpy()
        perturbation = perturbation.permute(1, 2, 0).numpy()

        # Clip to valid range
        orig_img = np.clip(orig_img, 0, 1)
        adv_img = np.clip(adv_img, 0, 1)

        # Original image
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f'Original\nTrue: {CIFAR10_CLASSES[original_labels[i]]}')
        axes[i, 0].axis('off')

        # Adversarial image
        axes[i, 1].imshow(adv_img)
        axes[i, 1].set_title(f'Adversarial\nPred: {CIFAR10_CLASSES[adversarial_predictions[i]]}')
        axes[i, 1].axis('off')

        # Perturbation (amplified for visibility)
        perturbation_vis = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min() + 1e-8)
        axes[i, 2].imshow(perturbation_vis)
        axes[i, 2].set_title('Perturbation\n(Amplified)')
        axes[i, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    plt.show()


def plot_adversarial_comparison(
    clean_accuracy,
    adversarial_accuracies,
    epsilon_values=None,
    attack_name='FGSM',
    save_path=None
):
    """
    Plot accuracy comparison across different epsilon values

    Args:
        clean_accuracy: Accuracy on clean images
        adversarial_accuracies: List of accuracies on adversarial images
        epsilon_values: List of epsilon values
        attack_name: Name of the attack
        save_path: Path to save the figure (optional)
    """
    if epsilon_values is None:
        epsilon_values = config.FGSM_EPSILONS

    plt.figure(figsize=(10, 6))

    plt.plot(epsilon_values, adversarial_accuracies, 'o-', linewidth=2, markersize=8, label=f'{attack_name} Attack')
    plt.axhline(y=clean_accuracy, color='green', linestyle='--', linewidth=2, label='Clean Accuracy')

    plt.xlabel('Epsilon (Perturbation Magnitude)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'Model Robustness vs {attack_name} Attack', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    # Add value annotations
    for i, (eps, acc) in enumerate(zip(epsilon_values, adversarial_accuracies)):
        plt.annotate(f'{acc:.1f}%', (eps, acc), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_training_history(train_losses, train_accuracies, test_accuracies, save_path=None):
    """
    Plot training history

    Args:
        train_losses: List of training losses
        train_accuracies: List of training accuracies
        test_accuracies: List of test accuracies
        save_path: Path to save the figure (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    ax1.plot(train_losses, label='Training Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2.plot(train_accuracies, label='Training Accuracy', linewidth=2)
    ax2.plot(test_accuracies, label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history saved to {save_path}")

    plt.show()
