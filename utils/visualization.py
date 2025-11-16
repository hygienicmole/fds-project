"""
Comprehensive Visualization Utilities for Adversarial Analysis

This module provides advanced visualization functions for analyzing
adversarial attacks, including:
- Side-by-side comparisons of original vs adversarial images
- Perturbation heatmaps
- Confusion matrices
- Class-wise attack success rates
- Confidence score distributions
- Statistical analysis plots
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix
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


# ============================================================================
# ADVANCED VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_comprehensive_comparison(
    model,
    original_images,
    adversarial_images,
    original_labels,
    num_samples=20,
    denormalize_fn=None,
    save_path=None
):
    """
    Generate comprehensive side-by-side comparison of original vs adversarial images

    Shows 20 samples in a grid with:
    - Original images with predictions
    - Adversarial images with predictions
    - Perturbation heatmaps (amplified for visibility)

    Args:
        model: Model for generating predictions
        original_images: Original clean images (B, C, H, W)
        adversarial_images: Adversarial images (B, C, H, W)
        original_labels: True labels (B,)
        num_samples: Number of samples to visualize (default: 20)
        denormalize_fn: Function to denormalize images (optional)
        save_path: Path to save visualization (optional)
    """
    num_samples = min(num_samples, len(original_images))

    # Get predictions
    model.eval()
    with torch.no_grad():
        orig_outputs = model(original_images[:num_samples])
        adv_outputs = model(adversarial_images[:num_samples])

        orig_preds = orig_outputs.argmax(dim=1).cpu().numpy()
        adv_preds = adv_outputs.argmax(dim=1).cpu().numpy()

        # Get confidence scores
        orig_probs = torch.softmax(orig_outputs, dim=1)
        adv_probs = torch.softmax(adv_outputs, dim=1)

        orig_conf = orig_probs.max(dim=1)[0].cpu().numpy()
        adv_conf = adv_probs.max(dim=1)[0].cpu().numpy()

    # Create figure with grid layout
    rows = (num_samples + 3) // 4  # 4 samples per row
    fig = plt.figure(figsize=(20, 5 * rows))
    gs = gridspec.GridSpec(rows, 4, figure=fig, hspace=0.4, wspace=0.3)

    # Denormalize images if function provided
    if denormalize_fn is None:
        denormalize_fn = lambda x: denormalize(x)

    for i in range(num_samples):
        row = i // 4
        col = i % 4

        # Create 3-column subplot for this sample
        ax_group = fig.add_subplot(gs[row, col])
        ax_group.axis('off')

        # Create inner grid for the 3 images
        inner_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[row, col],
                                                     wspace=0.05, hspace=0.05)

        # Denormalize images
        orig_img = denormalize_fn(original_images[i]).cpu()
        adv_img = denormalize_fn(adversarial_images[i]).cpu()

        # Calculate perturbation
        perturbation = (adversarial_images[i] - original_images[i]).cpu()

        # Convert to numpy
        orig_img_np = orig_img.permute(1, 2, 0).numpy()
        adv_img_np = adv_img.permute(1, 2, 0).numpy()
        pert_np = perturbation.permute(1, 2, 0).numpy()

        # Clip to valid range
        orig_img_np = np.clip(orig_img_np, 0, 1)
        adv_img_np = np.clip(adv_img_np, 0, 1)

        # Original image
        ax1 = fig.add_subplot(inner_gs[0, 0])
        ax1.imshow(orig_img_np)
        ax1.axis('off')
        true_label = CIFAR10_CLASSES[original_labels[i].item()]
        pred_label = CIFAR10_CLASSES[orig_preds[i]]
        color = 'green' if orig_preds[i] == original_labels[i].item() else 'red'
        ax1.set_title(f'{true_label}\n{pred_label}\n{orig_conf[i]:.2f}',
                     fontsize=8, color=color, fontweight='bold')

        # Adversarial image
        ax2 = fig.add_subplot(inner_gs[0, 1])
        ax2.imshow(adv_img_np)
        ax2.axis('off')
        adv_label = CIFAR10_CLASSES[adv_preds[i]]
        color = 'red' if adv_preds[i] != original_labels[i].item() else 'green'
        ax2.set_title(f'Adv\n{adv_label}\n{adv_conf[i]:.2f}',
                     fontsize=8, color=color, fontweight='bold')

        # Perturbation heatmap (amplified)
        ax3 = fig.add_subplot(inner_gs[0, 2])
        # Use L2 norm across channels for heatmap
        pert_magnitude = np.sqrt((pert_np ** 2).sum(axis=2))
        im = ax3.imshow(pert_magnitude, cmap='hot', interpolation='nearest')
        ax3.axis('off')
        ax3.set_title(f'Pert\n{pert_magnitude.max():.3f}',
                     fontsize=8, fontweight='bold')

    # Add overall title
    fig.suptitle(f'Adversarial Attack Comparison ({num_samples} samples)',
                fontsize=16, fontweight='bold', y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comprehensive comparison saved to: {save_path}")

    plt.close()


def visualize_perturbation_heatmaps(
    original_images,
    adversarial_images,
    num_samples=10,
    amplification_factor=10,
    denormalize_fn=None,
    save_path=None
):
    """
    Generate detailed perturbation heatmaps with multiple visualizations

    For each sample, shows:
    - Original image
    - Adversarial image
    - Perturbation heatmap (L2 norm across channels)
    - Amplified perturbation (RGB visualization)

    Args:
        original_images: Original clean images
        adversarial_images: Adversarial images
        num_samples: Number of samples to visualize
        amplification_factor: Factor to amplify perturbations for visibility
        denormalize_fn: Function to denormalize images
        save_path: Path to save visualization
    """
    num_samples = min(num_samples, len(original_images))

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    if denormalize_fn is None:
        denormalize_fn = lambda x: denormalize(x)

    for i in range(num_samples):
        # Denormalize images
        orig_img = denormalize_fn(original_images[i]).cpu()
        adv_img = denormalize_fn(adversarial_images[i]).cpu()

        # Calculate perturbation (in normalized space for accuracy)
        perturbation = (adversarial_images[i] - original_images[i]).cpu()

        # Convert to numpy
        orig_np = orig_img.permute(1, 2, 0).numpy()
        adv_np = adv_img.permute(1, 2, 0).numpy()
        pert_np = perturbation.permute(1, 2, 0).numpy()

        # Clip to valid range
        orig_np = np.clip(orig_np, 0, 1)
        adv_np = np.clip(adv_np, 0, 1)

        # Column 1: Original image
        axes[i, 0].imshow(orig_np)
        axes[i, 0].set_title('Original', fontsize=10, fontweight='bold')
        axes[i, 0].axis('off')

        # Column 2: Adversarial image
        axes[i, 1].imshow(adv_np)
        axes[i, 1].set_title('Adversarial', fontsize=10, fontweight='bold')
        axes[i, 1].axis('off')

        # Column 3: Perturbation heatmap (L2 norm)
        pert_magnitude = np.sqrt((pert_np ** 2).sum(axis=2))
        im1 = axes[i, 2].imshow(pert_magnitude, cmap='hot', interpolation='bilinear')
        axes[i, 2].set_title(f'Perturbation Heatmap\nMax: {pert_magnitude.max():.4f}',
                            fontsize=10, fontweight='bold')
        axes[i, 2].axis('off')
        plt.colorbar(im1, ax=axes[i, 2], fraction=0.046, pad=0.04)

        # Column 4: Amplified perturbation (RGB)
        pert_amplified = pert_np * amplification_factor
        pert_amplified = (pert_amplified - pert_amplified.min()) / (pert_amplified.max() - pert_amplified.min() + 1e-8)
        axes[i, 3].imshow(pert_amplified)
        axes[i, 3].set_title(f'Amplified Perturbation\n({amplification_factor}×)',
                            fontsize=10, fontweight='bold')
        axes[i, 3].axis('off')

    plt.suptitle('Perturbation Analysis', fontsize=14, fontweight='bold', y=0.998)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Perturbation heatmaps saved to: {save_path}")

    plt.close()


def plot_confusion_matrix(
    model,
    images,
    true_labels,
    adversarial=False,
    class_names=None,
    normalize=True,
    save_path=None
):
    """
    Generate confusion matrix for model predictions

    Args:
        model: Model for predictions
        images: Input images
        true_labels: True labels
        adversarial: Whether images are adversarial
        class_names: List of class names (default: CIFAR10_CLASSES)
        normalize: Whether to normalize confusion matrix
        save_path: Path to save visualization
    """
    if class_names is None:
        class_names = CIFAR10_CLASSES

    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu().numpy()

    true_labels = true_labels.cpu().numpy()

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predictions)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', square=True, cbar_kws={'label': 'Proportion' if normalize else 'Count'},
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor='gray', ax=ax)

    # Labels and title
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')

    title = 'Confusion Matrix - '
    title += 'Adversarial Predictions' if adversarial else 'Clean Predictions'
    if normalize:
        title += ' (Normalized)'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {save_path}")

    plt.close()

    return cm


def plot_class_wise_attack_success(
    model,
    original_images,
    adversarial_images,
    true_labels,
    class_names=None,
    save_path=None
):
    """
    Plot class-wise attack success rates

    Shows which classes are most vulnerable to adversarial attacks

    Args:
        model: Model for predictions
        original_images: Original clean images
        adversarial_images: Adversarial images
        true_labels: True labels
        class_names: List of class names
        save_path: Path to save visualization
    """
    if class_names is None:
        class_names = CIFAR10_CLASSES

    num_classes = len(class_names)

    # Get predictions
    model.eval()
    with torch.no_grad():
        clean_outputs = model(original_images)
        adv_outputs = model(adversarial_images)

        clean_preds = clean_outputs.argmax(dim=1).cpu().numpy()
        adv_preds = adv_outputs.argmax(dim=1).cpu().numpy()

    true_labels = true_labels.cpu().numpy()

    # Calculate per-class metrics
    class_total = np.zeros(num_classes)
    class_clean_correct = np.zeros(num_classes)
    class_adv_correct = np.zeros(num_classes)
    class_attack_success = np.zeros(num_classes)

    for i in range(len(true_labels)):
        label = true_labels[i]
        class_total[label] += 1

        if clean_preds[i] == label:
            class_clean_correct[label] += 1

            # Attack success only counts if clean prediction was correct
            if adv_preds[i] != label:
                class_attack_success[label] += 1

        if adv_preds[i] == label:
            class_adv_correct[label] += 1

    # Calculate rates
    clean_accuracy = class_clean_correct / (class_total + 1e-8) * 100
    adv_accuracy = class_adv_correct / (class_total + 1e-8) * 100
    attack_success_rate = class_attack_success / (class_total + 1e-8) * 100

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    x = np.arange(num_classes)
    width = 0.35

    # Plot 1: Clean vs Adversarial Accuracy
    bars1 = ax1.bar(x - width/2, clean_accuracy, width, label='Clean Accuracy',
                    color='green', alpha=0.7)
    bars2 = ax1.bar(x + width/2, adv_accuracy, width, label='Adversarial Accuracy',
                    color='red', alpha=0.7)

    ax1.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Class-wise Accuracy: Clean vs Adversarial', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 105])

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=8)

    # Plot 2: Attack Success Rate
    bars3 = ax2.bar(x, attack_success_rate, color='orange', alpha=0.7)

    ax2.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Class-wise Attack Success Rate', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 105])

    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9)

    # Add horizontal line at mean
    mean_asr = attack_success_rate.mean()
    ax2.axhline(y=mean_asr, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_asr:.1f}%')
    ax2.legend(fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Class-wise attack success rates saved to: {save_path}")

    plt.close()

    return {
        'clean_accuracy': clean_accuracy,
        'adversarial_accuracy': adv_accuracy,
        'attack_success_rate': attack_success_rate,
        'class_names': class_names
    }


def plot_confidence_distributions(
    model,
    original_images,
    adversarial_images,
    true_labels,
    save_path=None
):
    """
    Plot distribution of confidence scores for clean vs adversarial predictions

    Shows:
    - Histogram of confidence scores
    - Box plots comparing distributions
    - Statistical comparison

    Args:
        model: Model for predictions
        original_images: Original clean images
        adversarial_images: Adversarial images
        true_labels: True labels
        save_path: Path to save visualization
    """
    model.eval()
    with torch.no_grad():
        # Get predictions and probabilities
        clean_outputs = model(original_images)
        adv_outputs = model(adversarial_images)

        clean_probs = torch.softmax(clean_outputs, dim=1)
        adv_probs = torch.softmax(adv_outputs, dim=1)

        # Get confidence scores (max probability)
        clean_conf = clean_probs.max(dim=1)[0].cpu().numpy()
        adv_conf = adv_probs.max(dim=1)[0].cpu().numpy()

        # Get predictions
        clean_preds = clean_outputs.argmax(dim=1).cpu().numpy()
        adv_preds = adv_outputs.argmax(dim=1).cpu().numpy()

        # Get confidence for true class
        true_labels_np = true_labels.cpu().numpy()
        clean_conf_true = clean_probs[range(len(true_labels)), true_labels].cpu().numpy()
        adv_conf_true = adv_probs[range(len(true_labels)), true_labels].cpu().numpy()

    # Separate correct and incorrect predictions
    clean_correct_mask = clean_preds == true_labels_np
    adv_correct_mask = adv_preds == true_labels_np

    clean_conf_correct = clean_conf[clean_correct_mask]
    clean_conf_incorrect = clean_conf[~clean_correct_mask]
    adv_conf_correct = adv_conf[adv_correct_mask]
    adv_conf_incorrect = adv_conf[~adv_correct_mask]

    # Create figure with 4 subplots
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Plot 1: Histogram of max confidence
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(clean_conf, bins=50, alpha=0.6, label='Clean', color='green', density=True)
    ax1.hist(adv_conf, bins=50, alpha=0.6, label='Adversarial', color='red', density=True)
    ax1.set_xlabel('Confidence Score', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax1.set_title('Distribution of Maximum Confidence Scores', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Add mean lines
    ax1.axvline(clean_conf.mean(), color='green', linestyle='--', linewidth=2,
               label=f'Clean mean: {clean_conf.mean():.3f}')
    ax1.axvline(adv_conf.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Adv mean: {adv_conf.mean():.3f}')
    ax1.legend(fontsize=9)

    # Plot 2: Box plot comparison
    ax2 = fig.add_subplot(gs[0, 1])
    data_to_plot = [clean_conf, adv_conf]
    bp = ax2.boxplot(data_to_plot, labels=['Clean', 'Adversarial'],
                     patch_artist=True, showmeans=True)

    # Color the boxes
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor('red')
    bp['boxes'][1].set_alpha(0.6)

    ax2.set_ylabel('Confidence Score', fontsize=11, fontweight='bold')
    ax2.set_title('Confidence Score Comparison', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Confidence for true class
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(clean_conf_true, bins=50, alpha=0.6, label='Clean', color='green', density=True)
    ax3.hist(adv_conf_true, bins=50, alpha=0.6, label='Adversarial', color='red', density=True)
    ax3.set_xlabel('Confidence for True Class', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax3.set_title('Distribution of Confidence for True Class', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)

    # Plot 4: Correct vs Incorrect predictions
    ax4 = fig.add_subplot(gs[1, 1])
    data_correct_incorrect = [
        clean_conf_correct, clean_conf_incorrect,
        adv_conf_correct, adv_conf_incorrect
    ]
    labels = ['Clean\n(Correct)', 'Clean\n(Incorrect)',
             'Adv\n(Correct)', 'Adv\n(Incorrect)']

    bp2 = ax4.boxplot(data_correct_incorrect, labels=labels,
                      patch_artist=True, showmeans=True)

    # Color the boxes
    colors = ['green', 'lightcoral', 'darkgreen', 'red']
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax4.set_ylabel('Confidence Score', fontsize=11, fontweight='bold')
    ax4.set_title('Confidence: Correct vs Incorrect Predictions', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    # Add overall title
    fig.suptitle('Confidence Score Analysis: Clean vs Adversarial',
                fontsize=14, fontweight='bold', y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confidence distributions saved to: {save_path}")

    plt.close()

    # Return statistics
    return {
        'clean_mean': clean_conf.mean(),
        'clean_std': clean_conf.std(),
        'adv_mean': adv_conf.mean(),
        'adv_std': adv_conf.std(),
        'clean_conf_true_mean': clean_conf_true.mean(),
        'adv_conf_true_mean': adv_conf_true.mean()
    }


def generate_all_visualizations(
    model,
    original_images,
    adversarial_images,
    true_labels,
    attack_name='FGSM',
    epsilon=0.03,
    save_dir='./results/visualizations',
    denormalize_fn=None
):
    """
    Generate all comprehensive visualizations in one call

    Creates:
    1. Comprehensive comparison (20 samples)
    2. Perturbation heatmaps (10 samples)
    3. Confusion matrix (clean)
    4. Confusion matrix (adversarial)
    5. Class-wise attack success rates
    6. Confidence score distributions

    Args:
        model: Model for predictions
        original_images: Original clean images
        adversarial_images: Adversarial images
        true_labels: True labels
        attack_name: Name of the attack (for titles)
        epsilon: Epsilon value used (for titles)
        save_dir: Directory to save all visualizations
        denormalize_fn: Function to denormalize images

    Returns:
        Dictionary with paths to all generated visualizations
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f"GENERATING COMPREHENSIVE VISUALIZATIONS: {attack_name} (ε={epsilon:.4f})")
    print(f"{'=' * 80}\n")

    paths = {}

    # 1. Comprehensive comparison
    print("1. Generating comprehensive comparison (20 samples)...")
    paths['comparison'] = os.path.join(save_dir, f'{attack_name.lower()}_comparison.png')
    visualize_comprehensive_comparison(
        model, original_images, adversarial_images, true_labels,
        num_samples=20, denormalize_fn=denormalize_fn,
        save_path=paths['comparison']
    )

    # 2. Perturbation heatmaps
    print("\n2. Generating perturbation heatmaps (10 samples)...")
    paths['heatmaps'] = os.path.join(save_dir, f'{attack_name.lower()}_perturbation_heatmaps.png')
    visualize_perturbation_heatmaps(
        original_images, adversarial_images,
        num_samples=10, denormalize_fn=denormalize_fn,
        save_path=paths['heatmaps']
    )

    # 3. Clean confusion matrix
    print("\n3. Generating confusion matrix (clean predictions)...")
    paths['confusion_clean'] = os.path.join(save_dir, 'confusion_matrix_clean.png')
    plot_confusion_matrix(
        model, original_images, true_labels,
        adversarial=False, normalize=True,
        save_path=paths['confusion_clean']
    )

    # 4. Adversarial confusion matrix
    print("\n4. Generating confusion matrix (adversarial predictions)...")
    paths['confusion_adv'] = os.path.join(save_dir, f'confusion_matrix_{attack_name.lower()}.png')
    plot_confusion_matrix(
        model, adversarial_images, true_labels,
        adversarial=True, normalize=True,
        save_path=paths['confusion_adv']
    )

    # 5. Class-wise attack success
    print("\n5. Generating class-wise attack success rates...")
    paths['class_wise'] = os.path.join(save_dir, f'{attack_name.lower()}_class_wise_success.png')
    class_stats = plot_class_wise_attack_success(
        model, original_images, adversarial_images, true_labels,
        save_path=paths['class_wise']
    )

    # 6. Confidence distributions
    print("\n6. Generating confidence score distributions...")
    paths['confidence'] = os.path.join(save_dir, f'{attack_name.lower()}_confidence_distributions.png')
    conf_stats = plot_confidence_distributions(
        model, original_images, adversarial_images, true_labels,
        save_path=paths['confidence']
    )

    print(f"\n{'=' * 80}")
    print("ALL VISUALIZATIONS GENERATED!")
    print(f"{'=' * 80}")
    print(f"\nSaved to: {save_dir}/")
    for name, path in paths.items():
        print(f"  - {os.path.basename(path)}")
    print(f"{'=' * 80}\n")

    return {
        'paths': paths,
        'class_statistics': class_stats,
        'confidence_statistics': conf_stats
    }
