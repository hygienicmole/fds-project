"""
Generate example training curves to show expected results

This script creates example visualizations showing typical ResNet18
training behavior on CIFAR-10. These serve as a reference for what
to expect when running train_baseline.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def generate_realistic_training_curves():
    """Generate realistic training curves for ResNet18 on CIFAR-10"""

    epochs = 50
    x = np.arange(1, epochs + 1)

    # Training loss (decreases exponentially with some noise)
    train_loss_base = 2.3 * np.exp(-x / 10) + 0.05
    train_loss_noise = np.random.normal(0, 0.02, epochs)
    train_loss = train_loss_base + train_loss_noise
    train_loss = np.maximum(train_loss, 0.01)  # Floor at 0.01

    # Validation loss (similar but higher, with more variation)
    val_loss_base = 2.3 * np.exp(-x / 12) + 0.35
    val_loss_noise = np.random.normal(0, 0.04, epochs)
    val_loss = val_loss_base + val_loss_noise
    val_loss = np.maximum(val_loss, 0.3)  # Floor at 0.3

    # Training accuracy (logistic growth with noise)
    train_acc_base = 100 / (1 + np.exp(-(x - 15) / 5))
    train_acc_noise = np.random.normal(0, 0.5, epochs)
    train_acc = train_acc_base + train_acc_noise
    train_acc = np.clip(train_acc, 0, 99.5)

    # Validation accuracy (similar but lower plateau)
    val_acc_base = 91.5 / (1 + np.exp(-(x - 18) / 6))
    val_acc_noise = np.random.normal(0, 0.3, epochs)
    val_acc = val_acc_base + val_acc_noise
    val_acc = np.clip(val_acc, 0, 92)

    # Learning rate schedule (MultiStepLR at [50, 75])
    lr = np.full(epochs, 0.1)

    # Epoch times (roughly constant with small variation)
    epoch_times = np.random.normal(25, 1, epochs)

    return {
        'epochs': x,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'lr': lr,
        'epoch_times': epoch_times
    }


def plot_training_curves(data, output_dir):
    """Create comprehensive training curve visualizations"""

    os.makedirs(output_dir, exist_ok=True)

    epochs = data['epochs']
    best_val_acc = np.max(data['val_acc'])
    best_epoch = np.argmax(data['val_acc']) + 1

    # Create main figure with all plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Loss curves
    axes[0, 0].plot(epochs, data['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, data['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[0, 1].plot(epochs, data['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, data['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 1].axhline(y=best_val_acc, color='g', linestyle='--',
                       label=f'Best Val Acc: {best_val_acc:.2f}%', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Learning rate schedule
    axes[1, 0].plot(epochs, data['lr'], 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0.001, 1])

    # Epoch times
    axes[1, 1].plot(epochs, data['epoch_times'], 'm-', linewidth=2)
    axes[1, 1].axhline(y=np.mean(data['epoch_times']), color='r', linestyle='--',
                       label=f'Mean: {np.mean(data["epoch_times"]):.2f}s', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Time (seconds)', fontsize=12)
    axes[1, 1].set_title('Epoch Training Time', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'expected_training_curves.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'expected_training_curves.png')}")
    plt.close()

    # Individual plots
    # Loss only
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, data['train_loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, data['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Expected Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'expected_loss_curve.png'), dpi=150)
    print(f"Saved: {os.path.join(output_dir, 'expected_loss_curve.png')}")
    plt.close()

    # Accuracy only
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, data['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, data['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    plt.axhline(y=best_val_acc, color='g', linestyle='--',
                label=f'Best: {best_val_acc:.2f}%', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Expected Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'expected_accuracy_curve.png'), dpi=150)
    print(f"Saved: {os.path.join(output_dir, 'expected_accuracy_curve.png')}")
    plt.close()

    print(f"\n[OK] All example training curves generated successfully!")
    print(f"\nExpected Results:")
    print(f"  Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"  Final Training Accuracy: {data['train_acc'][-1]:.2f}%")
    print(f"  Final Validation Accuracy: {data['val_acc'][-1]:.2f}%")


def main():
    """Generate example training curves"""
    print("Generating example training curves for ResNet18 on CIFAR-10...")
    print("These show expected results when running train_baseline.py\n")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate data
    data = generate_realistic_training_curves()

    # Create plots
    output_dir = './results/baseline_training'
    plot_training_curves(data, output_dir)

    print(f"\nView the generated plots in: {output_dir}/")
    print("\nTo run actual training:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Run training: python train_baseline.py")


if __name__ == '__main__':
    main()
