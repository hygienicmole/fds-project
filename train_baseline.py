"""
Baseline Training Script for ResNet18 on CIFAR-10

This script trains a ResNet18 model on CIFAR-10 for 50 epochs and generates
comprehensive training metrics and visualizations.

Usage:
    python train_baseline.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import time
from datetime import datetime
from tqdm import tqdm

from models import get_resnet18, save_checkpoint, count_parameters
from utils import get_cifar10_loaders
import config


class BaselineTrainer:
    """
    Baseline Trainer for ResNet18 on CIFAR-10

    This class handles the complete training pipeline for a ResNet18 model on the
    CIFAR-10 dataset, including training, validation, metrics tracking, and
    visualization generation.

    The trainer implements standard supervised learning with:
    - SGD optimizer with momentum
    - MultiStepLR learning rate scheduling
    - Cross-entropy loss
    - Automatic best model checkpointing
    - Comprehensive metrics logging

    Attributes:
        model (nn.Module): ResNet18 model to train
        train_loader (DataLoader): Training data loader
        test_loader (DataLoader): Validation/test data loader
        optimizer (Optimizer): PyTorch optimizer (typically SGD)
        scheduler (LRScheduler): Learning rate scheduler (typically MultiStepLR)
        criterion (nn.Module): Loss function (typically CrossEntropyLoss)
        device (torch.device): Device to train on (cuda or cpu)
        num_epochs (int): Total number of training epochs

        train_losses (list): Training loss history
        train_accuracies (list): Training accuracy history
        val_losses (list): Validation loss history
        val_accuracies (list): Validation accuracy history
        learning_rates (list): Learning rate history
        epoch_times (list): Time taken per epoch

        best_val_acc (float): Best validation accuracy achieved
        best_epoch (int): Epoch number of best validation accuracy
        results_dir (str): Directory to save results and visualizations

    Example:
        >>> model = get_resnet18(num_classes=10, device='cuda')
        >>> train_loader, test_loader = get_cifar10_loaders()
        >>> optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        >>> criterion = nn.CrossEntropyLoss()
        >>>
        >>> trainer = BaselineTrainer(
        ...     model, train_loader, test_loader, optimizer, scheduler,
        ...     criterion, device='cuda', num_epochs=200
        ... )
        >>> trainer.train()
        >>> trainer.save_metrics()
    """

    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        criterion,
        device,
        num_epochs=50
    ):
        """
        Initialize the baseline trainer.

        Args:
            model (nn.Module): PyTorch model to train
            train_loader (DataLoader): Training data loader
            test_loader (DataLoader): Validation/test data loader
            optimizer (Optimizer): PyTorch optimizer
            scheduler (LRScheduler): Learning rate scheduler
            criterion (nn.Module): Loss function
            device (torch.device or str): Device to train on ('cuda' or 'cpu')
            num_epochs (int, optional): Number of training epochs. Default: 50
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.num_epochs = num_epochs

        # Metrics tracking
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        self.epoch_times = []

        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0

        # Create results directory
        self.results_dir = os.path.join(config.RESULTS_DIR, 'baseline_training')
        os.makedirs(self.results_dir, exist_ok=True)

    def train_epoch(self, epoch):
        """
        Train the model for one epoch.

        Performs one complete pass through the training data, computing gradients
        and updating model parameters. Tracks loss and accuracy metrics.

        Args:
            epoch (int): Current epoch number (0-indexed)

        Returns:
            tuple: (average_loss, accuracy)
                - average_loss (float): Mean training loss for the epoch
                - accuracy (float): Training accuracy as percentage (0-100)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Train]')

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} [Val]')

            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                pbar.set_postfix({
                    'loss': f'{running_loss/(batch_idx+1):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })

        val_loss = running_loss / len(self.test_loader)
        val_acc = 100. * correct / total

        return val_loss, val_acc

    def train(self):
        """Main training loop"""
        print("=" * 80)
        print("BASELINE TRAINING: ResNet18 on CIFAR-10")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Batch size: {config.BATCH_SIZE}")
        print(f"Initial LR: {config.LEARNING_RATE}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")

        total_params, trainable_params = count_parameters(self.model)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("=" * 80)

        start_time = time.time()

        for epoch in range(self.num_epochs):
            epoch_start = time.time()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)

            print(f"\nEpoch {epoch+1}/{self.num_epochs} | LR: {current_lr:.6f}")

            # Train and validate
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)

            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            # Calculate epoch time
            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)

            # Print summary
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"Time: {epoch_time:.2f}s")

            # Update learning rate
            self.scheduler.step()

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                print(f"★ New best validation accuracy: {val_acc:.2f}%")

                # Save best model
                best_path = os.path.join(config.MODEL_DIR, 'baseline_best_model.pth')
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_acc, val_loss,
                    best_path, self.scheduler, val_acc
                )

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(
                    config.CHECKPOINT_DIR,
                    f'baseline_checkpoint_epoch_{epoch+1}.pth'
                )
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_acc, val_loss,
                    checkpoint_path, self.scheduler, self.best_val_acc
                )

        total_time = time.time() - start_time

        # Print final summary
        self._print_summary(total_time)

        # Generate plots
        self._plot_training_curves()

        # Save metrics
        self._save_metrics(total_time)

        return {
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'final_train_acc': self.train_accuracies[-1],
            'final_val_acc': self.val_accuracies[-1],
            'total_time': total_time
        }

    def _print_summary(self, total_time):
        """Print training summary"""
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE!")
        print("=" * 80)
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch+1})")
        print(f"Final Train Accuracy: {self.train_accuracies[-1]:.2f}%")
        print(f"Final Validation Accuracy: {self.val_accuracies[-1]:.2f}%")
        print(f"Total Training Time: {total_time/60:.2f} minutes")
        print(f"Average Epoch Time: {np.mean(self.epoch_times):.2f}s")
        print("=" * 80)

    def _plot_training_curves(self):
        """Generate and save training curves"""
        print("\nGenerating training curves...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        epochs = range(1, self.num_epochs + 1)

        # Loss curves
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy curves
        axes[0, 1].plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        axes[0, 1].axhline(y=self.best_val_acc, color='g', linestyle='--',
                           label=f'Best Val Acc: {self.best_val_acc:.2f}%', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)

        # Learning rate schedule
        axes[1, 0].plot(epochs, self.learning_rates, 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

        # Epoch times
        axes[1, 1].plot(epochs, self.epoch_times, 'm-', linewidth=2)
        axes[1, 1].axhline(y=np.mean(self.epoch_times), color='r', linestyle='--',
                           label=f'Mean: {np.mean(self.epoch_times):.2f}s', linewidth=2)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Time (seconds)', fontsize=12)
        axes[1, 1].set_title('Epoch Training Time', fontsize=14, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.results_dir, 'baseline_training_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to: {plot_path}")

        # Also save individual plots
        self._save_individual_plots(epochs)

    def _save_individual_plots(self, epochs):
        """Save individual plots for easier viewing"""

        # Loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'loss_curve.png'), dpi=150)
        plt.close()

        # Accuracy plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        plt.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        plt.axhline(y=self.best_val_acc, color='g', linestyle='--',
                    label=f'Best: {self.best_val_acc:.2f}%', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'accuracy_curve.png'), dpi=150)
        plt.close()

        print(f"Individual plots saved to: {self.results_dir}/")

    def _save_metrics(self, total_time):
        """Save metrics to JSON file"""
        metrics = {
            'model': 'ResNet18',
            'dataset': 'CIFAR-10',
            'training_type': 'baseline',
            'num_epochs': self.num_epochs,
            'batch_size': config.BATCH_SIZE,
            'initial_lr': config.LEARNING_RATE,
            'optimizer': 'SGD',
            'momentum': config.MOMENTUM,
            'weight_decay': config.WEIGHT_DECAY,
            'lr_schedule': {
                'type': 'MultiStepLR',
                'milestones': config.LR_MILESTONES,
                'gamma': config.LR_GAMMA
            },
            'results': {
                'best_validation_accuracy': float(self.best_val_acc),
                'best_epoch': int(self.best_epoch + 1),
                'final_train_accuracy': float(self.train_accuracies[-1]),
                'final_validation_accuracy': float(self.val_accuracies[-1]),
                'final_train_loss': float(self.train_losses[-1]),
                'final_validation_loss': float(self.val_losses[-1]),
            },
            'training_history': {
                'train_losses': [float(x) for x in self.train_losses],
                'train_accuracies': [float(x) for x in self.train_accuracies],
                'val_losses': [float(x) for x in self.val_losses],
                'val_accuracies': [float(x) for x in self.val_accuracies],
                'learning_rates': [float(x) for x in self.learning_rates],
                'epoch_times': [float(x) for x in self.epoch_times],
            },
            'timing': {
                'total_training_time_seconds': float(total_time),
                'total_training_time_minutes': float(total_time / 60),
                'average_epoch_time_seconds': float(np.mean(self.epoch_times)),
            },
            'model_info': {
                'total_parameters': int(count_parameters(self.model)[0]),
                'trainable_parameters': int(count_parameters(self.model)[1]),
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        # Save to JSON
        json_path = os.path.join(self.results_dir, 'baseline_metrics.json')
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        print(f"\nMetrics saved to: {json_path}")

        # Also save a summary text file
        summary_path = os.path.join(self.results_dir, 'baseline_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("BASELINE TRAINING SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Model: ResNet18\n")
            f.write(f"Dataset: CIFAR-10\n")
            f.write(f"Training Epochs: {self.num_epochs}\n\n")
            f.write(f"Best Validation Accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch+1})\n")
            f.write(f"Final Train Accuracy: {self.train_accuracies[-1]:.2f}%\n")
            f.write(f"Final Validation Accuracy: {self.val_accuracies[-1]:.2f}%\n\n")
            f.write(f"Total Training Time: {total_time/60:.2f} minutes\n")
            f.write(f"Average Epoch Time: {np.mean(self.epoch_times):.2f} seconds\n\n")
            f.write(f"Total Parameters: {count_parameters(self.model)[0]:,}\n")
            f.write(f"Trainable Parameters: {count_parameters(self.model)[1]:,}\n\n")
            f.write("=" * 80 + "\n")

        print(f"Summary saved to: {summary_path}")


def main():
    """Main function"""
    print("Initializing baseline training for ResNet18 on CIFAR-10...")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Set random seed for reproducibility
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.SEED)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")

    # Load data
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=config.BATCH_SIZE,
        test_batch_size=config.TEST_BATCH_SIZE
    )
    print(f"✓ Dataset loaded\n")

    # Create model
    print("Creating ResNet18 model...")
    model = get_resnet18(num_classes=10, pretrained=False, device=device)
    print("✓ Model created\n")

    # Create optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY
    )

    # Create scheduler
    scheduler = MultiStepLR(
        optimizer,
        milestones=config.LR_MILESTONES,
        gamma=config.LR_GAMMA
    )

    # Create criterion
    criterion = nn.CrossEntropyLoss()

    # Create trainer
    trainer = BaselineTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        num_epochs=50
    )

    # Train
    results = trainer.train()

    # Print final results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Best Validation Accuracy: {results['best_val_acc']:.2f}% (Epoch {results['best_epoch']+1})")
    print(f"Final Training Accuracy: {results['final_train_acc']:.2f}%")
    print(f"Final Validation Accuracy: {results['final_val_acc']:.2f}%")
    print(f"Total Training Time: {results['total_time']/60:.2f} minutes")
    print("=" * 80)

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n✓ Baseline training completed successfully!")


if __name__ == '__main__':
    main()
