"""
Training Module for ResNet18 on CIFAR-10

Provides comprehensive training and validation functionality with:
- Progress tracking with tqdm
- Loss and accuracy logging
- Checkpoint management
- Resume training capability
- Early stopping
- Learning rate scheduling
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import os
import sys
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet import get_resnet18, save_checkpoint, load_checkpoint, count_parameters
from utils.data_loader import get_cifar10_loaders
import config


class TrainingMetrics:
    """Track and manage training metrics"""

    def __init__(self):
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        self.epoch_times = []
        self.best_val_acc = 0.0
        self.best_epoch = 0

    def update_train(self, loss, accuracy):
        """Update training metrics"""
        self.train_losses.append(loss)
        self.train_accuracies.append(accuracy)

    def update_val(self, loss, accuracy):
        """Update validation metrics"""
        self.val_losses.append(loss)
        self.val_accuracies.append(accuracy)

    def update_best(self, accuracy, epoch):
        """Update best validation accuracy"""
        if accuracy > self.best_val_acc:
            self.best_val_acc = accuracy
            self.best_epoch = epoch
            return True
        return False

    def add_learning_rate(self, lr):
        """Add current learning rate"""
        self.learning_rates.append(lr)

    def add_epoch_time(self, time_elapsed):
        """Add epoch time"""
        self.epoch_times.append(time_elapsed)

    def get_summary(self):
        """Get training summary"""
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'epoch_times': self.epoch_times,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch
        }

    def save_to_file(self, filepath):
        """Save metrics to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.get_summary(), f, indent=4)
        print(f"Metrics saved to {filepath}")


class Trainer:
    """
    Trainer class for ResNet18 on CIFAR-10

    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        criterion: Loss function
        scheduler: Learning rate scheduler (optional)
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        use_tensorboard: Whether to use TensorBoard logging
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        scheduler=None,
        device='cuda',
        checkpoint_dir='./models/checkpoints',
        use_tensorboard=True
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.use_tensorboard = use_tensorboard

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize metrics
        self.metrics = TrainingMetrics()

        # Initialize TensorBoard writer
        if use_tensorboard:
            log_dir = os.path.join(config.TENSORBOARD_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            self.writer = SummaryWriter(log_dir)
            print(f"TensorBoard logs will be saved to: {log_dir}")

        # Training state
        self.current_epoch = 0
        self.global_step = 0

    def train_epoch(self):
        """
        Train for one epoch

        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1} [Train]', leave=False)

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Calculate statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            current_loss = running_loss / (batch_idx + 1)
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })

            # TensorBoard logging (every N batches)
            if self.use_tensorboard and batch_idx % config.LOG_INTERVAL == 0:
                self.writer.add_scalar('Train/BatchLoss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/BatchAcc', current_acc, self.global_step)

            self.global_step += 1

        # Calculate epoch statistics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self):
        """
        Validate the model

        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch+1} [Val]', leave=False)

            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Calculate statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Update progress bar
                current_loss = running_loss / (batch_idx + 1)
                current_acc = 100. * correct / total
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{current_acc:.2f}%'
                })

        # Calculate validation statistics
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total

        return val_loss, val_acc

    def train(self, num_epochs, save_every=10, early_stopping_patience=None):
        """
        Main training loop

        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            early_stopping_patience: Number of epochs to wait before early stopping (None to disable)

        Returns:
            TrainingMetrics: Final training metrics
        """
        print("=" * 80)
        print("Starting Training")
        print("=" * 80)
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        total_params, trainable_params = count_parameters(self.model)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("=" * 80)

        # Early stopping counter
        patience_counter = 0

        # Training loop
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.metrics.add_learning_rate(current_lr)

            print(f"\nEpoch {epoch+1}/{num_epochs} | LR: {current_lr:.6f}")

            # Train
            train_loss, train_acc = self.train_epoch()
            self.metrics.update_train(train_loss, train_acc)

            # Validate
            val_loss, val_acc = self.validate()
            self.metrics.update_val(val_loss, val_acc)

            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            self.metrics.add_epoch_time(epoch_time)

            # Print epoch summary
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"Epoch Time: {epoch_time:.2f}s")

            # TensorBoard logging
            if self.use_tensorboard:
                self.writer.add_scalar('Epoch/TrainLoss', train_loss, epoch)
                self.writer.add_scalar('Epoch/TrainAcc', train_acc, epoch)
                self.writer.add_scalar('Epoch/ValLoss', val_loss, epoch)
                self.writer.add_scalar('Epoch/ValAcc', val_acc, epoch)
                self.writer.add_scalar('Epoch/LearningRate', current_lr, epoch)

            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Save best model
            is_best = self.metrics.update_best(val_acc, epoch)
            if is_best:
                print(f"★ New best validation accuracy: {val_acc:.2f}%")
                best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_acc, val_loss,
                    best_path, self.scheduler, val_acc
                )
                patience_counter = 0
            else:
                patience_counter += 1

            # Save checkpoint at intervals
            if (epoch + 1) % save_every == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_acc, val_loss,
                    checkpoint_path, self.scheduler, self.metrics.best_val_acc
                )

            # Early stopping
            if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                print(f"\n⚠ Early stopping triggered after {early_stopping_patience} epochs without improvement")
                break

        # Save final model
        final_path = os.path.join(self.checkpoint_dir, 'final_model.pth')
        save_checkpoint(
            self.model, self.optimizer, self.current_epoch, val_acc, val_loss,
            final_path, self.scheduler, self.metrics.best_val_acc
        )

        # Save metrics
        metrics_path = os.path.join(self.checkpoint_dir, 'training_metrics.json')
        self.metrics.save_to_file(metrics_path)

        # Close TensorBoard writer
        if self.use_tensorboard:
            self.writer.close()

        # Print final summary
        self._print_summary()

        return self.metrics

    def resume_from_checkpoint(self, checkpoint_path):
        """
        Resume training from a checkpoint

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            int: Epoch number to resume from
        """
        print(f"Resuming training from checkpoint: {checkpoint_path}")

        info = load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.scheduler,
            self.device
        )

        # Update current epoch
        self.current_epoch = info['epoch'] + 1

        # Update best accuracy if available
        if info['best_accuracy'] is not None:
            self.metrics.best_val_acc = info['best_accuracy']

        print(f"Resuming from epoch {self.current_epoch}")

        return self.current_epoch

    def _print_summary(self):
        """Print training summary"""
        print("\n" + "=" * 80)
        print("Training Complete!")
        print("=" * 80)
        print(f"Best Validation Accuracy: {self.metrics.best_val_acc:.2f}% (Epoch {self.metrics.best_epoch+1})")
        print(f"Final Train Accuracy: {self.metrics.train_accuracies[-1]:.2f}%")
        print(f"Final Validation Accuracy: {self.metrics.val_accuracies[-1]:.2f}%")
        print(f"Total Training Time: {sum(self.metrics.epoch_times):.2f}s")
        print(f"Average Epoch Time: {sum(self.metrics.epoch_times)/len(self.metrics.epoch_times):.2f}s")
        print("=" * 80)


def create_optimizer(model, optimizer_name='sgd', lr=0.1, momentum=0.9, weight_decay=5e-4):
    """
    Create optimizer

    Args:
        model: Model to optimize
        optimizer_name: Name of optimizer ('sgd', 'adam', 'adamw')
        lr: Learning rate
        momentum: Momentum (for SGD)
        weight_decay: Weight decay

    Returns:
        Optimizer
    """
    optimizer_name = optimizer_name.lower()

    if optimizer_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer


def create_scheduler(optimizer, scheduler_name='multistep', **kwargs):
    """
    Create learning rate scheduler

    Args:
        optimizer: Optimizer
        scheduler_name: Name of scheduler ('multistep', 'cosine', 'plateau')
        **kwargs: Scheduler-specific arguments

    Returns:
        Scheduler
    """
    scheduler_name = scheduler_name.lower()

    if scheduler_name == 'multistep':
        milestones = kwargs.get('milestones', [50, 75])
        gamma = kwargs.get('gamma', 0.1)
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    elif scheduler_name == 'cosine':
        T_max = kwargs.get('T_max', 100)
        eta_min = kwargs.get('eta_min', 0)
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    elif scheduler_name == 'plateau':
        mode = kwargs.get('mode', 'min')
        factor = kwargs.get('factor', 0.1)
        patience = kwargs.get('patience', 10)
        scheduler = ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience, verbose=True
        )

    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    return scheduler


def main():
    """Main training function"""
    import argparse

    parser = argparse.ArgumentParser(description='Train ResNet18 on CIFAR-10')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--scheduler', type=str, default='multistep', choices=['multistep', 'cosine', 'plateau'])
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--save-every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--early-stopping', type=int, default=None, help='Early stopping patience')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--no-tensorboard', action='store_true', help='Disable TensorBoard logging')

    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.SEED)

    # Load data
    print("\nLoading CIFAR-10 dataset...")
    train_loader, val_loader = get_cifar10_loaders(batch_size=args.batch_size)

    # Create model
    print("\nCreating ResNet18 model...")
    model = get_resnet18(
        num_classes=10,
        pretrained=args.pretrained,
        dropout=args.dropout,
        device=device
    )

    # Create optimizer
    optimizer = create_optimizer(
        model,
        optimizer_name=args.optimizer,
        lr=args.lr,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY
    )

    # Create scheduler
    scheduler = create_scheduler(
        optimizer,
        scheduler_name=args.scheduler,
        milestones=config.LR_MILESTONES,
        gamma=config.LR_GAMMA,
        T_max=args.epochs
    )

    # Create criterion
    criterion = nn.CrossEntropyLoss()

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=config.CHECKPOINT_DIR,
        use_tensorboard=not args.no_tensorboard
    )

    # Resume from checkpoint if specified
    if args.resume is not None:
        trainer.resume_from_checkpoint(args.resume)

    # Train
    metrics = trainer.train(
        num_epochs=args.epochs,
        save_every=args.save_every,
        early_stopping_patience=args.early_stopping
    )

    print("\n✓ Training completed successfully!")


if __name__ == '__main__':
    main()
