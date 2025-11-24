"""
Training script for ResNet18 on CIFAR-10
Supports both standard training and adversarial training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import argparse
import os

import config
from utils import get_cifar10_loaders, get_model, save_checkpoint, count_parameters
from attacks import PGD


def train_epoch(model, train_loader, criterion, optimizer, device, use_adv_training=False):
    """
    Train for one epoch

    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        use_adv_training: Whether to use adversarial training

    Returns:
        Average loss and accuracy for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Setup PGD attack for adversarial training
    if use_adv_training:
        pgd = PGD(
            model=model,
            epsilon=config.ADV_TRAIN_EPSILON,
            alpha=config.ADV_TRAIN_ALPHA,
            iterations=config.ADV_TRAIN_ITERATIONS,
            random_start=True
        )

    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        # Generate adversarial examples if adversarial training
        if use_adv_training:
            with torch.no_grad():
                inputs = pgd.generate(inputs, targets)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    """
    Evaluate model on test set

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        Test loss and accuracy
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total

    return test_loss, test_acc


def main(args):
    """Main training function"""

    # Set random seed for reproducibility
    torch.manual_seed(config.SEED)

    # Get data loaders
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders()

    # Get model
    print(f"\nInitializing {config.MODEL_NAME}...")
    model = get_model(num_classes=config.NUM_CLASSES, device=config.DEVICE)
    print(f"Number of parameters: {count_parameters(model):,}")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY
    )

    # Learning rate scheduler
    scheduler = MultiStepLR(
        optimizer,
        milestones=config.LR_MILESTONES,
        gamma=config.LR_GAMMA
    )

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"Device: {config.DEVICE}")
    print(f"Adversarial Training: {args.adv_training}")
    print("-" * 60)

    best_acc = 0.0
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE,
            use_adv_training=args.adv_training
        )

        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, config.DEVICE)

        # Update scheduler
        scheduler.step()

        # Print statistics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

        # Save statistics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, test_acc, save_path)

        # Save checkpoint at intervals
        if epoch % args.save_interval == 0:
            save_path = os.path.join(config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, test_acc, save_path)

    print("\n" + "=" * 60)
    print(f"Training complete! Best Test Accuracy: {best_acc:.2f}%")
    print("=" * 60)

    # Save final model
    final_path = os.path.join(config.MODEL_DIR, 'final_model.pth')
    save_checkpoint(model, optimizer, args.epochs, test_acc, final_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ResNet18 on CIFAR-10')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                        help='Number of epochs to train')
    parser.add_argument('--adv-training', action='store_true',
                        help='Use adversarial training')
    parser.add_argument('--save-interval', type=int, default=config.SAVE_MODEL_INTERVAL,
                        help='Save checkpoint every N epochs')

    args = parser.parse_args()
    main(args)
