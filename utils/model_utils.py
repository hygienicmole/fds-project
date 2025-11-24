"""
Model utilities for loading and saving models
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18
import config
import os


def get_model(num_classes=10, pretrained=False, device=None):
    """
    Get ResNet18 model for CIFAR-10

    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        device: Device to load model on

    Returns:
        model: ResNet18 model
    """
    if device is None:
        device = config.DEVICE

    # Load ResNet18
    model = resnet18(pretrained=pretrained)

    # Modify first conv layer for CIFAR-10 (32x32 images instead of 224x224)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool for CIFAR-10

    # Modify final layer for number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model = model.to(device)

    return model


def save_checkpoint(model, optimizer, epoch, accuracy, filepath):
    """
    Save model checkpoint

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        accuracy: Model accuracy
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
    }

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, filepath, optimizer=None, device=None):
    """
    Load model checkpoint

    Args:
        model: Model to load weights into
        filepath: Path to checkpoint file
        optimizer: Optimizer to load state into (optional)
        device: Device to load model on

    Returns:
        epoch: Epoch number from checkpoint
        accuracy: Accuracy from checkpoint
    """
    if device is None:
        device = config.DEVICE

    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    accuracy = checkpoint.get('accuracy', 0.0)

    print(f"Checkpoint loaded from {filepath}")
    print(f"Epoch: {epoch}, Accuracy: {accuracy:.2f}%")

    return epoch, accuracy


def count_parameters(model):
    """
    Count trainable parameters in model

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
