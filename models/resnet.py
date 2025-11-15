"""
ResNet18 Model for CIFAR-10

This module provides ResNet18 implementation optimized for CIFAR-10 dataset
with utilities for model creation, checkpoint management, and model summary.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from collections import OrderedDict
import os


class ResNet18CIFAR10(nn.Module):
    """
    ResNet18 model adapted for CIFAR-10 dataset

    Modifications from standard ResNet18:
    - First conv layer uses 3x3 kernel with stride 1 (instead of 7x7 with stride 2)
    - Removed max pooling layer after first conv
    - Final FC layer outputs 10 classes instead of 1000

    Args:
        num_classes: Number of output classes (default: 10)
        pretrained: Whether to use pretrained ImageNet weights (default: False)
        dropout: Dropout probability (default: 0.0)
    """

    def __init__(self, num_classes=10, pretrained=False, dropout=0.0):
        super(ResNet18CIFAR10, self).__init__()

        # Load base ResNet18 model
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            base_model = resnet18(weights=weights)
        else:
            base_model = resnet18(weights=None)

        # Modify first convolutional layer for CIFAR-10 (32x32 images)
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # Modified: Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = base_model.bn1
        self.relu = base_model.relu

        # Remove max pooling (not needed for small CIFAR-10 images)
        # self.maxpool = nn.Identity()

        # Use ResNet18 residual layers
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        # Global average pooling
        self.avgpool = base_model.avgpool

        # Optional dropout before final layer
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Modify final fully connected layer for CIFAR-10 (10 classes)
        self.fc = nn.Linear(512, num_classes)

        # Initialize weights
        self._initialize_weights(pretrained)

    def _initialize_weights(self, pretrained):
        """Initialize model weights"""
        if not pretrained:
            # Initialize conv1 since we created it from scratch
            nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')

        # Always initialize the final FC layer
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        """Forward pass"""
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # Note: no maxpool for CIFAR-10

        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Dropout and classification
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def get_features(self, x):
        """
        Extract features before the final classification layer

        Args:
            x: Input tensor

        Returns:
            Features from the last layer before FC
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


def get_resnet18(num_classes=10, pretrained=False, dropout=0.0, device='cuda'):
    """
    Factory function to create ResNet18 model for CIFAR-10

    Args:
        num_classes: Number of output classes (default: 10)
        pretrained: Use pretrained ImageNet weights (default: False)
        dropout: Dropout probability (default: 0.0)
        device: Device to load model on (default: 'cuda')

    Returns:
        ResNet18CIFAR10 model
    """
    model = ResNet18CIFAR10(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    model = model.to(device)
    return model


def model_summary(model, input_size=(3, 32, 32), batch_size=1, device='cuda'):
    """
    Print model summary with layer information and parameter counts

    Args:
        model: PyTorch model
        input_size: Input tensor size (C, H, W)
        batch_size: Batch size for summary
        device: Device to run summary on
    """
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = f"{class_name}-{module_idx+1}"
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["output_shape"] = list(output.size())

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size()))).item()
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size()))).item()
            summary[m_key]["nb_params"] = params

        if not isinstance(module, nn.Sequential) and \
           not isinstance(module, nn.ModuleList) and \
           not (module == model):
            hooks.append(module.register_forward_hook(hook))

    # Create properties
    summary = OrderedDict()
    hooks = []

    # Register hook
    model.apply(register_hook)

    # Make a forward pass
    x = torch.zeros(batch_size, *input_size).to(device)
    model(x)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Print summary
    print("=" * 100)
    print(f"{'Layer (type)':<30} {'Output Shape':<25} {'Param #':<15}")
    print("=" * 100)

    total_params = 0
    total_output = 0
    trainable_params = 0

    for layer in summary:
        line = f"{layer:<30} {str(summary[layer]['output_shape']):<25} {summary[layer]['nb_params']:<15,}"
        print(line)

        total_params += summary[layer]["nb_params"]
        total_output += torch.prod(torch.LongTensor(summary[layer]["output_shape"])).item()

        if "trainable" in summary[layer]:
            if summary[layer]["trainable"]:
                trainable_params += summary[layer]["nb_params"]

    # Print statistics
    print("=" * 100)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {total_params - trainable_params:,}")
    print("=" * 100)

    # Calculate size
    total_input_size = abs(batch_size * torch.prod(torch.LongTensor(list(input_size))).item() * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print(f"Input size (MB): {total_input_size:.2f}")
    print(f"Forward/backward pass size (MB): {total_output_size:.2f}")
    print(f"Params size (MB): {total_params_size:.2f}")
    print(f"Estimated Total Size (MB): {total_size:.2f}")
    print("=" * 100)


def count_parameters(model):
    """
    Count total and trainable parameters in the model

    Args:
        model: PyTorch model

    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def save_checkpoint(model, optimizer, epoch, accuracy, loss, filepath,
                   scheduler=None, best_acc=None, additional_info=None):
    """
    Save model checkpoint with training state

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch number
        accuracy: Current accuracy
        loss: Current loss
        filepath: Path to save checkpoint
        scheduler: Learning rate scheduler (optional)
        best_acc: Best accuracy achieved (optional)
        additional_info: Dictionary with additional information (optional)
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        'loss': loss,
        'model_architecture': model.__class__.__name__,
    }

    # Add optional components
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if best_acc is not None:
        checkpoint['best_accuracy'] = best_acc

    if additional_info is not None:
        checkpoint['additional_info'] = additional_info

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")
    print(f"  Epoch: {epoch}, Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}")


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device='cuda'):
    """
    Load model checkpoint and restore training state

    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to restore state (optional)
        scheduler: Scheduler to restore state (optional)
        device: Device to load checkpoint on

    Returns:
        dict: Checkpoint information (epoch, accuracy, loss, etc.)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Extract checkpoint info
    info = {
        'epoch': checkpoint.get('epoch', 0),
        'accuracy': checkpoint.get('accuracy', 0.0),
        'loss': checkpoint.get('loss', 0.0),
        'best_accuracy': checkpoint.get('best_accuracy', None),
        'additional_info': checkpoint.get('additional_info', None)
    }

    print(f"Checkpoint loaded from {filepath}")
    print(f"  Epoch: {info['epoch']}, Accuracy: {info['accuracy']:.2f}%, Loss: {info['loss']:.4f}")
    if info['best_accuracy'] is not None:
        print(f"  Best Accuracy: {info['best_accuracy']:.2f}%")

    return info


def save_model_weights(model, filepath):
    """
    Save only model weights (lighter than full checkpoint)

    Args:
        model: Model to save
        filepath: Path to save weights
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model weights saved to {filepath}")


def load_model_weights(filepath, model, device='cuda'):
    """
    Load only model weights

    Args:
        filepath: Path to weights file
        model: Model to load weights into
        device: Device to load weights on
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Weights file not found: {filepath}")

    model.load_state_dict(torch.load(filepath, map_location=device))
    print(f"Model weights loaded from {filepath}")


# Example usage
if __name__ == '__main__':
    # Test model creation
    print("Creating ResNet18 for CIFAR-10...\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_resnet18(num_classes=10, pretrained=False, device=device)

    # Print model summary
    print("\nModel Summary:")
    model_summary(model, input_size=(3, 32, 32), batch_size=1, device=device)

    # Count parameters
    total, trainable = count_parameters(model)
    print(f"\nParameter Count:")
    print(f"Total: {total:,}")
    print(f"Trainable: {trainable:,}")

    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(4, 3, 32, 32).to(device)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.2f}, {output.max():.2f}]")

    print("\nResNet18 model created successfully!")
