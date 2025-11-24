"""
CIFAR-10 Data Loading Module

This module provides comprehensive data loading functionality for CIFAR-10:
- Dataset downloading and loading
- Data transformations (normalization, augmentation)
- Train and test dataloader creation
- Sample visualization
- Dataset statistics computation and storage
"""

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from typing import Tuple, Optional, Dict


# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Default CIFAR-10 statistics (computed from training set)
DEFAULT_MEAN = [0.4914, 0.4822, 0.4465]
DEFAULT_STD = [0.2023, 0.1994, 0.2010]


def compute_dataset_statistics(data_dir: str = './data', save_path: Optional[str] = None) -> Dict[str, list]:
    """
    Compute mean and standard deviation of CIFAR-10 training dataset.

    Args:
        data_dir: Directory to store/load CIFAR-10 dataset
        save_path: Path to save statistics JSON file (if None, saves to data_dir/cifar10_stats.json)

    Returns:
        Dictionary containing 'mean' and 'std' lists for each channel
    """
    print("Computing CIFAR-10 dataset statistics...")

    # Load dataset without normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=2
    )

    # Compute mean and std
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0

    print("Computing mean...")
    for images, _ in train_loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        total_samples += batch_samples

    mean /= total_samples

    print("Computing standard deviation...")
    for images, _ in train_loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        std += ((images - mean.unsqueeze(1))**2).sum([0, 2])

    std = torch.sqrt(std / (total_samples * 32 * 32))

    # Convert to lists
    stats = {
        'mean': mean.tolist(),
        'std': std.tolist()
    }

    print(f"Computed statistics:")
    print(f"  Mean: {stats['mean']}")
    print(f"  Std:  {stats['std']}")

    # Save statistics
    if save_path is None:
        save_path = os.path.join(data_dir, 'cifar10_stats.json')

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Statistics saved to: {save_path}")

    return stats


def load_dataset_statistics(stats_path: str) -> Dict[str, list]:
    """
    Load dataset statistics from JSON file.

    Args:
        stats_path: Path to statistics JSON file

    Returns:
        Dictionary containing 'mean' and 'std' lists
    """
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Statistics file not found: {stats_path}")

    with open(stats_path, 'r') as f:
        stats = json.load(f)

    return stats


def get_transforms(mean: list = None, std: list = None, augment: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get train and test transforms for CIFAR-10.

    Args:
        mean: Mean values for normalization (uses default if None)
        std: Standard deviation values for normalization (uses default if None)
        augment: Whether to apply data augmentation to training set

    Returns:
        Tuple of (train_transform, test_transform)
    """
    if mean is None:
        mean = DEFAULT_MEAN
    if std is None:
        std = DEFAULT_STD

    # Training transforms with augmentation
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, test_transform


def get_cifar10_dataloaders(
    data_dir: str = './data',
    batch_size: int = 128,
    test_batch_size: int = 100,
    num_workers: int = 4,
    mean: list = None,
    std: list = None,
    augment: bool = True,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create CIFAR-10 train and test dataloaders.

    Args:
        data_dir: Directory to store/load CIFAR-10 dataset
        batch_size: Batch size for training
        test_batch_size: Batch size for testing
        num_workers: Number of worker processes for data loading
        mean: Mean values for normalization (uses default if None)
        std: Standard deviation values for normalization (uses default if None)
        augment: Whether to apply data augmentation to training set
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Get transforms
    train_transform, test_transform = get_transforms(mean, std, augment)

    # Download and load datasets
    print("Loading CIFAR-10 dataset...")
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    return train_loader, test_loader


def visualize_samples(
    data_loader: DataLoader,
    num_samples: int = 16,
    mean: list = None,
    std: list = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Visualize sample images from CIFAR-10 dataset.

    Args:
        data_loader: DataLoader to get samples from
        num_samples: Number of samples to visualize
        mean: Mean values used for normalization (for denormalization)
        std: Std values used for normalization (for denormalization)
        save_path: Path to save the visualization (if None, doesn't save)
        show: Whether to display the plot
    """
    if mean is None:
        mean = DEFAULT_MEAN
    if std is None:
        std = DEFAULT_STD

    # Get a batch of images
    images, labels = next(iter(data_loader))

    # Limit to num_samples
    images = images[:num_samples]
    labels = labels[:num_samples]

    # Denormalize images
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)

    # Create grid
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()

    for idx in range(grid_size * grid_size):
        ax = axes[idx]
        if idx < num_samples:
            # Convert from CHW to HWC
            img = images[idx].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.set_title(f'{CIFAR10_CLASSES[labels[idx]]}', fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def visualize_augmentations(
    data_dir: str = './data',
    num_images: int = 3,
    augmentations_per_image: int = 5,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Visualize the effect of data augmentation on sample images.

    Args:
        data_dir: Directory containing CIFAR-10 dataset
        num_images: Number of original images to show
        augmentations_per_image: Number of augmented versions per image
        save_path: Path to save the visualization (if None, doesn't save)
        show: Whether to display the plot
    """
    # Load dataset without normalization for better visualization
    transform_original = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_augmented = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    # Get original images
    dataset_original = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform_original
    )

    dataset_augmented = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform_augmented
    )

    fig, axes = plt.subplots(num_images, augmentations_per_image + 1, figsize=(15, 3 * num_images))

    for i in range(num_images):
        # Get original image
        img_original, label = dataset_original[i]

        # Show original
        ax = axes[i, 0] if num_images > 1 else axes[0]
        ax.imshow(img_original.permute(1, 2, 0).numpy())
        ax.set_title(f'Original: {CIFAR10_CLASSES[label]}', fontsize=10)
        ax.axis('off')

        # Show augmented versions
        for j in range(augmentations_per_image):
            img_aug, _ = dataset_augmented[i]
            ax = axes[i, j + 1] if num_images > 1 else axes[j + 1]
            ax.imshow(img_aug.permute(1, 2, 0).numpy())
            ax.set_title(f'Augmented {j+1}', fontsize=10)
            ax.axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Augmentation visualization saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def get_class_distribution(data_loader: DataLoader) -> Dict[str, int]:
    """
    Compute class distribution in the dataset.

    Args:
        data_loader: DataLoader to analyze

    Returns:
        Dictionary mapping class names to counts
    """
    class_counts = {class_name: 0 for class_name in CIFAR10_CLASSES}

    print("Computing class distribution...")
    for _, labels in data_loader:
        for label in labels:
            class_counts[CIFAR10_CLASSES[label.item()]] += 1

    return class_counts


def print_dataset_info(train_loader: DataLoader, test_loader: DataLoader) -> None:
    """
    Print comprehensive information about the dataloaders.

    Args:
        train_loader: Training dataloader
        test_loader: Test dataloader
    """
    print("\n" + "="*60)
    print("CIFAR-10 Dataset Information")
    print("="*60)

    # Basic info
    print(f"\nTraining set:")
    print(f"  Total samples: {len(train_loader.dataset)}")
    print(f"  Batch size: {train_loader.batch_size}")
    print(f"  Number of batches: {len(train_loader)}")

    print(f"\nTest set:")
    print(f"  Total samples: {len(test_loader.dataset)}")
    print(f"  Batch size: {test_loader.batch_size}")
    print(f"  Number of batches: {len(test_loader)}")

    # Image info
    sample_image, _ = next(iter(train_loader))
    print(f"\nImage shape: {sample_image.shape}")
    print(f"  Batch size: {sample_image.shape[0]}")
    print(f"  Channels: {sample_image.shape[1]}")
    print(f"  Height: {sample_image.shape[2]}")
    print(f"  Width: {sample_image.shape[3]}")

    print(f"\nNumber of classes: {len(CIFAR10_CLASSES)}")
    print(f"Class names: {', '.join(CIFAR10_CLASSES)}")

    print("="*60 + "\n")


if __name__ == "__main__":
    # Example usage
    print("CIFAR-10 DataLoader Module")
    print("-" * 60)

    # Compute and save statistics
    stats = compute_dataset_statistics(data_dir='./data')

    # Create dataloaders
    train_loader, test_loader = get_cifar10_dataloaders(
        data_dir='./data',
        batch_size=128,
        test_batch_size=100,
        num_workers=2
    )

    # Print dataset info
    print_dataset_info(train_loader, test_loader)

    # Visualize samples
    visualize_samples(
        train_loader,
        num_samples=16,
        save_path='./data/sample_visualization.png',
        show=False
    )

    # Visualize augmentations
    visualize_augmentations(
        data_dir='./data',
        num_images=3,
        augmentations_per_image=5,
        save_path='./data/augmentation_visualization.png',
        show=False
    )

    print("\nDataLoader module test completed successfully!")
