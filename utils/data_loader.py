"""
Data loading utilities for CIFAR-10
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import config


def get_cifar10_loaders(batch_size=None, test_batch_size=None, num_workers=None):
    """
    Get CIFAR-10 train and test data loaders

    Args:
        batch_size: Batch size for training (default from config)
        test_batch_size: Batch size for testing (default from config)
        num_workers: Number of worker processes (default from config)

    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if test_batch_size is None:
        test_batch_size = config.TEST_BATCH_SIZE
    if num_workers is None:
        num_workers = config.NUM_WORKERS

    # Data augmentation and normalization for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ])

    # Normalization for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ])

    # Load datasets
    train_dataset = datasets.CIFAR10(
        root=config.DATA_DIR,
        train=True,
        download=True,
        transform=transform_train
    )

    test_dataset = datasets.CIFAR10(
        root=config.DATA_DIR,
        train=False,
        download=True,
        transform=transform_test
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


def get_unnormalized_loader(batch_size=100):
    """
    Get CIFAR-10 test loader without normalization (for visualization)

    Args:
        batch_size: Batch size

    Returns:
        DataLoader with unnormalized images
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_dataset = datasets.CIFAR10(
        root=config.DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    return test_loader


# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
