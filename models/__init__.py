"""
Models Module

Provides ResNet18 implementation and training utilities for CIFAR-10
"""

from .resnet import (
    ResNet18CIFAR10,
    get_resnet18,
    model_summary,
    count_parameters,
    save_checkpoint,
    load_checkpoint,
    save_model_weights,
    load_model_weights
)

from .train import (
    Trainer,
    TrainingMetrics,
    create_optimizer,
    create_scheduler
)

__all__ = [
    # ResNet model
    'ResNet18CIFAR10',
    'get_resnet18',
    'model_summary',
    'count_parameters',

    # Checkpoint management
    'save_checkpoint',
    'load_checkpoint',
    'save_model_weights',
    'load_model_weights',

    # Training
    'Trainer',
    'TrainingMetrics',
    'create_optimizer',
    'create_scheduler',
]
