"""
Fast Gradient Sign Method (FGSM) Attack Implementation

Reference:
Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014).
Explaining and harnessing adversarial examples.
arXiv preprint arXiv:1412.6572.
"""

import torch
import torch.nn as nn


class FGSM:
    """
    Fast Gradient Sign Method (FGSM) Attack

    The FGSM attack perturbs the input by taking a step in the direction
    of the gradient of the loss with respect to the input.

    Args:
        model: The target model to attack
        epsilon: Maximum perturbation magnitude (L-infinity norm)
        clip_min: Minimum value for clipping (default: 0)
        clip_max: Maximum value for clipping (default: 1)
    """

    def __init__(self, model, epsilon=0.03, clip_min=0.0, clip_max=1.0):
        self.model = model
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.criterion = nn.CrossEntropyLoss()

    def generate(self, x, y):
        """
        Generate adversarial examples using FGSM

        Args:
            x: Input images (batch_size, channels, height, width)
            y: True labels (batch_size,)

        Returns:
            x_adv: Adversarial examples
        """
        # Set model to evaluation mode
        self.model.eval()

        # Clone input and enable gradient computation
        x_adv = x.clone().detach()
        x_adv.requires_grad = True

        # Forward pass
        outputs = self.model(x_adv)

        # Calculate loss
        loss = self.criterion(outputs, y)

        # Backward pass to get gradients
        self.model.zero_grad()
        loss.backward()

        # Get the sign of the gradient
        grad_sign = x_adv.grad.sign()

        # Create adversarial example
        x_adv = x_adv.detach() + self.epsilon * grad_sign

        # Clip to valid range
        x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)

        return x_adv

    def __call__(self, x, y):
        """Allow calling the object as a function"""
        return self.generate(x, y)


def fgsm_attack(model, x, y, epsilon, device):
    """
    Standalone function for FGSM attack

    Args:
        model: Target model
        x: Input images
        y: True labels
        epsilon: Perturbation magnitude
        device: Device to run on (cpu/cuda)

    Returns:
        Adversarial examples
    """
    criterion = nn.CrossEntropyLoss()

    x_adv = x.clone().detach().to(device)
    x_adv.requires_grad = True

    outputs = model(x_adv)
    loss = criterion(outputs, y)

    model.zero_grad()
    loss.backward()

    grad_sign = x_adv.grad.sign()
    x_adv = x_adv.detach() + epsilon * grad_sign
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv
