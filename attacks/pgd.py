"""
Projected Gradient Descent (PGD) Attack Implementation

Reference:
Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2017).
Towards deep learning models resistant to adversarial attacks.
arXiv preprint arXiv:1706.06083.
"""

import torch
import torch.nn as nn


class PGD:
    """
    Projected Gradient Descent (PGD) Attack

    PGD is an iterative adversarial attack that performs multiple steps
    of FGSM with projection back to the epsilon ball.

    Args:
        model: The target model to attack
        epsilon: Maximum perturbation magnitude (L-infinity norm)
        alpha: Step size for each iteration
        iterations: Number of attack iterations
        random_start: Whether to start from a random point in epsilon ball
        clip_min: Minimum value for clipping (default: 0)
        clip_max: Maximum value for clipping (default: 1)
    """

    def __init__(
        self,
        model,
        epsilon=0.03,
        alpha=0.01,
        iterations=40,
        random_start=True,
        clip_min=0.0,
        clip_max=1.0
    ):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.iterations = iterations
        self.random_start = random_start
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.criterion = nn.CrossEntropyLoss()

    def generate(self, x, y):
        """
        Generate adversarial examples using PGD

        Args:
            x: Input images (batch_size, channels, height, width)
            y: True labels (batch_size,)

        Returns:
            x_adv: Adversarial examples
        """
        # Set model to evaluation mode
        self.model.eval()

        # Initialize adversarial example
        x_adv = x.clone().detach()

        # Random initialization within epsilon ball
        if self.random_start:
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-self.epsilon, self.epsilon)
            x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)

        # Perform PGD iterations
        for i in range(self.iterations):
            x_adv.requires_grad = True

            # Forward pass
            outputs = self.model(x_adv)

            # Calculate loss
            loss = self.criterion(outputs, y)

            # Backward pass
            self.model.zero_grad()
            loss.backward()

            # Perform gradient step
            with torch.no_grad():
                # Get gradient sign
                grad_sign = x_adv.grad.sign()

                # Update adversarial example
                x_adv = x_adv + self.alpha * grad_sign

                # Project back to epsilon ball around original image
                perturbation = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
                x_adv = x + perturbation

                # Clip to valid range
                x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)

        return x_adv.detach()

    def __call__(self, x, y):
        """Allow calling the object as a function"""
        return self.generate(x, y)


def pgd_attack(model, x, y, epsilon, alpha, iterations, device, random_start=True):
    """
    Standalone function for PGD attack

    Args:
        model: Target model
        x: Input images
        y: True labels
        epsilon: Maximum perturbation magnitude
        alpha: Step size
        iterations: Number of iterations
        device: Device to run on (cpu/cuda)
        random_start: Whether to use random initialization

    Returns:
        Adversarial examples
    """
    criterion = nn.CrossEntropyLoss()

    x_adv = x.clone().detach().to(device)

    # Random initialization
    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    # PGD iterations
    for i in range(iterations):
        x_adv.requires_grad = True

        outputs = model(x_adv)
        loss = criterion(outputs, y)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad_sign = x_adv.grad.sign()
            x_adv = x_adv + alpha * grad_sign

            # Project back
            perturbation = torch.clamp(x_adv - x, -epsilon, epsilon)
            x_adv = x + perturbation
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv.detach()
