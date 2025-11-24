"""
Evaluation script for adversarial attacks on CIFAR-10
Evaluates model robustness against FGSM and PGD attacks
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import os
import numpy as np

import config
from utils import get_cifar10_loaders, get_model, load_checkpoint
from utils import visualize_adversarial_examples, plot_adversarial_comparison
from attacks import FGSM, PGD


def evaluate_clean(model, test_loader, device):
    """
    Evaluate model on clean test data

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to evaluate on

    Returns:
        Clean accuracy
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Clean Evaluation'):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    return accuracy


def evaluate_fgsm(model, test_loader, device, epsilon_values=None):
    """
    Evaluate model against FGSM attack with various epsilon values

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to evaluate on
        epsilon_values: List of epsilon values to test

    Returns:
        Dictionary of accuracies for each epsilon
    """
    if epsilon_values is None:
        epsilon_values = config.FGSM_EPSILONS

    model.eval()
    results = {}

    for epsilon in epsilon_values:
        print(f"\nEvaluating FGSM with epsilon={epsilon:.4f}")

        fgsm = FGSM(model=model, epsilon=epsilon)
        correct = 0
        total = 0

        for inputs, targets in tqdm(test_loader, desc=f'FGSM ε={epsilon:.4f}'):
            inputs, targets = inputs.to(device), targets.to(device)

            # Generate adversarial examples
            adv_inputs = fgsm.generate(inputs, targets)

            # Evaluate on adversarial examples
            with torch.no_grad():
                outputs = model(adv_inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = 100. * correct / total
        results[epsilon] = accuracy
        print(f"FGSM (ε={epsilon:.4f}) Accuracy: {accuracy:.2f}%")

    return results


def evaluate_pgd(model, test_loader, device, configs=None):
    """
    Evaluate model against PGD attack with various configurations

    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to evaluate on
        configs: List of PGD configurations

    Returns:
        Dictionary of accuracies for each configuration
    """
    if configs is None:
        configs = config.PGD_CONFIGS

    model.eval()
    results = {}

    for cfg in configs:
        epsilon = cfg['epsilon']
        alpha = cfg['alpha']
        iterations = cfg['iterations']

        print(f"\nEvaluating PGD with ε={epsilon:.4f}, α={alpha:.4f}, iter={iterations}")

        pgd = PGD(
            model=model,
            epsilon=epsilon,
            alpha=alpha,
            iterations=iterations,
            random_start=True
        )

        correct = 0
        total = 0

        for inputs, targets in tqdm(test_loader, desc=f'PGD ε={epsilon:.4f} iter={iterations}'):
            inputs, targets = inputs.to(device), targets.to(device)

            # Generate adversarial examples
            adv_inputs = pgd.generate(inputs, targets)

            # Evaluate on adversarial examples
            with torch.no_grad():
                outputs = model(adv_inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = 100. * correct / total
        results[f"ε={epsilon:.4f}_iter={iterations}"] = accuracy
        print(f"PGD (ε={epsilon:.4f}, iter={iterations}) Accuracy: {accuracy:.2f}%")

    return results


def visualize_attack_examples(model, test_loader, device, attack_type='fgsm', num_samples=10):
    """
    Generate and visualize adversarial examples

    Args:
        model: Model to attack
        test_loader: Test data loader
        device: Device to run on
        attack_type: Type of attack ('fgsm' or 'pgd')
        num_samples: Number of samples to visualize
    """
    model.eval()

    # Get a batch of test images
    inputs, targets = next(iter(test_loader))
    inputs, targets = inputs.to(device), targets.to(device)

    # Select only first num_samples
    inputs = inputs[:num_samples]
    targets = targets[:num_samples]

    # Create attack
    if attack_type.lower() == 'fgsm':
        attack = FGSM(model=model, epsilon=config.FGSM_EPSILON)
        attack_name = 'FGSM'
    else:
        attack = PGD(
            model=model,
            epsilon=config.PGD_EPSILON,
            alpha=config.PGD_ALPHA,
            iterations=config.PGD_ITERATIONS,
            random_start=True
        )
        attack_name = 'PGD'

    # Generate adversarial examples
    adv_inputs = attack.generate(inputs, targets)

    # Get predictions
    with torch.no_grad():
        adv_outputs = model(adv_inputs)
        _, adv_predictions = adv_outputs.max(1)

    # Visualize
    save_path = os.path.join(config.VIS_SAVE_DIR, f'{attack_name.lower()}_examples.png')
    visualize_adversarial_examples(
        inputs, adv_inputs, targets.cpu(), adv_predictions.cpu(),
        num_samples=num_samples, save_path=save_path
    )


def main(args):
    """Main evaluation function"""

    # Set random seed
    torch.manual_seed(config.SEED)

    # Get data loader
    print("Loading CIFAR-10 test dataset...")
    _, test_loader = get_cifar10_loaders(test_batch_size=args.batch_size)

    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model = get_model(num_classes=config.NUM_CLASSES, device=config.DEVICE)

    if os.path.exists(args.model_path):
        load_checkpoint(model, args.model_path, device=config.DEVICE)
    else:
        print(f"Warning: Model file not found at {args.model_path}")
        print("Using untrained model for evaluation (for testing purposes only)")

    print("\n" + "=" * 60)
    print("ADVERSARIAL ROBUSTNESS EVALUATION")
    print("=" * 60)

    # Evaluate on clean data
    print("\n1. Clean Data Evaluation")
    print("-" * 60)
    clean_acc = evaluate_clean(model, test_loader, config.DEVICE)
    print(f"\nClean Accuracy: {clean_acc:.2f}%")

    # Evaluate FGSM
    if args.eval_fgsm:
        print("\n2. FGSM Attack Evaluation")
        print("-" * 60)
        fgsm_results = evaluate_fgsm(model, test_loader, config.DEVICE)

        # Plot results
        epsilons = list(fgsm_results.keys())
        accuracies = list(fgsm_results.values())
        save_path = os.path.join(config.RESULTS_DIR, 'fgsm_robustness.png')
        plot_adversarial_comparison(
            clean_acc, accuracies, epsilons, attack_name='FGSM', save_path=save_path
        )

    # Evaluate PGD
    if args.eval_pgd:
        print("\n3. PGD Attack Evaluation")
        print("-" * 60)
        pgd_results = evaluate_pgd(model, test_loader, config.DEVICE)

    # Visualize examples
    if args.visualize:
        print("\n4. Visualizing Adversarial Examples")
        print("-" * 60)
        visualize_attack_examples(model, test_loader, config.DEVICE,
                                  attack_type='fgsm', num_samples=args.num_vis)
        visualize_attack_examples(model, test_loader, config.DEVICE,
                                  attack_type='pgd', num_samples=args.num_vis)

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate adversarial robustness')
    parser.add_argument('--model-path', type=str,
                        default='./models/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for evaluation')
    parser.add_argument('--eval-fgsm', action='store_true', default=True,
                        help='Evaluate FGSM attack')
    parser.add_argument('--eval-pgd', action='store_true', default=True,
                        help='Evaluate PGD attack')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Visualize adversarial examples')
    parser.add_argument('--num-vis', type=int, default=10,
                        help='Number of examples to visualize')

    args = parser.parse_args()
    main(args)
