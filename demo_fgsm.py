"""
FGSM Attack Demonstration Script

This script demonstrates all features of the FGSM attack implementation:
1. Untargeted attacks
2. Targeted attacks
3. Perturbation analysis
4. Visualization
5. Success rate evaluation

Usage:
    python demo_fgsm.py --model-path ./models/baseline_best_model.pth
"""

import torch
import argparse
import os
import sys

from attacks.fgsm import FGSM, visualize_fgsm_attack
from utils import get_cifar10_loaders, CIFAR10_CLASSES
from utils.visualization import denormalize
import config

# Try to import model
try:
    from models import get_resnet18, load_checkpoint
    MODELS_AVAILABLE = True
except ImportError:
    print("Warning: models module not fully available. Install dependencies first.")
    MODELS_AVAILABLE = False


def demo_untargeted_attack(model, test_loader, device, epsilon=8/255):
    """
    Demonstrate untargeted FGSM attack

    Goal: Cause any misclassification
    """
    print("\n" + "=" * 80)
    print("DEMO 1: UNTARGETED FGSM ATTACK")
    print("=" * 80)
    print(f"Goal: Cause model to misclassify (any wrong prediction)")
    print(f"Epsilon: {epsilon:.6f} ({epsilon*255:.2f}/255)")
    print()

    # Get a batch of test images
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    # Create untargeted FGSM attack
    fgsm = FGSM(model, epsilon=epsilon, targeted=False)

    # Generate adversarial examples
    print("Generating adversarial examples...")
    x_adv, perturbation = fgsm.generate(images, labels, return_perturbation=True)

    # Analyze perturbation
    print("\n📊 PERTURBATION ANALYSIS:")
    print("-" * 80)
    stats = fgsm.analyze_perturbation(images, x_adv, verbose=True)

    # Evaluate attack success
    print("\n📈 ATTACK EVALUATION:")
    print("-" * 80)
    results = fgsm.evaluate_attack_success(images, labels, x_adv, verbose=True)

    print("\n✓ Untargeted attack demonstration complete!")

    return x_adv, stats, results


def demo_targeted_attack(model, test_loader, device, epsilon=8/255):
    """
    Demonstrate targeted FGSM attack

    Goal: Force model to predict a specific (wrong) class
    """
    print("\n" + "=" * 80)
    print("DEMO 2: TARGETED FGSM ATTACK")
    print("=" * 80)
    print(f"Goal: Force model to predict specific target class")
    print(f"Epsilon: {epsilon:.6f} ({epsilon*255:.2f}/255)")
    print()

    # Get a batch of test images
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    # Create target labels (different from true labels)
    # Strategy: target = (true_label + 1) mod 10
    target_labels = (labels + 1) % 10

    print(f"Examples of targets:")
    for i in range(min(5, len(labels))):
        print(f"  Image {i+1}: {CIFAR10_CLASSES[labels[i]]} → {CIFAR10_CLASSES[target_labels[i]]}")
    print()

    # Create targeted FGSM attack
    fgsm_targeted = FGSM(model, epsilon=epsilon, targeted=True)

    # Generate adversarial examples
    print("Generating targeted adversarial examples...")
    x_adv, perturbation = fgsm_targeted.generate(
        images, target_labels, return_perturbation=True
    )

    # Analyze perturbation
    print("\n📊 PERTURBATION ANALYSIS:")
    print("-" * 80)
    stats = fgsm_targeted.analyze_perturbation(images, x_adv, verbose=True)

    # Evaluate attack success
    print("\n📈 ATTACK EVALUATION:")
    print("-" * 80)
    results = fgsm_targeted.evaluate_attack_success(
        images, labels, x_adv, target_labels=target_labels, verbose=True
    )

    print("\n✓ Targeted attack demonstration complete!")

    return x_adv, target_labels, stats, results


def demo_epsilon_comparison(model, test_loader, device):
    """
    Compare attack effectiveness across different epsilon values
    """
    print("\n" + "=" * 80)
    print("DEMO 3: EPSILON COMPARISON")
    print("=" * 80)
    print("Testing different epsilon values to see trade-off between")
    print("attack success and perturbation visibility.")
    print()

    # Get a batch of test images
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    # Test different epsilon values
    epsilons = [0, 2/255, 4/255, 8/255, 16/255, 32/255]

    print(f"{'Epsilon':<15} {'ε (0-255)':<12} {'Success Rate':<15} {'Accuracy Drop':<15} {'L∞ Norm':<12}")
    print("-" * 80)

    results_list = []

    for eps in epsilons:
        if eps == 0:
            # No attack
            model.eval()
            with torch.no_grad():
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                acc = (preds == labels).float().mean().item()

            print(f"{eps:.6f}    {eps*255:6.2f}      {'N/A':<15} {0.0:<15.2f} {0.0:<12.6f}")
            results_list.append({
                'epsilon': eps,
                'success_rate': 0.0,
                'accuracy_drop': 0.0,
                'linf_norm': 0.0
            })
        else:
            # Run attack
            fgsm = FGSM(model, epsilon=eps, targeted=False)
            x_adv = fgsm.generate(images, labels)

            # Evaluate
            stats = fgsm.analyze_perturbation(images, x_adv, verbose=False)
            results = fgsm.evaluate_attack_success(images, labels, x_adv, verbose=False)

            print(f"{eps:.6f}    {eps*255:6.2f}      "
                  f"{results['attack_success_rate']*100:6.2f}%        "
                  f"{results['accuracy_drop']*100:6.2f}%        "
                  f"{stats['linf_norm']:.6f}")

            results_list.append({
                'epsilon': eps,
                'success_rate': results['attack_success_rate'],
                'accuracy_drop': results['accuracy_drop'],
                'linf_norm': stats['linf_norm']
            })

    print("-" * 80)
    print("\nObservations:")
    print("  • Larger epsilon → Higher success rate")
    print("  • Larger epsilon → More visible perturbations")
    print("  • Trade-off: effectiveness vs imperceptibility")
    print()
    print("✓ Epsilon comparison complete!")

    return results_list


def demo_visualization(model, test_loader, device, epsilon=8/255):
    """
    Create comprehensive visualizations of FGSM attacks
    """
    print("\n" + "=" * 80)
    print("DEMO 4: VISUALIZATION")
    print("=" * 80)
    print("Creating visualizations of adversarial examples...")
    print()

    # Get a batch of test images
    images, labels = next(iter(test_loader))

    # Denormalization function for CIFAR-10
    def denormalize_cifar10(tensor):
        mean = torch.tensor(config.MEAN).view(3, 1, 1)
        std = torch.tensor(config.STD).view(3, 1, 1)
        return tensor * std + mean

    # Visualization 1: Untargeted attack
    print("1. Untargeted FGSM attack visualization...")
    save_path_untargeted = os.path.join(config.VIS_SAVE_DIR, 'fgsm_untargeted_demo.png')

    visualize_fgsm_attack(
        model=model,
        images=images,
        labels=labels,
        epsilon=epsilon,
        class_names=CIFAR10_CLASSES,
        device=device,
        num_samples=5,
        targeted=False,
        denormalize_fn=denormalize_cifar10,
        save_path=save_path_untargeted
    )

    # Visualization 2: Targeted attack
    print("\n2. Targeted FGSM attack visualization...")
    target_labels = (labels + 1) % 10  # Target next class
    save_path_targeted = os.path.join(config.VIS_SAVE_DIR, 'fgsm_targeted_demo.png')

    visualize_fgsm_attack(
        model=model,
        images=images,
        labels=labels,
        epsilon=epsilon,
        class_names=CIFAR10_CLASSES,
        device=device,
        num_samples=5,
        targeted=True,
        target_labels=target_labels,
        denormalize_fn=denormalize_cifar10,
        save_path=save_path_targeted
    )

    print("\n✓ Visualizations created successfully!")
    print(f"  Saved to: {config.VIS_SAVE_DIR}/")


def demo_batch_processing(model, test_loader, device, epsilon=8/255):
    """
    Demonstrate batch processing for large datasets
    """
    print("\n" + "=" * 80)
    print("DEMO 5: BATCH PROCESSING")
    print("=" * 80)
    print("Demonstrating efficient batch processing for large datasets...")
    print()

    # Get multiple batches
    all_images = []
    all_labels = []
    for i, (images, labels) in enumerate(test_loader):
        all_images.append(images)
        all_labels.append(labels)
        if i >= 4:  # Get 5 batches
            break

    all_images = torch.cat(all_images, dim=0).to(device)
    all_labels = torch.cat(all_labels, dim=0).to(device)

    print(f"Total images: {len(all_images)}")
    print(f"Processing in batches of 32...")
    print()

    # Create FGSM attack
    fgsm = FGSM(model, epsilon=epsilon, targeted=False)

    # Generate adversarial examples in batches
    import time
    start_time = time.time()

    x_adv = fgsm.generate_batch(all_images, all_labels, batch_size=32)

    elapsed_time = time.time() - start_time

    print(f"✓ Generated {len(x_adv)} adversarial examples")
    print(f"  Time: {elapsed_time:.2f} seconds")
    print(f"  Speed: {len(x_adv)/elapsed_time:.2f} images/second")

    # Quick evaluation
    results = fgsm.evaluate_attack_success(all_images, all_labels, x_adv, verbose=False)
    print(f"\n  Attack success rate: {results['attack_success_rate']*100:.2f}%")
    print(f"  Accuracy drop: {results['accuracy_drop']*100:.2f}%")

    print("\n✓ Batch processing demonstration complete!")


def main():
    """Main demonstration function"""
    parser = argparse.ArgumentParser(description='FGSM Attack Demonstration')
    parser.add_argument('--model-path', type=str,
                        default='./models/baseline_best_model.pth',
                        help='Path to trained model')
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='FGSM epsilon (default: 8/255)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for testing')
    parser.add_argument('--skip-viz', action='store_true',
                        help='Skip visualization (faster)')

    args = parser.parse_args()

    # Check if models available
    if not MODELS_AVAILABLE:
        print("Error: Cannot import models. Please install dependencies:")
        print("  pip install -r requirements.txt")
        return

    print("=" * 80)
    print("FGSM ATTACK - COMPREHENSIVE DEMONSTRATION")
    print("=" * 80)
    print()
    print("This script demonstrates all features of the FGSM attack:")
    print("  1. Untargeted attacks (cause any misclassification)")
    print("  2. Targeted attacks (force specific misclassification)")
    print("  3. Epsilon comparison (trade-off analysis)")
    print("  4. Visualization (see adversarial examples)")
    print("  5. Batch processing (efficient large-scale attacks)")
    print()
    print("=" * 80)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load data
    print("\nLoading CIFAR-10 test data...")
    _, test_loader = get_cifar10_loaders(test_batch_size=args.batch_size)

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = get_resnet18(num_classes=10, device=device)

    if os.path.exists(args.model_path):
        load_checkpoint(args.model_path, model, device=device)
        print("✓ Model loaded successfully")
    else:
        print("⚠ Warning: Model checkpoint not found. Using untrained model.")
        print("  Train a model first: python train_baseline.py")

    model.eval()

    # Run demonstrations
    try:
        # Demo 1: Untargeted attack
        demo_untargeted_attack(model, test_loader, device, args.epsilon)

        # Demo 2: Targeted attack
        demo_targeted_attack(model, test_loader, device, args.epsilon)

        # Demo 3: Epsilon comparison
        demo_epsilon_comparison(model, test_loader, device)

        # Demo 4: Visualization (if not skipped)
        if not args.skip_viz:
            demo_visualization(model, test_loader, device, args.epsilon)
        else:
            print("\n(Skipping visualization as requested)")

        # Demo 5: Batch processing
        demo_batch_processing(model, test_loader, device, args.epsilon)

        # Final summary
        print("\n" + "=" * 80)
        print("ALL DEMONSTRATIONS COMPLETED!")
        print("=" * 80)
        print()
        print("Key Takeaways:")
        print("  ✓ FGSM is a simple yet effective one-step attack")
        print("  ✓ Works for both untargeted and targeted scenarios")
        print("  ✓ Trade-off between attack success and perturbation visibility")
        print("  ✓ L∞ norm constraint ensures bounded perturbations")
        print("  ✓ Demonstrates vulnerability of standard-trained models")
        print()
        print("Next Steps:")
        print("  • Try the PGD attack (stronger, iterative)")
        print("  • Implement adversarial training for robustness")
        print("  • Explore certified defenses")
        print()
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\nDemonstration interrupted by user.")
    except Exception as e:
        print(f"\n\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
