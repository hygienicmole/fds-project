"""
PGD Attack Demonstration Script

This script demonstrates all features of the PGD attack implementation:
1. Untargeted PGD attacks
2. Targeted PGD attacks
3. Attack progression visualization
4. Comparison with FGSM
5. Iteration vs success rate analysis

Usage:
    python demo_pgd.py --model-path ./models/baseline_best_model.pth
"""

import torch
import argparse
import os
import sys

from attacks.pgd import PGD, visualize_pgd_progression
from attacks.fgsm import FGSM
from utils import get_cifar10_loaders, CIFAR10_CLASSES
import config

# Try to import model
try:
    from models import get_resnet18, load_checkpoint
    MODELS_AVAILABLE = True
except ImportError:
    print("Warning: models module not fully available. Install dependencies first.")
    MODELS_AVAILABLE = False


def demo_untargeted_pgd(model, test_loader, device, epsilon=8/255, alpha=2/255, iterations=20):
    """
    Demonstrate untargeted PGD attack with progress tracking
    """
    print("\n" + "=" * 80)
    print("DEMO 1: UNTARGETED PGD ATTACK WITH PROGRESS TRACKING")
    print("=" * 80)
    print(f"Goal: Maximize loss to cause misclassification")
    print(f"Epsilon (ε): {epsilon:.6f} ({epsilon*255:.2f}/255)")
    print(f"Alpha (α): {alpha:.6f} ({alpha*255:.2f}/255)")
    print(f"Iterations: {iterations}")
    print()

    # Get a batch of test images
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    # Create PGD attack with progress tracking
    pgd = PGD(
        model,
        epsilon=epsilon,
        alpha=alpha,
        iterations=iterations,
        random_start=True,
        targeted=False
    )

    # Generate adversarial examples with progress tracking
    print("Generating adversarial examples...")
    x_adv, history = pgd.generate(images, labels, track_progress=True)

    # Analyze perturbation
    print("\n📊 PERTURBATION ANALYSIS:")
    print("-" * 80)
    stats = pgd.analyze_perturbation(images, x_adv, verbose=True)

    # Evaluate attack success
    print("\n📈 ATTACK EVALUATION:")
    print("-" * 80)
    results = pgd.evaluate_attack_success(images, labels, x_adv, verbose=True)

    # Visualize progression
    print("\n📉 ATTACK PROGRESSION:")
    print("-" * 80)
    print(f"Initial loss: {history['losses'][0]:.4f}")
    print(f"Final loss: {history['losses'][-1]:.4f}")
    print(f"Initial success rate: {history['success_rates'][0]*100:.2f}%")
    print(f"Final success rate: {history['success_rates'][-1]*100:.2f}%")
    print()

    # Create visualization
    save_path = os.path.join(config.VIS_SAVE_DIR, 'pgd_progression_untargeted.png')
    visualize_pgd_progression(history, save_path=save_path)

    print("\n[OK] Untargeted PGD attack demonstration complete!")

    return x_adv, history, stats, results


def demo_targeted_pgd(model, test_loader, device, epsilon=8/255, alpha=2/255, iterations=20):
    """
    Demonstrate targeted PGD attack
    """
    print("\n" + "=" * 80)
    print("DEMO 2: TARGETED PGD ATTACK")
    print("=" * 80)
    print(f"Goal: Minimize loss for specific target class")
    print(f"Epsilon (ε): {epsilon:.6f} ({epsilon*255:.2f}/255)")
    print(f"Alpha (α): {alpha:.6f} ({alpha*255:.2f}/255)")
    print(f"Iterations: {iterations}")
    print()

    # Get a batch of test images
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    # Create target labels
    target_labels = (labels + 1) % 10

    print(f"Examples of targets:")
    for i in range(min(5, len(labels))):
        print(f"  Image {i+1}: {CIFAR10_CLASSES[labels[i]]} -> {CIFAR10_CLASSES[target_labels[i]]}")
    print()

    # Create targeted PGD attack
    pgd_targeted = PGD(
        model,
        epsilon=epsilon,
        alpha=alpha,
        iterations=iterations,
        random_start=True,
        targeted=True
    )

    # Generate adversarial examples with progress tracking
    print("Generating targeted adversarial examples...")
    x_adv, history = pgd_targeted.generate(images, target_labels, track_progress=True)

    # Analyze perturbation
    print("\n📊 PERTURBATION ANALYSIS:")
    print("-" * 80)
    stats = pgd_targeted.analyze_perturbation(images, x_adv, verbose=True)

    # Evaluate attack success
    print("\n📈 ATTACK EVALUATION:")
    print("-" * 80)
    results = pgd_targeted.evaluate_attack_success(
        images, labels, x_adv, target_labels=target_labels, verbose=True
    )

    # Visualize progression
    save_path = os.path.join(config.VIS_SAVE_DIR, 'pgd_progression_targeted.png')
    visualize_pgd_progression(history, save_path=save_path)

    print("\n[OK] Targeted PGD attack demonstration complete!")

    return x_adv, history, stats, results


def demo_pgd_vs_fgsm(model, test_loader, device, epsilon=8/255):
    """
    Compare PGD with FGSM to show strength difference
    """
    print("\n" + "=" * 80)
    print("DEMO 3: PGD VS FGSM COMPARISON")
    print("=" * 80)
    print("Comparing single-step FGSM with iterative PGD")
    print()

    # Get a batch of test images
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    print(f"Testing with epsilon = {epsilon:.6f} ({epsilon*255:.2f}/255)")
    print()

    # FGSM Attack
    print("1. FGSM Attack (single step):")
    fgsm = FGSM(model, epsilon=epsilon, targeted=False)
    x_adv_fgsm = fgsm.generate(images, labels)
    results_fgsm = fgsm.evaluate_attack_success(images, labels, x_adv_fgsm, verbose=False)
    print(f"   Success rate: {results_fgsm['attack_success_rate']*100:.2f}%")
    print(f"   Accuracy drop: {results_fgsm['accuracy_drop']*100:.2f}%")

    # PGD-7 Attack
    print("\n2. PGD-7 Attack (7 iterations):")
    pgd7 = PGD(model, epsilon=epsilon, alpha=2/255, iterations=7, random_start=True)
    x_adv_pgd7 = pgd7.generate(images, labels)
    results_pgd7 = pgd7.evaluate_attack_success(images, labels, x_adv_pgd7, verbose=False)
    print(f"   Success rate: {results_pgd7['attack_success_rate']*100:.2f}%")
    print(f"   Accuracy drop: {results_pgd7['accuracy_drop']*100:.2f}%")

    # PGD-20 Attack
    print("\n3. PGD-20 Attack (20 iterations):")
    pgd20 = PGD(model, epsilon=epsilon, alpha=2/255, iterations=20, random_start=True)
    x_adv_pgd20 = pgd20.generate(images, labels)
    results_pgd20 = pgd20.evaluate_attack_success(images, labels, x_adv_pgd20, verbose=False)
    print(f"   Success rate: {results_pgd20['attack_success_rate']*100:.2f}%")
    print(f"   Accuracy drop: {results_pgd20['accuracy_drop']*100:.2f}%")

    # PGD-40 Attack
    print("\n4. PGD-40 Attack (40 iterations):")
    pgd40 = PGD(model, epsilon=epsilon, alpha=2/255, iterations=40, random_start=True)
    x_adv_pgd40 = pgd40.generate(images, labels)
    results_pgd40 = pgd40.evaluate_attack_success(images, labels, x_adv_pgd40, verbose=False)
    print(f"   Success rate: {results_pgd40['attack_success_rate']*100:.2f}%")
    print(f"   Accuracy drop: {results_pgd40['accuracy_drop']*100:.2f}%")

    # Summary table
    print("\n" + "-" * 80)
    print(f"{'Attack':<15} {'Iterations':<12} {'Success Rate':<15} {'Accuracy Drop':<15}")
    print("-" * 80)
    print(f"{'FGSM':<15} {'1':<12} {results_fgsm['attack_success_rate']*100:6.2f}%        "
          f"{results_fgsm['accuracy_drop']*100:6.2f}%")
    print(f"{'PGD-7':<15} {'7':<12} {results_pgd7['attack_success_rate']*100:6.2f}%        "
          f"{results_pgd7['accuracy_drop']*100:6.2f}%")
    print(f"{'PGD-20':<15} {'20':<12} {results_pgd20['attack_success_rate']*100:6.2f}%        "
          f"{results_pgd20['accuracy_drop']*100:6.2f}%")
    print(f"{'PGD-40':<15} {'40':<12} {results_pgd40['attack_success_rate']*100:6.2f}%        "
          f"{results_pgd40['accuracy_drop']*100:6.2f}%")
    print("-" * 80)

    print("\nObservations:")
    print("  * PGD is significantly stronger than FGSM")
    print("  * More iterations -> higher success rate")
    print("  * Diminishing returns after ~20 iterations")
    print("  * PGD-20+ can achieve near 100% success rate")

    print("\n[OK] PGD vs FGSM comparison complete!")

    return {
        'fgsm': results_fgsm,
        'pgd7': results_pgd7,
        'pgd20': results_pgd20,
        'pgd40': results_pgd40
    }


def demo_random_start_effect(model, test_loader, device, epsilon=8/255, iterations=20):
    """
    Demonstrate the effect of random initialization
    """
    print("\n" + "=" * 80)
    print("DEMO 4: RANDOM INITIALIZATION EFFECT")
    print("=" * 80)
    print("Comparing PGD with and without random start")
    print()

    # Get a batch of test images
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    # PGD without random start
    print("1. PGD without random start:")
    pgd_no_random = PGD(
        model, epsilon=epsilon, alpha=2/255, iterations=iterations,
        random_start=False
    )
    x_adv_no_random = pgd_no_random.generate(images, labels)
    results_no_random = pgd_no_random.evaluate_attack_success(
        images, labels, x_adv_no_random, verbose=False
    )
    print(f"   Success rate: {results_no_random['attack_success_rate']*100:.2f}%")

    # PGD with random start
    print("\n2. PGD with random start:")
    pgd_random = PGD(
        model, epsilon=epsilon, alpha=2/255, iterations=iterations,
        random_start=True
    )
    x_adv_random = pgd_random.generate(images, labels)
    results_random = pgd_random.evaluate_attack_success(
        images, labels, x_adv_random, verbose=False
    )
    print(f"   Success rate: {results_random['attack_success_rate']*100:.2f}%")

    # Comparison
    print("\n" + "-" * 80)
    improvement = results_random['attack_success_rate'] - results_no_random['attack_success_rate']
    print(f"Improvement with random start: {improvement*100:+.2f}%")
    print("-" * 80)

    print("\nWhy random start helps:")
    print("  * Escapes local minima")
    print("  * Finds stronger adversarial examples")
    print("  * Increases attack diversity")
    print("  * Standard practice for PGD attacks")

    print("\n[OK] Random initialization demonstration complete!")


def demo_alpha_tuning(model, test_loader, device, epsilon=8/255, iterations=20):
    """
    Demonstrate the effect of different alpha (step size) values
    """
    print("\n" + "=" * 80)
    print("DEMO 5: ALPHA (STEP SIZE) TUNING")
    print("=" * 80)
    print("Testing different step sizes for PGD")
    print()

    # Get a batch of test images
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    # Test different alpha values
    alpha_values = [epsilon/10, epsilon/4, epsilon/2, epsilon, 2*epsilon]

    print(f"{'Alpha':<20} {'α/ε Ratio':<15} {'Success Rate':<15} {'Note':<20}")
    print("-" * 80)

    for alpha in alpha_values:
        pgd = PGD(model, epsilon=epsilon, alpha=alpha, iterations=iterations, random_start=True)
        x_adv = pgd.generate(images, labels)
        results = pgd.evaluate_attack_success(images, labels, x_adv, verbose=False)

        ratio = alpha / epsilon
        note = ""
        if ratio < 0.2:
            note = "Too small"
        elif ratio > 1.5:
            note = "Too large"
        else:
            note = "Good range"

        print(f"{alpha:.6f}         {ratio:.2f}            "
              f"{results['attack_success_rate']*100:6.2f}%        {note}")

    print("-" * 80)
    print("\nCommon alpha choices:")
    print(f"  * α = ε/4 = {epsilon/4:.6f}")
    print(f"  * α = 2ε/T = {2*epsilon/iterations:.6f} (where T={iterations})")
    print(f"  * Typical: α = 2/255 for ε = 8/255")

    print("\n[OK] Alpha tuning demonstration complete!")


def main():
    """Main demonstration function"""
    parser = argparse.ArgumentParser(description='PGD Attack Demonstration')
    parser.add_argument('--model-path', type=str,
                        default='./models/baseline_best_model.pth',
                        help='Path to trained model')
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='PGD epsilon (default: 8/255)')
    parser.add_argument('--alpha', type=float, default=2/255,
                        help='PGD alpha (default: 2/255)')
    parser.add_argument('--iterations', type=int, default=20,
                        help='Number of PGD iterations (default: 20)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for testing')

    args = parser.parse_args()

    # Check if models available
    if not MODELS_AVAILABLE:
        print("Error: Cannot import models. Please install dependencies:")
        print("  pip install -r requirements.txt")
        return

    print("=" * 80)
    print("PGD ATTACK - COMPREHENSIVE DEMONSTRATION")
    print("=" * 80)
    print()
    print("This script demonstrates all features of the PGD attack:")
    print("  1. Untargeted PGD with progress tracking")
    print("  2. Targeted PGD attacks")
    print("  3. Comparison with FGSM (strength difference)")
    print("  4. Random initialization effect")
    print("  5. Alpha (step size) tuning")
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
        print("[OK] Model loaded successfully")
    else:
        print("[WARNING] Warning: Model checkpoint not found. Using untrained model.")
        print("  Train a model first: python train_baseline.py")

    model.eval()

    # Run demonstrations
    try:
        # Demo 1: Untargeted PGD with progress tracking
        demo_untargeted_pgd(model, test_loader, device,
                           args.epsilon, args.alpha, args.iterations)

        # Demo 2: Targeted PGD
        demo_targeted_pgd(model, test_loader, device,
                         args.epsilon, args.alpha, args.iterations)

        # Demo 3: PGD vs FGSM comparison
        demo_pgd_vs_fgsm(model, test_loader, device, args.epsilon)

        # Demo 4: Random start effect
        demo_random_start_effect(model, test_loader, device,
                                args.epsilon, args.iterations)

        # Demo 5: Alpha tuning
        demo_alpha_tuning(model, test_loader, device,
                         args.epsilon, args.iterations)

        # Final summary
        print("\n" + "=" * 80)
        print("ALL DEMONSTRATIONS COMPLETED!")
        print("=" * 80)
        print()
        print("Key Takeaways:")
        print("  [OK] PGD is significantly stronger than FGSM")
        print("  [OK] Iterative refinement finds better adversarial examples")
        print("  [OK] Random initialization helps escape local minima")
        print("  [OK] Projection ensures L∞ constraint satisfaction")
        print("  [OK] 20-40 iterations typically sufficient")
        print("  [OK] Alpha ≈ ε/4 is a good default choice")
        print()
        print("Visualizations saved to:")
        print(f"  {config.VIS_SAVE_DIR}/pgd_progression_untargeted.png")
        print(f"  {config.VIS_SAVE_DIR}/pgd_progression_targeted.png")
        print()
        print("Next Steps:")
        print("  * Compare with baseline model performance")
        print("  * Try adversarial training for robustness")
        print("  * Explore certified defenses")
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
