"""
Generate Comprehensive Adversarial Attack Visualizations

This script generates all advanced visualizations for analyzing adversarial attacks:
1. Side-by-side comparison of 20 original vs adversarial images
2. Perturbation heatmaps (amplified for visibility)
3. Confusion matrices (clean and adversarial)
4. Class-wise attack success rates
5. Confidence score distributions

Usage:
    python generate_visualizations.py --model-path ./models/baseline_best_model.pth
"""

import torch
import argparse
import os

from attacks.fgsm import FGSM
from attacks.pgd import PGD
from models import get_resnet18, load_checkpoint
from utils import get_cifar10_loaders
from utils.visualization import generate_all_visualizations, denormalize
import config


def main():
    parser = argparse.ArgumentParser(description='Generate Comprehensive Visualizations')
    parser.add_argument('--model-path', type=str,
                       default='./models/baseline_best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--attack', type=str, default='fgsm',
                       choices=['fgsm', 'pgd'],
                       help='Attack type to visualize')
    parser.add_argument('--epsilon', type=float, default=0.03,
                       help='Epsilon value for attack (default: 0.03)')
    parser.add_argument('--pgd-alpha', type=float, default=None,
                       help='PGD alpha (default: epsilon/4)')
    parser.add_argument('--pgd-iterations', type=int, default=20,
                       help='PGD iterations (default: 20)')
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='Number of samples to evaluate (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size (default: 100)')
    parser.add_argument('--save-dir', type=str,
                       default='./results/visualizations',
                       help='Directory to save visualizations')

    args = parser.parse_args()

    print("=" * 80)
    print("COMPREHENSIVE ADVERSARIAL ATTACK VISUALIZATION GENERATOR")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_path}")
    print(f"  Attack: {args.attack.upper()}")
    print(f"  Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.2f}/255)")
    if args.attack == 'pgd':
        alpha = args.pgd_alpha if args.pgd_alpha else args.epsilon * 0.25
        print(f"  PGD Alpha: {alpha:.4f}")
        print(f"  PGD Iterations: {args.pgd_iterations}")
    print(f"  Samples: {args.num_samples}")
    print(f"  Save directory: {args.save_dir}")
    print()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load data
    print("Loading CIFAR-10 test data...")
    _, test_loader = get_cifar10_loaders(test_batch_size=args.batch_size)
    print("[OK] Data loaded\n")

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = get_resnet18(num_classes=10, device=device)

    if os.path.exists(args.model_path):
        load_checkpoint(args.model_path, model, device=device)
        print("[OK] Model loaded successfully\n")
    else:
        print("[WARNING] Model checkpoint not found. Using untrained model.\n")

    model.eval()

    # Collect samples
    print(f"Collecting {args.num_samples} samples...")
    all_images = []
    all_labels = []

    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size

    for i, (images, labels) in enumerate(test_loader):
        if i >= num_batches:
            break
        all_images.append(images)
        all_labels.append(labels)

    all_images = torch.cat(all_images, dim=0)[:args.num_samples].to(device)
    all_labels = torch.cat(all_labels, dim=0)[:args.num_samples].to(device)

    print(f"[OK] Collected {len(all_images)} samples\n")

    # Create attack
    print(f"Creating {args.attack.upper()} attack...")
    if args.attack == 'fgsm':
        attack = FGSM(model, epsilon=args.epsilon, targeted=False)
        attack_name = 'FGSM'
    else:
        alpha = args.pgd_alpha if args.pgd_alpha else args.epsilon * 0.25
        attack = PGD(
            model,
            epsilon=args.epsilon,
            alpha=alpha,
            iterations=args.pgd_iterations,
            random_start=True,
            targeted=False
        )
        attack_name = f'PGD-{args.pgd_iterations}'

    print(f"[OK] {attack_name} attack created\n")

    # Generate adversarial examples
    print("Generating adversarial examples...")
    print("(This may take a few minutes depending on attack and sample count)")

    if args.attack == 'fgsm':
        adv_images = attack.generate(all_images, all_labels)
    else:
        # Generate PGD in batches to avoid memory issues
        adv_images_list = []
        for i in range(0, len(all_images), args.batch_size):
            batch_images = all_images[i:i+args.batch_size]
            batch_labels = all_labels[i:i+args.batch_size]
            batch_adv = attack.generate(batch_images, batch_labels)
            adv_images_list.append(batch_adv)
            print(f"  Processed {min(i+args.batch_size, len(all_images))}/{len(all_images)} samples", end='\r')
        adv_images = torch.cat(adv_images_list, dim=0)
        print()

    print(f"[OK] Generated {len(adv_images)} adversarial examples\n")

    # Denormalization function
    def denormalize_fn(tensor):
        mean = torch.tensor(config.MEAN).view(3, 1, 1).to(tensor.device)
        std = torch.tensor(config.STD).view(3, 1, 1).to(tensor.device)
        return tensor * std + mean

    # Generate all visualizations
    results = generate_all_visualizations(
        model=model,
        original_images=all_images,
        adversarial_images=adv_images,
        true_labels=all_labels,
        attack_name=attack_name,
        epsilon=args.epsilon,
        save_dir=args.save_dir,
        denormalize_fn=denormalize_fn
    )

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    class_stats = results['class_statistics']
    conf_stats = results['confidence_statistics']

    print("\nClass-wise Attack Success Rates:")
    print("-" * 80)
    for i, class_name in enumerate(class_stats['class_names']):
        print(f"  {class_name:<12}: Clean {class_stats['clean_accuracy'][i]:5.1f}%  "
              f"Adv {class_stats['adversarial_accuracy'][i]:5.1f}%  "
              f"ASR {class_stats['attack_success_rate'][i]:5.1f}%")

    mean_asr = class_stats['attack_success_rate'].mean()
    print(f"\n  {'Mean ASR':<12}: {mean_asr:5.1f}%")

    print("\nConfidence Score Statistics:")
    print("-" * 80)
    print(f"  Clean predictions (mean):       {conf_stats['clean_mean']:.4f} ± {conf_stats['clean_std']:.4f}")
    print(f"  Adversarial predictions (mean): {conf_stats['adv_mean']:.4f} ± {conf_stats['adv_std']:.4f}")
    print(f"  Confidence drop:                {conf_stats['clean_mean'] - conf_stats['adv_mean']:.4f}")
    print(f"\n  Confidence for true class:")
    print(f"    Clean:       {conf_stats['clean_conf_true_mean']:.4f}")
    print(f"    Adversarial: {conf_stats['adv_conf_true_mean']:.4f}")
    print(f"    Drop:        {conf_stats['clean_conf_true_mean'] - conf_stats['adv_conf_true_mean']:.4f}")

    print("\n" + "=" * 80)
    print("VISUALIZATION GENERATION COMPLETE!")
    print("=" * 80)
    print(f"\nAll visualizations saved to: {args.save_dir}/")
    print("\nGenerated files:")
    for name, path in results['paths'].items():
        print(f"  - {os.path.basename(path)}")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
