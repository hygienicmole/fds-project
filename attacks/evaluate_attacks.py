"""
Comprehensive Attack Evaluation Script

This script evaluates the robustness of the trained baseline model against
FGSM and PGD adversarial attacks with various epsilon values and settings.

Features:
- Clean accuracy evaluation
- FGSM attack with multiple epsilon values
- PGD attack with multiple configurations
- Comprehensive metrics (ACC, ASR, adversarial accuracy)
- Comparison plots (accuracy vs epsilon curves)
- Visual grid of original vs adversarial examples
- JSON results export

Usage:
    python attacks/evaluate_attacks.py --model-path ./models/baseline_best_model.pth
"""

import torch
import torch.nn as nn
import argparse
import os
import json
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from tqdm import tqdm

from attacks.fgsm import FGSM
from attacks.pgd import PGD
from models import get_resnet18, load_checkpoint
from utils import get_cifar10_loaders, CIFAR10_CLASSES
import config


class AttackEvaluator:
    """
    Comprehensive evaluator for adversarial attacks

    Evaluates model robustness against FGSM and PGD attacks with
    various configurations and generates detailed reports.
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
        num_eval_batches: int = None
    ):
        """
        Initialize attack evaluator

        Args:
            model: Model to evaluate
            test_loader: DataLoader for test data
            device: Device to use (cuda/cpu)
            num_eval_batches: Number of batches to evaluate (None = all)
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.num_eval_batches = num_eval_batches

        self.model.eval()

        # Results storage
        self.results = {
            'clean': {},
            'fgsm': {},
            'pgd': {},
            'metadata': {
                'device': str(device),
                'num_eval_batches': num_eval_batches,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }

    def evaluate_clean_accuracy(self) -> Dict:
        """
        Evaluate clean accuracy (no attack)

        Returns:
            Dictionary with clean accuracy metrics
        """
        print("\n" + "=" * 80)
        print("EVALUATING CLEAN ACCURACY (No Attack)")
        print("=" * 80)

        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Clean Eval")
            for batch_idx, (images, labels) in enumerate(pbar):
                if self.num_eval_batches and batch_idx >= self.num_eval_batches:
                    break

                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                predictions = outputs.argmax(dim=1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                # Update progress bar
                accuracy = 100.0 * correct / total
                pbar.set_postfix({'acc': f'{accuracy:.2f}%'})

        clean_accuracy = correct / total

        print(f"\n✓ Clean Accuracy: {clean_accuracy * 100:.2f}%")
        print(f"  Correct: {correct}/{total}")

        self.results['clean'] = {
            'accuracy': clean_accuracy,
            'correct': correct,
            'total': total
        }

        return self.results['clean']

    def evaluate_fgsm(self, epsilon_values: List[float]) -> Dict:
        """
        Evaluate FGSM attack with different epsilon values

        Args:
            epsilon_values: List of epsilon values to test

        Returns:
            Dictionary with FGSM results for each epsilon
        """
        print("\n" + "=" * 80)
        print("EVALUATING FGSM ATTACK")
        print("=" * 80)
        print(f"Testing epsilon values: {epsilon_values}")
        print()

        fgsm_results = {}

        for epsilon in epsilon_values:
            print(f"\n--- FGSM with ε = {epsilon:.4f} ({epsilon*255:.2f}/255) ---")

            # Create FGSM attack
            fgsm = FGSM(self.model, epsilon=epsilon, targeted=False)

            correct_clean = 0
            correct_adv = 0
            total = 0

            pbar = tqdm(self.test_loader, desc=f"FGSM ε={epsilon:.4f}")
            for batch_idx, (images, labels) in enumerate(pbar):
                if self.num_eval_batches and batch_idx >= self.num_eval_batches:
                    break

                images, labels = images.to(self.device), labels.to(self.device)

                # Clean predictions
                with torch.no_grad():
                    clean_outputs = self.model(images)
                    clean_preds = clean_outputs.argmax(dim=1)
                    correct_clean += (clean_preds == labels).sum().item()

                # Generate adversarial examples
                adv_images = fgsm.generate(images, labels)

                # Adversarial predictions
                with torch.no_grad():
                    adv_outputs = self.model(adv_images)
                    adv_preds = adv_outputs.argmax(dim=1)
                    correct_adv += (adv_preds == labels).sum().item()

                total += labels.size(0)

                # Update progress bar
                clean_acc = 100.0 * correct_clean / total
                adv_acc = 100.0 * correct_adv / total
                asr = 100.0 * (correct_clean - correct_adv) / total
                pbar.set_postfix({
                    'clean_acc': f'{clean_acc:.1f}%',
                    'adv_acc': f'{adv_acc:.1f}%',
                    'asr': f'{asr:.1f}%'
                })

            # Calculate metrics
            clean_accuracy = correct_clean / total
            adv_accuracy = correct_adv / total
            attack_success_rate = (correct_clean - correct_adv) / total
            accuracy_drop = clean_accuracy - adv_accuracy

            # Store results
            fgsm_results[epsilon] = {
                'epsilon': epsilon,
                'clean_accuracy': clean_accuracy,
                'adversarial_accuracy': adv_accuracy,
                'attack_success_rate': attack_success_rate,
                'accuracy_drop': accuracy_drop,
                'correct_clean': correct_clean,
                'correct_adv': correct_adv,
                'total': total
            }

            # Print summary
            print(f"\n  Clean Accuracy:       {clean_accuracy * 100:.2f}%")
            print(f"  Adversarial Accuracy: {adv_accuracy * 100:.2f}%")
            print(f"  Attack Success Rate:  {attack_success_rate * 100:.2f}%")
            print(f"  Accuracy Drop:        {accuracy_drop * 100:.2f}%")

        self.results['fgsm'] = fgsm_results
        return fgsm_results

    def evaluate_pgd(
        self,
        epsilon_values: List[float],
        alpha_ratio: float = 0.25,
        iterations_list: List[int] = [7, 20, 40]
    ) -> Dict:
        """
        Evaluate PGD attack with different configurations

        Args:
            epsilon_values: List of epsilon values to test
            alpha_ratio: Alpha as fraction of epsilon (default: 0.25)
            iterations_list: List of iteration counts to test

        Returns:
            Dictionary with PGD results
        """
        print("\n" + "=" * 80)
        print("EVALUATING PGD ATTACK")
        print("=" * 80)
        print(f"Testing epsilon values: {epsilon_values}")
        print(f"Testing iterations: {iterations_list}")
        print(f"Alpha ratio: {alpha_ratio} (alpha = epsilon * {alpha_ratio})")
        print()

        pgd_results = {}

        for epsilon in epsilon_values:
            pgd_results[epsilon] = {}
            alpha = epsilon * alpha_ratio

            for iterations in iterations_list:
                config_name = f"eps_{epsilon:.4f}_iter_{iterations}"
                print(f"\n--- PGD: ε={epsilon:.4f}, α={alpha:.4f}, iters={iterations} ---")

                # Create PGD attack
                pgd = PGD(
                    self.model,
                    epsilon=epsilon,
                    alpha=alpha,
                    iterations=iterations,
                    random_start=True,
                    targeted=False
                )

                correct_clean = 0
                correct_adv = 0
                total = 0

                pbar = tqdm(self.test_loader, desc=f"PGD ε={epsilon:.4f} i={iterations}")
                for batch_idx, (images, labels) in enumerate(pbar):
                    if self.num_eval_batches and batch_idx >= self.num_eval_batches:
                        break

                    images, labels = images.to(self.device), labels.to(self.device)

                    # Clean predictions
                    with torch.no_grad():
                        clean_outputs = self.model(images)
                        clean_preds = clean_outputs.argmax(dim=1)
                        correct_clean += (clean_preds == labels).sum().item()

                    # Generate adversarial examples
                    adv_images = pgd.generate(images, labels)

                    # Adversarial predictions
                    with torch.no_grad():
                        adv_outputs = self.model(adv_images)
                        adv_preds = adv_outputs.argmax(dim=1)
                        correct_adv += (adv_preds == labels).sum().item()

                    total += labels.size(0)

                    # Update progress bar
                    clean_acc = 100.0 * correct_clean / total
                    adv_acc = 100.0 * correct_adv / total
                    asr = 100.0 * (correct_clean - correct_adv) / total
                    pbar.set_postfix({
                        'clean_acc': f'{clean_acc:.1f}%',
                        'adv_acc': f'{adv_acc:.1f}%',
                        'asr': f'{asr:.1f}%'
                    })

                # Calculate metrics
                clean_accuracy = correct_clean / total
                adv_accuracy = correct_adv / total
                attack_success_rate = (correct_clean - correct_adv) / total
                accuracy_drop = clean_accuracy - adv_accuracy

                # Store results
                pgd_results[epsilon][iterations] = {
                    'epsilon': epsilon,
                    'alpha': alpha,
                    'iterations': iterations,
                    'clean_accuracy': clean_accuracy,
                    'adversarial_accuracy': adv_accuracy,
                    'attack_success_rate': attack_success_rate,
                    'accuracy_drop': accuracy_drop,
                    'correct_clean': correct_clean,
                    'correct_adv': correct_adv,
                    'total': total
                }

                # Print summary
                print(f"\n  Clean Accuracy:       {clean_accuracy * 100:.2f}%")
                print(f"  Adversarial Accuracy: {adv_accuracy * 100:.2f}%")
                print(f"  Attack Success Rate:  {attack_success_rate * 100:.2f}%")
                print(f"  Accuracy Drop:        {accuracy_drop * 100:.2f}%")

        self.results['pgd'] = pgd_results
        return pgd_results

    def generate_comparison_plots(self, save_dir: str = './results/attack_evaluation'):
        """
        Generate comparison plots for attack effectiveness

        Creates:
        - Accuracy vs Epsilon curves (FGSM and PGD)
        - Attack Success Rate vs Epsilon
        - FGSM vs PGD comparison

        Args:
            save_dir: Directory to save plots
        """
        print("\n" + "=" * 80)
        print("GENERATING COMPARISON PLOTS")
        print("=" * 80)

        os.makedirs(save_dir, exist_ok=True)

        # Extract FGSM data
        fgsm_epsilons = sorted(self.results['fgsm'].keys())
        fgsm_clean_acc = [self.results['fgsm'][e]['clean_accuracy'] * 100 for e in fgsm_epsilons]
        fgsm_adv_acc = [self.results['fgsm'][e]['adversarial_accuracy'] * 100 for e in fgsm_epsilons]
        fgsm_asr = [self.results['fgsm'][e]['attack_success_rate'] * 100 for e in fgsm_epsilons]

        # Extract PGD data (use highest iteration count for comparison)
        pgd_epsilons = sorted(self.results['pgd'].keys())
        if pgd_epsilons:
            # Get the highest iteration count used
            max_iters = max(self.results['pgd'][pgd_epsilons[0]].keys())
            pgd_adv_acc = [self.results['pgd'][e][max_iters]['adversarial_accuracy'] * 100 for e in pgd_epsilons]
            pgd_asr = [self.results['pgd'][e][max_iters]['attack_success_rate'] * 100 for e in pgd_epsilons]

        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1: Accuracy vs Epsilon
        axes[0].plot(fgsm_epsilons, fgsm_clean_acc, 'o-', linewidth=2, markersize=8,
                     label='Clean', color='green', alpha=0.7)
        axes[0].plot(fgsm_epsilons, fgsm_adv_acc, 's-', linewidth=2, markersize=8,
                     label='FGSM', color='red', alpha=0.7)
        if pgd_epsilons:
            axes[0].plot(pgd_epsilons, pgd_adv_acc, '^-', linewidth=2, markersize=8,
                         label=f'PGD-{max_iters}', color='purple', alpha=0.7)
        axes[0].set_xlabel('Epsilon (ε)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        axes[0].set_title('Model Accuracy vs Attack Strength', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 105])

        # Plot 2: Attack Success Rate vs Epsilon
        axes[1].plot(fgsm_epsilons, fgsm_asr, 's-', linewidth=2, markersize=8,
                     label='FGSM', color='red', alpha=0.7)
        if pgd_epsilons:
            axes[1].plot(pgd_epsilons, pgd_asr, '^-', linewidth=2, markersize=8,
                         label=f'PGD-{max_iters}', color='purple', alpha=0.7)
        axes[1].set_xlabel('Epsilon (ε)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
        axes[1].set_title('Attack Success Rate vs Epsilon', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 105])

        # Plot 3: PGD iterations comparison (if available)
        if pgd_epsilons and len(self.results['pgd'][pgd_epsilons[0]]) > 1:
            # Pick a middle epsilon value
            middle_eps = pgd_epsilons[len(pgd_epsilons) // 2]
            iterations = sorted(self.results['pgd'][middle_eps].keys())
            iter_acc = [self.results['pgd'][middle_eps][i]['adversarial_accuracy'] * 100 for i in iterations]
            iter_asr = [self.results['pgd'][middle_eps][i]['attack_success_rate'] * 100 for i in iterations]

            axes[2].plot(iterations, iter_acc, 'o-', linewidth=2, markersize=8,
                         label='Adversarial Accuracy', color='blue', alpha=0.7)
            axes[2].plot(iterations, iter_asr, 's-', linewidth=2, markersize=8,
                         label='Attack Success Rate', color='orange', alpha=0.7)
            axes[2].set_xlabel('PGD Iterations', fontsize=12, fontweight='bold')
            axes[2].set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
            axes[2].set_title(f'PGD Iterations Effect (ε={middle_eps:.3f})', fontsize=14, fontweight='bold')
            axes[2].legend(fontsize=11)
            axes[2].grid(True, alpha=0.3)
            axes[2].set_ylim([0, 105])

        plt.tight_layout()

        save_path = os.path.join(save_dir, 'attack_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plots saved to: {save_path}")
        plt.close()

    def generate_visual_grid(
        self,
        epsilon_fgsm: float = 0.03,
        epsilon_pgd: float = 0.03,
        num_samples: int = 8,
        save_dir: str = './results/attack_evaluation'
    ):
        """
        Generate grid showing original vs adversarial examples

        Creates a grid with 4 columns:
        - Original images
        - FGSM adversarial examples
        - PGD adversarial examples
        - Perturbation comparisons

        Args:
            epsilon_fgsm: Epsilon for FGSM visualization
            epsilon_pgd: Epsilon for PGD visualization
            num_samples: Number of samples to show
            save_dir: Directory to save visualization
        """
        print("\n" + "=" * 80)
        print("GENERATING VISUAL GRID")
        print("=" * 80)
        print(f"FGSM epsilon: {epsilon_fgsm:.4f}")
        print(f"PGD epsilon:  {epsilon_pgd:.4f}")
        print(f"Samples:      {num_samples}")

        os.makedirs(save_dir, exist_ok=True)

        # Get a batch of test images
        images, labels = next(iter(self.test_loader))
        images = images[:num_samples].to(self.device)
        labels = labels[:num_samples].to(self.device)

        # Create attacks
        fgsm = FGSM(self.model, epsilon=epsilon_fgsm, targeted=False)
        pgd = PGD(self.model, epsilon=epsilon_pgd, alpha=epsilon_pgd*0.25,
                  iterations=20, random_start=True, targeted=False)

        # Generate adversarial examples
        print("Generating FGSM examples...")
        adv_fgsm = fgsm.generate(images, labels)

        print("Generating PGD examples...")
        adv_pgd = pgd.generate(images, labels)

        # Get predictions
        with torch.no_grad():
            pred_clean = self.model(images).argmax(dim=1)
            pred_fgsm = self.model(adv_fgsm).argmax(dim=1)
            pred_pgd = self.model(adv_pgd).argmax(dim=1)

        # Denormalize for visualization
        def denormalize(tensor):
            mean = torch.tensor(config.MEAN).view(3, 1, 1).to(tensor.device)
            std = torch.tensor(config.STD).view(3, 1, 1).to(tensor.device)
            return tensor * std + mean

        images_denorm = denormalize(images).cpu()
        adv_fgsm_denorm = denormalize(adv_fgsm).cpu()
        adv_pgd_denorm = denormalize(adv_pgd).cpu()

        # Compute perturbations (amplified for visibility)
        pert_fgsm = (adv_fgsm - images).cpu()
        pert_pgd = (adv_pgd - images).cpu()

        # Create figure
        fig = plt.figure(figsize=(20, num_samples * 2.5))
        gs = gridspec.GridSpec(num_samples, 4, figure=fig, hspace=0.3, wspace=0.2)

        for i in range(num_samples):
            # Column 1: Original
            ax1 = fig.add_subplot(gs[i, 0])
            img = images_denorm[i].permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            ax1.imshow(img)
            ax1.axis('off')
            true_label = CIFAR10_CLASSES[labels[i].item()]
            pred_label = CIFAR10_CLASSES[pred_clean[i].item()]
            color = 'green' if pred_clean[i] == labels[i] else 'red'
            ax1.set_title(f'Original\nTrue: {true_label}\nPred: {pred_label}',
                         fontsize=10, color=color, fontweight='bold')

            # Column 2: FGSM
            ax2 = fig.add_subplot(gs[i, 1])
            img_fgsm = adv_fgsm_denorm[i].permute(1, 2, 0).numpy()
            img_fgsm = np.clip(img_fgsm, 0, 1)
            ax2.imshow(img_fgsm)
            ax2.axis('off')
            pred_label_fgsm = CIFAR10_CLASSES[pred_fgsm[i].item()]
            color = 'red' if pred_fgsm[i] != labels[i] else 'green'
            ax2.set_title(f'FGSM (ε={epsilon_fgsm:.3f})\nPred: {pred_label_fgsm}',
                         fontsize=10, color=color, fontweight='bold')

            # Column 3: PGD
            ax3 = fig.add_subplot(gs[i, 2])
            img_pgd = adv_pgd_denorm[i].permute(1, 2, 0).numpy()
            img_pgd = np.clip(img_pgd, 0, 1)
            ax3.imshow(img_pgd)
            ax3.axis('off')
            pred_label_pgd = CIFAR10_CLASSES[pred_pgd[i].item()]
            color = 'red' if pred_pgd[i] != labels[i] else 'green'
            ax3.set_title(f'PGD-20 (ε={epsilon_pgd:.3f})\nPred: {pred_label_pgd}',
                         fontsize=10, color=color, fontweight='bold')

            # Column 4: Perturbation comparison
            ax4 = fig.add_subplot(gs[i, 3])
            # Show FGSM perturbation (amplified)
            pert_vis = pert_fgsm[i].permute(1, 2, 0).numpy()
            pert_vis = (pert_vis - pert_vis.min()) / (pert_vis.max() - pert_vis.min() + 1e-8)
            ax4.imshow(pert_vis)
            ax4.axis('off')
            linf_fgsm = pert_fgsm[i].abs().max().item()
            linf_pgd = pert_pgd[i].abs().max().item()
            ax4.set_title(f'Perturbation\nFGSM L∞: {linf_fgsm:.4f}\nPGD L∞: {linf_pgd:.4f}',
                         fontsize=10, fontweight='bold')

        # Add main title
        fig.suptitle('Adversarial Attack Examples: Original vs FGSM vs PGD',
                    fontsize=16, fontweight='bold', y=0.995)

        save_path = os.path.join(save_dir, 'attack_visual_grid.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visual grid saved to: {save_path}")
        plt.close()

    def save_results(self, save_path: str = './results/attack_results.json'):
        """
        Save evaluation results to JSON file

        Args:
            save_path: Path to save JSON file
        """
        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            else:
                return obj

        serializable_results = convert_to_serializable(self.results)

        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"✓ Results saved to: {save_path}")

        # Also save a human-readable summary
        summary_path = save_path.replace('.json', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ADVERSARIAL ATTACK EVALUATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            # Clean accuracy
            f.write(f"Clean Accuracy: {self.results['clean']['accuracy'] * 100:.2f}%\n\n")

            # FGSM results
            f.write("-" * 80 + "\n")
            f.write("FGSM ATTACK RESULTS\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Epsilon':<12} {'Clean Acc':<12} {'Adv Acc':<12} {'ASR':<12} {'Drop':<12}\n")
            f.write("-" * 80 + "\n")
            for eps in sorted(self.results['fgsm'].keys()):
                r = self.results['fgsm'][eps]
                f.write(f"{eps:<12.4f} {r['clean_accuracy']*100:<12.2f} "
                       f"{r['adversarial_accuracy']*100:<12.2f} "
                       f"{r['attack_success_rate']*100:<12.2f} "
                       f"{r['accuracy_drop']*100:<12.2f}\n")

            # PGD results
            if self.results['pgd']:
                f.write("\n" + "-" * 80 + "\n")
                f.write("PGD ATTACK RESULTS\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'Epsilon':<12} {'Iters':<8} {'Clean Acc':<12} {'Adv Acc':<12} {'ASR':<12} {'Drop':<12}\n")
                f.write("-" * 80 + "\n")
                for eps in sorted(self.results['pgd'].keys()):
                    for iters in sorted(self.results['pgd'][eps].keys()):
                        r = self.results['pgd'][eps][iters]
                        f.write(f"{eps:<12.4f} {iters:<8} {r['clean_accuracy']*100:<12.2f} "
                               f"{r['adversarial_accuracy']*100:<12.2f} "
                               f"{r['attack_success_rate']*100:<12.2f} "
                               f"{r['accuracy_drop']*100:<12.2f}\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"✓ Summary saved to: {summary_path}")

    def print_summary_table(self):
        """Print a comprehensive summary table of all results"""
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)

        # Clean accuracy
        print(f"\nClean Accuracy: {self.results['clean']['accuracy'] * 100:.2f}%\n")

        # FGSM table
        print("-" * 80)
        print("FGSM ATTACK RESULTS")
        print("-" * 80)
        print(f"{'Epsilon':<12} {'ε (0-255)':<12} {'Clean Acc':<12} {'Adv Acc':<12} {'ASR':<12} {'Drop':<12}")
        print("-" * 80)
        for eps in sorted(self.results['fgsm'].keys()):
            r = self.results['fgsm'][eps]
            print(f"{eps:<12.4f} {eps*255:<12.2f} {r['clean_accuracy']*100:<12.2f} "
                  f"{r['adversarial_accuracy']*100:<12.2f} "
                  f"{r['attack_success_rate']*100:<12.2f} "
                  f"{r['accuracy_drop']*100:<12.2f}")

        # PGD table
        if self.results['pgd']:
            print("\n" + "-" * 80)
            print("PGD ATTACK RESULTS")
            print("-" * 80)
            print(f"{'Epsilon':<12} {'Iters':<8} {'Clean Acc':<12} {'Adv Acc':<12} {'ASR':<12} {'Drop':<12}")
            print("-" * 80)
            for eps in sorted(self.results['pgd'].keys()):
                for iters in sorted(self.results['pgd'][eps].keys()):
                    r = self.results['pgd'][eps][iters]
                    print(f"{eps:<12.4f} {iters:<8} {r['clean_accuracy']*100:<12.2f} "
                          f"{r['adversarial_accuracy']*100:<12.2f} "
                          f"{r['attack_success_rate']*100:<12.2f} "
                          f"{r['accuracy_drop']*100:<12.2f}")

        print("=" * 80)


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Comprehensive Attack Evaluation')
    parser.add_argument('--model-path', type=str,
                       default='./models/baseline_best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for evaluation')
    parser.add_argument('--num-batches', type=int, default=None,
                       help='Number of batches to evaluate (None = all)')
    parser.add_argument('--fgsm-epsilons', type=float, nargs='+',
                       default=[0.01, 0.03, 0.05, 0.1],
                       help='Epsilon values for FGSM')
    parser.add_argument('--pgd-epsilons', type=float, nargs='+',
                       default=[0.01, 0.03, 0.05, 0.1],
                       help='Epsilon values for PGD')
    parser.add_argument('--pgd-iterations', type=int, nargs='+',
                       default=[7, 20, 40],
                       help='Iteration counts for PGD')
    parser.add_argument('--skip-pgd', action='store_true',
                       help='Skip PGD evaluation (faster)')
    parser.add_argument('--skip-viz', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--results-dir', type=str,
                       default='./results/attack_evaluation',
                       help='Directory to save results')

    args = parser.parse_args()

    print("=" * 80)
    print("COMPREHENSIVE ADVERSARIAL ATTACK EVALUATION")
    print("=" * 80)
    print(f"\nModel: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"FGSM epsilons: {args.fgsm_epsilons}")
    if not args.skip_pgd:
        print(f"PGD epsilons: {args.pgd_epsilons}")
        print(f"PGD iterations: {args.pgd_iterations}")
    print(f"Results directory: {args.results_dir}")
    print()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load data
    print("Loading CIFAR-10 test data...")
    _, test_loader = get_cifar10_loaders(test_batch_size=args.batch_size)
    print(f"✓ Test loader ready (batch size: {args.batch_size})\n")

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = get_resnet18(num_classes=10, device=device)

    if os.path.exists(args.model_path):
        load_checkpoint(args.model_path, model, device=device)
        print("✓ Model loaded successfully\n")
    else:
        print("⚠ Warning: Model checkpoint not found. Using untrained model.")
        print("  Train a model first: python train_baseline.py\n")

    # Create evaluator
    evaluator = AttackEvaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        num_eval_batches=args.num_batches
    )

    # Run evaluations
    start_time = time.time()

    try:
        # 1. Clean accuracy
        evaluator.evaluate_clean_accuracy()

        # 2. FGSM evaluation
        evaluator.evaluate_fgsm(args.fgsm_epsilons)

        # 3. PGD evaluation (if not skipped)
        if not args.skip_pgd:
            evaluator.evaluate_pgd(args.pgd_epsilons, iterations_list=args.pgd_iterations)

        # 4. Generate visualizations (if not skipped)
        if not args.skip_viz:
            evaluator.generate_comparison_plots(save_dir=args.results_dir)
            evaluator.generate_visual_grid(
                epsilon_fgsm=0.03,
                epsilon_pgd=0.03,
                num_samples=8,
                save_dir=args.results_dir
            )

        # 5. Save results
        results_path = os.path.join(args.results_dir, 'attack_results.json')
        evaluator.save_results(save_path=results_path)

        # 6. Print summary
        evaluator.print_summary_table()

        # Final summary
        elapsed_time = time.time() - start_time
        print(f"\n" + "=" * 80)
        print("EVALUATION COMPLETE!")
        print("=" * 80)
        print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print(f"\nResults saved to: {args.results_dir}/")
        print(f"  - attack_results.json (machine-readable)")
        print(f"  - attack_results_summary.txt (human-readable)")
        if not args.skip_viz:
            print(f"  - attack_comparison.png (comparison plots)")
            print(f"  - attack_visual_grid.png (visual examples)")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"\n\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
