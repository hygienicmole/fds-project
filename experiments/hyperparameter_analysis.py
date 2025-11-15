"""
Comprehensive Hyperparameter Analysis for Adversarial Attacks

This script performs an in-depth analysis of how hyperparameters affect
the effectiveness of FGSM and PGD adversarial attacks.

Analysis includes:
- Epsilon range: 0.001 to 0.3 (15 values)
- PGD iterations: [5, 10, 20, 40]
- Statistical analysis (mean, std across batches)
- Detailed visualizations (ASR, accuracy, perturbation visibility)
- Comparison tables and reports

Usage:
    python experiments/hyperparameter_analysis.py --model-path ./models/baseline_best_model.pth
"""

import torch
import torch.nn as nn
import argparse
import os
import json
import time
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from collections import defaultdict

from attacks.fgsm import FGSM
from attacks.pgd import PGD
from models import get_resnet18, load_checkpoint
from utils import get_cifar10_loaders
import config


class HyperparameterAnalyzer:
    """
    Comprehensive hyperparameter analysis for adversarial attacks

    Analyzes the effect of epsilon and iteration count on attack
    effectiveness with detailed metrics and visualizations.
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
        num_eval_batches: int = None
    ):
        """
        Initialize hyperparameter analyzer

        Args:
            model: Model to analyze
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
            'epsilon_analysis': {
                'fgsm': {},
                'pgd': {}
            },
            'iteration_analysis': {},
            'statistical_summary': {},
            'metadata': {
                'device': str(device),
                'num_eval_batches': num_eval_batches,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }

    def analyze_epsilon_range(
        self,
        epsilon_values: List[float],
        attack_type: str = 'fgsm',
        pgd_iterations: int = 20,
        pgd_alpha_ratio: float = 0.25
    ) -> Dict:
        """
        Analyze attack effectiveness across epsilon range

        Args:
            epsilon_values: List of epsilon values to test
            attack_type: 'fgsm' or 'pgd'
            pgd_iterations: Number of PGD iterations (if attack_type='pgd')
            pgd_alpha_ratio: Alpha as fraction of epsilon (if attack_type='pgd')

        Returns:
            Dictionary with results for each epsilon
        """
        print(f"\n{'=' * 80}")
        print(f"EPSILON ANALYSIS: {attack_type.upper()}")
        print(f"{'=' * 80}")
        print(f"Testing {len(epsilon_values)} epsilon values: {epsilon_values}")
        if attack_type == 'pgd':
            print(f"PGD iterations: {pgd_iterations}")
            print(f"Alpha ratio: {pgd_alpha_ratio}")
        print()

        results = {}

        for epsilon in epsilon_values:
            print(f"\n--- Testing ε = {epsilon:.4f} ({epsilon*255:.2f}/255) ---")

            # Create attack
            if attack_type == 'fgsm':
                attack = FGSM(self.model, epsilon=epsilon, targeted=False)
            else:  # pgd
                alpha = epsilon * pgd_alpha_ratio
                attack = PGD(
                    self.model,
                    epsilon=epsilon,
                    alpha=alpha,
                    iterations=pgd_iterations,
                    random_start=True,
                    targeted=False
                )

            # Collect metrics for statistical analysis
            batch_metrics = []

            correct_clean_total = 0
            correct_adv_total = 0
            total_samples = 0

            perturbation_norms = {
                'l0': [],
                'l1': [],
                'l2': [],
                'linf': []
            }

            pbar = tqdm(self.test_loader, desc=f"{attack_type.upper()} ε={epsilon:.4f}")
            for batch_idx, (images, labels) in enumerate(pbar):
                if self.num_eval_batches and batch_idx >= self.num_eval_batches:
                    break

                images, labels = images.to(self.device), labels.to(self.device)
                batch_size = labels.size(0)

                # Clean predictions
                with torch.no_grad():
                    clean_outputs = self.model(images)
                    clean_preds = clean_outputs.argmax(dim=1)
                    correct_clean_batch = (clean_preds == labels).sum().item()

                # Generate adversarial examples
                adv_images = attack.generate(images, labels)

                # Adversarial predictions
                with torch.no_grad():
                    adv_outputs = self.model(adv_images)
                    adv_preds = adv_outputs.argmax(dim=1)
                    correct_adv_batch = (adv_preds == labels).sum().item()

                # Calculate batch metrics
                batch_clean_acc = correct_clean_batch / batch_size
                batch_adv_acc = correct_adv_batch / batch_size
                batch_asr = (correct_clean_batch - correct_adv_batch) / batch_size

                batch_metrics.append({
                    'clean_accuracy': batch_clean_acc,
                    'adversarial_accuracy': batch_adv_acc,
                    'attack_success_rate': batch_asr
                })

                # Accumulate totals
                correct_clean_total += correct_clean_batch
                correct_adv_total += correct_adv_batch
                total_samples += batch_size

                # Calculate perturbation norms
                perturbation = adv_images - images

                # L0 norm (number of changed pixels)
                l0 = (perturbation.abs() > 1e-6).float().sum(dim=[1,2,3])
                perturbation_norms['l0'].extend(l0.cpu().numpy())

                # L1 norm (sum of absolute values)
                l1 = perturbation.abs().sum(dim=[1,2,3])
                perturbation_norms['l1'].extend(l1.cpu().numpy())

                # L2 norm (Euclidean distance)
                l2 = perturbation.pow(2).sum(dim=[1,2,3]).sqrt()
                perturbation_norms['l2'].extend(l2.cpu().numpy())

                # L∞ norm (maximum absolute value)
                linf = perturbation.abs().view(batch_size, -1).max(dim=1)[0]
                perturbation_norms['linf'].extend(linf.cpu().numpy())

                # Update progress bar
                overall_clean_acc = 100.0 * correct_clean_total / total_samples
                overall_adv_acc = 100.0 * correct_adv_total / total_samples
                overall_asr = 100.0 * (correct_clean_total - correct_adv_total) / total_samples
                pbar.set_postfix({
                    'clean_acc': f'{overall_clean_acc:.1f}%',
                    'adv_acc': f'{overall_adv_acc:.1f}%',
                    'asr': f'{overall_asr:.1f}%'
                })

            # Calculate overall metrics
            clean_accuracy = correct_clean_total / total_samples
            adv_accuracy = correct_adv_total / total_samples
            attack_success_rate = (correct_clean_total - correct_adv_total) / total_samples
            accuracy_drop = clean_accuracy - adv_accuracy

            # Calculate statistical metrics across batches
            clean_acc_list = [m['clean_accuracy'] for m in batch_metrics]
            adv_acc_list = [m['adversarial_accuracy'] for m in batch_metrics]
            asr_list = [m['attack_success_rate'] for m in batch_metrics]

            # Store comprehensive results
            results[epsilon] = {
                'epsilon': epsilon,
                'clean_accuracy': clean_accuracy,
                'adversarial_accuracy': adv_accuracy,
                'attack_success_rate': attack_success_rate,
                'accuracy_drop': accuracy_drop,
                'total_samples': total_samples,

                # Statistical metrics
                'statistics': {
                    'clean_accuracy': {
                        'mean': np.mean(clean_acc_list),
                        'std': np.std(clean_acc_list),
                        'min': np.min(clean_acc_list),
                        'max': np.max(clean_acc_list)
                    },
                    'adversarial_accuracy': {
                        'mean': np.mean(adv_acc_list),
                        'std': np.std(adv_acc_list),
                        'min': np.min(adv_acc_list),
                        'max': np.max(adv_acc_list)
                    },
                    'attack_success_rate': {
                        'mean': np.mean(asr_list),
                        'std': np.std(asr_list),
                        'min': np.min(asr_list),
                        'max': np.max(asr_list)
                    }
                },

                # Perturbation norms
                'perturbation_norms': {
                    'l0': {
                        'mean': np.mean(perturbation_norms['l0']),
                        'std': np.std(perturbation_norms['l0']),
                        'median': np.median(perturbation_norms['l0'])
                    },
                    'l1': {
                        'mean': np.mean(perturbation_norms['l1']),
                        'std': np.std(perturbation_norms['l1']),
                        'median': np.median(perturbation_norms['l1'])
                    },
                    'l2': {
                        'mean': np.mean(perturbation_norms['l2']),
                        'std': np.std(perturbation_norms['l2']),
                        'median': np.median(perturbation_norms['l2'])
                    },
                    'linf': {
                        'mean': np.mean(perturbation_norms['linf']),
                        'std': np.std(perturbation_norms['linf']),
                        'median': np.median(perturbation_norms['linf'])
                    }
                }
            }

            # Print summary
            print(f"\n  Clean Accuracy:       {clean_accuracy * 100:.2f}% ± {np.std(clean_acc_list) * 100:.2f}%")
            print(f"  Adversarial Accuracy: {adv_accuracy * 100:.2f}% ± {np.std(adv_acc_list) * 100:.2f}%")
            print(f"  Attack Success Rate:  {attack_success_rate * 100:.2f}% ± {np.std(asr_list) * 100:.2f}%")
            print(f"  Accuracy Drop:        {accuracy_drop * 100:.2f}%")
            print(f"  L∞ Norm (mean):       {results[epsilon]['perturbation_norms']['linf']['mean']:.6f}")
            print(f"  L2 Norm (mean):       {results[epsilon]['perturbation_norms']['l2']['mean']:.6f}")

        return results

    def analyze_pgd_iterations(
        self,
        epsilon: float,
        iteration_values: List[int],
        alpha_ratio: float = 0.25
    ) -> Dict:
        """
        Analyze effect of PGD iteration count

        Args:
            epsilon: Fixed epsilon value to use
            iteration_values: List of iteration counts to test
            alpha_ratio: Alpha as fraction of epsilon

        Returns:
            Dictionary with results for each iteration count
        """
        print(f"\n{'=' * 80}")
        print(f"ITERATION ANALYSIS: PGD at ε={epsilon:.4f}")
        print(f"{'=' * 80}")
        print(f"Testing iteration counts: {iteration_values}")
        print(f"Alpha ratio: {alpha_ratio} (α = {epsilon * alpha_ratio:.4f})")
        print()

        results = {}
        alpha = epsilon * alpha_ratio

        for iterations in iteration_values:
            print(f"\n--- PGD with {iterations} iterations ---")

            # Create PGD attack
            pgd = PGD(
                self.model,
                epsilon=epsilon,
                alpha=alpha,
                iterations=iterations,
                random_start=True,
                targeted=False
            )

            # Collect metrics
            batch_metrics = []
            correct_clean_total = 0
            correct_adv_total = 0
            total_samples = 0

            pbar = tqdm(self.test_loader, desc=f"PGD iters={iterations}")
            for batch_idx, (images, labels) in enumerate(pbar):
                if self.num_eval_batches and batch_idx >= self.num_eval_batches:
                    break

                images, labels = images.to(self.device), labels.to(self.device)
                batch_size = labels.size(0)

                # Clean predictions
                with torch.no_grad():
                    clean_outputs = self.model(images)
                    clean_preds = clean_outputs.argmax(dim=1)
                    correct_clean_batch = (clean_preds == labels).sum().item()

                # Generate adversarial examples
                adv_images = pgd.generate(images, labels)

                # Adversarial predictions
                with torch.no_grad():
                    adv_outputs = self.model(adv_images)
                    adv_preds = adv_outputs.argmax(dim=1)
                    correct_adv_batch = (adv_preds == labels).sum().item()

                # Batch metrics
                batch_clean_acc = correct_clean_batch / batch_size
                batch_adv_acc = correct_adv_batch / batch_size
                batch_asr = (correct_clean_batch - correct_adv_batch) / batch_size

                batch_metrics.append({
                    'clean_accuracy': batch_clean_acc,
                    'adversarial_accuracy': batch_adv_acc,
                    'attack_success_rate': batch_asr
                })

                # Accumulate totals
                correct_clean_total += correct_clean_batch
                correct_adv_total += correct_adv_batch
                total_samples += batch_size

                # Update progress bar
                overall_adv_acc = 100.0 * correct_adv_total / total_samples
                overall_asr = 100.0 * (correct_clean_total - correct_adv_total) / total_samples
                pbar.set_postfix({
                    'adv_acc': f'{overall_adv_acc:.1f}%',
                    'asr': f'{overall_asr:.1f}%'
                })

            # Calculate metrics
            clean_accuracy = correct_clean_total / total_samples
            adv_accuracy = correct_adv_total / total_samples
            attack_success_rate = (correct_clean_total - correct_adv_total) / total_samples

            adv_acc_list = [m['adversarial_accuracy'] for m in batch_metrics]
            asr_list = [m['attack_success_rate'] for m in batch_metrics]

            # Store results
            results[iterations] = {
                'iterations': iterations,
                'epsilon': epsilon,
                'alpha': alpha,
                'clean_accuracy': clean_accuracy,
                'adversarial_accuracy': adv_accuracy,
                'attack_success_rate': attack_success_rate,
                'total_samples': total_samples,
                'statistics': {
                    'adversarial_accuracy': {
                        'mean': np.mean(adv_acc_list),
                        'std': np.std(adv_acc_list)
                    },
                    'attack_success_rate': {
                        'mean': np.mean(asr_list),
                        'std': np.std(asr_list)
                    }
                }
            }

            print(f"\n  Adversarial Accuracy: {adv_accuracy * 100:.2f}% ± {np.std(adv_acc_list) * 100:.2f}%")
            print(f"  Attack Success Rate:  {attack_success_rate * 100:.2f}% ± {np.std(asr_list) * 100:.2f}%")

        return results

    def generate_comprehensive_plots(
        self,
        save_dir: str = './results/hyperparameter_analysis'
    ):
        """
        Generate comprehensive visualization plots

        Creates multiple detailed plots analyzing hyperparameter effects

        Args:
            save_dir: Directory to save plots
        """
        print(f"\n{'=' * 80}")
        print("GENERATING COMPREHENSIVE PLOTS")
        print(f"{'=' * 80}")

        os.makedirs(save_dir, exist_ok=True)

        # Plot 1: Epsilon Analysis (ASR and Accuracy vs Epsilon)
        self._plot_epsilon_analysis(save_dir)

        # Plot 2: Iteration Analysis
        if self.results['iteration_analysis']:
            self._plot_iteration_analysis(save_dir)

        # Plot 3: Perturbation Visibility Analysis
        self._plot_perturbation_analysis(save_dir)

        # Plot 4: Statistical Comparison
        self._plot_statistical_comparison(save_dir)

        print(f"\n✓ All plots saved to: {save_dir}/")

    def _plot_epsilon_analysis(self, save_dir: str):
        """Plot ASR and Accuracy vs Epsilon"""
        print("\n1. Generating epsilon analysis plots...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Extract FGSM data
        fgsm_epsilons = sorted(self.results['epsilon_analysis']['fgsm'].keys())
        fgsm_clean_acc = [self.results['epsilon_analysis']['fgsm'][e]['clean_accuracy'] * 100
                          for e in fgsm_epsilons]
        fgsm_adv_acc = [self.results['epsilon_analysis']['fgsm'][e]['adversarial_accuracy'] * 100
                        for e in fgsm_epsilons]
        fgsm_asr = [self.results['epsilon_analysis']['fgsm'][e]['attack_success_rate'] * 100
                    for e in fgsm_epsilons]
        fgsm_adv_std = [self.results['epsilon_analysis']['fgsm'][e]['statistics']['adversarial_accuracy']['std'] * 100
                        for e in fgsm_epsilons]

        # Extract PGD data (if available)
        if self.results['epsilon_analysis']['pgd']:
            pgd_epsilons = sorted(self.results['epsilon_analysis']['pgd'].keys())
            pgd_adv_acc = [self.results['epsilon_analysis']['pgd'][e]['adversarial_accuracy'] * 100
                          for e in pgd_epsilons]
            pgd_asr = [self.results['epsilon_analysis']['pgd'][e]['attack_success_rate'] * 100
                      for e in pgd_epsilons]
            pgd_adv_std = [self.results['epsilon_analysis']['pgd'][e]['statistics']['adversarial_accuracy']['std'] * 100
                          for e in pgd_epsilons]

        # Plot 1: Accuracy vs Epsilon
        axes[0, 0].plot(fgsm_epsilons, fgsm_clean_acc, 'o-', linewidth=2, markersize=6,
                       label='Clean', color='green', alpha=0.7)
        axes[0, 0].plot(fgsm_epsilons, fgsm_adv_acc, 's-', linewidth=2, markersize=6,
                       label='FGSM', color='red', alpha=0.7)
        axes[0, 0].fill_between(fgsm_epsilons,
                                np.array(fgsm_adv_acc) - np.array(fgsm_adv_std),
                                np.array(fgsm_adv_acc) + np.array(fgsm_adv_std),
                                color='red', alpha=0.2)
        if self.results['epsilon_analysis']['pgd']:
            axes[0, 0].plot(pgd_epsilons, pgd_adv_acc, '^-', linewidth=2, markersize=6,
                           label='PGD-20', color='purple', alpha=0.7)
            axes[0, 0].fill_between(pgd_epsilons,
                                    np.array(pgd_adv_acc) - np.array(pgd_adv_std),
                                    np.array(pgd_adv_acc) + np.array(pgd_adv_std),
                                    color='purple', alpha=0.2)
        axes[0, 0].set_xlabel('Epsilon (ε)', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('Model Accuracy vs Attack Strength (ε)', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 105])

        # Plot 2: ASR vs Epsilon
        axes[0, 1].plot(fgsm_epsilons, fgsm_asr, 's-', linewidth=2, markersize=6,
                       label='FGSM', color='red', alpha=0.7)
        if self.results['epsilon_analysis']['pgd']:
            axes[0, 1].plot(pgd_epsilons, pgd_asr, '^-', linewidth=2, markersize=6,
                           label='PGD-20', color='purple', alpha=0.7)
        axes[0, 1].set_xlabel('Epsilon (ε)', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Attack Success Rate vs Epsilon', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 105])

        # Plot 3: Accuracy Drop vs Epsilon
        fgsm_drop = [self.results['epsilon_analysis']['fgsm'][e]['accuracy_drop'] * 100
                     for e in fgsm_epsilons]
        axes[1, 0].plot(fgsm_epsilons, fgsm_drop, 's-', linewidth=2, markersize=6,
                       label='FGSM', color='red', alpha=0.7)
        if self.results['epsilon_analysis']['pgd']:
            pgd_drop = [self.results['epsilon_analysis']['pgd'][e]['accuracy_drop'] * 100
                       for e in pgd_epsilons]
            axes[1, 0].plot(pgd_epsilons, pgd_drop, '^-', linewidth=2, markersize=6,
                           label='PGD-20', color='purple', alpha=0.7)
        axes[1, 0].set_xlabel('Epsilon (ε)', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Accuracy Drop (%)', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('Accuracy Drop vs Epsilon', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 105])

        # Plot 4: Log scale version for fine details
        axes[1, 1].semilogx(fgsm_epsilons, fgsm_adv_acc, 's-', linewidth=2, markersize=6,
                           label='FGSM', color='red', alpha=0.7)
        if self.results['epsilon_analysis']['pgd']:
            axes[1, 1].semilogx(pgd_epsilons, pgd_adv_acc, '^-', linewidth=2, markersize=6,
                               label='PGD-20', color='purple', alpha=0.7)
        axes[1, 1].set_xlabel('Epsilon (ε) - Log Scale', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Adversarial Accuracy (%)', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Adversarial Accuracy vs Epsilon (Log Scale)', fontsize=14, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3, which='both')
        axes[1, 1].set_ylim([0, 105])

        plt.tight_layout()
        save_path = os.path.join(save_dir, 'epsilon_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: {save_path}")
        plt.close()

    def _plot_iteration_analysis(self, save_dir: str):
        """Plot PGD iteration effect"""
        print("\n2. Generating iteration analysis plot...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        iterations = sorted(self.results['iteration_analysis'].keys())
        adv_acc = [self.results['iteration_analysis'][i]['adversarial_accuracy'] * 100
                  for i in iterations]
        asr = [self.results['iteration_analysis'][i]['attack_success_rate'] * 100
              for i in iterations]
        adv_std = [self.results['iteration_analysis'][i]['statistics']['adversarial_accuracy']['std'] * 100
                  for i in iterations]

        # Plot 1: Adversarial Accuracy vs Iterations
        axes[0].plot(iterations, adv_acc, 'o-', linewidth=2, markersize=8,
                    color='blue', alpha=0.7)
        axes[0].fill_between(iterations,
                            np.array(adv_acc) - np.array(adv_std),
                            np.array(adv_acc) + np.array(adv_std),
                            color='blue', alpha=0.2)
        axes[0].set_xlabel('PGD Iterations', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Adversarial Accuracy (%)', fontsize=12, fontweight='bold')
        epsilon = self.results['iteration_analysis'][iterations[0]]['epsilon']
        axes[0].set_title(f'PGD Adversarial Accuracy vs Iterations (ε={epsilon:.3f})',
                         fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, max(adv_acc) * 1.2])

        # Plot 2: ASR vs Iterations
        axes[1].plot(iterations, asr, 's-', linewidth=2, markersize=8,
                    color='orange', alpha=0.7)
        axes[1].set_xlabel('PGD Iterations', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
        axes[1].set_title(f'PGD Attack Success Rate vs Iterations (ε={epsilon:.3f})',
                         fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([min(asr) * 0.9, 105])

        plt.tight_layout()
        save_path = os.path.join(save_dir, 'iteration_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: {save_path}")
        plt.close()

    def _plot_perturbation_analysis(self, save_dir: str):
        """Plot perturbation visibility analysis"""
        print("\n3. Generating perturbation analysis plots...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Extract FGSM perturbation data
        fgsm_epsilons = sorted(self.results['epsilon_analysis']['fgsm'].keys())

        # L∞ norms
        fgsm_linf = [self.results['epsilon_analysis']['fgsm'][e]['perturbation_norms']['linf']['mean']
                    for e in fgsm_epsilons]

        # L2 norms
        fgsm_l2 = [self.results['epsilon_analysis']['fgsm'][e]['perturbation_norms']['l2']['mean']
                  for e in fgsm_epsilons]

        # L1 norms
        fgsm_l1 = [self.results['epsilon_analysis']['fgsm'][e]['perturbation_norms']['l1']['mean']
                  for e in fgsm_epsilons]

        # L0 norms (pixels changed)
        fgsm_l0 = [self.results['epsilon_analysis']['fgsm'][e]['perturbation_norms']['l0']['mean']
                  for e in fgsm_epsilons]

        # Plot 1: L∞ norm vs Epsilon
        axes[0, 0].plot(fgsm_epsilons, fgsm_linf, 'o-', linewidth=2, markersize=6,
                       color='purple', label='Actual L∞')
        axes[0, 0].plot(fgsm_epsilons, fgsm_epsilons, '--', linewidth=2,
                       color='gray', alpha=0.5, label='Theoretical Max (ε)')
        axes[0, 0].set_xlabel('Epsilon (ε)', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('L∞ Norm', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('L∞ Perturbation Norm vs Epsilon', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: L2 norm vs Epsilon
        axes[0, 1].plot(fgsm_epsilons, fgsm_l2, 's-', linewidth=2, markersize=6,
                       color='blue')
        axes[0, 1].set_xlabel('Epsilon (ε)', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('L2 Norm', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('L2 Perturbation Norm vs Epsilon', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: L1 norm vs Epsilon
        axes[1, 0].plot(fgsm_epsilons, fgsm_l1, '^-', linewidth=2, markersize=6,
                       color='green')
        axes[1, 0].set_xlabel('Epsilon (ε)', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('L1 Norm', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('L1 Perturbation Norm vs Epsilon', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: L0 norm (pixels changed) vs Epsilon
        axes[1, 1].plot(fgsm_epsilons, fgsm_l0, 'D-', linewidth=2, markersize=6,
                       color='red')
        axes[1, 1].set_xlabel('Epsilon (ε)', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('L0 Norm (Pixels Changed)', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Number of Changed Pixels vs Epsilon', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(save_dir, 'perturbation_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: {save_path}")
        plt.close()

    def _plot_statistical_comparison(self, save_dir: str):
        """Plot statistical comparison with error bars"""
        print("\n4. Generating statistical comparison plot...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Extract data with statistics
        fgsm_epsilons = sorted(self.results['epsilon_analysis']['fgsm'].keys())
        fgsm_adv_mean = [self.results['epsilon_analysis']['fgsm'][e]['adversarial_accuracy'] * 100
                        for e in fgsm_epsilons]
        fgsm_adv_std = [self.results['epsilon_analysis']['fgsm'][e]['statistics']['adversarial_accuracy']['std'] * 100
                       for e in fgsm_epsilons]

        # Plot 1: Accuracy with error bars
        axes[0].errorbar(fgsm_epsilons, fgsm_adv_mean, yerr=fgsm_adv_std,
                        fmt='o-', linewidth=2, markersize=6, capsize=5,
                        label='FGSM', color='red', alpha=0.7)

        if self.results['epsilon_analysis']['pgd']:
            pgd_epsilons = sorted(self.results['epsilon_analysis']['pgd'].keys())
            pgd_adv_mean = [self.results['epsilon_analysis']['pgd'][e]['adversarial_accuracy'] * 100
                          for e in pgd_epsilons]
            pgd_adv_std = [self.results['epsilon_analysis']['pgd'][e]['statistics']['adversarial_accuracy']['std'] * 100
                          for e in pgd_epsilons]
            axes[0].errorbar(pgd_epsilons, pgd_adv_mean, yerr=pgd_adv_std,
                           fmt='^-', linewidth=2, markersize=6, capsize=5,
                           label='PGD-20', color='purple', alpha=0.7)

        axes[0].set_xlabel('Epsilon (ε)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Adversarial Accuracy (%)', fontsize=12, fontweight='bold')
        axes[0].set_title('Adversarial Accuracy with Standard Deviation', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 105])

        # Plot 2: Variance analysis
        axes[1].plot(fgsm_epsilons, fgsm_adv_std, 's-', linewidth=2, markersize=6,
                    label='FGSM Std Dev', color='red', alpha=0.7)
        if self.results['epsilon_analysis']['pgd']:
            axes[1].plot(pgd_epsilons, pgd_adv_std, '^-', linewidth=2, markersize=6,
                        label='PGD-20 Std Dev', color='purple', alpha=0.7)

        axes[1].set_xlabel('Epsilon (ε)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Standard Deviation (%)', fontsize=12, fontweight='bold')
        axes[1].set_title('Attack Variance Across Batches', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(save_dir, 'statistical_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: {save_path}")
        plt.close()

    def generate_comparison_tables(self, save_dir: str = './results/hyperparameter_analysis'):
        """Generate detailed comparison tables"""
        print(f"\n{'=' * 80}")
        print("GENERATING COMPARISON TABLES")
        print(f"{'=' * 80}")

        os.makedirs(save_dir, exist_ok=True)

        # Table 1: FGSM Epsilon Comparison
        self._generate_fgsm_table(save_dir)

        # Table 2: PGD Epsilon Comparison
        if self.results['epsilon_analysis']['pgd']:
            self._generate_pgd_epsilon_table(save_dir)

        # Table 3: PGD Iteration Comparison
        if self.results['iteration_analysis']:
            self._generate_iteration_table(save_dir)

        # Table 4: Perturbation Norms Table
        self._generate_perturbation_table(save_dir)

        print(f"\n✓ All tables saved to: {save_dir}/")

    def _generate_fgsm_table(self, save_dir: str):
        """Generate FGSM comparison table"""
        print("\n1. Generating FGSM epsilon comparison table...")

        table_path = os.path.join(save_dir, 'fgsm_epsilon_table.txt')
        with open(table_path, 'w') as f:
            f.write("=" * 120 + "\n")
            f.write("FGSM EPSILON ANALYSIS - COMPREHENSIVE RESULTS\n")
            f.write("=" * 120 + "\n\n")

            f.write(f"{'Epsilon':<12} {'ε(0-255)':<10} {'Clean Acc':<12} {'Adv Acc':<12} {'ASR':<10} "
                   f"{'Drop':<10} {'L∞':<12} {'L2':<12} {'L0':<12}\n")
            f.write("-" * 120 + "\n")

            for eps in sorted(self.results['epsilon_analysis']['fgsm'].keys()):
                r = self.results['epsilon_analysis']['fgsm'][eps]
                adv_std = r['statistics']['adversarial_accuracy']['std']
                asr_std = r['statistics']['attack_success_rate']['std']

                f.write(f"{eps:<12.4f} {eps*255:<10.2f} "
                       f"{r['clean_accuracy']*100:<12.2f} "
                       f"{r['adversarial_accuracy']*100:5.2f}±{adv_std*100:4.2f} "
                       f"{r['attack_success_rate']*100:5.2f}%   "
                       f"{r['accuracy_drop']*100:<10.2f} "
                       f"{r['perturbation_norms']['linf']['mean']:<12.6f} "
                       f"{r['perturbation_norms']['l2']['mean']:<12.4f} "
                       f"{r['perturbation_norms']['l0']['mean']:<12.0f}\n")

            f.write("=" * 120 + "\n")

        print(f"   ✓ Saved: {table_path}")

    def _generate_pgd_epsilon_table(self, save_dir: str):
        """Generate PGD epsilon comparison table"""
        print("\n2. Generating PGD epsilon comparison table...")

        table_path = os.path.join(save_dir, 'pgd_epsilon_table.txt')
        with open(table_path, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("PGD EPSILON ANALYSIS - COMPREHENSIVE RESULTS\n")
            f.write("=" * 100 + "\n\n")

            f.write(f"{'Epsilon':<12} {'ε(0-255)':<10} {'Adv Acc':<12} {'ASR':<10} "
                   f"{'Drop':<10} {'L∞':<12} {'L2':<12}\n")
            f.write("-" * 100 + "\n")

            for eps in sorted(self.results['epsilon_analysis']['pgd'].keys()):
                r = self.results['epsilon_analysis']['pgd'][eps]
                adv_std = r['statistics']['adversarial_accuracy']['std']

                f.write(f"{eps:<12.4f} {eps*255:<10.2f} "
                       f"{r['adversarial_accuracy']*100:5.2f}±{adv_std*100:4.2f} "
                       f"{r['attack_success_rate']*100:5.2f}%   "
                       f"{r['accuracy_drop']*100:<10.2f} "
                       f"{r['perturbation_norms']['linf']['mean']:<12.6f} "
                       f"{r['perturbation_norms']['l2']['mean']:<12.4f}\n")

            f.write("=" * 100 + "\n")

        print(f"   ✓ Saved: {table_path}")

    def _generate_iteration_table(self, save_dir: str):
        """Generate PGD iteration comparison table"""
        print("\n3. Generating PGD iteration comparison table...")

        table_path = os.path.join(save_dir, 'pgd_iteration_table.txt')
        epsilon = list(self.results['iteration_analysis'].values())[0]['epsilon']

        with open(table_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"PGD ITERATION ANALYSIS (ε={epsilon:.4f})\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"{'Iterations':<12} {'Adv Acc':<15} {'ASR':<15} {'Improvement':<15}\n")
            f.write("-" * 80 + "\n")

            prev_asr = 0
            for iters in sorted(self.results['iteration_analysis'].keys()):
                r = self.results['iteration_analysis'][iters]
                adv_std = r['statistics']['adversarial_accuracy']['std']
                asr = r['attack_success_rate'] * 100
                improvement = asr - prev_asr if prev_asr > 0 else 0

                f.write(f"{iters:<12} "
                       f"{r['adversarial_accuracy']*100:5.2f}±{adv_std*100:4.2f}%   "
                       f"{asr:5.2f}%         "
                       f"+{improvement:5.2f}%\n")

                prev_asr = asr

            f.write("=" * 80 + "\n")

        print(f"   ✓ Saved: {table_path}")

    def _generate_perturbation_table(self, save_dir: str):
        """Generate perturbation norms table"""
        print("\n4. Generating perturbation norms table...")

        table_path = os.path.join(save_dir, 'perturbation_norms_table.txt')
        with open(table_path, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("PERTURBATION NORMS ANALYSIS - FGSM\n")
            f.write("=" * 100 + "\n\n")

            f.write(f"{'Epsilon':<12} {'L0 (mean±std)':<20} {'L1 (mean±std)':<20} "
                   f"{'L2 (mean±std)':<20} {'L∞ (mean±std)':<20}\n")
            f.write("-" * 100 + "\n")

            for eps in sorted(self.results['epsilon_analysis']['fgsm'].keys()):
                r = self.results['epsilon_analysis']['fgsm'][eps]
                p = r['perturbation_norms']

                f.write(f"{eps:<12.4f} "
                       f"{p['l0']['mean']:8.0f}±{p['l0']['std']:6.0f}     "
                       f"{p['l1']['mean']:8.4f}±{p['l1']['std']:6.4f}   "
                       f"{p['l2']['mean']:8.4f}±{p['l2']['std']:6.4f}   "
                       f"{p['linf']['mean']:8.6f}±{p['linf']['std']:6.6f}\n")

            f.write("=" * 100 + "\n")

        print(f"   ✓ Saved: {table_path}")

    def save_results(self, save_path: str = './results/hyperparameter_analysis/results.json'):
        """Save all results to JSON"""
        print(f"\n{'=' * 80}")
        print("SAVING RESULTS")
        print(f"{'=' * 80}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Convert to serializable format
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        serializable_results = convert_to_serializable(self.results)

        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"✓ Results saved to: {save_path}")


def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(description='Comprehensive Hyperparameter Analysis')
    parser.add_argument('--model-path', type=str,
                       default='./models/baseline_best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for evaluation')
    parser.add_argument('--num-batches', type=int, default=None,
                       help='Number of batches to evaluate (None = all)')
    parser.add_argument('--epsilon-min', type=float, default=0.001,
                       help='Minimum epsilon value')
    parser.add_argument('--epsilon-max', type=float, default=0.3,
                       help='Maximum epsilon value')
    parser.add_argument('--num-epsilons', type=int, default=15,
                       help='Number of epsilon values to test')
    parser.add_argument('--pgd-iterations', type=int, nargs='+',
                       default=[5, 10, 20, 40],
                       help='PGD iteration counts to test')
    parser.add_argument('--skip-pgd-epsilon', action='store_true',
                       help='Skip PGD epsilon analysis (faster)')
    parser.add_argument('--skip-iteration-analysis', action='store_true',
                       help='Skip iteration analysis')
    parser.add_argument('--results-dir', type=str,
                       default='./results/hyperparameter_analysis',
                       help='Directory to save results')

    args = parser.parse_args()

    print("=" * 80)
    print("COMPREHENSIVE HYPERPARAMETER ANALYSIS")
    print("=" * 80)
    print(f"\nModel: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epsilon range: {args.epsilon_min} to {args.epsilon_max} ({args.num_epsilons} values)")
    if not args.skip_iteration_analysis:
        print(f"PGD iterations: {args.pgd_iterations}")
    print(f"Results directory: {args.results_dir}")
    print()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load data
    print("Loading CIFAR-10 test data...")
    _, test_loader = get_cifar10_loaders(test_batch_size=args.batch_size)
    print(f"✓ Test loader ready\n")

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = get_resnet18(num_classes=10, device=device)

    if os.path.exists(args.model_path):
        load_checkpoint(args.model_path, model, device=device)
        print("✓ Model loaded successfully\n")
    else:
        print("⚠ Warning: Model checkpoint not found. Using untrained model.\n")

    # Create analyzer
    analyzer = HyperparameterAnalyzer(
        model=model,
        test_loader=test_loader,
        device=device,
        num_eval_batches=args.num_batches
    )

    # Generate epsilon values (logarithmic spacing for better coverage)
    epsilon_values = np.logspace(
        np.log10(args.epsilon_min),
        np.log10(args.epsilon_max),
        args.num_epsilons
    ).tolist()

    start_time = time.time()

    try:
        # Analysis 1: FGSM epsilon analysis
        print("\nStarting FGSM epsilon analysis...")
        analyzer.results['epsilon_analysis']['fgsm'] = analyzer.analyze_epsilon_range(
            epsilon_values=epsilon_values,
            attack_type='fgsm'
        )

        # Analysis 2: PGD epsilon analysis (if not skipped)
        if not args.skip_pgd_epsilon:
            print("\nStarting PGD epsilon analysis...")
            analyzer.results['epsilon_analysis']['pgd'] = analyzer.analyze_epsilon_range(
                epsilon_values=epsilon_values,
                attack_type='pgd',
                pgd_iterations=20  # Use 20 iterations for epsilon analysis
            )

        # Analysis 3: PGD iteration analysis (if not skipped)
        if not args.skip_iteration_analysis:
            print("\nStarting PGD iteration analysis...")
            # Use middle epsilon value for iteration analysis
            middle_epsilon = epsilon_values[len(epsilon_values) // 2]
            analyzer.results['iteration_analysis'] = analyzer.analyze_pgd_iterations(
                epsilon=middle_epsilon,
                iteration_values=args.pgd_iterations
            )

        # Generate visualizations
        analyzer.generate_comprehensive_plots(save_dir=args.results_dir)

        # Generate comparison tables
        analyzer.generate_comparison_tables(save_dir=args.results_dir)

        # Save results
        results_path = os.path.join(args.results_dir, 'results.json')
        analyzer.save_results(save_path=results_path)

        # Final summary
        elapsed_time = time.time() - start_time
        print(f"\n{'=' * 80}")
        print("ANALYSIS COMPLETE!")
        print(f"{'=' * 80}")
        print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print(f"\nResults saved to: {args.results_dir}/")
        print("  - results.json (complete data)")
        print("  - epsilon_analysis.png (4-panel epsilon plot)")
        if not args.skip_iteration_analysis:
            print("  - iteration_analysis.png (iteration effect)")
        print("  - perturbation_analysis.png (norm analysis)")
        print("  - statistical_comparison.png (with error bars)")
        print("  - *_table.txt (detailed tables)")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\n\nError during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
