#!/usr/bin/env python3
"""
Automated Test Suite for Adversarial ML Project
Tests model loading, attack generation, API endpoints, and key functionality

Usage:
    python demo/test_all.py              # Run all tests
    python demo/test_all.py --quick      # Skip slow tests
    python demo/test_all.py --verbose    # Detailed output
"""

import os
import sys
import argparse
import time
import requests
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import io
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class TestRunner:
    """Comprehensive test suite for the adversarial ML project"""

    def __init__(self, verbose=False, quick=False):
        self.verbose = verbose
        self.quick = quick
        self.tests_passed = 0
        self.tests_failed = 0
        self.api_base_url = "http://localhost:8000"

    def print_header(self, text: str):
        """Print section header"""
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{text:^70}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")

    def print_test(self, name: str, passed: bool, message: str = ""):
        """Print test result"""
        if passed:
            status = f"{Colors.OKGREEN}[OK] PASS{Colors.ENDC}"
            self.tests_passed += 1
        else:
            status = f"{Colors.FAIL}[X] FAIL{Colors.ENDC}"
            self.tests_failed += 1

        print(f"{status} | {name}")
        if message and (not passed or self.verbose):
            print(f"       {message}")

    def print_summary(self):
        """Print final test summary"""
        total = self.tests_passed + self.tests_failed
        pass_rate = (self.tests_passed / total * 100) if total > 0 else 0

        print(f"\n{Colors.BOLD}{'='*70}{Colors.ENDC}")
        print(f"{Colors.BOLD}Test Summary{Colors.ENDC}")
        print(f"{Colors.BOLD}{'='*70}{Colors.ENDC}")
        print(f"Total Tests: {total}")
        print(f"{Colors.OKGREEN}Passed: {self.tests_passed}{Colors.ENDC}")
        print(f"{Colors.FAIL}Failed: {self.tests_failed}{Colors.ENDC}")
        print(f"Pass Rate: {pass_rate:.1f}%")

        if self.tests_failed == 0:
            print(f"\n{Colors.OKGREEN}{Colors.BOLD}🎉 All tests passed!{Colors.ENDC}")
        else:
            print(f"\n{Colors.WARNING}{Colors.BOLD}[WARNING]️  Some tests failed{Colors.ENDC}")
        print(f"{Colors.BOLD}{'='*70}{Colors.ENDC}\n")

    # =========================================================================
    # Model Tests
    # =========================================================================

    def test_model_loading(self):
        """Test if model checkpoint can be loaded"""
        self.print_header("Model Loading Tests")

        try:
            from models.resnet import ResNet18

            # Test model initialization
            model = ResNet18(num_classes=10)
            self.print_test(
                "Model initialization",
                True,
                "ResNet18 initialized successfully"
            )

            # Test checkpoint loading
            checkpoint_path = "checkpoints/best_model.pth"
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
                self.print_test(
                    "Checkpoint loading",
                    True,
                    f"Loaded checkpoint from {checkpoint_path}"
                )

                # Verify checkpoint contents
                has_epoch = 'epoch' in checkpoint
                has_accuracy = 'test_accuracy' in checkpoint
                self.print_test(
                    "Checkpoint metadata",
                    has_epoch and has_accuracy,
                    f"Epoch: {checkpoint.get('epoch', 'N/A')}, "
                    f"Accuracy: {checkpoint.get('test_accuracy', 'N/A'):.2f}%"
                )
            else:
                self.print_test(
                    "Checkpoint loading",
                    False,
                    f"Checkpoint not found: {checkpoint_path}"
                )
        except Exception as e:
            self.print_test("Model loading", False, f"Error: {str(e)}")

    def test_model_inference(self):
        """Test model forward pass"""
        try:
            from models.resnet import ResNet18

            model = ResNet18(num_classes=10)
            model.eval()

            # Test with dummy input
            dummy_input = torch.randn(1, 3, 32, 32)
            with torch.no_grad():
                output = model(dummy_input)

            # Verify output shape
            shape_correct = output.shape == (1, 10)
            self.print_test(
                "Model forward pass",
                shape_correct,
                f"Output shape: {output.shape}"
            )

            # Verify output is probability distribution
            probs = F.softmax(output, dim=1)
            sum_correct = torch.allclose(probs.sum(), torch.tensor(1.0))
            self.print_test(
                "Output probability distribution",
                sum_correct,
                f"Sum of probabilities: {probs.sum().item():.6f}"
            )

        except Exception as e:
            self.print_test("Model inference", False, f"Error: {str(e)}")

    # =========================================================================
    # Attack Tests
    # =========================================================================

    def test_attack_implementations(self):
        """Test FGSM and PGD attack implementations"""
        self.print_header("Attack Implementation Tests")

        try:
            from models.resnet import ResNet18
            from attacks.fgsm import FGSM
            from attacks.pgd import PGD
            import torch.nn as nn

            # Setup
            model = ResNet18(num_classes=10)
            model.eval()

            # Test data
            test_image = torch.randn(1, 3, 32, 32)
            test_label = torch.tensor([3])

            # Test FGSM
            try:
                fgsm = FGSM(model=model, epsilon=0.03)
                adv_image_fgsm = fgsm.generate(test_image, test_label)

                # Verify perturbation is bounded
                perturbation = (adv_image_fgsm - test_image).abs()
                max_pert = perturbation.max().item()
                bounded = max_pert <= 0.03 + 1e-6

                self.print_test(
                    "FGSM attack generation",
                    True,
                    f"Max perturbation: {max_pert:.6f}"
                )
                self.print_test(
                    "FGSM epsilon constraint",
                    bounded,
                    f"Max perturbation <= epsilon: {bounded}"
                )
            except Exception as e:
                self.print_test("FGSM attack", False, f"Error: {str(e)}")

            # Test PGD
            try:
                pgd = PGD(model=model, epsilon=0.03, alpha=0.01, iterations=20)
                adv_image_pgd = pgd.generate(test_image, test_label)

                # Verify perturbation is bounded
                perturbation = (adv_image_pgd - test_image).abs()
                max_pert = perturbation.max().item()
                bounded = max_pert <= 0.03 + 1e-6

                self.print_test(
                    "PGD attack generation",
                    True,
                    f"Max perturbation: {max_pert:.6f}"
                )
                self.print_test(
                    "PGD epsilon constraint",
                    bounded,
                    f"Max perturbation <= epsilon: {bounded}"
                )

                # Verify images are clamped to [0,1]
                in_range = (adv_image_pgd >= 0).all() and (adv_image_pgd <= 1).all()
                self.print_test(
                    "PGD pixel range constraint",
                    in_range,
                    f"All pixels in [0, 1]: {in_range}"
                )

            except Exception as e:
                self.print_test("PGD attack", False, f"Error: {str(e)}")

        except Exception as e:
            self.print_test("Attack implementations", False, f"Error: {str(e)}")

    def test_attack_effectiveness(self):
        """Test if attacks successfully fool the model"""
        if self.quick:
            print("Skipping effectiveness test (--quick mode)")
            return

        try:
            from models.resnet import ResNet18
            from attacks.fgsm import FGSM
            from attacks.pgd import PGD
            import torchvision
            import torchvision.transforms as transforms

            # Load a pre-trained model
            model = ResNet18(num_classes=10)
            checkpoint_path = "checkpoints/best_model.pth"

            if not os.path.exists(checkpoint_path):
                print("Skipping effectiveness test (no checkpoint)")
                return

            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            # Load CIFAR-10 test data
            transform = transforms.Compose([transforms.ToTensor()])
            testset = torchvision.datasets.CIFAR10(
                root='./data',
                train=False,
                download=True,
                transform=transform
            )

            # Test on 100 samples
            num_samples = 100
            fgsm_success = 0
            pgd_success = 0

            fgsm = FGSM(model=model, epsilon=0.03)
            pgd = PGD(model=model, epsilon=0.03, alpha=0.0075, iterations=10)

            for i in range(num_samples):
                image, label = testset[i]
                image = image.unsqueeze(0)
                label = torch.tensor([label])

                # Check if originally correct
                with torch.no_grad():
                    pred = model(image).argmax(dim=1)
                if pred.item() != label.item():
                    continue  # Skip if already misclassified

                # Test FGSM
                adv_fgsm = fgsm.generate(image, label)
                with torch.no_grad():
                    pred_fgsm = model(adv_fgsm).argmax(dim=1)
                if pred_fgsm.item() != label.item():
                    fgsm_success += 1

                # Test PGD
                adv_pgd = pgd.generate(image, label)
                with torch.no_grad():
                    pred_pgd = model(adv_pgd).argmax(dim=1)
                if pred_pgd.item() != label.item():
                    pgd_success += 1

            fgsm_asr = (fgsm_success / num_samples) * 100
            pgd_asr = (pgd_success / num_samples) * 100

            # Expect reasonable ASR (>20% for FGSM, >50% for PGD)
            self.print_test(
                "FGSM attack effectiveness",
                fgsm_asr > 20,
                f"ASR: {fgsm_asr:.1f}% ({fgsm_success}/{num_samples})"
            )
            self.print_test(
                "PGD attack effectiveness",
                pgd_asr > 50,
                f"ASR: {pgd_asr:.1f}% ({pgd_success}/{num_samples})"
            )

        except Exception as e:
            self.print_test("Attack effectiveness", False, f"Error: {str(e)}")

    # =========================================================================
    # API Tests
    # =========================================================================

    def test_api_health(self):
        """Test API health check endpoint"""
        self.print_header("API Endpoint Tests")

        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            passed = response.status_code == 200
            self.print_test(
                "Health check endpoint",
                passed,
                f"Status: {response.status_code}, Response: {response.json()}"
            )
        except requests.exceptions.ConnectionError:
            self.print_test(
                "Health check endpoint",
                False,
                f"Connection failed. Is the backend running on {self.api_base_url}?"
            )
        except Exception as e:
            self.print_test("Health check endpoint", False, f"Error: {str(e)}")

    def test_api_model_info(self):
        """Test model info endpoint"""
        try:
            response = requests.get(f"{self.api_base_url}/model-info", timeout=5)
            passed = response.status_code == 200

            if passed:
                data = response.json()
                has_fields = all(k in data for k in ['model_type', 'num_classes', 'input_size'])
                self.print_test(
                    "Model info endpoint",
                    has_fields,
                    f"Model: {data.get('model_type')}, "
                    f"Classes: {data.get('num_classes')}, "
                    f"Accuracy: {data.get('test_accuracy', 'N/A')}"
                )
            else:
                self.print_test(
                    "Model info endpoint",
                    False,
                    f"Status: {response.status_code}"
                )
        except Exception as e:
            self.print_test("Model info endpoint", False, f"Error: {str(e)}")

    def test_api_upload(self):
        """Test image upload endpoint"""
        try:
            # Create a test image
            test_image = Image.new('RGB', (32, 32), color='red')
            img_bytes = io.BytesIO()
            test_image.save(img_bytes, format='PNG')
            img_bytes.seek(0)

            files = {'file': ('test.png', img_bytes, 'image/png')}
            response = requests.post(
                f"{self.api_base_url}/upload",
                files=files,
                timeout=10
            )

            passed = response.status_code == 200
            if passed:
                data = response.json()
                has_prediction = 'predicted_class' in data and 'confidence' in data
                self.print_test(
                    "Image upload endpoint",
                    has_prediction,
                    f"Prediction: {data.get('predicted_class')}, "
                    f"Confidence: {data.get('confidence', 0):.2f}"
                )
            else:
                self.print_test(
                    "Image upload endpoint",
                    False,
                    f"Status: {response.status_code}"
                )
        except Exception as e:
            self.print_test("Image upload endpoint", False, f"Error: {str(e)}")

    def test_api_attack(self):
        """Test attack generation endpoint"""
        if self.quick:
            print("Skipping attack endpoint test (--quick mode)")
            return

        try:
            # Create a test image
            test_image = Image.new('RGB', (32, 32), color='blue')
            img_bytes = io.BytesIO()
            test_image.save(img_bytes, format='PNG')
            img_bytes.seek(0)

            files = {'file': ('test.png', img_bytes, 'image/png')}
            data = {
                'attack_type': 'fgsm',
                'epsilon': '0.03'
            }

            response = requests.post(
                f"{self.api_base_url}/attack",
                files=files,
                data=data,
                timeout=30
            )

            passed = response.status_code == 200
            if passed:
                result = response.json()
                has_fields = all(k in result for k in [
                    'original_class', 'adversarial_class',
                    'attack_success', 'adversarial_image'
                ])
                self.print_test(
                    "Attack generation endpoint",
                    has_fields,
                    f"Attack: {data['attack_type']}, "
                    f"Success: {result.get('attack_success')}"
                )
            else:
                self.print_test(
                    "Attack generation endpoint",
                    False,
                    f"Status: {response.status_code}"
                )
        except Exception as e:
            self.print_test("Attack generation endpoint", False, f"Error: {str(e)}")

    def test_api_examples(self):
        """Test example results endpoint"""
        try:
            response = requests.get(
                f"{self.api_base_url}/example-results",
                timeout=10
            )

            passed = response.status_code == 200
            if passed:
                data = response.json()
                is_list = isinstance(data, list)
                self.print_test(
                    "Example results endpoint",
                    is_list,
                    f"Returned {len(data) if is_list else 0} examples"
                )
            else:
                self.print_test(
                    "Example results endpoint",
                    False,
                    f"Status: {response.status_code}"
                )
        except Exception as e:
            self.print_test("Example results endpoint", False, f"Error: {str(e)}")

    # =========================================================================
    # Integration Tests
    # =========================================================================

    def test_end_to_end_workflow(self):
        """Test complete attack workflow"""
        self.print_header("Integration Tests")

        if self.quick:
            print("Skipping end-to-end test (--quick mode)")
            return

        try:
            # 1. Check health
            health_response = requests.get(f"{self.api_base_url}/health", timeout=5)
            if health_response.status_code != 200:
                self.print_test("E2E workflow", False, "Health check failed")
                return

            # 2. Get model info
            info_response = requests.get(f"{self.api_base_url}/model-info", timeout=5)
            if info_response.status_code != 200:
                self.print_test("E2E workflow", False, "Model info failed")
                return

            # 3. Upload image
            test_image = Image.new('RGB', (32, 32), color='green')
            img_bytes = io.BytesIO()
            test_image.save(img_bytes, format='PNG')
            img_bytes.seek(0)

            files = {'file': ('test.png', img_bytes, 'image/png')}
            upload_response = requests.post(
                f"{self.api_base_url}/upload",
                files=files,
                timeout=10
            )

            if upload_response.status_code != 200:
                self.print_test("E2E workflow", False, "Image upload failed")
                return

            # 4. Generate FGSM attack
            img_bytes.seek(0)
            files = {'file': ('test.png', img_bytes, 'image/png')}
            attack_response = requests.post(
                f"{self.api_base_url}/attack",
                files=files,
                data={'attack_type': 'fgsm', 'epsilon': '0.03'},
                timeout=30
            )

            if attack_response.status_code != 200:
                self.print_test("E2E workflow", False, "Attack generation failed")
                return

            # 5. Verify response structure
            result = attack_response.json()
            required_fields = [
                'original_class', 'adversarial_class', 'attack_success',
                'original_confidence', 'adversarial_confidence',
                'perturbation_linf', 'perturbation_l2'
            ]

            has_all_fields = all(f in result for f in required_fields)

            self.print_test(
                "End-to-end workflow",
                has_all_fields,
                "Complete workflow: health -> info -> upload -> attack -> results"
            )

        except Exception as e:
            self.print_test("End-to-end workflow", False, f"Error: {str(e)}")

    def test_frontend_build(self):
        """Test if frontend can build successfully"""
        try:
            frontend_dir = Path(__file__).parent.parent / "frontend"

            # Check if package.json exists
            package_json = frontend_dir / "package.json"
            if not package_json.exists():
                self.print_test(
                    "Frontend package.json",
                    False,
                    "package.json not found"
                )
                return

            self.print_test(
                "Frontend package.json",
                True,
                "package.json exists"
            )

            # Check if node_modules exists
            node_modules = frontend_dir / "node_modules"
            if node_modules.exists():
                self.print_test(
                    "Frontend dependencies",
                    True,
                    "node_modules exists (dependencies installed)"
                )
            else:
                self.print_test(
                    "Frontend dependencies",
                    False,
                    "node_modules not found (run 'npm install')"
                )

        except Exception as e:
            self.print_test("Frontend build", False, f"Error: {str(e)}")

    # =========================================================================
    # Performance Tests
    # =========================================================================

    def test_performance(self):
        """Test attack generation performance"""
        self.print_header("Performance Tests")

        if self.quick:
            print("Skipping performance tests (--quick mode)")
            return

        try:
            from models.resnet import ResNet18
            from attacks.fgsm import FGSM
            from attacks.pgd import PGD

            model = ResNet18(num_classes=10)
            model.eval()

            test_image = torch.randn(1, 3, 32, 32)
            test_label = torch.tensor([3])

            # Test FGSM performance
            fgsm = FGSM(model=model, epsilon=0.03)
            start_time = time.time()
            for _ in range(10):
                _ = fgsm.generate(test_image, test_label)
            fgsm_time = (time.time() - start_time) / 10

            self.print_test(
                "FGSM performance",
                fgsm_time < 0.1,  # Should be under 100ms
                f"Average time: {fgsm_time*1000:.2f}ms"
            )

            # Test PGD performance
            pgd = PGD(model=model, epsilon=0.03, alpha=0.0075, iterations=20)
            start_time = time.time()
            for _ in range(10):
                _ = pgd.generate(test_image, test_label)
            pgd_time = (time.time() - start_time) / 10

            self.print_test(
                "PGD performance (20 iterations)",
                pgd_time < 2.0,  # Should be under 2s
                f"Average time: {pgd_time*1000:.2f}ms"
            )

            # Compare speedup
            speedup = pgd_time / fgsm_time
            self.print_test(
                "PGD vs FGSM speedup",
                True,
                f"PGD is {speedup:.1f}x slower than FGSM"
            )

        except Exception as e:
            self.print_test("Performance tests", False, f"Error: {str(e)}")

    # =========================================================================
    # Main Test Runner
    # =========================================================================

    def run_all_tests(self):
        """Run all tests"""
        print(f"\n{Colors.BOLD}{Colors.OKCYAN}Adversarial ML Project - Test Suite{Colors.ENDC}")
        print(f"{Colors.OKCYAN}Mode: {'Quick' if self.quick else 'Full'}{Colors.ENDC}")
        print(f"{Colors.OKCYAN}Verbose: {self.verbose}{Colors.ENDC}\n")

        # Model tests
        self.test_model_loading()
        self.test_model_inference()

        # Attack tests
        self.test_attack_implementations()
        self.test_attack_effectiveness()

        # API tests
        self.test_api_health()
        self.test_api_model_info()
        self.test_api_upload()
        self.test_api_attack()
        self.test_api_examples()

        # Integration tests
        self.test_end_to_end_workflow()
        self.test_frontend_build()

        # Performance tests
        self.test_performance()

        # Print summary
        self.print_summary()

        # Return exit code
        return 0 if self.tests_failed == 0 else 1


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Run automated tests for Adversarial ML project'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Skip slow tests (attack effectiveness, performance)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed output for all tests'
    )
    args = parser.parse_args()

    runner = TestRunner(verbose=args.verbose, quick=args.quick)
    exit_code = runner.run_all_tests()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
