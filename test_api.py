"""
Test Script for Adversarial ML API

This script tests all API endpoints to ensure they're working correctly.

Usage:
    # Start the API server first
    uvicorn app.backend:app --reload --host 0.0.0.0 --port 8000

    # Then run this test script
    python test_api.py
"""

import requests
import io
import base64
from PIL import Image
import numpy as np
import time


class APITester:
    """Test suite for Adversarial ML API"""

    def __init__(self, base_url="http://localhost:8000"):
        """
        Initialize the API tester.

        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url
        self.passed = 0
        self.failed = 0

    def create_test_image(self, size=(32, 32)):
        """
        Create a random test image.

        Args:
            size: Image size (width, height)

        Returns:
            BytesIO: Image file in memory
        """
        # Create random RGB image
        image_array = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        image = Image.fromarray(image_array, 'RGB')

        # Convert to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        return img_bytes

    def test_endpoint(self, name, test_func):
        """
        Run a test and report results.

        Args:
            name: Test name
            test_func: Test function to run
        """
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")

        try:
            test_func()
            print(f"✓ PASSED: {name}")
            self.passed += 1
        except Exception as e:
            print(f"✗ FAILED: {name}")
            print(f"  Error: {str(e)}")
            self.failed += 1

    def test_health(self):
        """Test health check endpoint"""
        response = requests.get(f"{self.base_url}/health")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert "status" in data, "Missing 'status' in response"
        assert data["status"] == "healthy", f"Status is {data['status']}, expected 'healthy'"
        assert "model_loaded" in data, "Missing 'model_loaded' in response"

        print(f"  Status: {data['status']}")
        print(f"  Model Loaded: {data['model_loaded']}")
        print(f"  Device: {data.get('device', 'unknown')}")

    def test_root(self):
        """Test root endpoint"""
        response = requests.get(f"{self.base_url}/")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert "name" in data, "Missing 'name' in response"
        assert "version" in data, "Missing 'version' in response"
        assert "endpoints" in data, "Missing 'endpoints' in response"

        print(f"  API Name: {data['name']}")
        print(f"  Version: {data['version']}")
        print(f"  Endpoints: {len(data['endpoints'])} available")

    def test_model_info(self):
        """Test model info endpoint"""
        response = requests.get(f"{self.base_url}/model-info")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert "model_name" in data, "Missing 'model_name' in response"
        assert "num_classes" in data, "Missing 'num_classes' in response"
        assert "class_names" in data, "Missing 'class_names' in response"
        assert "parameters" in data, "Missing 'parameters' in response"
        assert "model_loaded" in data, "Missing 'model_loaded' in response"

        print(f"  Model: {data['model_name']}")
        print(f"  Classes: {data['num_classes']}")
        print(f"  Parameters: {data['parameters']:,}")
        print(f"  Device: {data['device']}")
        print(f"  Model Loaded: {data['model_loaded']}")

        if data.get('checkpoint_info'):
            info = data['checkpoint_info']
            print(f"  Checkpoint Epoch: {info.get('epoch')}")
            print(f"  Checkpoint Accuracy: {info.get('accuracy')}")

    def test_upload(self):
        """Test image upload endpoint"""
        # Create test image
        img_bytes = self.create_test_image()

        # Upload
        files = {"file": ("test.png", img_bytes, "image/png")}
        response = requests.post(f"{self.base_url}/upload", files=files)

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert "prediction" in data, "Missing 'prediction' in response"
        assert "image" in data, "Missing 'image' in response"

        pred = data['prediction']
        assert "predicted_class" in pred, "Missing 'predicted_class'"
        assert "predicted_label" in pred, "Missing 'predicted_label'"
        assert "confidence" in pred, "Missing 'confidence'"
        assert "all_probabilities" in pred, "Missing 'all_probabilities'"

        print(f"  Predicted Class: {pred['predicted_label']} (class {pred['predicted_class']})")
        print(f"  Confidence: {pred['confidence']:.4f}")
        print(f"  Image Encoded: {len(data['image'])} characters")

    def test_fgsm_attack(self):
        """Test FGSM attack endpoint"""
        # Create test image
        img_bytes = self.create_test_image()

        # Generate attack
        files = {"file": ("test.png", img_bytes, "image/png")}
        data = {
            "attack_type": "fgsm",
            "epsilon": "0.03"
        }

        start_time = time.time()
        response = requests.post(f"{self.base_url}/attack", files=files, data=data)
        elapsed = time.time() - start_time

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        result = response.json()
        assert "success" in result, "Missing 'success'"
        assert "original_prediction" in result, "Missing 'original_prediction'"
        assert "adversarial_prediction" in result, "Missing 'adversarial_prediction'"
        assert "original_image" in result, "Missing 'original_image'"
        assert "adversarial_image" in result, "Missing 'adversarial_image'"
        assert "perturbation" in result, "Missing 'perturbation'"
        assert "attack_parameters" in result, "Missing 'attack_parameters'"
        assert "perturbation_stats" in result, "Missing 'perturbation_stats'"

        orig_pred = result['original_prediction']
        adv_pred = result['adversarial_prediction']
        stats = result['perturbation_stats']

        print(f"  Attack Success: {result['success']}")
        print(f"  Original: {orig_pred['predicted_label']} ({orig_pred['confidence']:.4f})")
        print(f"  Adversarial: {adv_pred['predicted_label']} ({adv_pred['confidence']:.4f})")
        print(f"  L∞ Norm: {stats['linf_norm']:.6f}")
        print(f"  L2 Norm: {stats['l2_norm']:.6f}")
        print(f"  Processing Time: {elapsed:.2f}s")

    def test_pgd_attack(self):
        """Test PGD attack endpoint"""
        # Create test image
        img_bytes = self.create_test_image()

        # Generate attack
        files = {"file": ("test.png", img_bytes, "image/png")}
        data = {
            "attack_type": "pgd",
            "epsilon": "0.03",
            "pgd_iterations": "20",
            "pgd_alpha": "0.01"
        }

        start_time = time.time()
        response = requests.post(f"{self.base_url}/attack", files=files, data=data)
        elapsed = time.time() - start_time

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        result = response.json()
        assert "success" in result, "Missing 'success'"

        orig_pred = result['original_prediction']
        adv_pred = result['adversarial_prediction']
        stats = result['perturbation_stats']
        params = result['attack_parameters']

        print(f"  Attack Success: {result['success']}")
        print(f"  Original: {orig_pred['predicted_label']} ({orig_pred['confidence']:.4f})")
        print(f"  Adversarial: {adv_pred['predicted_label']} ({adv_pred['confidence']:.4f})")
        print(f"  L∞ Norm: {stats['linf_norm']:.6f}")
        print(f"  L2 Norm: {stats['l2_norm']:.6f}")
        print(f"  PGD Iterations: {params['pgd_iterations']}")
        print(f"  Processing Time: {elapsed:.2f}s")

    def test_targeted_attack(self):
        """Test targeted attack"""
        # Create test image
        img_bytes = self.create_test_image()

        # Generate targeted attack
        files = {"file": ("test.png", img_bytes, "image/png")}
        data = {
            "attack_type": "pgd",
            "epsilon": "0.05",
            "targeted": "true",
            "target_class": "5",  # target: dog
            "pgd_iterations": "40"
        }

        response = requests.post(f"{self.base_url}/attack", files=files, data=data)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        result = response.json()
        adv_pred = result['adversarial_prediction']
        params = result['attack_parameters']

        print(f"  Target Class: {params['target_class']}")
        print(f"  Adversarial: {adv_pred['predicted_label']} (class {adv_pred['predicted_class']})")
        print(f"  Attack Success: {result['success']}")
        print(f"  Confidence: {adv_pred['confidence']:.4f}")

    def test_example_results(self):
        """Test example results endpoint"""
        response = requests.get(f"{self.base_url}/example-results")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        data = response.json()
        assert isinstance(data, list), "Expected list of examples"

        if len(data) > 0:
            example = data[0]
            assert "image_id" in example, "Missing 'image_id'"
            assert "attack_type" in example, "Missing 'attack_type'"
            assert "epsilon" in example, "Missing 'epsilon'"
            assert "attack_success" in example, "Missing 'attack_success'"

            print(f"  Number of Examples: {len(data)}")
            print(f"  First Example:")
            print(f"    Attack: {example['attack_type']}")
            print(f"    Epsilon: {example['epsilon']}")
            print(f"    Success: {example['attack_success']}")
        else:
            print(f"  Number of Examples: 0 (no results file found)")

    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*60)
        print("ADVERSARIAL ML API TEST SUITE")
        print("="*60)

        # Run all tests
        self.test_endpoint("Health Check", self.test_health)
        self.test_endpoint("Root Endpoint", self.test_root)
        self.test_endpoint("Model Info", self.test_model_info)
        self.test_endpoint("Image Upload", self.test_upload)
        self.test_endpoint("FGSM Attack", self.test_fgsm_attack)
        self.test_endpoint("PGD Attack", self.test_pgd_attack)
        self.test_endpoint("Targeted Attack", self.test_targeted_attack)
        self.test_endpoint("Example Results", self.test_example_results)

        # Print summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"  Total Tests: {self.passed + self.failed}")
        print(f"  ✓ Passed: {self.passed}")
        print(f"  ✗ Failed: {self.failed}")
        print(f"  Success Rate: {self.passed / (self.passed + self.failed) * 100:.1f}%")
        print("="*60)

        return self.failed == 0


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Test Adversarial ML API")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the API server"
    )

    args = parser.parse_args()

    # Create tester
    tester = APITester(base_url=args.url)

    # Run tests
    success = tester.run_all_tests()

    # Exit with appropriate code
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
