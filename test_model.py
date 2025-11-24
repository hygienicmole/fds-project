"""
Test script for ResNet18 model

This script tests the model creation, summary, and basic functionality
Run this after installing dependencies to verify everything works.
"""

import torch
import sys

def test_model_creation():
    """Test model creation and basic operations"""
    print("=" * 80)
    print("Testing ResNet18 Model for CIFAR-10")
    print("=" * 80)

    # Import model functions
    from models import get_resnet18, count_parameters, model_summary
    from models import save_checkpoint, load_checkpoint

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Create model
    print("\n1. Creating ResNet18 model...")
    model = get_resnet18(num_classes=10, pretrained=False, dropout=0.0, device=device)
    print("[OK] Model created successfully")

    # Count parameters
    print("\n2. Counting parameters...")
    total, trainable = count_parameters(model)
    print(f"   Total parameters: {total:,}")
    print(f"   Trainable parameters: {trainable:,}")
    print("[OK] Parameter counting successful")

    # Test forward pass
    print("\n3. Testing forward pass...")
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32).to(device)
    output = model(x)
    print(f"   Input shape: {list(x.shape)}")
    print(f"   Output shape: {list(output.shape)}")
    assert output.shape == (batch_size, 10), "Output shape mismatch!"
    print("[OK] Forward pass successful")

    # Test feature extraction
    print("\n4. Testing feature extraction...")
    features = model.get_features(x)
    print(f"   Feature shape: {list(features.shape)}")
    assert features.shape == (batch_size, 512), "Feature shape mismatch!"
    print("[OK] Feature extraction successful")

    # Test model summary
    print("\n5. Generating model summary...")
    model_summary(model, input_size=(3, 32, 32), batch_size=1, device=device)

    # Test checkpoint save/load
    print("\n6. Testing checkpoint save/load...")
    import torch.optim as optim
    import os
    import tempfile

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.pth')

        # Save checkpoint
        save_checkpoint(
            model, optimizer, epoch=10, accuracy=85.5, loss=0.45,
            filepath=checkpoint_path
        )

        # Load checkpoint
        info = load_checkpoint(checkpoint_path, model, optimizer, device=device)
        assert info['epoch'] == 10, "Epoch mismatch!"
        assert info['accuracy'] == 85.5, "Accuracy mismatch!"
        print("[OK] Checkpoint save/load successful")

    print("\n" + "=" * 80)
    print("All tests passed! [OK]")
    print("=" * 80)


def test_training_setup():
    """Test training setup"""
    print("\n" + "=" * 80)
    print("Testing Training Setup")
    print("=" * 80)

    from models import create_optimizer, create_scheduler, TrainingMetrics
    from models import get_resnet18

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_resnet18(num_classes=10, pretrained=False, device=device)

    # Test optimizer creation
    print("\n1. Testing optimizer creation...")
    optimizers = ['sgd', 'adam', 'adamw']
    for opt_name in optimizers:
        optimizer = create_optimizer(model, optimizer_name=opt_name, lr=0.1)
        print(f"   [OK] {opt_name.upper()} optimizer created")

    # Test scheduler creation
    print("\n2. Testing scheduler creation...")
    optimizer = create_optimizer(model, optimizer_name='sgd', lr=0.1)

    schedulers = ['multistep', 'cosine', 'plateau']
    for sched_name in schedulers:
        scheduler = create_scheduler(optimizer, scheduler_name=sched_name, T_max=100)
        print(f"   [OK] {sched_name.capitalize()} scheduler created")

    # Test metrics tracking
    print("\n3. Testing metrics tracking...")
    metrics = TrainingMetrics()
    metrics.update_train(loss=0.5, accuracy=85.0)
    metrics.update_val(loss=0.6, accuracy=83.0)
    metrics.update_best(accuracy=83.0, epoch=0)

    summary = metrics.get_summary()
    assert len(summary['train_losses']) == 1, "Metrics tracking failed!"
    print("   [OK] Metrics tracking successful")

    print("\n" + "=" * 80)
    print("Training setup tests passed! [OK]")
    print("=" * 80)


def main():
    """Run all tests"""
    try:
        # Test PyTorch installation
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        # Run tests
        test_model_creation()
        test_training_setup()

        print("\n" + "=" * 80)
        print("SUCCESS: All tests completed!")
        print("=" * 80)
        print("\nYou can now:")
        print("1. Train a model: python models/train.py --epochs 100")
        print("2. Or use the root training script: python train.py --epochs 100")
        print("3. Run the Jupyter notebook: jupyter notebook notebooks/adversarial_attacks_demo.ipynb")

    except ImportError as e:
        print(f"Error: {e}")
        print("\nPlease install dependencies first:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
