"""
Test script for CIFAR-10 dataloader module

This script tests all functionalities of the data/dataloader.py module:
- Dataset statistics computation
- Dataloader creation
- Sample visualization
- Augmentation visualization
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataloader import (
    compute_dataset_statistics,
    load_dataset_statistics,
    get_cifar10_dataloaders,
    visualize_samples,
    visualize_augmentations,
    get_class_distribution,
    print_dataset_info,
    CIFAR10_CLASSES
)


def test_statistics_computation():
    """Test dataset statistics computation and saving."""
    print("\n" + "="*60)
    print("TEST 1: Computing Dataset Statistics")
    print("="*60)

    stats = compute_dataset_statistics(
        data_dir='./data',
        save_path='./data/cifar10_stats.json'
    )

    assert 'mean' in stats, "Statistics should contain 'mean'"
    assert 'std' in stats, "Statistics should contain 'std'"
    assert len(stats['mean']) == 3, "Mean should have 3 values (RGB)"
    assert len(stats['std']) == 3, "Std should have 3 values (RGB)"

    print("✓ Statistics computation test passed!")
    return stats


def test_statistics_loading():
    """Test loading saved statistics."""
    print("\n" + "="*60)
    print("TEST 2: Loading Saved Statistics")
    print("="*60)

    stats = load_dataset_statistics('./data/cifar10_stats.json')

    print(f"Loaded statistics:")
    print(f"  Mean: {stats['mean']}")
    print(f"  Std: {stats['std']}")

    print("✓ Statistics loading test passed!")
    return stats


def test_dataloader_creation():
    """Test creating train and test dataloaders."""
    print("\n" + "="*60)
    print("TEST 3: Creating Dataloaders")
    print("="*60)

    train_loader, test_loader = get_cifar10_dataloaders(
        data_dir='./data',
        batch_size=128,
        test_batch_size=100,
        num_workers=2,
        augment=True
    )

    # Verify dataloaders
    assert train_loader is not None, "Train loader should not be None"
    assert test_loader is not None, "Test loader should not be None"
    assert len(train_loader.dataset) == 50000, "Training set should have 50000 samples"
    assert len(test_loader.dataset) == 10000, "Test set should have 10000 samples"

    print("✓ Dataloader creation test passed!")
    return train_loader, test_loader


def test_batch_loading(train_loader, test_loader):
    """Test loading batches from dataloaders."""
    print("\n" + "="*60)
    print("TEST 4: Loading Batches")
    print("="*60)

    # Test train loader
    train_images, train_labels = next(iter(train_loader))
    print(f"\nTrain batch:")
    print(f"  Images shape: {train_images.shape}")
    print(f"  Labels shape: {train_labels.shape}")
    print(f"  Images dtype: {train_images.dtype}")
    print(f"  Images min: {train_images.min():.4f}, max: {train_images.max():.4f}")

    # Test test loader
    test_images, test_labels = next(iter(test_loader))
    print(f"\nTest batch:")
    print(f"  Images shape: {test_images.shape}")
    print(f"  Labels shape: {test_labels.shape}")

    # Verify shapes
    assert train_images.shape == (128, 3, 32, 32), "Train batch shape incorrect"
    assert test_images.shape == (100, 3, 32, 32), "Test batch shape incorrect"
    assert train_labels.shape == (128,), "Train labels shape incorrect"
    assert test_labels.shape == (100,), "Test labels shape incorrect"

    print("✓ Batch loading test passed!")


def test_visualization(train_loader):
    """Test sample visualization."""
    print("\n" + "="*60)
    print("TEST 5: Sample Visualization")
    print("="*60)

    visualize_samples(
        train_loader,
        num_samples=16,
        save_path='./data/test_samples.png',
        show=False
    )

    # Verify file was created
    assert os.path.exists('./data/test_samples.png'), "Visualization file should be created"

    print("✓ Sample visualization test passed!")


def test_augmentation_visualization():
    """Test augmentation visualization."""
    print("\n" + "="*60)
    print("TEST 6: Augmentation Visualization")
    print("="*60)

    visualize_augmentations(
        data_dir='./data',
        num_images=3,
        augmentations_per_image=5,
        save_path='./data/test_augmentations.png',
        show=False
    )

    # Verify file was created
    assert os.path.exists('./data/test_augmentations.png'), "Augmentation visualization file should be created"

    print("✓ Augmentation visualization test passed!")


def test_class_distribution(test_loader):
    """Test class distribution computation."""
    print("\n" + "="*60)
    print("TEST 7: Class Distribution")
    print("="*60)

    distribution = get_class_distribution(test_loader)

    print("\nClass distribution in test set:")
    for class_name, count in distribution.items():
        print(f"  {class_name:12s}: {count:4d} samples")

    # Verify all classes present
    assert len(distribution) == 10, "Should have 10 classes"
    total_samples = sum(distribution.values())
    assert total_samples == 10000, "Test set should have 10000 samples total"

    print("✓ Class distribution test passed!")


def test_dataset_info(train_loader, test_loader):
    """Test dataset info printing."""
    print("\n" + "="*60)
    print("TEST 8: Dataset Information")
    print("="*60)

    print_dataset_info(train_loader, test_loader)

    print("✓ Dataset info test passed!")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print(" "*20 + "CIFAR-10 DATALOADER TEST SUITE")
    print("="*80)

    try:
        # Test 1: Compute statistics
        stats = test_statistics_computation()

        # Test 2: Load statistics
        loaded_stats = test_statistics_loading()

        # Test 3: Create dataloaders
        train_loader, test_loader = test_dataloader_creation()

        # Test 4: Load batches
        test_batch_loading(train_loader, test_loader)

        # Test 5: Visualize samples
        test_visualization(train_loader)

        # Test 6: Visualize augmentations
        test_augmentation_visualization()

        # Test 7: Class distribution
        test_class_distribution(test_loader)

        # Test 8: Dataset info
        test_dataset_info(train_loader, test_loader)

        # Summary
        print("\n" + "="*80)
        print(" "*25 + "ALL TESTS PASSED! ✓")
        print("="*80)

        print("\nGenerated files:")
        print("  - ./data/cifar10_stats.json")
        print("  - ./data/test_samples.png")
        print("  - ./data/test_augmentations.png")

        print("\nThe dataloader module is working correctly!")

        return 0

    except Exception as e:
        print("\n" + "="*80)
        print(" "*30 + "TEST FAILED! ✗")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
