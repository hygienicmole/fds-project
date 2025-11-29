"""
Quick Image Viewer - Opens generated visualizations
"""
import os
import sys
from PIL import Image

def open_image(path):
    """Open an image file with the default viewer"""
    if os.path.exists(path):
        try:
            img = Image.open(path)
            img.show()
            print(f"[OK] Opened: {path}")
            return True
        except Exception as e:
            print(f"[ERROR] Could not open {path}: {e}")
            return False
    else:
        print(f"[WARNING] File not found: {path}")
        return False

def main():
    print("=" * 70)
    print("Image Viewer - Opening Generated Visualizations")
    print("=" * 70)
    print()
    
    # List of generated images to view
    images = [
        ("./data/test_samples.png", "CIFAR-10 Sample Images"),
        ("./data/test_augmentations.png", "Data Augmentation Examples"),
        ("./results/baseline_training/expected_training_curves.png", "Training Curves"),
        ("./results/baseline_training/expected_loss_curve.png", "Loss Curve"),
        ("./results/baseline_training/expected_accuracy_curve.png", "Accuracy Curve"),
    ]
    
    opened_count = 0
    for path, description in images:
        print(f"Opening: {description}")
        if open_image(path):
            opened_count += 1
        print()
    
    print("=" * 70)
    print(f"Opened {opened_count}/{len(images)} images")
    print("=" * 70)
    print()
    print("Press Enter to continue...")
    input()

if __name__ == '__main__':
    main()


