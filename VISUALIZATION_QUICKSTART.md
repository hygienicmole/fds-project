# 📊 Visualization Guide - Quick Start

This guide shows you how to view data and generate visualizations in your Adversarial ML project.

---

## 🖼️ Already Generated Visualizations

The images are automatically opened in your default viewer. You can also find them here:

### 📁 Dataset Visualizations
Location: `./data/`

1. **`test_samples.png`** - Grid of CIFAR-10 sample images
   - Shows all 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
   
2. **`test_augmentations.png`** - Data augmentation examples
   - Shows how images are transformed during training (crops, flips, etc.)

### 📁 Training Curves
Location: `./results/baseline_training/`

1. **`expected_training_curves.png`** - Combined training metrics
2. **`expected_loss_curve.png`** - Training and validation loss over epochs
3. **`expected_accuracy_curve.png`** - Training and validation accuracy over epochs

---

## 🎯 How to Generate More Visualizations

### 1️⃣ View Dataset Samples
```powershell
python test_dataloader.py
```
**Generates:**
- `./data/test_samples.png` - Sample images
- `./data/test_augmentations.png` - Augmentation examples

---

### 2️⃣ Open All Images at Once
```powershell
python view_images.py
```
Opens all generated visualizations automatically!

---

### 3️⃣ Interactive Jupyter Notebook (BEST FOR EXPLORATION!)
```powershell
jupyter notebook notebooks/adversarial_attacks_demo.ipynb
```

**Features:**
- ✅ Interactive data exploration
- ✅ Live visualizations
- ✅ Step-by-step demonstrations
- ✅ Modify parameters in real-time
- ✅ Generate custom plots

---

### 4️⃣ Generate Adversarial Attack Visualizations

**Option A: Quick Demos**
```powershell
# FGSM attack demo
python demo_fgsm.py

# PGD attack demo  
python demo_pgd.py
```

**Option B: Comprehensive Visualizations (After Training)**
```powershell
# Generate 6 types of visualizations
python generate_visualizations.py --attack fgsm --epsilon 0.03

# Or for PGD
python generate_visualizations.py --attack pgd --epsilon 0.03 --pgd-iterations 20
```

**Generates:**
1. Comprehensive comparison (clean vs adversarial)
2. Perturbation heatmaps
3. Confusion matrix (clean)
4. Confusion matrix (adversarial)
5. Class-wise attack success rates
6. Confidence distributions

**Saved to:** `./results/visualizations/`

---

### 5️⃣ Hyperparameter Analysis Plots

```powershell
python experiments/hyperparameter_analysis.py --num-batches 10
```

**Generates:**
- Accuracy vs Epsilon plots
- Attack success rate analysis
- PGD iteration effect plots
- Perturbation analysis

**Saved to:** `./results/hyperparameter_analysis/`

---

## 📂 Where Are Images Saved?

```
fds-project/
├── data/                           # Dataset visualizations
│   ├── test_samples.png           ← Sample images
│   └── test_augmentations.png     ← Augmentation examples
│
├── results/
│   ├── baseline_training/         # Training curves
│   │   ├── expected_training_curves.png
│   │   ├── expected_loss_curve.png
│   │   └── expected_accuracy_curve.png
│   │
│   ├── visualizations/            # Attack visualizations
│   │   ├── fgsm_comparison.png
│   │   ├── fgsm_heatmaps.png
│   │   └── ... (more after running attacks)
│   │
│   └── hyperparameter_analysis/  # Analysis plots
│       └── ... (after running experiments)
```

---

## 🚀 Quick Commands Summary

| What You Want | Command |
|---------------|---------|
| View dataset samples | `python test_dataloader.py` |
| Open all images | `python view_images.py` |
| Interactive exploration | `jupyter notebook notebooks/adversarial_attacks_demo.ipynb` |
| FGSM demo | `python demo_fgsm.py` |
| PGD demo | `python demo_pgd.py` |
| Training curves | `python generate_example_curves.py` |
| Full attack viz | `python generate_visualizations.py --attack fgsm` |

---

## 💡 Tips

1. **For Learning:** Start with Jupyter notebook - it's interactive!
2. **For Quick Checks:** Use `view_images.py` to open everything at once
3. **For Reports:** Use `generate_visualizations.py` for publication-quality plots
4. **Customization:** All visualization functions are in `utils/visualization.py`

---

## 🎨 Example Output

Your visualizations will show:

### Dataset Samples
- 32x32 color images from CIFAR-10
- 10 different classes
- Multiple samples per class

### Training Curves
- Loss decreasing over time
- Accuracy increasing over time
- Train vs validation comparison

### Adversarial Examples
- Original vs adversarial images (look identical!)
- Perturbation magnified to show differences
- Prediction changes (correct → wrong)
- Success rates per class

---

Need help? Check the main README.md or run any script with `--help`


