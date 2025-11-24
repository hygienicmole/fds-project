# 🚀 Complete Project Guide: A to Z

**Run your Adversarial Machine Learning project from start to finish!**

---

## 📋 Table of Contents

1. [Environment Setup](#step-1-environment-setup-done)
2. [Train the Model](#step-2-train-the-model)
3. [Evaluate Adversarial Attacks](#step-3-evaluate-adversarial-attacks)
4. [Run Demos](#step-4-run-interactive-demos)
5. [Generate Visualizations](#step-5-generate-comprehensive-visualizations)
6. [Hyperparameter Analysis](#step-6-hyperparameter-analysis-optional)
7. [Generate Reports](#step-7-generate-publication-reports)
8. [Web Interface](#step-8-web-interface-optional)
9. [Explore with Jupyter](#step-9-interactive-exploration)

**Estimated Total Time:** 2-4 hours (mostly training)

---

## ✅ Step 1: Environment Setup (✓ DONE!)

You've already completed this! Your environment is ready with:
- Python 3.13.5
- PyTorch 2.9.1+cpu
- All dependencies installed
- Virtual environment activated

**To activate venv in future sessions:**
```powershell
cd C:\Users\tanay\OneDrive\Desktop\fds\fds-project
.\venv\Scripts\Activate.ps1
```

---

## 🎯 Step 2: Train the Model

This is the most important step! Train a ResNet18 model on CIFAR-10.

### Option A: Quick Training (Recommended for Testing)
**Time:** ~20-30 minutes on CPU, ~5 minutes on GPU

```powershell
python train.py --epochs 10
```

### Option B: Full Baseline Training (Recommended for Best Results)
**Time:** ~2-3 hours on CPU, ~30-40 minutes on GPU

```powershell
python train_baseline.py
```

**This will:**
- Train for 200 epochs
- Save checkpoints every 10 epochs
- Track all metrics
- Generate training curves
- Save best model automatically

**Output:**
- Model saved to: `./models/baseline_best_model.pth`
- Metrics saved to: `./results/baseline_training/baseline_metrics.json`
- Training curves: `./results/baseline_training/training_curves.png`

### Option C: Advanced Training (Most Flexible)
```powershell
python models/train.py --epochs 100 --optimizer sgd --scheduler multistep --save-every 10
```

**What You'll See:**
```
Epoch 1/100: 100%|████| 391/391 [02:45<00:00]
  Train Loss: 1.8234, Train Acc: 32.45%
  Val Loss: 1.6543, Val Acc: 38.21%
  Best Val Acc: 38.21% (improved!)
```

**Model Location:** `./models/checkpoints/` or `./models/baseline_best_model.pth`

---

## 🎭 Step 3: Evaluate Adversarial Attacks

Now test how vulnerable your trained model is to attacks!

### Comprehensive Attack Evaluation
**Time:** ~10-15 minutes

```powershell
python attacks/evaluate_attacks.py --model-path ./models/baseline_best_model.pth
```

**This evaluates:**
- ✅ FGSM with 4 epsilon values (0.01, 0.03, 0.05, 0.1)
- ✅ PGD with 12 configurations (4 epsilons × 3 iteration counts)
- ✅ Generates comparison plots
- ✅ Visual grids of adversarial examples

**Output:**
- Results: `./results/attack_evaluation/attack_results.json`
- Plots: `./results/attack_evaluation/comparison_plots.png`
- Visual grid: `./results/attack_evaluation/visual_grid.png`

**Example Results:**
```
Clean Accuracy: 93.45%
FGSM (ε=0.03): 42.18%  ← 51% accuracy drop!
PGD-20 (ε=0.03): 8.76% ← 85% accuracy drop!
```

---

## 🎪 Step 4: Run Interactive Demos

See attacks in action with visual examples!

### Demo 1: FGSM Attack
```powershell
python demo_fgsm.py
```

**Shows:**
- Untargeted attacks (cause any misclassification)
- Targeted attacks (force specific misclassification)
- Epsilon comparison (different attack strengths)
- Visual examples with predictions

### Demo 2: PGD Attack
```powershell
python demo_pgd.py
```

**Shows:**
- PGD vs FGSM comparison
- Effect of iteration count
- Random initialization benefit
- Step size tuning

**Visualizations saved to:** `./results/demos/`

---

## 📊 Step 5: Generate Comprehensive Visualizations

Create publication-quality visualizations!

### For FGSM
```powershell
python generate_visualizations.py --attack fgsm --epsilon 0.03 --num-samples 2000
```

### For PGD
```powershell
python generate_visualizations.py --attack pgd --epsilon 0.03 --pgd-iterations 20 --num-samples 2000
```

**Generates 6 types of visualizations:**
1. ✅ Comprehensive comparison (clean vs adversarial)
2. ✅ Perturbation heatmaps
3. ✅ Confusion matrix (clean predictions)
4. ✅ Confusion matrix (adversarial predictions)
5. ✅ Class-wise attack success rates
6. ✅ Confidence distributions

**Saved to:** `./results/visualizations/`

**Example files:**
- `fgsm_comprehensive_comparison.png`
- `fgsm_perturbation_heatmaps.png`
- `fgsm_confusion_matrix_clean.png`
- `fgsm_confusion_matrix_adversarial.png`
- `fgsm_class_success_rates.png`
- `fgsm_confidence_distributions.png`

---

## 🔬 Step 6: Hyperparameter Analysis (Optional)

Deep dive into how epsilon and iterations affect attacks.

### Quick Analysis (~10 minutes)
```powershell
python experiments/hyperparameter_analysis.py --num-batches 10
```

### Full Analysis (~60 minutes)
```powershell
python experiments/hyperparameter_analysis.py
```

**Generates:**
- 📈 Accuracy vs Epsilon curves
- 📈 Attack success rate analysis
- 📈 PGD iteration effect plots
- 📈 Perturbation norm analysis
- 📊 Statistical comparison tables

**Saved to:** `./results/hyperparameter_analysis/`

---

## 📄 Step 7: Generate Publication Reports

Create LaTeX tables and markdown summaries for your paper/presentation!

```powershell
python utils/report_generator.py
```

**Generates:**
1. **LaTeX Tables** (`reports/latex_tables.tex`)
   - Attack evaluation results
   - FGSM epsilon comparison
   - PGD iteration analysis
   - Baseline training metrics
   - Hyperparameter analysis summary

2. **Markdown Summary** (`reports/EXPERIMENTAL_SUMMARY.md`)
   - Complete experimental results
   - Statistical analysis
   - Key findings
   - Recommendations

**Use in LaTeX:**
```latex
\usepackage{booktabs}
\input{reports/latex_tables.tex}

% Reference tables
As shown in Table~\ref{tab:attack_eval}...
```

---

## 🌐 Step 8: Web Interface (Optional)

Run a web-based interface to interact with the model!

### Start the Backend API
```powershell
uvicorn app.backend:app --reload --port 8000
```

**Access:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- API: http://localhost:8000

**Features:**
- Upload images
- Run attacks via API
- Get predictions
- Real-time results

### Test the API
```powershell
python test_api.py
```

---

## 📓 Step 9: Interactive Exploration

The best way to understand and experiment!

```powershell
jupyter notebook notebooks/adversarial_attacks_demo.ipynb
```

**Features:**
- 📖 Step-by-step tutorial
- 🎨 Interactive visualizations
- 🔧 Modify parameters in real-time
- 📊 Generate custom plots
- 💡 Educational explanations

**Perfect for:**
- Learning how attacks work
- Experimenting with parameters
- Creating custom visualizations
- Understanding the mathematics

---

## 📁 Complete File Structure After Running

```
fds-project/
├── models/
│   ├── baseline_best_model.pth          ← Your trained model!
│   └── checkpoints/                     ← Training checkpoints
│
├── results/
│   ├── baseline_training/
│   │   ├── baseline_metrics.json       ← Training metrics
│   │   └── training_curves.png         ← Loss/accuracy plots
│   │
│   ├── attack_evaluation/
│   │   ├── attack_results.json         ← Attack results
│   │   ├── comparison_plots.png        ← FGSM vs PGD
│   │   └── visual_grid.png             ← Example attacks
│   │
│   ├── visualizations/
│   │   ├── fgsm_comprehensive_comparison.png
│   │   ├── fgsm_perturbation_heatmaps.png
│   │   ├── fgsm_confusion_matrix_clean.png
│   │   ├── fgsm_confusion_matrix_adversarial.png
│   │   ├── fgsm_class_success_rates.png
│   │   └── fgsm_confidence_distributions.png
│   │
│   └── hyperparameter_analysis/
│       ├── results.json                ← Analysis results
│       ├── epsilon_analysis.png        ← Epsilon plots
│       └── iteration_analysis.png      ← Iteration plots
│
├── reports/
│   ├── latex_tables.tex                ← For papers
│   └── EXPERIMENTAL_SUMMARY.md         ← Complete summary
│
└── data/
    ├── test_samples.png                ← Dataset samples
    └── test_augmentations.png          ← Augmentations
```

---

## ⚡ Quick Command Cheatsheet

```powershell
# 1. Train model (quick)
python train.py --epochs 10

# 2. Train model (full - RECOMMENDED)
python train_baseline.py

# 3. Evaluate attacks
python attacks/evaluate_attacks.py

# 4. Run demos
python demo_fgsm.py
python demo_pgd.py

# 5. Generate visualizations
python generate_visualizations.py --attack fgsm --epsilon 0.03

# 6. Hyperparameter analysis
python experiments/hyperparameter_analysis.py --num-batches 10

# 7. Generate reports
python utils/report_generator.py

# 8. Start web API
uvicorn app.backend:app --reload --port 8000

# 9. Open Jupyter notebook
jupyter notebook notebooks/adversarial_attacks_demo.ipynb

# 10. View all images
python view_images.py
```

---

## 🎯 Recommended Workflow

### For Quick Demo (30 minutes):
```powershell
# 1. Train quick model
python train.py --epochs 10

# 2. Run demos
python demo_fgsm.py
python demo_pgd.py

# 3. View results
python view_images.py
```

### For Complete Experiment (3-4 hours):
```powershell
# 1. Train full model (2-3 hours)
python train_baseline.py

# 2. Evaluate attacks (10 min)
python attacks/evaluate_attacks.py

# 3. Generate visualizations (5 min)
python generate_visualizations.py --attack fgsm --epsilon 0.03
python generate_visualizations.py --attack pgd --epsilon 0.03

# 4. Hyperparameter analysis (10 min quick, 60 min full)
python experiments/hyperparameter_analysis.py --num-batches 10

# 5. Generate reports (1 min)
python utils/report_generator.py

# 6. View everything
python view_images.py
```

### For Learning/Exploration:
```powershell
# Start Jupyter notebook
jupyter notebook notebooks/adversarial_attacks_demo.ipynb

# Interactive, self-paced learning!
```

---

## 💡 Tips & Tricks

1. **Start with quick training** (10 epochs) to test everything works
2. **Use Jupyter notebook** for interactive learning
3. **Run demos first** to understand what attacks look like
4. **Full training takes time** - run overnight or during lunch
5. **Check results folder** after each step
6. **Generate reports last** - needs all results to be complete

---

## 🐛 Troubleshooting

### "No module named 'torch'"
```powershell
# Make sure venv is activated
.\venv\Scripts\Activate.ps1
```

### "Model checkpoint not found"
```powershell
# Train a model first
python train.py --epochs 10
```

### "Out of memory"
```powershell
# Reduce batch size in config.py
# or use fewer samples: --num-samples 100
```

---

## 📊 Expected Results

After running everything, you should see:

**Training:**
- Clean Test Accuracy: ~93-95%

**Attacks:**
- FGSM (ε=0.03): ~40-50% accuracy (50% drop!)
- PGD-20 (ε=0.03): ~5-15% accuracy (85% drop!)

**Key Insight:** Models are highly vulnerable to adversarial attacks!

---

## 🎓 What You'll Learn

By running this project A to Z, you'll understand:

1. ✅ How to train deep learning models
2. ✅ How adversarial attacks work
3. ✅ Why models are vulnerable
4. ✅ How to evaluate robustness
5. ✅ Differences between attack types
6. ✅ Impact of hyperparameters
7. ✅ How to visualize results
8. ✅ How to present findings

---

## 🚀 Ready? Start Here!

```powershell
# Quick start (30 min demo)
python train.py --epochs 10
python demo_fgsm.py
python view_images.py

# OR Full pipeline (3-4 hours)
python train_baseline.py
# ... (follow complete workflow above)
```

**Good luck! 🎉**

---

**Questions?** Check README.md or the documentation in `docs/` folder.

