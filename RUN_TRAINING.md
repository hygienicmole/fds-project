# Running Baseline Training

This guide explains how to run the baseline training for ResNet18 on CIFAR-10.

## Prerequisites

Ensure you have installed all dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

Run the baseline training for 50 epochs:

```bash
python train_baseline.py
```

## What the Script Does

The `train_baseline.py` script:

1. **Trains ResNet18** on CIFAR-10 for 50 epochs
2. **Uses SGD optimizer** with momentum (0.9) and weight decay (5e-4)
3. **Learning rate schedule**: Starts at 0.1, reduces by 10x at epochs 50 and 75
4. **Tracks metrics**: Training/validation loss and accuracy for every epoch
5. **Saves checkpoints**:
   - Best model (based on validation accuracy)
   - Checkpoints every 10 epochs
6. **Generates visualizations**: Training curves (loss, accuracy, learning rate, epoch times)
7. **Exports metrics**: Detailed JSON file with all training statistics

## Expected Output

### Console Output

```
================================================================================
BASELINE TRAINING: ResNet18 on CIFAR-10
================================================================================
Device: cuda
Epochs: 50
Batch size: 128
Initial LR: 0.1
Training samples: 50000
Test samples: 10000
Total parameters: 11,173,962
Trainable parameters: 11,173,962
================================================================================

Epoch 1/50 | LR: 0.100000
Epoch 1/50 [Train]: 100%|████████| 391/391 [00:24<00:00, loss: 1.8234, acc: 32.45%]
Epoch 1/50 [Val]:   100%|████████| 79/79 [00:03<00:00, loss: 1.6543, acc: 38.76%]
Train Loss: 1.8234 | Train Acc: 32.45%
Val Loss:   1.6543 | Val Acc:   38.76%
Time: 24.89s
★ New best validation accuracy: 38.76%
Checkpoint saved to ./models/baseline_best_model.pth
  Epoch: 0, Accuracy: 38.76%, Loss: 1.6543

...

Epoch 50/50 | LR: 0.100000
...
Train Loss: 0.0456 | Train Acc: 98.76%
Val Loss:   0.3842 | Val Acc:   91.32%
Time: 24.95s

================================================================================
TRAINING COMPLETE!
================================================================================
Best Validation Accuracy: 91.45% (Epoch 48)
Final Train Accuracy: 98.76%
Final Validation Accuracy: 91.32%
Total Training Time: 20.79 minutes
Average Epoch Time: 24.95s
================================================================================
```

### Files Generated

After training completes, you'll find:

#### 1. Model Checkpoints

```
models/
├── baseline_best_model.pth              # Best model (highest val accuracy)
└── checkpoints/
    ├── baseline_checkpoint_epoch_10.pth
    ├── baseline_checkpoint_epoch_20.pth
    ├── baseline_checkpoint_epoch_30.pth
    ├── baseline_checkpoint_epoch_40.pth
    └── baseline_checkpoint_epoch_50.pth
```

#### 2. Training Visualizations

```
results/baseline_training/
├── baseline_training_curves.png   # All 4 plots in one figure
├── loss_curve.png                 # Training/validation loss
└── accuracy_curve.png             # Training/validation accuracy
```

#### 3. Metrics and Summary

```
results/baseline_training/
├── baseline_metrics.json          # Detailed JSON metrics
└── baseline_summary.txt           # Human-readable summary
```

## Expected Performance

| Metric | Expected Value |
|--------|----------------|
| Best Validation Accuracy | 90-93% |
| Final Training Accuracy | 98-99% |
| Training Time (GPU) | 15-30 minutes |
| Training Time (CPU) | 2-3 hours |
| Parameters | ~11.2M |

## Training Curves

The script generates comprehensive training curves showing:

1. **Loss Curve**: Training and validation loss over epochs
   - Shows convergence behavior
   - Helps identify overfitting (if val loss increases while train loss decreases)

2. **Accuracy Curve**: Training and validation accuracy over epochs
   - Shows learning progression
   - Highlights best validation accuracy

3. **Learning Rate Schedule**: How LR changes over epochs
   - MultiStepLR: Constant 0.1 until epoch 50, then 0.01, then 0.001

4. **Epoch Times**: Time taken for each epoch
   - Helps identify performance bottlenecks
   - Should be relatively constant

## Understanding the Results

### Training Progression

**Phase 1 (Epochs 1-10): Rapid Learning**
- Accuracy jumps from ~30% to ~70%
- High loss values
- Model learns basic features

**Phase 2 (Epochs 10-30): Steady Improvement**
- Accuracy improves from ~70% to ~88%
- Loss steadily decreases
- Model refines features

**Phase 3 (Epochs 30-50): Fine-tuning**
- Accuracy slowly improves from ~88% to ~91%
- Small loss improvements
- Model learns subtle patterns

**Epoch 50+: Learning Rate Reduction**
- If you train beyond 50 epochs, LR drops to 0.01
- Enables finer optimization
- May improve accuracy by 0.5-1%

### Signs of Good Training

✅ **Validation accuracy increases steadily**
✅ **Training and validation curves track each other**
✅ **No sudden spikes in loss**
✅ **Epoch times are consistent**

### Signs of Problems

⚠️ **Validation accuracy plateaus early** → Learning rate too low or high
⚠️ **Large gap between train and val accuracy** → Overfitting (try dropout)
⚠️ **Loss increases suddenly** → Learning rate too high
⚠️ **Very slow training** → Hardware issue or batch size too small

## Using the Trained Model

After training, load the best model for evaluation:

```python
from models import get_resnet18, load_checkpoint

model = get_resnet18(num_classes=10, device='cuda')
load_checkpoint('./models/baseline_best_model.pth', model, device='cuda')

# Now evaluate against adversarial attacks
python evaluate.py --model-path ./models/baseline_best_model.pth
```

## Customizing Training

To modify training parameters, edit `config.py`:

```python
# Change batch size
BATCH_SIZE = 256  # Larger = faster but needs more memory

# Change learning rate
LEARNING_RATE = 0.05  # Lower = more stable, higher = faster but risky

# Change LR schedule
LR_MILESTONES = [30, 45]  # Earlier drops for faster convergence

# Change number of epochs (modify train_baseline.py)
num_epochs = 100  # Train longer for potentially better accuracy
```

## Troubleshooting

### Out of Memory Error

**Error**: `RuntimeError: CUDA out of memory`

**Solution**: Reduce batch size in `config.py`:
```python
BATCH_SIZE = 64  # or even 32
```

### Training is Very Slow

**On CPU**:
- Expected! Training on CPU takes 2-3 hours
- Consider using a GPU environment

**On GPU**:
- Check GPU utilization: `nvidia-smi`
- Increase batch size if GPU memory allows
- Reduce `NUM_WORKERS` if CPU is bottleneck

### Poor Accuracy (< 85%)

Possible causes:
- Training interrupted early
- Learning rate too high/low
- Data loading issue
- Random seed causing poor initialization

**Solution**: Run again, training should be reproducible with same seed.

### Model Checkpoint Not Found

**Error**: `FileNotFoundError: Checkpoint file not found`

**Solution**: Ensure training completed successfully and check:
```bash
ls -l models/baseline_best_model.pth
```

## Next Steps

After baseline training:

1. **Evaluate adversarial robustness**:
   ```bash
   python evaluate.py --model-path ./models/baseline_best_model.pth --visualize
   ```

2. **Compare with adversarial training**:
   ```bash
   python train.py --epochs 50 --adv-training
   ```

3. **Experiment in Jupyter**:
   ```bash
   jupyter notebook notebooks/adversarial_attacks_demo.ipynb
   ```

4. **Analyze training curves**:
   - Open `results/baseline_training/baseline_training_curves.png`
   - Review `results/baseline_training/baseline_metrics.json`

## Citation

If you use this training script in your coursework, reference:

```
Baseline ResNet18 training on CIFAR-10 for CS685 Adversarial ML Project
Configuration: SGD (momentum=0.9, lr=0.1, MultiStepLR at [50,75])
Expected accuracy: 90-93% on clean test data
```

---

**Questions?** Check the main README.md or create an issue in the repository.
