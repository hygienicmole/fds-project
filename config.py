"""
Configuration file for Adversarial ML Project
Contains all hyperparameters and settings
"""

import torch
import os

# ===========================
# GENERAL SETTINGS
# ===========================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
NUM_WORKERS = 4

# ===========================
# PATHS
# ===========================
DATA_DIR = './data'
MODEL_DIR = './models'
RESULTS_DIR = './results'
CHECKPOINT_DIR = './models/checkpoints'

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ===========================
# DATASET SETTINGS
# ===========================
DATASET = 'CIFAR10'
NUM_CLASSES = 10
IMAGE_SIZE = 32
MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]

# ===========================
# MODEL SETTINGS
# ===========================
MODEL_NAME = 'ResNet18'
PRETRAINED = False

# ===========================
# TRAINING SETTINGS
# ===========================
BATCH_SIZE = 128
TEST_BATCH_SIZE = 100
EPOCHS = 100
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# Learning rate schedule
LR_MILESTONES = [50, 75]
LR_GAMMA = 0.1

# ===========================
# ADVERSARIAL ATTACK SETTINGS
# ===========================

# FGSM Attack
FGSM_EPSILON = 8/255  # Maximum perturbation (in [0,1] range)
FGSM_EPSILONS = [0, 2/255, 4/255, 8/255, 16/255, 32/255]  # For evaluation

# PGD Attack
PGD_EPSILON = 8/255  # Maximum perturbation
PGD_ALPHA = 2/255    # Step size
PGD_ITERATIONS = 7   # Number of iterations
PGD_RANDOM_START = True

# PGD variants for evaluation
PGD_CONFIGS = [
    {'epsilon': 8/255, 'alpha': 2/255, 'iterations': 7},
    {'epsilon': 8/255, 'alpha': 2/255, 'iterations': 20},
    {'epsilon': 8/255, 'alpha': 2/255, 'iterations': 40},
]

# ===========================
# ADVERSARIAL TRAINING SETTINGS
# ===========================
ADV_TRAINING = False  # Set to True for adversarial training
ADV_TRAIN_EPSILON = 8/255
ADV_TRAIN_ALPHA = 2/255
ADV_TRAIN_ITERATIONS = 7

# ===========================
# EVALUATION SETTINGS
# ===========================
EVAL_BATCH_SIZE = 100
EVAL_SUBSET_SIZE = None  # Set to a number to evaluate on subset, None for full test set

# ===========================
# VISUALIZATION SETTINGS
# ===========================
VIS_NUM_SAMPLES = 10  # Number of samples to visualize
VIS_SAVE_DIR = os.path.join(RESULTS_DIR, 'visualizations')
os.makedirs(VIS_SAVE_DIR, exist_ok=True)

# ===========================
# LOGGING SETTINGS
# ===========================
LOG_INTERVAL = 100  # Log every N batches during training
SAVE_MODEL_INTERVAL = 10  # Save model every N epochs
TENSORBOARD_DIR = './runs'
os.makedirs(TENSORBOARD_DIR, exist_ok=True)
