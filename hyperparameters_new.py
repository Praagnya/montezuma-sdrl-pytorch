# Hoang M. Le
# California Institute of Technology
# hmle@caltech.edu
# ===================================================================================================================
# PyTorch implementation by Manus AI

SEED = 1337

import numpy as np
np.random.seed(SEED)
import torch
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
import random
random.seed(SEED)

# Configure PyTorch to use deterministic algorithms for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = 'Desktop'
GPU = 0
VERSION = 1

# Training hyperparameters
BATCH = 128
TRAIN_FREQ = 4
EXP_MEMORY = 500000
HARD_UPDATE_FREQUENCY = 2000
LEARNING_RATE = 0.0001

# Episode constants
maxStepsPerEpisode = 500

# Goals to train
goal_to_train = [0, 1, 2, 3, 4, 5, 6]

# Record keeping
recordFolder = f"summary_v{VERSION}"
recordFileName = f"hybrid_atari_result_v{VERSION}"

# Training thresholds
STOP_TRAINING_THRESHOLD = 0.90

# Network architecture
HIDDEN_NODES = 512

# RL parameters
defaultGamma = 0.99

# Environment parameters
nb_Action = 8
nb_Option = 7
TRAIN_HIST_SIZE = 10000
EPISODE_LIMIT = 80000
STEPS_LIMIT = 8000000

# PyTorch specific configurations
def get_optimizer(model, learning_rate=LEARNING_RATE):
    """Get Adam optimizer for the model with the specified learning rate"""
    return torch.optim.Adam(model.parameters(), lr=learning_rate)

def set_device_config():
    """Configure device settings for PyTorch"""
    if torch.cuda.is_available():
        # Set PyTorch to use the specified GPU
        torch.cuda.set_device(GPU)
        print(f"Using GPU: {torch.cuda.get_device_name(GPU)}")
    else:
        print("CUDA is not available. Using CPU.")
    
    # Return the device for model placement
    return device

# Initialize device configuration
device = set_device_config()