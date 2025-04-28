# Hoang M. Le
# California Institute of Technology
# hmle@caltech.edu
# ===================================================================================================================
# PyTorch implementation of the original Keras/TensorFlow code

# from hyperparameters import *
from hyperparameters_new import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gc
import random

# Set seeds for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

HUBER_DELTA = 0.5

def mean_q(y_true, y_pred):
    """Calculate mean of maximum Q-values across batch"""
    return torch.mean(torch.max(y_pred, dim=-1)[0])

def huber_loss(y_true, y_pred, clip_value=HUBER_DELTA):
    """PyTorch implementation of Huber loss"""
    assert clip_value > 0.
    
    x = y_true - y_pred
    if np.isinf(clip_value):
        return 0.5 * torch.square(x)
    
    condition = torch.abs(x) < clip_value
    squared_loss = 0.5 * torch.square(x)
    linear_loss = clip_value * (torch.abs(x) - 0.5 * clip_value)
    
    return torch.where(condition, squared_loss, linear_loss)

def clipped_masked_error(y_true, y_pred, mask):
    """Apply huber loss with masking"""
    loss = huber_loss(y_true, y_pred, 1.0)
    loss = loss * mask  # apply element-wise mask
    return torch.sum(loss, dim=-1)

class DQNModel(nn.Module):
    """PyTorch implementation of the DQN model"""
    def __init__(self, input_shape=(4, 84, 84), hidden_nodes=HIDDEN_NODES, nb_actions=nb_Action, enable_dueling=False):
        super(DQNModel, self).__init__()
        self.enable_dueling = enable_dueling
        
        # CNN layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of the flattened features
        self.feature_size = self._get_conv_output(input_shape)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size, hidden_nodes)
        
        if not self.enable_dueling:
            self.fc2 = nn.Linear(hidden_nodes, nb_actions)
        else:
            self.value_stream = nn.Linear(hidden_nodes, 1)
            self.advantage_stream = nn.Linear(hidden_nodes, nb_actions)
    
    def _get_conv_output(self, shape):
        """Helper function to calculate the size of the flattened features"""
        bs = 1
        input = torch.rand(bs, *shape)
        output = self._forward_conv(input)
        return int(np.prod(output.size()))
    
    def _forward_conv(self, x):
        """Forward pass through convolutional layers"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x
    
    def forward(self, x):
        """Forward pass through the network"""
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        
        if not self.enable_dueling:
            return self.fc2(x)
        else:
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            # Combine value and advantage streams (dueling architecture)
            return value + advantage - advantage.mean(dim=1, keepdim=True)

    def init_weights(self):
        """Initialize weights with normal distribution"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

class Hdqn:
    def __init__(self, gpu):
        self.enable_dueling_network = False
        self.gpu = gpu
        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        
        # Create controller network and target network
        self.controllerNet = DQNModel(
            input_shape=(4, 84, 84),  # PyTorch uses channels-first format
            hidden_nodes=HIDDEN_NODES,
            nb_actions=nb_Action,
            enable_dueling=self.enable_dueling_network
        ).to(self.device)
        
        # Initialize weights
        self.controllerNet.init_weights()
        
        # Create target network as a copy of the controller network
        self.targetControllerNet = DQNModel(
            input_shape=(4, 84, 84),
            hidden_nodes=HIDDEN_NODES,
            nb_actions=nb_Action,
            enable_dueling=self.enable_dueling_network
        ).to(self.device)
        
        # Copy weights from controller to target
        self.targetControllerNet.load_state_dict(self.controllerNet.state_dict())
        
        # Set optimizer (equivalent to RMSprop in Keras)
        self.optimizer = optim.RMSprop(
            self.controllerNet.parameters(),
            lr=LEARNING_RATE,
            alpha=0.95,  # equivalent to rho in Keras
            eps=1e-08,
            weight_decay=0.0  # equivalent to decay in Keras
        )
    
    def saveWeight(self, subgoal):
        """Save model weights"""
        torch.save(self.controllerNet.state_dict(), f"{recordFolder}/policy_subgoal_{subgoal}.pt")

    def loadWeight(self, subgoal):
        """Load model weights"""
        self.controllerNet.load_state_dict(torch.load(f"{recordFolder}/policy_subgoal_{subgoal}.pt", map_location=self.device))
        # Copy weights to target network
        self.targetControllerNet.load_state_dict(self.controllerNet.state_dict())

    def clear_memory(self):
        """Clear memory by deleting models and collecting garbage"""
        del self.controllerNet
        del self.targetControllerNet
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
