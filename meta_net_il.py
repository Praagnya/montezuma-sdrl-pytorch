# hmle@caltech.edu
# ===================================================================================================================
# PyTorch implementation by Manus AI - Optimized Version

# Import the PyTorch hyperparameters

from hyperparameters_new import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, List, Optional, Union, Any

BATCH_SIZE = 32

class MetaNN_PyTorch_Optimized:
    """
    Optimized PyTorch implementation of the MetaNN class for reinforcement learning.
    This class implements a meta-controller network for hierarchical reinforcement learning.
    """
    
    def __init__(self):
        """Initialize the MetaNN_PyTorch class with optimized components."""
        # Set up device
        self.device = torch.device("cuda:"+str(GPU) if torch.cuda.is_available() else "cpu")
        
        # Create the meta controller network
        self.meta_controller = MetaControllerNetwork_Optimized().to(self.device)
        
        # Initialize optimizer (equivalent to RMSprop in Keras)
        self.optimizer = optim.RMSprop(
            self.meta_controller.parameters(), 
            lr=0.00025, 
            alpha=0.95, 
            eps=1e-08, 
            weight_decay=0.0
        )
        
        # Save initial random weights
        torch.save(self.meta_controller.state_dict(), "initial_random_weights_metacontroller.pt")
        
        # Initialize replay history with pre-allocated tensors
        self.replay_hist = [None] * TRAIN_HIST_SIZE
        self.ind = 0
        self.count = 0
        self.meta_ind = 0
        
        # PyTorch uses channels-first format
        self.input_shape = (4, 84, 84)
        self.losses = []
        self.num_pass = 1
        
        # Pre-allocate tensors for efficiency
        self._x_batch = torch.zeros((BATCH_SIZE, 4, 84, 84), dtype=torch.float32, device=self.device)
        self._y_batch = torch.zeros((BATCH_SIZE, nb_Option), dtype=torch.float32, device=self.device)
    
    def reset(self):
        """Reset the model to initial weights."""
        # Load initial random weights
        self.meta_controller.load_state_dict(torch.load("initial_random_weights_metacontroller.pt"))
        
        # Reset optimizer
        self.optimizer = optim.RMSprop(
            self.meta_controller.parameters(), 
            lr=0.00025, 
            alpha=0.95, 
            eps=1e-08, 
            weight_decay=0.0
        )
    
    def check_training_clock(self) -> bool:
        """Check if it's time to train the model."""
        return (self.meta_ind >= 100)
    
    def collect(self, processed: np.ndarray, expert_a: np.ndarray) -> None:
        """
        Collect experience for training.
        
        Args:
            processed: Processed state (4D tensor with shape matching input_shape)
            expert_a: Expert action (one-hot encoded)
        """
        if processed is not None:
            # Convert numpy arrays to PyTorch tensors
            processed_tensor = torch.FloatTensor(processed.astype(np.float32))
            expert_a_tensor = torch.FloatTensor(expert_a.astype(np.float32))
            
            # Ensure correct shape for processed tensor (channels first)
            if processed_tensor.dim() == 3 and processed_tensor.shape[0] != 4:
                # If (H, W, C) format, convert to (C, H, W)
                processed_tensor = processed_tensor.permute(2, 0, 1)
            
            # Store in replay history
            self.replay_hist[self.ind] = (processed_tensor, expert_a_tensor)
            self.ind = (self.ind + 1) % TRAIN_HIST_SIZE
            self.count += 1
            self.meta_ind += 1
    
    def end_collect(self) -> List[float]:
        """End collection phase and train the model."""
        try:
            return self.train()
        except Exception as e:
            print(f"Training error: {e}")
            return []
    
    def train(self) -> List[float]:
        """
        Train the model using collected experiences.
        
        Returns:
            List of loss values from training
        """
        # If not reached TRAIN_HIST_SIZE yet, then get the number of samples
        self._num_valid = self.ind if self.replay_hist[-1] is None else TRAIN_HIST_SIZE
        
        try:
            self._samples = range(self._num_valid)
            batch_size = len(self._samples)
        except:
            self._samples = list(range(self._num_valid)) + [0] * (BATCH_SIZE - len(range(self._num_valid)))
            batch_size = BATCH_SIZE
        
        # Convert replay data to trainable data
        self._selected_replay_data = [self.replay_hist[i] for i in self._samples]
        
        # Reset batch tensors
        self._x_batch.zero_()
        self._y_batch.zero_()
        
        # Fill batch tensors
        for i in range(batch_size):
            if self._selected_replay_data[i] is not None:
                x, y = self._selected_replay_data[i]
                
                # Ensure x is in the right format (C, H, W)
                if x.dim() == 3 and x.shape[0] != 4:  # If (H, W, C)
                    x = x.permute(2, 0, 1)  # Convert to (C, H, W)
                
                # Copy data to pre-allocated tensors
                self._x_batch[i].copy_(x)
                self._y_batch[i].copy_(y)
        
        # Training loop
        self.losses = []
        for epoch in range(self.num_pass):
            # Set model to training mode
            self.meta_controller.train()
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.meta_controller(self._x_batch[:batch_size])
            
            # Calculate loss (categorical cross-entropy)
            loss = F.cross_entropy(outputs, self._y_batch[:batch_size])
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Store loss
            self.losses.append(loss.item())
        
        self.count = 0  # Reset the count clock
        return self.losses
    
    def predict(self, x: Union[np.ndarray, torch.Tensor], batch_size: int = 1) -> np.ndarray:
        """
        Predict on (a batch of) x.
        
        Args:
            x: Input state (numpy array or PyTorch tensor)
            batch_size: Batch size for prediction
            
        Returns:
            Numpy array of predictions
        """
        # Convert numpy array to PyTorch tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        
        # Ensure x is in the right format (C, H, W)
        if x.dim() == 3 and x.shape[0] != 4:  # If (H, W, C)
            x = x.permute(2, 0, 1)  # Convert to (C, H, W)
        elif x.dim() == 4 and x.shape[1] != 4:  # If (B, H, W, C)
            x = x.permute(0, 3, 1, 2)  # Convert to (B, C, H, W)
        
        # Add batch dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        # Move to device and make prediction
        x = x.to(self.device)
        
        # Set model to evaluation mode
        self.meta_controller.eval()
        
        with torch.no_grad():
            output = self.meta_controller(x)
            
        # Return as numpy array
        return output[0].cpu().numpy()
    
    def set_pass(self, num_pass: int) -> None:
        """Set the number of training passes."""
        self.num_pass = num_pass
    
    def sample(self, prob_vec: Union[np.ndarray, torch.Tensor], temperature: float = 0.1) -> int:
        """
        Sample from probability vector using temperature.
        
        Args:
            prob_vec: Probability vector
            temperature: Temperature for sampling (lower = more deterministic)
            
        Returns:
            Sampled action index
        """
        # Convert to PyTorch tensor if needed
        if isinstance(prob_vec, np.ndarray):
            prob_vec = torch.FloatTensor(prob_vec)
        
        # Apply temperature scaling
        logits = torch.log(prob_vec) / temperature
        probs = F.softmax(logits, dim=0)
        
        # Sample from the distribution
        action = torch.multinomial(probs, 1).item()
        
        return action
    
    def get_model_size(self) -> int:
        """
        Get the size of the model in parameters.
        
        Returns:
            Number of parameters in the model
        """
        return sum(p.numel() for p in self.meta_controller.parameters())
    
    def save_model(self, path: str) -> None:
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.meta_controller.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses,
            'ind': self.ind,
            'count': self.count,
            'meta_ind': self.meta_ind,
            'num_pass': self.num_pass
        }, path)
    
    def load_model(self, path: str) -> None:
        """
        Load the model from a file.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.meta_controller.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.losses = checkpoint['losses']
        self.ind = checkpoint['ind']
        self.count = checkpoint['count']
        self.meta_ind = checkpoint['meta_ind']
        self.num_pass = checkpoint['num_pass']


class MetaControllerNetwork_Optimized(nn.Module):
    """
    Optimized PyTorch implementation of the meta controller network.
    Uses batch normalization and efficient layer configurations.
    """
    
    def __init__(self):
        super(MetaControllerNetwork_Optimized, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            # Third convolutional block
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Calculate the size of the flattened features
        # For input (4, 84, 84), after conv layers, the size will be (64, 7, 7)
        self.fc_input_size = 7 * 7 * 64
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, HIDDEN_NODES),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(HIDDEN_NODES, nb_Option)
        )
        
        # Initialize weights with normal distribution
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 4, 84, 84)
            
        Returns:
            Output tensor of shape (batch_size, nb_Option)
        """
        # Extract features
        x = self.features(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classify
        x = self.classifier(x)
        
        # Apply softmax
        x = F.softmax(x, dim=1)
        
        return x
    
    def _initialize_weights(self) -> None:
        """Initialize weights with normal distribution."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.zeros_(m.bias)
