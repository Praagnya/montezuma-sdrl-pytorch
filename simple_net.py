import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# Set constants
nb_Action = 8
# Set random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)

class Net(nn.Module):
    def __init__(self, device=None):
        """
        PyTorch implementation of the neural network originally written in Keras/TensorFlow.
        
        Parameters:
        -----------
        device : torch.device, optional
            Device to run the model on (cuda or cpu). If None, will use cuda if available.
        """
        super(Net, self).__init__()
        
        # Set device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # Define convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Define fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),  # Size after convolutions: 7x7x64
            nn.ReLU(),
            nn.Linear(512, nb_Action),
            nn.ReLU()  # Original uses ReLU for final layer
        )
        
        # Initialize weights with normal distribution (stddev=0.01)
        self._initialize_weights()
        
        # Move model to device
        self.to(self.device)
        
        # Define optimizer (equivalent to RMSprop in Keras)
        self.optimizer = optim.RMSprop(self.parameters(), lr=0.00025, alpha=0.95, eps=1e-08)
        
        # Define loss function
        self.loss_fn = nn.MSELoss()
        
        # Initialize action history tracking
        self.last_actions = []
        self.action_counts = [0] * nb_Action
    
    def _initialize_weights(self):
        """Initialize weights with normal distribution (stddev=0.01)"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 4, 84, 84)
            
        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, nb_Action)
        """
        x = self.features(x)
        x = self.fc_layers(x)
        return x
    
    def sample(self, q_vec, temperature=0.1):
        """
        Sample an action from the Q-values using softmax with temperature.
        
        Parameters:
        -----------
        q_vec : numpy.ndarray or torch.Tensor
            Q-values for each action
        temperature : float, optional
            Temperature parameter for softmax (default: 0.1)
            
        Returns:
        --------
        int
            Sampled action index
        """
        # Convert to numpy if tensor
        if isinstance(q_vec, torch.Tensor):
            q_vec = q_vec.cpu().numpy()
            
        # Apply softmax with temperature
        prob_pred = np.log(q_vec) / temperature
        dist = np.exp(prob_pred) / np.sum(np.exp(prob_pred))
        choices = range(len(prob_pred))
        
        # Sample from distribution
        return np.random.choice(choices, p=dist)
    
    def selectMove(self, state, goal=None):
        """
        Select action with forced diversity to prevent getting stuck.
        
        Parameters:
        -----------
        state : numpy.ndarray or torch.Tensor
            State representation of shape (84, 84, 4)
        goal : any, optional
            Goal representation (not used in this implementation)
            
        Returns:
        --------
        int
            Index of the selected action
        """
        # Process state tensor
        if isinstance(state, torch.Tensor):
            if state.dim() == 3:  # [H, W, C]
                state_tensor = state.float().permute(2, 0, 1).unsqueeze(0).to(self.device)
            elif state.dim() == 4:  # [B, H, W, C]
                state_tensor = state.float().permute(0, 3, 1, 2).to(self.device)
            else:
                raise ValueError(f"Unexpected tensor shape: {state.shape}")
        else:
            state_tensor = torch.from_numpy(state).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        state_tensor = state_tensor.permute(0, 2, 1, 3)  # Correct dimensions to (batch, channels, height, width)
        
        # Get Q-values
        with torch.no_grad():
            q_array = self(state_tensor)
        q_array_np = q_array.cpu().numpy()[0]
        
        # Add penalty for recently used actions
        for i, count in enumerate(self.action_counts):
            if count > 0:
                q_array_np[i] -= count * 0.5  # Penalize repeated actions
        
        # Add noise to break ties
        q_array_np += np.random.normal(0, 0.2, size=q_array_np.shape)
        
        # Force exploration occasionally
        if np.random.random() < 0.3:  # 30% chance to explore
            # Exclude the most common action from exploration
            most_common_action = np.argmax(self.action_counts) if any(self.action_counts) else -1
            available_actions = [a for a in range(nb_Action) if a != most_common_action]
            action = np.random.choice(available_actions)
            print(f"Forcing exploration, choosing random action: {action}")
        else:
            action = np.argmax(q_array_np)
        
        # Update action history
        self.last_actions.append(action)
        if len(self.last_actions) > 20:  # Keep track of last 20 actions
            self.last_actions.pop(0)
        
        # Update action counts
        self.action_counts = [0] * nb_Action
        for a in self.last_actions:
            self.action_counts[a] += 1
        
        return action
    
    def loadWeight(self, subgoal):
        """
        Load weights for a specific subgoal.
        
        Parameters:
        -----------
        subgoal : int
            Subgoal index
        """
        try:
            # Try loading PyTorch weights first
            self.load_state_dict(torch.load(f'trained_models_for_test/policy_subgoal_{subgoal}.pt'))
            print(f"Loaded PyTorch weights for subgoal {subgoal}")
        except Exception:
            keras_path = f'trained_models_for_test/policy_subgoal_{subgoal}.h5'
            if os.path.exists(keras_path):
                import h5py

                # Load weights manually from h5 file
                with h5py.File(keras_path, 'r') as f:
                    keras_weights = []
                    for layer_name in f['model_weights']:
                        for weight_name in f['model_weights'][layer_name]:
                            keras_weights.append(np.array(f['model_weights'][layer_name][weight_name]))

                # Map to PyTorch layers
                mapping = [
                    (self.features[0], keras_weights[0], keras_weights[1]),  # Conv1
                    (self.features[2], keras_weights[2], keras_weights[3]),  # Conv2
                    (self.features[4], keras_weights[4], keras_weights[5]),  # Conv3
                    (self.fc_layers[0], keras_weights[6], keras_weights[7]),  # FC1
                    (self.fc_layers[2], keras_weights[8], keras_weights[9]),  # FC2
                ]

                for layer, weight, bias in mapping:
                    if len(weight.shape) == 4:
                        # Conv layer: transpose Keras (H, W, inC, outC) -> PyTorch (outC, inC, H, W)
                        layer.weight.data.copy_(torch.from_numpy(weight.transpose(3, 2, 0, 1)))
                    else:
                        # FC layer
                        layer.weight.data.copy_(torch.from_numpy(weight.T))
                    layer.bias.data.copy_(torch.from_numpy(bias))
                
                print(f"Loaded Keras (.h5) weights for subgoal {subgoal}")
            else:
                print(f"No weights file found for subgoal {subgoal}. Using random weights.")