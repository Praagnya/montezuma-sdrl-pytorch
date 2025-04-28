import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import gc
from replay_buffer import PrioritizedReplayBuffer

# Default hyperparameters
defaultGamma = 0.99
LEARNING_RATE = 0.00025
HIDDEN_NODES = 512
SEED = 42
nb_Action = 8

# Default architectures for the lower level controller/actor
defaultEpsilon = 1.0
defaultControllerEpsilon = 1.0

maxReward = 1
minReward = -1

prioritized_replay_alpha = 0.6
max_timesteps = 1000000
prioritized_replay_beta0 = 0.4
prioritized_replay_eps = 1e-6
prioritized_replay_beta_iters = max_timesteps * 0.5

class LinearSchedule:
    """
    Linear interpolation schedule for exploration and beta parameter
    """
    def __init__(self, schedule_timesteps, initial_p, final_p):
        """
        Initialize a linear schedule.
        
        Args:
            schedule_timesteps: Number of timesteps for schedule
            initial_p: Initial value
            final_p: Final value
        """
        self.schedule_timesteps = schedule_timesteps
        self.initial_p = initial_p
        self.final_p = final_p
    
    def value(self, t):
        """
        Get value at time t.
        
        Args:
            t: Current timestep
            
        Returns:
            Interpolated value
        """
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

class DuelingDQNNetwork(nn.Module):
    def __init__(self):
        super(DuelingDQNNetwork, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(7 * 7 * 64, HIDDEN_NODES),
            nn.ReLU(),
            nn.Linear(HIDDEN_NODES, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(7 * 7 * 64, HIDDEN_NODES),
            nn.ReLU(),
            nn.Linear(HIDDEN_NODES, nb_Action)
        )
        
        # Forward model for intrinsic motivation
        self.forward_model = nn.Sequential(
            nn.Linear(7 * 7 * 64 + nb_Action, HIDDEN_NODES),
            nn.ReLU(),
            nn.Linear(HIDDEN_NODES, 7 * 7 * 64)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Ensure input is in the right format
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        if x.shape[1] == 84 and x.shape[3] == 4:
            x = x.permute(0, 3, 1, 2)
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Flatten
        
        # Split into value and advantage streams
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
    
    def predict_next_state(self, state, action):
        # Convert state to features
        features = self.get_features(state)
        
        # One-hot encode action
        action_one_hot = torch.zeros(action.size(0), nb_Action, device=action.device)
        action_one_hot.scatter_(1, action.unsqueeze(1), 1)
        
        # Concatenate features and action
        combined = torch.cat([features, action_one_hot], dim=1)
        
        # Predict next state features
        next_features = self.forward_model(combined)
        return next_features
    
    def get_features(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        if x.shape[1] == 84 and x.shape[3] == 4:
            x = x.permute(0, 3, 1, 2)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.reshape(x.size(0), -1)

class HybridNetwork:
    """
    Network container for controller and target networks
    """
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.controllerNet = DuelingDQNNetwork().to(self.device)
        self.targetControllerNet = DuelingDQNNetwork().to(self.device)
        
        # Copy weights from controller to target
        self.targetControllerNet.load_state_dict(self.controllerNet.state_dict())
        
        # Set target network to evaluation mode
        self.targetControllerNet.eval()
        
        # Optimizer with gradient clipping
        self.optimizer = optim.Adam(
            self.controllerNet.parameters(),
            lr=LEARNING_RATE,
            eps=1e-7
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )

class Agent:
    """
    PyTorch implementation of hybrid reinforcement learning and imitation learning agent
    """
    def __init__(self, net, actionSet, goalSet, defaultNSample, defaultRandomPlaySteps, 
                 controllerMemCap, explorationSteps, trainFreq, hard_update,
                 controllerEpsilon=defaultControllerEpsilon):
        self.actionSet = actionSet
        self.controllerEpsilon = controllerEpsilon
        self.goalSet = goalSet
        self.nSamples = defaultNSample
        self.gamma = defaultGamma
        self.net = net
        self.memory = PrioritizedReplayBuffer(controllerMemCap, alpha=prioritized_replay_alpha)
        self.enable_double_dqn = True
        self.exploration = LinearSchedule(schedule_timesteps=explorationSteps, initial_p=1.0, final_p=0.02)
        self.defaultRandomPlaySteps = defaultRandomPlaySteps
        self.trainFreq = trainFreq
        self.randomPlay = True
        self.learning_done = False
        self.hard_update = hard_update
        
        # Initialize optimizer
        self.optimizer = optim.RMSprop(
            self.net.controllerNet.parameters(),
            lr=LEARNING_RATE,
            alpha=0.95,
            eps=1e-8
        )
        
        # Beta schedule for importance sampling
        self.beta_schedule = LinearSchedule(
            schedule_timesteps=prioritized_replay_beta_iters,
            initial_p=prioritized_replay_beta0,
            final_p=1.0
        )
        
        # Device
        self.device = self.net.device
    
    def compile(self):
        """
        PyTorch equivalent of Keras compile method.
        This method exists for compatibility with code converted from Keras/TensorFlow.
        In PyTorch, we don't need to compile models explicitly as in Keras.
        """
        # Optimizer is already initialized in __init__, so nothing to do here
        # This method exists just to prevent the AttributeError
        pass
    
    def selectMove(self, state):
        """
        Select an action using epsilon-greedy policy
        
        Args:
            state: Current state observation
            
        Returns:
            Selected action index
        """
        if not self.learning_done:
            if self.controllerEpsilon < random.random():
                # Convert state to tensor
                if isinstance(state, np.ndarray):
                    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                else:
                    state_tensor = state.unsqueeze(0).to(self.device)
                
                # Ensure state is in the right format
                if state_tensor.shape[1] == 84 and state_tensor.shape[3] == 4:
                    state_tensor = state_tensor.permute(0, 3, 1, 2)
                
                with torch.no_grad():
                    q_values = self.net.controllerNet(state_tensor)
                    return torch.argmax(q_values).item()
            return random.choice(self.actionSet)
        else:
            # Use simple network after learning is done
            if isinstance(state, np.ndarray):
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            else:
                state_tensor = state.unsqueeze(0).to(self.device)
            
            # Ensure state is in the right format
            if state_tensor.shape[1] == 84 and state_tensor.shape[3] == 4:
                state_tensor = state_tensor.permute(0, 3, 1, 2)
            
            with torch.no_grad():
                q_values = self.simple_net(state_tensor)
                return torch.argmax(q_values).item()
    
    def setControllerEpsilon(self, epsilonArr):
        """
        Set controller epsilon value
        
        Args:
            epsilonArr: New epsilon value
        """
        self.controllerEpsilon = epsilonArr
    
    def criticize(self, reachGoal, action, die, distanceReward, useSparseReward):
        reward = 0.0
        if reachGoal:
            reward += 50.0  # Increased reward for reaching the goal
        if die:
            reward -= 1.0
        if not useSparseReward:
            reward += distanceReward * 50.0  # Scale distance reward
        reward = max(min(reward, 1.0), -1.0)  # Clip reward
        print(f"[Criticize] reachGoal={reachGoal}, die={die}, distanceReward={distanceReward:.6f}, finalReward={reward:.3f}")
        return reward
    
    def store(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode ended
        """
        self.memory.add(state, action, reward, next_state, done)
    
    def _update(self, stepCount):
        """
        Update networks based on sampled experiences
        
        Args:
            stepCount: Current step count
            
        Returns:
            Tuple of (loss, mean_q_values, mean_td_errors)
        """
        # Skip update if in random play phase
        if stepCount < self.defaultRandomPlaySteps:
            return 0.0, 0.0, 0.0
        
        # Sample from replay buffer with importance sampling
        beta = self.beta_schedule.value(stepCount)
        batches = self.memory.sample(self.nSamples, beta=beta)
        (states, actions, rewards, next_states, dones, importance_weights, idxes) = batches
        
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array([1.0 - float(done) for done in dones]), dtype=torch.float32).to(self.device)
        importance_weights = torch.tensor(np.array(importance_weights), dtype=torch.float32).to(self.device)
        
        # Ensure states are in the right format
        if states.ndim == 5:
            # Collapse middle dimensions if input shape is [B, 1, 84, 4, 84]
            states = states.squeeze(1).permute(0, 1, 3, 2)  # [B, 4, 84, 84]
        if states.shape[1] == 84 and states.shape[3] == 4:
            states = states.permute(0, 3, 1, 2)
        if next_states.ndim == 5:
            next_states = next_states.squeeze(1).permute(0, 1, 3, 2)  # [B, 4, 84, 84]
        if next_states.shape[1] == 84 and next_states.shape[3] == 4:
            next_states = next_states.permute(0, 3, 1, 2)
        
        # Get current Q values
        q_values = self.net.controllerNet(states)
        q_values_for_actions = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            if self.enable_double_dqn:
                # Double DQN: use online network to select actions, target network to evaluate
                next_q_values = self.net.controllerNet(next_states)
                next_actions = torch.argmax(next_q_values, dim=1)
                next_q_values_target = self.net.targetControllerNet(next_states)
                next_q_values_for_actions = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN: use target network for both selection and evaluation
                next_q_values_target = self.net.targetControllerNet(next_states)
                next_q_values_for_actions = torch.max(next_q_values_target, dim=1)[0]
            
            # Calculate target values
            target_q_values = rewards + self.gamma * next_q_values_for_actions * dones
        
        # Calculate TD errors for priority update
        td_errors = target_q_values - q_values_for_actions
        
        # Update priorities in replay buffer
        new_priorities = torch.abs(td_errors).detach().cpu().numpy() + prioritized_replay_eps
        self.memory.update_priorities(idxes, new_priorities)
        
        # Calculate Huber loss
        loss = F.smooth_l1_loss(q_values_for_actions, target_q_values, reduction='none')
        
        # Apply importance sampling weights
        weighted_loss = (loss * importance_weights).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        weighted_loss.backward()
        
        # Clip gradients (optional)
        for param in self.net.controllerNet.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()

        # Update target network if needed
        if stepCount > self.defaultRandomPlaySteps and stepCount % self.hard_update == 0:
            self.net.targetControllerNet.load_state_dict(self.net.controllerNet.state_dict())

        return (
            weighted_loss.item(),
            q_values.mean().item(),
            torch.abs(td_errors).mean().item()
        )

    def update(self, stepCount):
        """
        Update networks and return loss information
        
        Args:
            stepCount: Current step count
            
        Returns:
            Tuple of (loss, mean_q_values, mean_td_errors)
        """
        return self._update(stepCount)
    
    def annealControllerEpsilon(self, stepCount, option_learned):
        """
        Anneal controller epsilon based on step count and learning status
        
        Args:
            stepCount: Current step count
            option_learned: Whether the option has been learned
        """
        if not self.randomPlay:
            if option_learned:
                self.controllerEpsilon = 0.0
            else:
                if stepCount > self.defaultRandomPlaySteps:
                    self.controllerEpsilon = self.exploration.value(stepCount - self.defaultRandomPlaySteps)
    
    def clear_memory(self, goal):
        """
        Clear memory and switch to simple network after learning is done
        
        Args:
            goal: Goal index for loading weights
        """
        self.learning_done = True  # Set the done learning flag
        
        # Clear memory
        del self.memory
        
        # Create simple network
        self.simple_net = DuelingDQNNetwork().to(self.device)
        
        # Load weights for the goal
        weight_path = f"weights/policy_subgoal_{goal}_pytorch.pth"
        try:
            self.simple_net.load_state_dict(torch.load(weight_path, map_location=self.device))
            print(f"Loaded weights for goal {goal}")
        except:
            print(f"No weights found for goal {goal}, using random initialization")
        
        # Set to evaluation mode
        self.simple_net.eval()
        
        # Clean up
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


