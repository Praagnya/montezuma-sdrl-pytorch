# ale_pytorch_interface_with_alepy.py
# This implements a PyTorch version of the arcade learning environment interface
# using the official ale-py package instead of direct C library access.

__all__ = ['ALEPyTorchInterface']

import numpy as np
import os
import torch
import warnings
from typing import Optional, Tuple, Union

# Import from ale-py package instead of using ctypes directly
try:
    import ale_py
    from ale_py import ALEInterface as ALEPyInterface
except ImportError:
    raise ImportError("The ale-py package is required. Install it with: pip install ale-py")

class ALEPyTorchInterface:
    """
    PyTorch implementation of the Arcade Learning Environment interface.
    This class extends the functionality of the official ale-py interface by
    providing PyTorch tensor outputs and GPU acceleration capabilities.
    
    Optimized for performance and memory efficiency.
    """
    
    # Logger enum
    class Logger:
        Info = 0
        Warning = 1
        Error = 2

    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        """
        Initialize the ALE PyTorch interface.
        
        Args:
            device: The device to use for PyTorch tensors.
                   If None, will use CUDA if available, otherwise CPU.
        """
        # Initialize the underlying ALE interface from ale-py
        self.ale = ALEPyInterface()
        
        # Set the device for PyTorch tensors
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device if isinstance(device, torch.device) else torch.device(device)
            
        # Cache for screen dimensions to avoid repeated calls
        self._width = None
        self._height = None
        
        # Cache for action sets to avoid repeated calls
        self._legal_actions = None
        self._minimal_actions = None
        
        # Cache for RAM size to avoid repeated calls
        self._ram_size = None
        
        # Reusable buffers for common operations
        self._screen_buffer = None
        self._rgb_buffer = None
        self._grayscale_buffer = None
        self._ram_buffer = None
        
        # Flag to track if ROM is loaded
        self._rom_loaded = False

    def getString(self, key):
        """Get a string value from ALE."""
        return self.ale.getString(key.encode('utf-8') if isinstance(key, str) else key)
    
    def getInt(self, key):
        """Get an integer value from ALE."""
        return self.ale.getInt(key.encode('utf-8') if isinstance(key, str) else key)
    
    def getBool(self, key):
        """Get a boolean value from ALE."""
        return self.ale.getBool(key.encode('utf-8') if isinstance(key, str) else key)
    
    def getFloat(self, key):
        """Get a float value from ALE."""
        return self.ale.getFloat(key.encode('utf-8') if isinstance(key, str) else key)

    def setString(self, key, value):
        """Set a string value in ALE."""
        self.ale.setString(
            key.encode('utf-8') if isinstance(key, str) else key,
            value.encode('utf-8') if isinstance(value, str) else value
        )
    
    def setInt(self, key, value):
        """Set an integer value in ALE."""
        self.ale.setInt(
            key.encode('utf-8') if isinstance(key, str) else key, 
            int(value)
        )
    
    def setBool(self, key, value):
        """Set a boolean value in ALE."""
        self.ale.setBool(
            key.encode('utf-8') if isinstance(key, str) else key, 
            bool(value)
        )
    
    def setFloat(self, key, value):
        """Set a float value in ALE."""
        self.ale.setFloat(
            key.encode('utf-8') if isinstance(key, str) else key, 
            float(value)
        )

    def loadROM(self, rom_file):
        """
        Load a ROM file into ALE.
        
        Args:
            rom_file: Path to the ROM file.
        """
        self.ale.loadROM(rom_file.encode('utf-8') if isinstance(rom_file, str) else rom_file)
        
        # Reset caches since a new ROM might have different dimensions
        self._width = None
        self._height = None
        self._legal_actions = None
        self._minimal_actions = None
        self._screen_buffer = None
        self._rgb_buffer = None
        self._grayscale_buffer = None
        
        # Mark ROM as loaded
        self._rom_loaded = True

    def act(self, action):
        """
        Take an action in the game.
        
        Args:
            action: The action to take (int or torch.Tensor).
            
        Returns:
            int: The reward received after taking the action.
        """
        if not self._rom_loaded:
            warnings.warn("No ROM loaded, action may not have an effect")
            
        if isinstance(action, torch.Tensor):
            action = action.item()
        return self.ale.act(int(action))

    def game_over(self):
        """
        Check if the game is over.
        
        Returns:
            bool: True if the game is over, False otherwise.
        """
        if not self._rom_loaded:
            warnings.warn("No ROM loaded, game_over status may not be accurate")
        return self.ale.game_over()

    def reset_game(self):
        """Reset the game."""
        if not self._rom_loaded:
            warnings.warn("No ROM loaded, reset_game may not have an effect")
        self.ale.reset_game()

    def getLegalActionSet(self):
        """
        Get the set of legal actions for the current game.
        
        Returns:
            torch.Tensor: A tensor containing the legal action set.
        """
        if not self._rom_loaded:
            warnings.warn("No ROM loaded, action set may not be accurate")
            
        # Use cached value if available
        if self._legal_actions is not None:
            return self._legal_actions
            
        # Get legal actions from ale-py
        act_np = self.ale.getLegalActionSet()
        
        # Cache the result
        self._legal_actions = torch.tensor(act_np, dtype=torch.int32, device=self.device)
        return self._legal_actions

    def getMinimalActionSet(self):
        """
        Get the minimal set of actions needed to play the current game.
        
        Returns:
            torch.Tensor: A tensor containing the minimal action set.
        """
        if not self._rom_loaded:
            warnings.warn("No ROM loaded, action set may not be accurate")
            
        # Use cached value if available
        if self._minimal_actions is not None:
            return self._minimal_actions
            
        # Get minimal actions from ale-py
        act_np = self.ale.getMinimalActionSet()
        
        # Cache the result
        self._minimal_actions = torch.tensor(act_np, dtype=torch.int32, device=self.device)
        return self._minimal_actions

    def getFrameNumber(self):
        """
        Get the current frame number.
        
        Returns:
            int: The current frame number.
        """
        if not self._rom_loaded:
            warnings.warn("No ROM loaded, frame number may not be accurate")
        return self.ale.getFrameNumber()

    def lives(self):
        """
        Get the number of lives remaining.
        
        Returns:
            int: The number of lives remaining.
        """
        if not self._rom_loaded:
            warnings.warn("No ROM loaded, lives count may not be accurate")
        return self.ale.lives()

    def getEpisodeFrameNumber(self):
        """
        Get the current episode frame number.
        
        Returns:
            int: The current episode frame number.
        """
        if not self._rom_loaded:
            warnings.warn("No ROM loaded, episode frame number may not be accurate")
        return self.ale.getEpisodeFrameNumber()

    def getScreenDims(self):
        """
        Get the dimensions of the game screen.
        
        Returns:
            tuple: A tuple containing (screen_width, screen_height).
        """
        # Use cached values if available
        if self._width is not None and self._height is not None:
            return (self._width, self._height)
            
        if not self._rom_loaded:
            warnings.warn("No ROM loaded, screen dimensions may not be accurate")
            
        self._width = self.ale.getScreenWidth()
        self._height = self.ale.getScreenHeight()
        return (self._width, self._height)

    def getScreen(self, screen_data=None):
        """
        Get the raw screen data.
        
        Args:
            screen_data: A tensor to fill with screen data.
                        If None, a new tensor will be created.
                                                         
        Returns:
            torch.Tensor: A tensor containing the raw screen data.
        """
        if not self._rom_loaded:
            warnings.warn("No ROM loaded, screen data may not be accurate")
            
        # Get screen data from ale-py
        screen_np = self.ale.getScreen()
        
        # Use the provided tensor or create a new one
        if screen_data is None:
            # Convert to PyTorch tensor and move to the specified device
            return torch.tensor(screen_np, dtype=torch.uint8, device=self.device)
        else:
            # Handle tensor on different devices
            if screen_data.device != torch.device('cpu'):
                # If tensor is on GPU, we need to move it to CPU first
                cpu_tensor = torch.tensor(screen_np, dtype=torch.uint8)
                # Move back to the original device
                screen_data.copy_(cpu_tensor.to(screen_data.device))
            else:
                # If tensor is already on CPU, we can copy directly
                screen_data.copy_(torch.tensor(screen_np, dtype=torch.uint8))
            
            return screen_data

    def getScreenRGB(self, screen_data=None):
        """
        Get the screen data in RGB format.
        
        Args:
            screen_data: A tensor to fill with RGB screen data.
                        If None, a new tensor will be created.
                                                         
        Returns:
            torch.Tensor: A tensor of shape (height, width, 3) containing the RGB screen data.
        """
        if not self._rom_loaded:
            warnings.warn("No ROM loaded, screen data may not be accurate")
            
        # Get RGB screen data from ale-py
        screen_np = self.ale.getScreenRGB()
        
        # Use the provided tensor or create a new one
        if screen_data is None:
            # Convert to PyTorch tensor and move to the specified device
            return torch.tensor(screen_np, dtype=torch.uint8, device=self.device)
        else:
            # Handle tensor on different devices
            if screen_data.device != torch.device('cpu'):
                # If tensor is on GPU, we need to move it to CPU first
                cpu_tensor = torch.tensor(screen_np, dtype=torch.uint8)
                # Move back to the original device
                screen_data.copy_(cpu_tensor.to(screen_data.device))
            else:
                # If tensor is already on CPU, we can copy directly
                screen_data.copy_(torch.tensor(screen_np, dtype=torch.uint8))
            
            return screen_data

    def getScreenGrayscale(self, screen_data=None):
        """
        Get the screen data in grayscale format.
        
        Args:
            screen_data: A tensor to fill with grayscale screen data.
                        If None, a new tensor will be created.
                                                         
        Returns:
            torch.Tensor: A tensor of shape (height, width, 1) containing the grayscale screen data.
        """
        if not self._rom_loaded:
            warnings.warn("No ROM loaded, screen data may not be accurate")
            
        # Get grayscale screen data from ale-py
        screen_np = self.ale.getScreenGrayscale()
        
        # Use the provided tensor or create a new one
        if screen_data is None:
            # Convert to PyTorch tensor and move to the specified device
            return torch.tensor(screen_np, dtype=torch.uint8, device=self.device)
        else:
            # Handle tensor on different devices
            if screen_data.device != torch.device('cpu'):
                # If tensor is on GPU, we need to move it to CPU first
                cpu_tensor = torch.tensor(screen_np, dtype=torch.uint8)
                # Move back to the original device
                screen_data.copy_(cpu_tensor.to(screen_data.device))
            else:
                # If tensor is already on CPU, we can copy directly
                screen_data.copy_(torch.tensor(screen_np, dtype=torch.uint8))
            
            return screen_data

    def getRAMSize(self):
        """
        Get the size of the game's RAM.
        
        Returns:
            int: The size of the RAM.
        """
        # Use cached value if available
        if self._ram_size is not None:
            return self._ram_size
            
        if not self._rom_loaded:
            warnings.warn("No ROM loaded, RAM size may not be accurate")
            
        self._ram_size = self.ale.getRAMSize()
        return self._ram_size

    def getRAM(self, ram=None):
        """
        Get the contents of the game's RAM.
        
        Args:
            ram: A tensor to fill with RAM data.
                If None, a new tensor will be created.
                                                 
        Returns:
            torch.Tensor: A tensor containing the RAM data.
        """
        if not self._rom_loaded:
            warnings.warn("No ROM loaded, RAM data may not be accurate")
            
        # Get RAM data from ale-py
        ram_np = self.ale.getRAM()
        
        # Use the provided tensor or create a new one
        if ram is None:
            # Convert to PyTorch tensor and move to the specified device
            return torch.tensor(ram_np, dtype=torch.uint8, device=self.device)
        else:
            # Handle tensor on different devices
            if ram.device != torch.device('cpu'):
                # If tensor is on GPU, we need to move it to CPU first
                cpu_tensor = torch.tensor(ram_np, dtype=torch.uint8)
                # Move back to the original device
                ram.copy_(cpu_tensor.to(ram.device))
            else:
                # If tensor is already on CPU, we can copy directly
                ram.copy_(torch.tensor(ram_np, dtype=torch.uint8))
            
            return ram

    def saveScreenPNG(self, filename):
        """
        Save the current screen as a PNG file.
        
        Args:
            filename: The filename to save the screen to.
            
        Returns:
            None
        """
        if not self._rom_loaded:
            warnings.warn("No ROM loaded, saved screen may not be accurate")
            
        return self.ale.saveScreenPNG(filename.encode('utf-8') if isinstance(filename, str) else filename)

    def saveState(self):
        """
        Save the state of the system.
        
        Returns:
            None
        """
        if not self._rom_loaded:
            warnings.warn("No ROM loaded, state save may not be effective")
            
        return self.ale.saveState()

    def loadState(self):
        """
        Load the state of the system.
        
        Returns:
            None
        """
        if not self._rom_loaded:
            warnings.warn("No ROM loaded, state load may not be effective")
            
        return self.ale.loadState()

    def cloneState(self):
        """
        Clone the state of the environment (excluding pseudorandomness).
        
        Returns:
            object: A cloned state object.
        """
        if not self._rom_loaded:
            warnings.warn("No ROM loaded, state clone may not be accurate")
            
        return self.ale.cloneState()

    def restoreState(self, state):
        """
        Restore a previously cloned state.
        
        Args:
            state: A state object to restore.
            
        Returns:
            None
        """
        if not self._rom_loaded:
            warnings.warn("No ROM loaded, state restore may not be effective")
            
        self.ale.restoreState(state)

    def cloneSystemState(self):
        """
        Clone the system state (including pseudorandomness).
        
        Returns:
            object: A cloned system state object.
        """
        if not self._rom_loaded:
            warnings.warn("No ROM loaded, system state clone may not be accurate")
            
        return self.ale.cloneSystemState()

    def restoreSystemState(self, state):
        """
        Restore a previously cloned system state.
        
        Args:
            state: A system state object to restore.
            
        Returns:
            None
        """
        if not self._rom_loaded:
            warnings.warn("No ROM loaded, system state restore may not be effective")
            
        self.ale.restoreSystemState(state)

    def encodeState(self, state, buf=None):
        """
        Encode a state.
        
        Args:
            state: A state object to encode.
            buf: A buffer to store the encoded state.
                If None, a new buffer will be created.
                                                 
        Returns:
            torch.Tensor: A tensor containing the encoded state.
        """
        # Get encoded state from ale-py
        encoded_np = self.ale.encodeState(state)
        
        if buf is None or not isinstance(buf, torch.Tensor) or buf.numel() < len(encoded_np):
            # Convert to PyTorch tensor and move to the specified device
            return torch.tensor(encoded_np, dtype=torch.uint8, device=self.device)
        else:
            # Handle tensor on different devices
            if buf.device != torch.device('cpu'):
                # If tensor is on GPU, we need to move it to CPU first
                cpu_tensor = torch.tensor(encoded_np, dtype=torch.uint8)
                # Move back to the original device
                buf.copy_(cpu_tensor.to(buf.device))
            else:
                # If tensor is already on CPU, we can copy directly
                buf.copy_(torch.tensor(encoded_np, dtype=torch.uint8))
            
            return buf

    def decodeState(self, serialized):
        """
        Decode a state.
        
        Args:
            serialized: A tensor or array containing the encoded state.
            
        Returns:
            object: A decoded state object.
        """
        if isinstance(serialized, torch.Tensor):
            # If tensor is on GPU, we need to move it to CPU first
            if serialized.device != torch.device('cpu'):
                serialized = serialized.cpu()
            # Convert to numpy array
            serialized_np = serialized.numpy()
            return self.ale.decodeState(serialized_np)
        else:
            # If not a tensor, assume it's a numpy array
            return self.ale.decodeState(serialized)

    def __del__(self):
        """Clean up resources when the object is deleted."""
        # The ale-py interface handles cleanup automatically

    @staticmethod
    def setLoggerMode(mode):
        """
        Set the logger mode.
        
        Args:
            mode: The logger mode (0: info, 1: warning, 2: error).
            
        Returns:
            None
        """
        dic = {'info': 0, 'warning': 1, 'error': 2}
        mode = dic.get(mode, mode)
        assert mode in [0, 1, 2], "Invalid Mode! Mode must be one of 0: info, 1: warning, 2: error"
        ALEPyInterface.setLoggerMode(mode)

    # Additional PyTorch-specific methods

    def getScreenRGBTensor(self, channel_first=True):
        """
        Get the screen data in RGB format as a PyTorch tensor.
        
        Args:
            channel_first: If True, returns tensor in (C, H, W) format,
                          otherwise in (H, W, C) format.
        
        Returns:
            torch.Tensor: A tensor containing the RGB screen data.
        """
        # Get the screen in RGB format
        screen = self.getScreenRGB()
        
        # Convert from (H, W, C) to (C, H, W) format if requested
        if channel_first:
            screen = screen.permute(2, 0, 1).contiguous()
        
        return screen

    def getScreenGrayscaleTensor(self, channel_first=True):
        """
        Get the screen data in grayscale format as a PyTorch tensor.
        
        Args:
            channel_first: If True, returns tensor in (1, H, W) format,
                          otherwise in (H, W, 1) format.
        
        Returns:
            torch.Tensor: A tensor containing the grayscale screen data.
        """
        # Get the screen in grayscale format
        screen = self.getScreenGrayscale()
        
        # Convert from (H, W, 1) to (1, H, W) format if requested
        if channel_first:
            screen = screen.permute(2, 0, 1).contiguous()
        
        return screen

    def getScreenNormalized(self, channel_first=True):
        """
        Get the screen data in RGB format as a normalized PyTorch tensor.
        
        Args:
            channel_first: If True, returns tensor in (C, H, W) format,
                          otherwise in (H, W, C) format.
        
        Returns:
            torch.Tensor: A tensor containing normalized RGB screen data.
        """
        # Get the screen in RGB format
        screen = self.getScreenRGBTensor(channel_first=channel_first)
        
        # Normalize to [0, 1] range
        screen = screen.float() / 255.0
        
        return screen

    def getScreenGrayscaleNormalized(self, channel_first=True):
        """
        Get the screen data in grayscale format as a normalized PyTorch tensor.
        
        Args:
            channel_first: If True, returns tensor in (1, H, W) format,
                          otherwise in (H, W, 1) format.
        
        Returns:
            torch.Tensor: A tensor containing normalized grayscale screen data.
        """
        # Get the screen in grayscale format
        screen = self.getScreenGrayscaleTensor(channel_first=channel_first)
        
        # Normalize to [0, 1] range
        screen = screen.float() / 255.0
        
        return screen

    def getRAMTensor(self):
        """
        Get the contents of the game's RAM as a PyTorch tensor.
        
        Returns:
            torch.Tensor: A tensor containing the RAM data.
        """
        return self.getRAM()

    def act_with_tensor(self, action_tensor):
        """
        Take an action in the game using a PyTorch tensor.
        
        Args:
            action_tensor: A tensor containing the action to take.
            
        Returns:
            int: The reward received after taking the action.
        """
        # Extract the action from the tensor
        if action_tensor.numel() == 1:
            action = action_tensor.item()
        else:
            action = action_tensor[0].item()
        
        # Take the action
        return self.act(action)
        
    def get_batch_dimensions(self):
        """
        Get the dimensions for batched tensor operations.
        
        Returns:
            tuple: A tuple containing (channels, height, width) for RGB
                  or (1, height, width) for grayscale.
        """
        width, height = self.getScreenDims()
        return (3, height, width)  # RGB format
        
    def get_batch_grayscale_dimensions(self):
        """
        Get the dimensions for batched grayscale tensor operations.
        
        Returns:
            tuple: A tuple containing (1, height, width).
        """
        width, height = self.getScreenDims()
        return (1, height, width)  # Grayscale format
        
    def create_state_buffer(self, batch_size=1, grayscale=True, channel_first=True):
        """
        Create an optimized buffer for storing multiple game states.
        
        Args:
            batch_size: Number of states to store in the buffer.
            grayscale: If True, creates grayscale buffer, otherwise RGB.
            channel_first: If True, uses (C, H, W) format, otherwise (H, W, C).
            
        Returns:
            torch.Tensor: A zeroed tensor for storing game states.
        """
        width, height = self.getScreenDims()
        
        if grayscale:
            channels = 1
        else:
            channels = 3
            
        if channel_first:
            shape = (batch_size, channels, height, width)
        else:
            shape = (batch_size, height, width, channels)
            
        return torch.zeros(shape, dtype=torch.uint8, device=self.device)
        
    def fill_state_buffer(self, buffer, index=0, grayscale=True, channel_first=True, normalize=False):
        """
        Fill a state buffer at the specified index with the current screen.
        
        Args:
            buffer: The buffer tensor to fill.
            index: The index in the buffer to fill.
            grayscale: If True, fills with grayscale data, otherwise RGB.
            channel_first: If True, assumes buffer is in (C, H, W) format.
            normalize: If True, normalizes values to [0, 1] range.
            
        Returns:
            torch.Tensor: The updated buffer.
        """
        if grayscale:
            if channel_first:
                screen = self.getScreenGrayscaleTensor(channel_first=True)
            else:
                screen = self.getScreenGrayscale()
        else:
            if channel_first:
                screen = self.getScreenRGBTensor(channel_first=True)
            else:
                screen = self.getScreenRGB()
                
        if normalize:
            screen = screen.float() / 255.0
            
        # Fill the buffer at the specified index
        buffer[index] = screen
        
        return buffer