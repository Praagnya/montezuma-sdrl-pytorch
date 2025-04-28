# Completely isolated PyTorch TensorBoard Visualizer
# Avoids all conflicts with TensorFlow by using file-based communication
# Original credits:
# Hoang M. Le
# California Institute of Technology
# hmle@caltech.edu
# ===================================================================================================================

from __future__ import absolute_import

import os
import six
import torch
import numpy as np
import json
import time
import subprocess
import sys
from pathlib import Path
import threading

class BaseVisualizer:
    """
    Base class for visualizers
    """
    def __init__(self):
        pass
    
    def initialize(self, *args, **kwargs):
        raise NotImplementedError()
    
    def add_entry(self, *args, **kwargs):
        raise NotImplementedError()
    
    def close(self):
        raise NotImplementedError()


class TensorboardVisualizer(BaseVisualizer):
    """
    Visualize the generated results in Tensorboard using PyTorch
    with complete isolation from TensorFlow
    """

    def __init__(self):
        super(TensorboardVisualizer, self).__init__()
        self._logdir = None
        self._event_dir = None
        self._image_dir = None
        self._tb_process = None
        self._tb_launcher_path = None
        self._is_initialized = False

    def initialize(self, logdir, model=None, converter=None):
        """
        Initialize the visualizer with a log directory
        
        Args:
            logdir: Directory to store TensorBoard logs
            model: Optional model to visualize
            converter: Optional converter to convert model for visualization
        """
        assert logdir is not None, "logdir cannot be None"
        assert isinstance(logdir, six.string_types), "logdir should be a string"

        # Create completely separate directory structure for PyTorch logs
        self._logdir = os.path.join(logdir, "pytorch_isolated")
        self._event_dir = os.path.join(self._logdir, "events")
        self._image_dir = os.path.join(self._logdir, "images")
        
        # Create directories
        os.makedirs(self._logdir, exist_ok=True)
        os.makedirs(self._event_dir, exist_ok=True)
        os.makedirs(self._image_dir, exist_ok=True)
        
        # Create TensorBoard launcher script
        self._tb_launcher_path = os.path.join(self._logdir, "launch_tensorboard.py")
        self._create_tensorboard_launcher()
        
        # If converter is provided, use it to convert the model
        if converter is not None and model is not None:
            assert isinstance(converter, TorchConverter), \
                        "converter should derive from TorchConverter"
            converter.convert(model, self._event_dir)
        
        self._is_initialized = True

    def _create_tensorboard_launcher(self):
        """Create a separate Python script to launch TensorBoard in isolation"""
        launcher_code = """
import os
import sys
import time
import json
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

# Set environment variables to avoid TensorFlow conflicts
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def main():
    if len(sys.argv) != 2:
        print("Usage: python launch_tensorboard.py <event_dir>")
        return
        
    event_dir = sys.argv[1]
    writer = SummaryWriter(log_dir=event_dir, flush_secs=30)
    
    # Monitor the events directory for new event files
    events_path = os.path.join(event_dir, "events")
    images_path = os.path.join(event_dir, "images")
    
    os.makedirs(events_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    
    print(f"TensorBoard writer initialized at {event_dir}")
    print(f"Monitoring for events at {events_path}")
    print(f"Monitoring for images at {images_path}")
    
    last_processed = set()
    
    try:
        while True:
            # Process scalar events
            event_files = [f for f in os.listdir(events_path) if f.endswith('.json')]
            for event_file in event_files:
                if event_file in last_processed:
                    continue
                    
                try:
                    with open(os.path.join(events_path, event_file), 'r') as f:
                        event_data = json.load(f)
                        
                    tag = event_data.get('tag')
                    value = event_data.get('value')
                    step = event_data.get('step', 0)
                    
                    if tag and value is not None:
                        writer.add_scalar(tag, value, global_step=step)
                        print(f"Added scalar: {tag}={value} (step {step})")
                    
                    # Mark as processed and remove file
                    last_processed.add(event_file)
                    os.remove(os.path.join(events_path, event_file))
                except Exception as e:
                    print(f"Error processing event file {event_file}: {e}")
            
            # Process image events
            image_dirs = [d for d in os.listdir(images_path) if os.path.isdir(os.path.join(images_path, d))]
            for image_dir in image_dirs:
                try:
                    # Each directory represents one image event
                    dir_path = os.path.join(images_path, image_dir)
                    meta_file = os.path.join(dir_path, 'meta.json')
                    
                    if not os.path.exists(meta_file):
                        continue
                        
                    with open(meta_file, 'r') as f:
                        meta_data = json.load(f)
                    
                    tag = meta_data.get('tag')
                    step = meta_data.get('step', 0)
                    image_file = os.path.join(dir_path, 'image.npy')
                    
                    if os.path.exists(image_file) and tag:
                        # Load image data
                        image_data = np.load(image_file)
                        
                        # Convert to tensor if needed
                        if image_data.ndim == 3 and image_data.shape[2] in [1, 3, 4]:
                            # Convert HWC to CHW format
                            image_tensor = torch.from_numpy(np.transpose(image_data, (2, 0, 1)))
                        else:
                            image_tensor = torch.from_numpy(image_data)
                            
                        writer.add_image(tag, image_tensor, global_step=step)
                        print(f"Added image: {tag} (step {step})")
                    
                    # Clean up directory after processing
                    import shutil
                    shutil.rmtree(dir_path)
                except Exception as e:
                    print(f"Error processing image directory {image_dir}: {e}")
            
            # Sleep to avoid high CPU usage
            time.sleep(1)
    except KeyboardInterrupt:
        print("TensorBoard launcher shutting down")
    finally:
        writer.close()
        print("TensorBoard writer closed")

if __name__ == "__main__":
    main()
"""
        with open(self._tb_launcher_path, 'w') as f:
            f.write(launcher_code)

    def _start_tensorboard_process(self):
        """Start TensorBoard process in a separate thread to avoid blocking"""
        def run_process():
            try:
                # Use subprocess to run the launcher script
                cmd = [sys.executable, self._tb_launcher_path, self._logdir]
                self._tb_process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Log output for debugging
                log_file = os.path.join(self._logdir, "tensorboard_log.txt")
                with open(log_file, 'w') as f:
                    for line in self._tb_process.stdout:
                        f.write(line)
                        f.flush()
            except Exception as e:
                error_log = os.path.join(self._logdir, "tensorboard_error.txt")
                with open(error_log, 'w') as f:
                    f.write(f"Error starting TensorBoard process: {str(e)}")

        # Start process in a separate thread
        thread = threading.Thread(target=run_process)
        thread.daemon = True
        thread.start()

    def add_entry(self, index, tag, value, **kwargs):
        """
        Add an entry to TensorBoard
        
        Args:
            index: Step index
            tag: Tag name
            value: Value to log
            **kwargs: Additional arguments, including 'image' flag
        """
        if not self._is_initialized:
            raise RuntimeError("Visualizer not initialized. Call initialize() first.")
            
        # Start TensorBoard process if not already started
        if self._tb_process is None:
            self._start_tensorboard_process()
            
        if "image" in kwargs and value is not None:
            self._add_image_entry(index, tag, value)
        else:
            self._add_scalar_entry(index, tag, value)

    def _add_scalar_entry(self, index, tag, value):
        """Add a scalar entry by writing to a JSON file"""
        event_data = {
            'tag': tag,
            'value': float(value),
            'step': int(index)
        }
        
        # Create a unique filename based on timestamp and tag
        timestamp = int(time.time() * 1000)
        filename = f"{timestamp}_{tag.replace('/', '_')}_{index}.json"
        filepath = os.path.join(self._event_dir, "events", filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Write event data to file
        with open(filepath, 'w') as f:
            json.dump(event_data, f)

    def _add_image_entry(self, index, tag, value):
        """Add an image entry by saving the image data to disk"""
        # Convert value to numpy array if it's a tensor
        if isinstance(value, torch.Tensor):
            # If tensor is in CHW format, convert to HWC for consistency
            if value.ndim == 3 and value.shape[0] in [1, 3, 4]:
                value = value.permute(1, 2, 0)
            value = value.detach().cpu().numpy()
        
        # Ensure value is a numpy array
        if not isinstance(value, np.ndarray):
            raise TypeError("Image value must be numpy array or torch tensor")
            
        # Create a unique directory for this image
        timestamp = int(time.time() * 1000)
        image_dir = f"{timestamp}_{tag.replace('/', '_')}_{index}"
        dir_path = os.path.join(self._image_dir, image_dir)
        os.makedirs(dir_path, exist_ok=True)
        
        # Save image data as numpy array
        image_path = os.path.join(dir_path, "image.npy")
        np.save(image_path, value)
        
        # Save metadata
        meta_path = os.path.join(dir_path, "meta.json")
        meta_data = {
            'tag': tag,
            'step': int(index),
            'shape': value.shape
        }
        with open(meta_path, 'w') as f:
            json.dump(meta_data, f)

    def close(self):
        """Close the visualizer and terminate the TensorBoard process"""
        if self._tb_process is not None:
            self._tb_process.terminate()
            self._tb_process = None
        self._is_initialized = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class TorchConverter(object):
    """
    Interface for model converters to be used with TensorboardVisualizer
    """
    def convert(self, network, event_dir):
        """
        Convert a network model for visualization
        
        Args:
            network: The network model to convert
            event_dir: Directory to store conversion events
        """
        raise NotImplementedError()
