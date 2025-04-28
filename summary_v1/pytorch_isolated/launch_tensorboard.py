
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
