import os
import sys
import torch
import h5py
import numpy as np

def extract_weights_from_h5(h5_file_path):
    """
    Extract weights from an h5 file into a list of PyTorch tensors.
    
    Args:
        h5_file_path: Path to the h5 file
        
    Returns:
        List of PyTorch tensors containing the weights
    """
    weight_list = []
    
    try:
        with h5py.File(h5_file_path, 'r') as f:
            # Print file structure for debugging
            print(f"H5 file structure: {list(f.keys())}")
            
            # Extract weights in the expected order
            layer_order = [
                'conv2d_1',
                'conv2d_2',
                'conv2d_3',
                'dense_1',
                'dense_2'
            ]
            
            for layer_name in layer_order:
                if layer_name not in f:
                    print(f"Warning: Layer {layer_name} not found in h5 file")
                    continue
                
                # Get all datasets in this layer
                layer = f[layer_name]
                
                # Extract kernel (weights)
                kernel_found = False
                for key in layer.keys():
                    if 'kernel' in key or (isinstance(layer[key], h5py.Group) and any('kernel' in k for k in layer[key].keys())):
                        if isinstance(layer[key], h5py.Group):
                            # If it's a group, find the kernel dataset within it
                            for subkey in layer[key].keys():
                                if 'kernel' in subkey:
                                    weight = layer[key][subkey][()]
                                    kernel_found = True
                                    break
                        else:
                            # If it's a dataset, get the data directly
                            weight = layer[key][()]
                            kernel_found = True
                        
                        if kernel_found:
                            # Convert to torch tensor
                            weight_tensor = torch.from_numpy(weight)
                            
                            # Transpose if needed
                            if len(weight_tensor.shape) == 4:  # Conv layer
                                weight_tensor = weight_tensor.permute(3, 2, 0, 1)
                            else:  # Dense layer
                                weight_tensor = weight_tensor.t()
                            
                            weight_list.append(weight_tensor)
                            print(f"Added {layer_name} kernel to weight list, shape: {weight_tensor.shape}")
                            break
                
                if not kernel_found:
                    print(f"Warning: Could not find kernel for {layer_name}")
                
                # Extract bias
                bias_found = False
                for key in layer.keys():
                    if 'bias' in key or (isinstance(layer[key], h5py.Group) and any('bias' in k for k in layer[key].keys())):
                        if isinstance(layer[key], h5py.Group):
                            # If it's a group, find the bias dataset within it
                            for subkey in layer[key].keys():
                                if 'bias' in subkey:
                                    bias = layer[key][subkey][()]
                                    bias_found = True
                                    break
                        else:
                            # If it's a dataset, get the data directly
                            bias = layer[key][()]
                            bias_found = True
                        
                        if bias_found:
                            # Convert to torch tensor
                            bias_tensor = torch.from_numpy(bias)
                            weight_list.append(bias_tensor)
                            print(f"Added {layer_name} bias to weight list, shape: {bias_tensor.shape}")
                            break
                
                if not bias_found:
                    print(f"Warning: Could not find bias for {layer_name}")
    
    except Exception as e:
        print(f"Error extracting weights: {e}")
    
    return weight_list

def add_load_from_h5_method(controller):
    """
    Add a load_from_h5 method to the controller if it doesn't already have one.
    
    Args:
        controller: The controller network
        
    Returns:
        The controller with the added method
    """
    if not hasattr(controller, 'load_from_h5'):
        def load_from_h5(self, weight_list):
            """
            Load weights from a list of tensors.
            
            Args:
                weight_list: List of torch tensors containing weights
            """
            try:
                # Get all parameters that should receive weights
                params = [p for p in self.parameters() if p.requires_grad]
                
                if len(params) != len(weight_list):
                    print(f"Warning: Number of parameters ({len(params)}) doesn't match "
                          f"number of weights ({len(weight_list)})")
                
                # Load weights in order
                for i, (param, weight) in enumerate(zip(params, weight_list)):
                    if param.shape != weight.shape:
                        print(f"Shape mismatch for parameter {i}: "
                              f"param shape {param.shape}, weight shape {weight.shape}")
                        continue
                    
                    param.data.copy_(weight)
                    print(f"Loaded weight {i} with shape {weight.shape}")
                
                print(f"Successfully loaded {len(weight_list)} weights")
                return True
            except Exception as e:
                print(f"Error in load_from_h5: {e}")
                return False
        
        # Add the method to the controller
        import types
        controller.load_from_h5 = types.MethodType(load_from_h5, controller)
    
    return controller

def load_h5_to_hdqn(h5_file_path, hdqn_instance):
    """
    Load weights from an h5 file into an Hdqn instance.
    
    Args:
        h5_file_path: Path to the h5 file
        hdqn_instance: Instance of the Hdqn class
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Make sure the controller has the load_from_h5 method
        add_load_from_h5_method(hdqn_instance.controller)
        
        # Extract weights from the h5 file
        weight_list = extract_weights_from_h5(h5_file_path)
        
        if not weight_list:
            print("Failed to extract weights from h5 file")
            return False
        
        # Load the weights into the controller
        success = hdqn_instance.controller.load_from_h5(weight_list)
        
        return success
    except Exception as e:
        print(f"Error loading weights to Hdqn: {e}")
        return False
