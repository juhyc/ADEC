import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from PIL import Image

###############################################
# * Create exposure saturation mask
###############################################

def visualize_mask(mask):
    mask = (mask*255).astype(np.uint8)
    return Image.fromarray(mask)

# Define the soft-binary thrapezoid function numpy version
def soft_binary_threshold_np(image_array, alpha = 0.04, beta = 0.96):
    """Soft-binary thrapezoid function

    Args:
        image_array (ndarray): gray scale image numpy array
        alpha (float, optional): _description_. Defaults to 0.04.
        beta (float, optional): _description_. Defaults to 0.96.

    Returns:
        saturation mask
    """
    assert isinstance(image_array, np.ndarray), "image_array should by numpy array."
    assert len(image_array.shape)==2, "image_array should be 1 channel image."
    
    normalized = image_array / 255.0

    mask_low = (1/alpha) * normalized
    mask_low[normalized >= alpha] = 1
    
    mask_high = 1 - (1/(1-beta)) * (normalized - beta)
    mask_high[normalized <= beta] = 1
    
    mask = np.minimum(mask_low, mask_high)
    
    mask = np.clip(mask, 0, 1)
    
    return mask

def soft_binary_threshold(image_tensor, alpha=0.1, beta=0.9):
    """Soft-binary trapezoid function for PyTorch tensors.

    Args:
        image_tensor (Tensor): gray scale image tensor
        alpha (float, optional): Lower threshold. Defaults to 0.02.
        beta (float, optional): Upper threshold. Defaults to 0.98.

    Returns:
        Tensor: saturation mask
    """
    assert torch.is_tensor(image_tensor), "image_tensor should be a PyTorch tensor."
    assert len(image_tensor.shape)==2, "image_tensor should be 1 channel image."
    
    normalized = image_tensor / 255.0

    mask_low = (1/alpha) * normalized
    mask_low = torch.where(normalized >= alpha, torch.ones_like(normalized), mask_low)
    
    mask_high = 1 - (1/(1-beta)) * (normalized - beta)
    mask_high = torch.where(normalized <= beta, torch.ones_like(normalized), mask_high)
    
    mask = torch.minimum(mask_low, mask_high)
    
    mask = torch.clamp(mask, 0, 1)
    
    return mask

def rgb_to_grayscale(image_tensor):
    """
    Args:
        image_tensor (Tensor): Batch of color image tensors (B, C, H, W).

    Returns:
        Tensor: Batch of grayscale images (B, 1, H, W).
    """
    r = image_tensor[:, 0:1, :, :]
    g = image_tensor[:, 1:2, :, :]
    b = image_tensor[:, 2:3, :, :]
    
    grayscale = 0.299 * r + 0.587 * g + 0.114 * b
    return grayscale


def soft_binary_threshold_batch(image_tensor, alpha=0.01, beta=0.99):
    """Soft-binary trapezoid function for PyTorch tensors in batch.

    Args:
        image_tensor (Tensor): Batch of gray scale image tensors (B, C, H, W).
        alpha (float, optional): Lower threshold. Defaults to 0.02.
        beta (float, optional): Upper threshold. Defaults to 0.98.

    Returns:
        Tensor: Batch of saturation masks.
    """
    assert torch.is_tensor(image_tensor), "image_tensor should be a PyTorch tensor."
    assert image_tensor.dim() == 4, "image_tensor should have shape [B, C, H, W]."
    
    # Using only the first channel for grayscale
    grayscale = rgb_to_grayscale(image_tensor)

    mask_low = (1/alpha) * grayscale
    mask_low = torch.where(grayscale >= alpha, torch.ones_like(grayscale), mask_low)
    
    mask_high = 1 - (1/(1-beta)) * (grayscale - beta)
    mask_high = torch.where(grayscale <= beta, torch.ones_like(grayscale), mask_high)
    
    mask = torch.minimum(mask_low, mask_high)
    
    mask = torch.clamp(mask, 0, 1)
    
    return mask

