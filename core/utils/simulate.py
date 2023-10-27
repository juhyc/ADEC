import numpy as np
import torch 
from PIL import Image
import random
from torchvision.transforms import ToTensor



M_exp = 4

# &  Adjust input HDR image to LDR image using scaling factor and a,b
def adjust_dr(image, s, select_range = (0,1)):
    """adjust dynamic range HDR -> LDR simulation

    Args:
        image (Image.png): original image file .png
        s (float): scaling factor, scale intensity
        select_range (tuple, optional): set ldr dynamic range  Defaults to (0,1).

    Returns:
        _type_: numpy array image range (0,1)
    """
    if isinstance(image, Image.Image):
        to_tensor = ToTensor()
        image = to_tensor(image)
        
    if not torch.is_tensor(image):
        raise ValueError("Input image should be a Pytorch Tensor")
        
    if not torch.is_tensor(s):
        s = torch.tensor(s).to(image.device)
    
    a,b = select_range[0], select_range[1]
    
    image = (s * image - a) / (b - a)
        
    image = torch.clamp(image , 0 ,1)

    return image

# & Poisson_gaussian noise modeling with ISO value
def poisson_gauss_noise(img, gauss_var=float(1e-6), poisson_scale=float(3.4e-4), iso=float(100.), clip=True):
    """
    Args:
        img (torch.Tensor): RGB image with float values in [0,1]
        gauss_var (float): Gaussian noise variance
        poisson_scale (float): Poisson noise scale
        iso (float): ISO level for noise scaling
        clip (bool): Whether to clip the output values to [0,1]

    Returns:
        torch.Tensor: An image with Poisson Gauss noise model.
    """
    
    # Ensure the input image is a tensor
    if not torch.is_tensor(img):
        raise ValueError("Input image should be a PyTorch Tensor.")
    
    # Rescale the noise parameters to take into account the ISO
    gauss_std = torch.sqrt(torch.tensor(gauss_var)) * iso / 100.0
    poisson_scale = torch.tensor(poisson_scale) * iso / 100.0
    
    # Move noise parameters to the same device as the image tensor
    gauss_std = gauss_std.to(img.device)
    poisson_scale = poisson_scale.to(img.device)
    
    # Add Poisson noise
    im_poisson = torch.poisson(img / poisson_scale) * poisson_scale
    
    # Add Gaussian noise
    im = im_poisson + gauss_std * torch.rand_like(img)
    
    if clip:
        im = torch.clamp(im, 0, 1)
        
    return im


# & Denormalize image to input stereo network
def denormalized_image(image, s, select_range=(0,1)):
    """denormalize image to input to depth estimation model

    Args:
        image (_type_): original image file
        s (_type_): scaling factor
        select_range (tuple, optional): _description_. Defaults to (0,1).

    Returns:
        torch tensor : denormalized image tensor 
    """
    # if not torch.is_tensor(image):
    #     raise ValueError("Input image should be a Pytorch Tensor")
    
    # if not torch.is_tensor(s):
    #     s = torch.tensor(s).to(image.device)
    
    a,b = select_range[0], select_range[1]
    
    image_denormalized = (b-a) * image + a
    image_denormalized = image_denormalized / s

    
    return image_denormalized/255.0

# & Calculation dynamic range [a,b] based on exposure factor
def cal_dynamic_range(image_tensor, exp):
    
    if not torch.is_tensor(image_tensor):
        raise ValueError("Input cal_dynamic range image should be a Pytorch Tensor")
    
    mean_intensity = torch.mean(image_tensor)
    std_intensity = torch.std(image_tensor)
    
    a = mean_intensity - exp * std_intensity
    b = mean_intensity + exp * std_intensity
    
    # 반환은 텐서값임
    # 스칼라 반환하고자하면 .item()으로 반환
    
    return a,b

# & Generate random exposure factor 
def generate_random_exposure():
    value1 = random.uniform(M_exp**(-1), M_exp)
    value2 = random.uniform(M_exp**(-1), M_exp)
        
    return value1, value2

