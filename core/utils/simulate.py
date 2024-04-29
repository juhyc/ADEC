import numpy as np
import torch 
from PIL import Image
import random
from torchvision.transforms import ToTensor

M_exp = 4
M_sensor = 2**8 - 1

###############################################
# * Image formation model, noise simulation
###############################################

# ^  Adjust input HDR image to LDR image using scaling factor and a,b
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

# ^ Poisson_gaussian noise modeling with ISO value
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

# ^ Denormalize image to input stereo network
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

# ^ Calculation dynamic range [a,b] based on exposure factor
def cal_dynamic_range(image_tensor, exp):
    
    if not torch.is_tensor(image_tensor):
        raise ValueError("Input cal_dynamic range image should be a Pytorch Tensor")
    
    mean_intensity = torch.mean(image_tensor)
    std_intensity = torch.std(image_tensor)
    
    a = mean_intensity - exp * std_intensity
    b = mean_intensity + exp * std_intensity
    
    return a,b

# ^ Generate random exposure factor 
def generate_random_exposures(batch_size, valid_mode=False, value = 1.0):
    exp_list = []
    
    if valid_mode:
        for _ in range(batch_size):
            exp = value
            exp_list.append([exp])
    
    else:
        for _ in range(batch_size):
            exp = random.uniform(M_exp**(-1), M_exp)
            exp_list.append([exp])
        
    return torch.tensor(exp_list)

# ^ Generate random exposure factor based on HDR scene dynamic range
def generate_adjusted_random_expousres(batch_size, d_r, gap_threshold = 0.5, large_gap  = 4, small_gap = 2):
    exp_list = []
    
    for _ in range(batch_size):
        exp1 = random.uniform(M_exp**(-1), M_exp)
        
        if d_r >= gap_threshold:
            adjust_factor = random.uniform(1, large_gap)
            exp2 = max(min(exp1 * adjust_factor, M_exp), M_exp**(-1))
        else:
            adjsut_factor = random.uniform(1/small_gap, 1)
            exp2 = max(min(exp1 * adjust_factor, M_exp), M_exp**(-1))
        
        exp_list.append(([exp1, exp2]))
    
    return torch.tensor(exp_list)

def calculate_dynamic_range_batch(images):
    # images shape: (batch_size, channels, height, width)
    
    # weight for luminance
    weights = torch.tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1).to(images.device)
    
    # rgb to luminance
    luminance = torch.sum(images * weights, dim=1)
    
    dr_values = []
    for i in range(images.shape[0]):
        luminance_flat = luminance[i].flatten()
        
        percentile_5, percentile_95 = torch.quantile(luminance_flat, torch.tensor([0.05, 0.95]).to(images.device))
        
        d_r = (percentile_95 - percentile_5)
        dr_values.append(d_r)
    
    return torch.stack(dr_values)


def min_max_scale(image):
    result = (image - torch.min(image))/(torch.max(image) - torch.min(image))
    return result

# ^ Image Simulation class (noise modeling, adjust dynamic range)
class ImageFormation:
    def __init__(self, image, exp, device = 'cpu'):
        """Initialize image formation model.
            Set image to phi.
            Calculate gain, shutter time from exposure value 'exp'.

        Args:
            image (Image): HDR image
            exp : exposure
        """
        if torch.any(exp <= 0):
            raise ValueError("'exp' should be a positive value.")
        
        self.device = torch.device(device)
        self.gauss_var = float(1e-5)
        self.poisson_scale = float(3.4e-7)
        
        # HDR scene
        # !
        if image.max() == 255.0:
            self.original_phi = image.to(self.device)/255.0
        else :
            # self.original_phi = min_max_scale(image).to(self.device)
            self.original_phi = (image).to(self.device)
        self.phi = self.original_phi.clone()
        self.exp = exp.to(self.device)
        
        # Max shutter speed 15ms
        self.T_max = 1.5
        
        # Camera gain and shutter speed for each image in the batch
        self.g = torch.max(torch.ones_like(self.exp) , self.exp/self.T_max)
        self.t = self.exp/self.g 
    
    def cal_dynamic_range(self, img, exp):
        """Calculate dynamic range from image statistic and exposure value.

        Args:
            img : image
            exp : exposure

        Returns:
            a,b : caculated dynamic range
        """
        mean_intensity = torch.mean(img, dim=[1,2,3], keepdim=True)
        std_intensity = torch.std(img, dim=[1,2,3], keepdim=True)
        
        a = mean_intensity - exp.view(-1,1,1,1) * std_intensity
        b = mean_intensity + exp.view(-1,1,1,1) * std_intensity
        
        return a,b
    
    def adjust_dr(self, img, exp):
                    
        # if not torch.is_tensor(exp):
        #     exp = torch.tensor(exp).to(img.device)
        
        a,b = self.cal_dynamic_range(img, exp)
        
        img = (exp * img - a) / (b - a)
        
        img = torch.clamp(img, 0, 1)
        
        return img
        
    def denormalized_image(self, img, exp):
        a, b = self.cal_dynamic_range(img, exp)
        img_denormalized = (b-a) * img + a
        img_denormalized = img_denormalized / exp
        
        return img_denormalized
        
    def noise_modeling(self, iso = float(100.)):
        """Add pre- post- noise and adjust dynamic range to simulate LDR captured image.

        Args:
            iso : Camera ISO. Defaults to float(100.).
        """
        # Validation check
        # if torch.any(self.phi < 0) or torch.any(self.phi > 1):
        #     raise ValueError("The image values should be between 0 and 1.")
        if iso <= 0:
            raise ValueError("ISO value should be positive.")
        
        iso_scale = iso / 100.0
        t_expanded = self.t.view(-1, 1, 1, 1)
        g_expanded = self.g.view(-1, 1, 1, 1)
        
        phi_scaled = self.phi * t_expanded * g_expanded * iso_scale
        
        noise_dark_scale = 10.0
        
        gauss_std = torch.sqrt(torch.tensor(self.gauss_var)).to(self.device) * (iso/100.0) * (1/t_expanded**2) * noise_dark_scale
        poisson_scale = torch.tensor(self.poisson_scale).to(self.device) * (iso/100.0) * (1/t_expanded**2) * noise_dark_scale
        
        # Shot noise
        shot_noise = torch.poisson(phi_scaled/poisson_scale) * poisson_scale * g_expanded
        # Readout noise
        readout_noise = gauss_std * torch.randn_like(phi_scaled) * g_expanded
        # ADC noise
        adc_noise = gauss_std * torch.randn_like(phi_scaled)
        
        noise_hdr = shot_noise + readout_noise + adc_noise
       
       # * Edit on 4/10
       # * Add quantization step, so claping bound is set to [0, 1]

        noise_ldr = torch.clamp(noise_hdr, 0, 1)
                
        return noise_ldr
    
    
    # * Calculate PSNR
    def calculate_psnr(original_image, processed_image, n = 8):
        """
        Calculate psnr between original image and processed image

        Parameters:
        original_image (numpy.ndarray)
        processed_image (numpy.ndarray): 

        Returns:
        float: calculated PSNR value (dB).
        """
        # MSE (Mean Squared Error)
        mse = torch.mean((original_image - processed_image) ** 2)
        
        if mse == 0:
            # MSE = 0 both orignal image and processed image are same
            return float('inf')
        
        # max pixel value ex) 8bit
        max_pixel = 2**n -1
        
        # PSNR
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))

        return psnr
