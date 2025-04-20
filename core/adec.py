import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models

DEVICE = 'cuda'
#  & Calculate histrogram by sub-images with different scale
def histogram_subimage(image, grid_size):
    """calculate histogram by Green Channel

    Args:
        image (tensor): batch of images [B, C, H, W]
        grid_size (int): grid size to divide original image

    Returns:
        list of tensors: histogram values based on green channel for each image in the batch
    """
    batch_size, _, height, width = image.shape
    
    grid_height, grid_width = height // grid_size, width // grid_size
    
    batch_histograms = []

    for b in range(batch_size):
        histograms = []
        for i in range(grid_size):
            for j in range(grid_size):
                # Green channel based
                sub_image_tensor = image[b, :, i*grid_height:(i+1)*grid_height, j*grid_width:(j+1)*grid_width]
                # histogram 
                hist_tensor = torch.histc(sub_image_tensor[1], bins= 256, min =0, max =1)
                histograms.append(hist_tensor)
        batch_histograms.append(histograms)
    
    return batch_histograms


# * Stack histograms that have different scales.
def stack_histogram(list_of_histogram):
    
    # Transpose the structure to: [[histogram_image1_from_batch1, histogram_image1_from_batch2, histogram_image1_from_batch3], ...]
    transposed_list = list(zip(*list_of_histogram))

    stacked_histograms = []
    for histograms_per_image in transposed_list:
        # Flatten the histograms for a single image
        merged_list = [item for sublist in histograms_per_image for item in sublist]
        stacked_histo_tensor = torch.stack(merged_list, dim=1)
        stacked_histograms.append(stacked_histo_tensor)

    # Stack all histograms for the entire batch
    batch_stacked_histograms = torch.stack(stacked_histograms, dim=0)

    return batch_stacked_histograms

# * Calculate left, right histograms with different scale.
def calculate_histograms(left_ldr_image, right_ldr_image):
    
    #^ Calculate histogram with multi scale
    histogram_coarest_l = histogram_subimage(left_ldr_image, 1)
    histogram_intermediate_l = histogram_subimage(left_ldr_image, 3)
    histogram_finest_l = histogram_subimage(left_ldr_image,7)
    
    histogram_coarest_r = histogram_subimage(right_ldr_image, 1)
    histogram_intermediate_r = histogram_subimage(right_ldr_image, 3)
    histogram_finest_r = histogram_subimage(right_ldr_image, 7)
    
    #^ Stack histogram [256,59]
    list_of_histograms_l = [histogram_coarest_l, histogram_intermediate_l, histogram_finest_l]
    stacked_histo_tensor_l = stack_histogram(list_of_histograms_l)

    list_of_histograms_r = [histogram_coarest_r, histogram_intermediate_r, histogram_finest_r]
    stacked_histo_tensor_r = stack_histogram(list_of_histograms_r)
    
    return stacked_histo_tensor_l, stacked_histo_tensor_r

def calculate_histograms2(left_ldr_image, right_ldr_image):
    
    #^ Calculate histogram with multi scale
    histogram_coarest_l = histogram_subimage(left_ldr_image, 1)
    histogram_intermediate_l = histogram_subimage(left_ldr_image, 3)
    histogram_finest_l = histogram_subimage(left_ldr_image,7)
    
    histogram_coarest_r = histogram_subimage(right_ldr_image, 1)
    histogram_intermediate_r = histogram_subimage(right_ldr_image, 3)
    histogram_finest_r = histogram_subimage(right_ldr_image, 7)
    
    #^ Stack histogram [256,118]
    list_of_histograms = [histogram_coarest_l, histogram_intermediate_l, histogram_finest_l,
                            histogram_coarest_r, histogram_intermediate_r, histogram_finest_r]
    
    stacked_histo_tensor = stack_histogram(list_of_histograms)
    
    return stacked_histo_tensor

# * Calculate histogram which is global scale.
def calculate_histogram_global(image):
    histo_global = histogram_subimage(image, 1)
    return histo_global[0]

# * Calculate skewness
def calculate_batch_skewness(batch_histograms):
    batch_skewness_level = []
    
    # device = batch_histograms[0][0].device
    for histogram in batch_histograms:
        frequencies = histogram
        

        pixel_values = torch.arange(len(frequencies), device=DEVICE)
        
        fixed_mean = 128
        skewness_numerator = torch.sum(frequencies * ((pixel_values - fixed_mean) **3))
        total_pixels = torch.sum(frequencies)
        skewness = skewness_numerator / total_pixels
        
        skewness /= ((255 - 0) ** 1.5)
        
        # Calculate mean skewness for combined multi-skewness value
        batch_skewness_level.append(skewness/255)
               
    return batch_skewness_level


# * Calculated combined skewness
def calculate_combined_skewness(image):
    grid_sizes = [1, 3, 7]
    all_skewness_values = []
    skewness_value_grid1 = []
    skewness_value_grid3 = []
    skewness_value_grid7 = []
    
    
    for grid_size in grid_sizes:
        batch_histograms = histogram_subimage(image, grid_size)
        for histograms in batch_histograms:
            skewness_values = calculate_batch_skewness(histograms)
            all_skewness_values.extend(skewness_values)
            
            if grid_size == 1:
                skewness_value_grid1.extend(skewness_values)
            elif grid_size == 3:
                skewness_value_grid3.extend(skewness_values)
            else:
                skewness_value_grid7.extend(skewness_values)
                
    #? Logging local skewness values
    combined_skewness = torch.mean(torch.tensor((all_skewness_values))) # len 59
    skewness_value_grid1 = torch.mean(torch.tensor(skewness_value_grid1))
    skewness_value_grid3 = torch.mean(torch.tensor(skewness_value_grid3))
    skewness_value_grid7 = torch.mean(torch.tensor(skewness_value_grid7))
    
    # print(combined_skewness, skewness_value_grid1, skewness_value_grid3, skewness_value_grid7)
    
    return combined_skewness, skewness_value_grid1, skewness_value_grid3, skewness_value_grid7

# * Calculated combined skewness
def calculate_combined_skewness_with_mask(image, mask):
    grid_sizes = [1, 3, 7]
    all_skewness_values = []
    skewness_value_grid1 = []
    skewness_value_grid3 = []
    skewness_value_grid7 = []
    
    # Apply mask to filter out unwanted region
    masked_image = image * mask.unsqueeze(0).unsqueeze(0)
    
    for grid_size in grid_sizes:
        batch_histograms = histogram_subimage(masked_image, grid_size)
        for histograms in batch_histograms:
            skewness_values = calculate_batch_skewness(histograms)
            all_skewness_values.extend(skewness_values)
            
            if grid_size == 1:
                skewness_value_grid1.extend(skewness_values)
            elif grid_size == 3:
                skewness_value_grid3.extend(skewness_values)
            else:
                skewness_value_grid7.extend(skewness_values)
                
    #? Logging local skewness values
    combined_skewness = torch.mean(torch.tensor((all_skewness_values))) # len 59
    skewness_value_grid1 = torch.mean(torch.tensor(skewness_value_grid1))
    skewness_value_grid3 = torch.mean(torch.tensor(skewness_value_grid3))
    skewness_value_grid7 = torch.mean(torch.tensor(skewness_value_grid7))
    
    print(combined_skewness, skewness_value_grid1, skewness_value_grid3, skewness_value_grid7)
    
    return combined_skewness, skewness_value_grid1, skewness_value_grid3, skewness_value_grid7

# * Calculate histogram sknewness by saturation mask threshold value
def calculate_exposure_adjustment(histogram, low_threshold=5, high_threshold=250):
    
    total_pixels = torch.sum(histogram)
    
    low_exposure_pixels = torch.sum(histogram[:low_threshold])
    low_exposure_ratio = low_exposure_pixels / total_pixels
    
    high_exposure_pixels = torch.sum(histogram[high_threshold:])
    high_exposure_ratio = high_exposure_pixels / total_pixels
    
    # print(f"Low exposure ratio : {low_exposure_ratio}")
    # print(f"High exposure ratio : {high_exposure_ratio}")

    return low_exposure_ratio, high_exposure_ratio

#^ Check histogram shape by pixel ratio
def is_bimodal(batch_histograms, low_threshold=0.1, high_threshold=0.9):
    bimodal_flags = []
    low_threshold = int(low_threshold*255)
    high_threshold = int(high_threshold*255)
    
    for histogram in batch_histograms:
        total_pixels = torch.sum(histogram)
        low_exposure_pixels  = torch.sum(histogram[:low_threshold])
        low_exposure_ratio = low_exposure_pixels / total_pixels
        high_exposure_pixels = torch.sum(histogram[high_threshold:])
        high_exposure_ratio = high_exposure_pixels / total_pixels
        
        if low_exposure_ratio > 0.1 and high_exposure_ratio > 0.1:
            bimodal_flags.append(True)
        else:
            bimodal_flags.append(False)
        
    return torch.tensor(bimodal_flags)

def clamping_ratio(batch_histograms, low_threshold=0.02, high_threshold=0.98):
    clamping_ratio_list = []
    low_threshold = int(low_threshold*255)
    high_threshold = int(high_threshold*255)
    
    for histogram in batch_histograms:
        total_pixels = torch.sum(histogram)
        low_exposure_pixels  = torch.sum(histogram[:low_threshold])
        low_exposure_ratio = low_exposure_pixels / total_pixels
        high_exposure_pixels = torch.sum(histogram[high_threshold:])
        high_exposure_ratio = high_exposure_pixels / total_pixels
        
        clamping_ratio_list.append((low_exposure_ratio, high_exposure_ratio))
        
    return torch.tensor(clamping_ratio_list)

def clamping_ratio_flag(batch_histograms, low_threshold=0.02, high_threshold=0.98):
    hdr_flag = []
    low_threshold = int(low_threshold*255)
    high_threshold = int(high_threshold*255)
    
    for histogram in batch_histograms:
        total_pixels = torch.sum(histogram)
        low_exposure_pixels  = torch.sum(histogram[:low_threshold])
        low_exposure_ratio = low_exposure_pixels / total_pixels
        high_exposure_pixels = torch.sum(histogram[high_threshold:])
        high_exposure_ratio = high_exposure_pixels / total_pixels
        
        if low_exposure_ratio > 0.1 and high_exposure_ratio > 0.1:
            hdr_flag.append(True)
        else:
            hdr_flag.append(False)
        
    return torch.tensor(hdr_flag)

def is_bimodal_print(batch_histograms, low_threshold=0.1, high_threshold=0.9):
    bimodal_flags = []
    low_threshold = int(low_threshold*255)
    high_threshold = int(high_threshold*255)
    
    for histogram in batch_histograms:
        total_pixels = torch.sum(histogram)
        low_exposure_pixels  = torch.sum(histogram[:low_threshold])
        low_exposure_ratio = low_exposure_pixels / total_pixels
        high_exposure_pixels = torch.sum(histogram[high_threshold:])
        high_exposure_ratio = high_exposure_pixels / total_pixels
        
        if low_exposure_ratio > 0.1 and high_exposure_ratio > 0.1:
            bimodal_flags.append(True)
        else:
            bimodal_flags.append(False)
        
    return torch.tensor(bimodal_flags)

# ^ Set saturation level based on skewness 
def calculate_batch_histogram_exposure(skewness_level, batch_histograms, symmetric_threshold = 0.1):
    
    batch_saturation_level = []
    
    # device = batch_histograms[0][0].device
    for skewness, histogram in zip(skewness_level, batch_histograms):
        histogram = histogram[0]
        # Under saturation
        if skewness < -symmetric_threshold:
            batch_saturation_level.append(torch.tensor(-1, device=DEVICE))
        # Over
        elif skewness > symmetric_threshold:
            batch_saturation_level.append(torch.tensor(1, device=DEVICE))
        # Symmetric
        # * divide into unimodal appro, bimodal 
        else:
            batch_saturation_level.append(torch.tensor(0, device=DEVICE))
        
    return batch_saturation_level

# * For batch histograms
def calculate_batch_exposure_adjustment(batch_histograms, low_threshold=5, high_threshold=250):
    batch_exp_saturation_level = []
    # -1 :under, +1 : over
    
    # [batch, list, bins = 256]
    # if grid == 1, list = 0
    for histogram in batch_histograms:
        low_exposure_ratio, high_exposure_ratio = calculate_exposure_adjustment(histogram[0], low_threshold, high_threshold)
        if low_exposure_ratio < high_exposure_ratio: # over saturation
            batch_exp_saturation_level.append(1)
        else: #under saturation
            batch_exp_saturation_level.append(-1)

    return torch.tensor(batch_exp_saturation_level).unsqueeze(-1)


def exposure_shift(exp, alpha = 0.1, decrease=False, maintain=False):
    if maintain:
        return exp
    if decrease:
        shifted_exp = exp - (alpha)
    else:
        shifted_exp = exp + (alpha)
        
    shifted_exp = torch.clamp(shifted_exp, min=0.1)
    
    return shifted_exp

def exposure_shift_with_clamp_ratio(exp, clamp_ratio, alpha = 0.1, decrease=False, maintain=False):
    if maintain:
        return exp
    if decrease:
        shifted_exp = exp - (alpha)
    else:
        shifted_exp = exp + (alpha)
        
    shifted_exp = torch.clamp(shifted_exp, min=0.1)
    
    return shifted_exp

def batch_exposure_adjustment(batch_low_ratios, batch_high_ratios, e_rand):
    adjusted_exposures = []

    for low_ratio, high_ratio in zip(batch_low_ratios, batch_high_ratios):
        if low_ratio < high_ratio:

            shifted_exp = exposure_shift(1, e_rand, high=True)
        else:

            shifted_exp = exposure_shift(1, e_rand, high=False)
        adjusted_exposures.append(shifted_exp)


    return torch.cat(adjusted_exposures, dim=0)
    
def exposure_shift_exp_diff(current_exp, exp_diff, alpha=0.1):
    next_exp = current_exp + alpha * exp_diff
    return next_exp

# Exposure shift with skewenss value, exposure shift in the direction where skewness approaches zero
def exposure_shift_skewness(current_exp, scene_skew, alpha=0.1):
    next_exp = current_exp - alpha * scene_skew
    return next_exp

def handle_ldr_scene_skewness(exp_f1, exp_f2, skewness_f1, skewness_f2, alpha = 0.1):
    new_exp_f1 = exposure_shift_skewness(exp_f1, skewness_f1, alpha)
    new_exp_f2 = exposure_shift_skewness(exp_f2, skewness_f2, alpha)

    return new_exp_f1, new_exp_f2

def clamping_ratio(batch_histograms, low_threshold=0.02, high_threshold=0.98):
    clamping_ratio_list = []
    low_threshold = int(low_threshold*255)
    high_threshold = int(high_threshold*255)

    for histogram in batch_histograms:
        total_pixels = torch.sum(histogram)
        low_exposure_pixels  = torch.sum(histogram[:low_threshold])
        low_exposure_ratio = low_exposure_pixels / total_pixels
        high_exposure_pixels = torch.sum(histogram[high_threshold:])
        high_exposure_ratio = high_exposure_pixels / total_pixels
        
        clamping_ratio_list.append((low_exposure_ratio, high_exposure_ratio))
        
    return torch.tensor(clamping_ratio_list)

def clamping_ratio_flag(batch_histograms, low_threshold=0.02, high_threshold=0.98):
    clamping_flags = []
    for ratio in clamping_ratio(batch_histograms, low_threshold, high_threshold):
        hdr_flag = (ratio[0] > 0.05 and ratio[1] > 0.05)
        clamping_flags.append(hdr_flag)
    return torch.tensor(clamping_flags)


def calculate_batch_skewness(batch_histograms):
    batch_skewness_level = []
    
    for histogram in batch_histograms:
        frequencies = histogram  
        
        pixel_values = torch.arange(len(frequencies), device=frequencies.device)
        
        fixed_mean = 128
        skewness_numerator = torch.sum(frequencies * ((pixel_values - fixed_mean) ** 3))
        total_pixels = torch.sum(frequencies)
        skewness = skewness_numerator / total_pixels
        
        variance = torch.sum(frequencies * ((pixel_values - fixed_mean) ** 2)) / total_pixels
        std_dev = torch.sqrt(variance)

        if std_dev != 0: 
            skewness /= std_dev ** 3
        else:
            skewness = torch.tensor(0.0, device=frequencies.device) 
        
        batch_skewness_level.append(skewness)
    
    return batch_skewness_level


def adjust_exposure_gap(exp_f1, exp_f2, clamping_ratio_f1, clamping_ratio_f2, alpha1, alpha2):
    
    if exp_f1 > exp_f2:
        # print("Getting wide exposure gap if exp_f1 > exp_f2")
        new_exp_f1 = exp_f1 + alpha1 * clamping_ratio_f1[0]
        new_exp_f2 = exp_f2 - alpha2 * clamping_ratio_f2[1]
        
        # new_exp_f1 = exposure_shift(exp_f1, alpha=alpha1, decrease=False)
        # new_exp_f2 = exposure_shift(exp_f2, alpha=alpha2, decrease=True) 
    else:
        # print("Getting wide exposure gap if exp_f1 < exp_f2")
        new_exp_f1 = exp_f1 - alpha1 * clamping_ratio_f1[1]
        new_exp_f2 = exp_f2 + alpha2 * clamping_ratio_f2[0]
        # new_exp_f1 = exposure_shift(exp_f1, alpha=alpha1, decrease=True)
        # new_exp_f2 = exposure_shift(exp_f2, alpha=alpha2, decrease=False) 
               
   
    return (new_exp_f1, new_exp_f2)
    

def stereo_exposure_control(batch_exp_f1, batch_exp_f2, batch_histograms_f1, batch_histograms_f2, alpha1, alpha2, exp_gap_threshold=2.0):
    device = batch_exp_f1.device 
    
    shifted_exp_f1 = torch.zeros_like(batch_exp_f1, device=device) 
    shifted_exp_f2 = torch.zeros_like(batch_exp_f2, device=device) 
    
    # Skewness 
    skewness_f1 = calculate_batch_skewness(batch_histograms_f1)
    skewness_f2 = calculate_batch_skewness(batch_histograms_f2)
    
    # Check scene dynamic range
    hdr_f1 = clamping_ratio_flag(batch_histograms_f1).to(device) 
    hdr_f2 = clamping_ratio_flag(batch_histograms_f2).to(device) 
    
    clamping_ratio_f1 = clamping_ratio(batch_histograms_f1).to(device) 
    clamping_ratio_f2 = clamping_ratio(batch_histograms_f2).to(device) 
    
    for i in range(batch_exp_f1.size(0)):
        exp_f1, exp_f2 = batch_exp_f1[i].to(device), batch_exp_f2[i].to(device) 
        a1, a2 = alpha1[i].to(device), alpha2[i].to(device) 
        skewness_diff = abs(skewness_f1[i] - skewness_f2[i])
        
        new_exp_f1 = exp_f1
        new_exp_f2 = exp_f2
        exp_diff = torch.abs(new_exp_f1 - new_exp_f2)

        if hdr_f1[i] or hdr_f2[i]:  # HDR scene, clamping
            if exp_diff < exp_gap_threshold:
                new_exp_f1, new_exp_f2 = adjust_exposure_gap(exp_f1, exp_f2, 
                                                             clamping_ratio_f1[i].to(device), 
                                                             clamping_ratio_f2[i].to(device), 
                                                             a1, a2)
        else:  # LDR scene, skewness 
            new_exp_f1, new_exp_f2 = handle_ldr_scene_skewness(exp_f1, exp_f2, 
                                                               skewness_f1[i], 
                                                               skewness_f2[i], 
                                                               alpha=0.1)

        shifted_exp_f1[i] = new_exp_f1
        shifted_exp_f2[i] = new_exp_f2
        
    return shifted_exp_f1, shifted_exp_f2



        
 