import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models

###############################################
# * Exposure control network and its functions
###############################################

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


# def stack_histogram(list_of_histogram):
    
#     assert len(list_of_histogram)==3, "len(list_of_histogram)!=3"
    
#     merged_list = [item for sublist in list_of_histogram for item in sublist]
#     stacked_histo_tensor = torch.stack(merged_list, dim = 1)
    
#     return stacked_histo_tensor

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

# * Create mask to calculate saturation level on object
def create_mask(height, width, exclude_top_fraction=0.2, exclude_bottom_fraction=0.2):
    """Create a mask to exclude top and bottom regions.

    Args:
        height (int): height of the image
        width (int): width of the image
        exclude_top_fraction (float): fraction of top region to exclude
        exclude_bottom_fraction (float): fraction of bottom region to exclude

    Returns:
        numpy array: mask with excluded regions
    """
    mask = np.ones((height, width), dtype=np.float32)
    top_exclude = int(height * exclude_top_fraction)
    bottom_exclude = int(height * exclude_bottom_fraction)
    
    mask[:top_exclude, :] = 0
    mask[-bottom_exclude:, :] = 0
    mask_tensor = torch.tensor(mask, dtype=torch.float32)
    
    return mask_tensor

# #* Apply mask to image to filter out unwanted regions
# def apply_mask(image, mask):
#     """Apply mask to image to filter out unwanted regions.

#     Args:
#         image (tensor): batch of images [B, C, H, W]
#         mask (numpy array): mask to apply [H, W]

#     Returns:
#         tensor: masked image
#     """
#     masked_image = image * mask.unsqueeze(0).unsqueeze(0)
#     return masked_image

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

# Todo) calculate batch histogram saturation level based on skewness value
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


# Todo_04.23) Modify eposure adjustment by 3x3 criteria
# & Add alpha parameter to test alpha prediction network

# def batch_exp_adjustment(batch_exp_f1, batch_exp_f2, batch_saturation_level_f1, batch_saturation_level_f2, alpha1, alpha2):
#     shifted_exp_f1 = torch.zeros_like(batch_exp_f1)
#     shifted_exp_f2 = torch.zeros_like(batch_exp_f2)
    
#     for i in range(batch_exp_f1.size(0)):
        
#         exp_f1, exp_f2 = batch_exp_f1[i], batch_exp_f2[i]
#         sat_f1, sat_f2 = batch_saturation_level_f1[i], batch_saturation_level_f2[i]
#         a1 = alpha1[i]
#         a2 = alpha2[i]
        
#         # e_high,under   e_low,under case both increase
#         if sat_f1 < -0.5 and sat_f2 < -0.5:
#             shifted_exp_f1[i] = exposure_shift(exp_f1, alpha=a1, decrease=False)
#             shifted_exp_f2[i] = exposure_shift(exp_f2, alpha=a2, decrease=False)
            
#         # e_high,appro   e_low,under case both increase
#         elif -0.5<= sat_f1 <= 0.5 and sat_f2 < -0.5 and exp_f1 > exp_f2:
#             shifted_exp_f1[i] = exposure_shift(exp_f1, alpha=a1, decrease=False)
#             shifted_exp_f2[i] = exposure_shift(exp_f2, alpha=a2, decrease=False)
#         elif sat_f1 <-0.5 and -0.5<= sat_f2 <= 0.5 and exp_f1 < exp_f2:
#             shifted_exp_f1[i] = exposure_shift(exp_f1, alpha=a1, decrease=False)
#             shifted_exp_f2[i] = exposure_shift(exp_f2, alpha=a2, decrease=False) 
            
#         # e_high,appro  e_low,appro case
#         elif -0.5<= sat_f1 <= 0.5 and -0.5<=sat_f2<=0.5:
#             if exp_f1 > exp_f2: # exp_f1 is high exposure case, increase high, decrease low
#                 shifted_exp_f1[i] = exposure_shift(exp_f1, alpha=a1, decrease=False)
#                 shifted_exp_f2[i] = exposure_shift(exp_f2, alpha=a2, decrease=True)
#             else: # exp_f1 is low exposure case
#                 shifted_exp_f1[i] = exposure_shift(exp_f1, alpha=a1, decrease=True)
#                 shifted_exp_f2[i] = exposure_shift(exp_f2, alpha=a2, decrease=False)
                
#         # e_high,over  e_low,under case, increase low, decrease high
#         elif sat_f1 > 0.5 and sat_f2 < -0.5 and exp_f1 > exp_f2:
#             shifted_exp_f1[i] = exposure_shift(exp_f1, alpha=a1, decrease=True)
#             shifted_exp_f2[i] = exposure_shift(exp_f2, alpha=a2, decrease=False)
#         elif sat_f1 < -0.5 and sat_f2 > 0.5 and exp_f1 < exp_f2:
#             shifted_exp_f1[i] = exposure_shift(exp_f1, alpha=a1, decrease=False)
#             shifted_exp_f2[i] = exposure_shift(exp_f2, alpha=a2, decrease=True)
                
#         # e_high,over  e_low,appro, both decrease
#         elif sat_f1 > 0.5 and -0.5<=sat_f2<=0.5 and exp_f1 > exp_f2:
#             shifted_exp_f1[i] = exposure_shift(exp_f1, alpha=a1, decrease=True)
#             shifted_exp_f2[i] = exposure_shift(exp_f2, alpha=a2, decrease=True)
#         elif -0.5<=sat_f1<=0.5 and sat_f2 >0.5 and exp_f1 < exp_f2:
#             shifted_exp_f1[i] = exposure_shift(exp_f1, alpha=a1, decrease=True)
#             shifted_exp_f2[i] = exposure_shift(exp_f2, alpha=a2, decrease=True)
                    
#         # e_high,over  e_low,over, both decrease
#         elif sat_f1 > 0.5 and sat_f2 > 0.5:
#             shifted_exp_f1[i] = exposure_shift(exp_f1, decrease=True, alpha=a1)
#             shifted_exp_f2[i] = exposure_shift(exp_f2, decrease=True, alpha=a2)
            
#         else: # Not valid case
#             print("Not valid case")
#             shifted_exp_f1[i] = exp_f1
#             shifted_exp_f2[i] = exp_f2

#     return shifted_exp_f1, shifted_exp_f2

# Todo) 240627 Edit based on histogram shape.
def batch_exp_adjustment(batch_exp_f1, batch_exp_f2, batch_saturation_level_f1, batch_saturation_level_f2, batch_histograms_f1, batch_histograms_f2, alpha1, alpha2):
    shifted_exp_f1 = torch.zeros_like(batch_exp_f1)
    shifted_exp_f2 = torch.zeros_like(batch_exp_f2)
    
    bimodal_f1 = is_bimodal(batch_histograms_f1)
    bimodal_f2 = is_bimodal(batch_histograms_f2)
    
    for i in range(batch_exp_f1.size(0)):
        exp_f1, exp_f2 = batch_exp_f1[i], batch_exp_f2[i]
        sat_f1, sat_f2 = batch_saturation_level_f1[i], batch_saturation_level_f2[i]
        a1 = alpha1[i]
        a2 = alpha2[i]
        
        if bimodal_f1[i] or bimodal_f2[i]:
            # Handle bimodal cases specifically
            if bimodal_f1[i] and bimodal_f2[i]:
                # Both are bimodal
                if exp_f1 > exp_f2:
                    shifted_exp_f1[i] = exposure_shift(exp_f1, alpha=a1, decrease=True)  # Lower the high exposure
                    shifted_exp_f2[i] = exposure_shift(exp_f2, alpha=a2, decrease=False)  # Increase the low exposure
                else:
                    shifted_exp_f1[i] = exposure_shift(exp_f1, alpha=a1, decrease=False)
                    shifted_exp_f2[i] = exposure_shift(exp_f2, alpha=a2, decrease=True)
            elif bimodal_f1[i]:
                # Only f1 is bimodal
                if sat_f2 <= 0 and sat_f1 <=0:
                    # f2 is under
                    shifted_exp_f1[i] = exposure_shift(exp_f1, alpha=a1, decrease=False)
                    shifted_exp_f2[i] = exposure_shift(exp_f2, alpha=a2, decrease=False)
                elif sat_f2 <=0 and sat_f1 >0:
                    shifted_exp_f1[i] = exposure_shift(exp_f1, alpha=a1, decrease=True)
                    shifted_exp_f2[i] = exposure_shift(exp_f2, alpha=a2, decrease=False)
                elif sat_f2 >0 and sat_f1 >0:
                    shifted_exp_f1[i] = exposure_shift(exp_f1, alpha=a1, decrease=True)
                    shifted_exp_f2[i] = exposure_shift(exp_f2, alpha=a2, decrease=True)
                elif sat_f2 > 0 and sat_f1 <0:
                    # f2 is over
                    shifted_exp_f1[i] = exposure_shift(exp_f1, alpha=a1, decrease=False)
                    shifted_exp_f2[i] = exposure_shift(exp_f2, alpha=a2, decrease=True)
            elif bimodal_f2[i]:
                # Only f2 is bimodal
                if sat_f1 <=0 and sat_f2 <=0:
                    # f1 is under
                    shifted_exp_f1[i] = exposure_shift(exp_f1, alpha=a1, decrease=False)
                    shifted_exp_f2[i] = exposure_shift(exp_f2, alpha=a2, decrease=False)
                elif sat_f1 <=0 and sat_f2>0:
                    shifted_exp_f1[i] = exposure_shift(exp_f1, alpha=a1, decrease=False)
                    shifted_exp_f2[i] = exposure_shift(exp_f2, alpha=a2, decrease=True)
                elif sat_f1 > 0 and sat_f2 > 0:
                    shifted_exp_f1[i] = exposure_shift(exp_f1, alpha=a1, decrease=True)
                    shifted_exp_f2[i] = exposure_shift(exp_f2, alpha=a2, decrease=True)
                elif sat_f1 > 0 and sat_f2 <0:
                    # f1 is over
                    shifted_exp_f1[i] = exposure_shift(exp_f1, alpha=a1, decrease=True)
                    shifted_exp_f2[i] = exposure_shift(exp_f2, alpha=a2, decrease=False)
        
                    shifted_exp_f2[i] = exposure_shift(exp_f2, decrease=True, alpha=a2)
                else:
                    print("Not valid case")
                    shifted_exp_f1[i] = exp_f1
                    shifted_exp_f2[i] = exp_f2

        return shifted_exp_f1, shifted_exp_f2

def batch_exp_adjustment2(batch_exp_f1, batch_exp_f2, batch_histograms_f1, batch_histograms_f2, alpha1, alpha2, min_skew_diff=0.2, max_skew_diff=1.2):
    shifted_exp_f1 = torch.zeros_like(batch_exp_f1)
    shifted_exp_f2 = torch.zeros_like(batch_exp_f2)
    
    #* Calculate Skewness based on histogram (Global histgoram)
    # Todo) Global + local histgoram
    skewness_f1 = calculate_batch_skewness(batch_histograms_f1)
    skewness_f2 = calculate_batch_skewness(batch_histograms_f2)
    
    #* Calculate saturation level based on skewnesss value
    batch_saturation_level_f1 = calculate_batch_histogram_exposure(skewness_f1, batch_histograms_f1)
    batch_saturation_level_f2 = calculate_batch_histogram_exposure(skewness_f2, batch_histograms_f2)
    
    #* Check hdr scene based on histgoram shape
    bimodal_f1 = is_bimodal(batch_histograms_f1)
    bimodal_f2 = is_bimodal(batch_histograms_f2)
    
    for i in range(batch_exp_f1.size(0)):
        exp_f1, exp_f2 = batch_exp_f1[i], batch_exp_f2[i]
        sat_f1, sat_f2 = batch_saturation_level_f1[i], batch_saturation_level_f2[i]
        a1 = alpha1[i]
        a2 = alpha2[i]
        
        if bimodal_f1[i] or bimodal_f2[i]:
            # HDR scene
            if bimodal_f1[i] and bimodal_f2[i]:
                # Frame 1 and 2 bi-modal
                if exp_f1 > exp_f2:
                    print("Case 1")
                    new_exp_f1 = exposure_shift(exp_f1, alpha=a1, decrease=False)
                    new_exp_f2 = exposure_shift(exp_f2, alpha=a2, decrease=True)
                else:
                    print("Case 2")
                    new_exp_f1 = exposure_shift(exp_f1, alpha=a1, decrease=True)
                    new_exp_f2 = exposure_shift(exp_f2, alpha=a2, decrease=False)
            elif bimodal_f1[i]:
                # Only f1 is bimodal
                if sat_f2 <= 0 and sat_f1 <=0:
                    # f2 is under
                    print("Case 3")
                    new_exp_f1 = exposure_shift(exp_f1, alpha=a1, decrease=False)
                    new_exp_f2 = exposure_shift(exp_f2, alpha=a2, decrease=False)
                elif sat_f2 <=0 and sat_f1 >0:
                    print("Case 4")
                    new_exp_f1 = exposure_shift(exp_f1, alpha=a1, decrease=True)
                    new_exp_f2 = exposure_shift(exp_f2, alpha=a2, decrease=False)
                elif sat_f2 >0 and sat_f1 >0:
                    print("Case 5")
                    new_exp_f1 = exposure_shift(exp_f1, alpha=a1, decrease=True)
                    new_exp_f2 = exposure_shift(exp_f2, alpha=a2, decrease=True)
                elif sat_f2 > 0 and sat_f1 <0:
                    print("Case 6")
                    # f2 is over
                    new_exp_f1 = exposure_shift(exp_f1, alpha=a1, decrease=False)
                    new_exp_f2 = exposure_shift(exp_f2, alpha=a2, decrease=True)
                else:
                    print("Not valid in Case 6")
            elif bimodal_f2[i]:
                # Only f2 is bimodal
                if sat_f1 <=0 and sat_f2 <=0:
                    # f1 is under
                    print("Case 7")
                    new_exp_f1 = exposure_shift(exp_f1, alpha=a1, decrease=False)
                    new_exp_f2 = exposure_shift(exp_f2, alpha=a2, decrease=False)
                elif sat_f1 <=0 and sat_f2>0:
                    print("Case 8")
                    new_exp_f1 = exposure_shift(exp_f1, alpha=a1, decrease=False)
                    new_exp_f2 = exposure_shift(exp_f2, alpha=a2, decrease=True)
                elif sat_f1 > 0 and sat_f2 > 0:
                    print("Case 9")
                    new_exp_f1 = exposure_shift(exp_f1, alpha=a1, decrease=True)
                    new_exp_f2 = exposure_shift(exp_f2, alpha=a2, decrease=True)
                elif sat_f1 > 0 and sat_f2 <0:
                    # f1 is over
                    print("Case 10")
                    print(bimodal_f1[i], bimodal_f2[i])
                    new_exp_f1 = exposure_shift(exp_f1, alpha=a1, decrease=True)
                    new_exp_f2 = exposure_shift(exp_f2, alpha=a2, decrease=False)
                else:
                    print("Not valid in Case10")
            else:
                print("Not valid bimodal case")
            
            # Adjust to ensure skewness difference is within desired range
            new_skew_f1 = calculate_batch_skewness([batch_histograms_f1[i]])[0]
            new_skew_f2 = calculate_batch_skewness([batch_histograms_f2[i]])[0]
            skew_diff = abs(new_skew_f1 - new_skew_f2)
            print(f"skewness_diff : {skew_diff}")
            
            if skew_diff < min_skew_diff:
                if new_skew_f1 > new_skew_f2:
                    print("Case 11")
                    new_exp_f1 = exposure_shift(new_exp_f1, alpha=a1, decrease=False)
                    new_exp_f2 = exposure_shift(new_exp_f2, alpha=a2, decrease=True)
                else:
                    print("Case 12")
                    new_exp_f1 = exposure_shift(new_exp_f1, alpha=a1, decrease=True)
                    new_exp_f2 = exposure_shift(new_exp_f2, alpha=a2, decrease=False)
            else:
                print(f"min_skew_diff not valid")
            if skew_diff > max_skew_diff:
                if new_skew_f1 > new_skew_f2:
                    print("Case 13")
                    new_exp_f1 = exposure_shift(new_exp_f1, alpha=a1, decrease=True)
                    new_exp_f2 = exposure_shift(new_exp_f2, alpha=a2, decrease=False)
                else:
                    print("Case 14")
                    new_exp_f1 = exposure_shift(new_exp_f1, alpha=a1, decrease=False)
                    new_exp_f2 = exposure_shift(new_exp_f2, alpha=a2, decrease=True)
            else:
                print(f"max_skew_diff not valid")
            
        else:
            # LDR scene
            if sat_f1 < 0 and sat_f2 < 0:
                print("Case 15")
                new_exp_f1 = exposure_shift(exp_f1, alpha=a1, decrease=False)
                new_exp_f2 = exposure_shift(exp_f2, alpha=a2, decrease=False)
            elif sat_f1 == 0 and sat_f2 < 0 and exp_f1 > exp_f2:
                print("Case 16")
                new_exp_f1 = exposure_shift(exp_f1, alpha=a1, decrease=False)
                new_exp_f2 = exposure_shift(exp_f2, alpha=a2, decrease=False)
            elif sat_f1 < 0 and sat_f2 == 0 and exp_f1 < exp_f2:
                print("Case 17")
                new_exp_f1 = exposure_shift(exp_f1, alpha=a1, decrease=False)
                new_exp_f2 = exposure_shift(exp_f2, alpha=a2, decrease=False)
            elif sat_f1 == 0 and sat_f2 == 0:
                if exp_f1 > exp_f2:
                    print("Case 18")
                    new_exp_f1 = exposure_shift(exp_f1, alpha=a1, decrease=False)
                    new_exp_f2 = exposure_shift(exp_f2, alpha=a2, decrease=True)
                else:
                    print("Case 19")
                    new_exp_f1 = exposure_shift(exp_f1, alpha=a1, decrease=True)
                    new_exp_f2 = exposure_shift(exp_f2, alpha=a2, decrease=False)
            elif sat_f1 > 0 and sat_f2 < 0 and exp_f1 > exp_f2:
                print("Case 20")
                new_exp_f1 = exposure_shift(exp_f1, alpha=a1, decrease=True)
                new_exp_f2 = exposure_shift(exp_f2, alpha=a2, decrease=False)
            elif sat_f1 < 0 and sat_f2 > 0 and exp_f1 < exp_f2:
                print("Case 21")
                new_exp_f1 = exposure_shift(exp_f1, alpha=a1, decrease=False)
                new_exp_f2 = exposure_shift(exp_f2, alpha=a2, decrease=True)
            elif sat_f1 > 0 and sat_f2 == 0 and exp_f1 > exp_f2:
                print("Case 22")
                new_exp_f1 = exposure_shift(exp_f1, alpha=a1, decrease=True)
                new_exp_f2 = exposure_shift(exp_f2, alpha=a2, decrease=True)
            elif sat_f1 == 0 and sat_f2 > 0 and exp_f1 < exp_f2:
                print("Case 23")
                new_exp_f1 = exposure_shift(exp_f1, alpha=a1, decrease=True)
                new_exp_f2 = exposure_shift(exp_f2, alpha=a2, decrease=True)
            elif sat_f1 > 0 and sat_f2 > 0:
                print("Case 24")
                new_exp_f1 = exposure_shift(exp_f1, decrease=True, alpha=a1)
                new_exp_f2 = exposure_shift(exp_f2, decrease=True, alpha=a2)
            else:
                print("Not valid case")
                new_exp_f1 = exp_f1
                new_exp_f2 = exp_f2

        shifted_exp_f1[i] = new_exp_f1
        shifted_exp_f2[i] = new_exp_f2

    return shifted_exp_f1, shifted_exp_f2

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

def exposure_shift_smooth(exp, alpha, prev_exp=None, decrease=False, maintain=False, smoothing_factor=0.8):
    if maintain:
        return exp
    
    # 기본적인 노출 조정
    if decrease:
        shifted_exp = exp - (alpha * exp)
    else:
        shifted_exp = exp + (alpha * exp)
    
    # 이전 노출 값(prev_exp)이 제공되면, 부드러운 전환을 위해 가중 평균을 계산
    if prev_exp is not None:
        # 이동 평균 또는 평활화를 위해 이전 값과 현재 계산된 값을 결합
        shifted_exp = (smoothing_factor * prev_exp) + ((1 - smoothing_factor) * shifted_exp)
    
    # 노출 값이 너무 낮아지지 않도록 클램핑
    shifted_exp = torch.clamp(shifted_exp, min=0.1)
    
    return shifted_exp


def exposure_shift_batch(batch_size, exp, alpha=0.3, high=True):
    exp_list = []
    
    if high:
        for _ in range(batch_size):
            shifted_exp = exp - (alpha * exp)
            exp_list.append(shifted_exp)
    else:
        for _ in range(batch_size):
            shifted_exp = exp + (alpha * exp)
            exp_list.append(shifted_exp)

    return torch.tensor(exp_list).unsqueeze(-1)  # 차원 추가로 리스트 형태의 텐서 반환

def batch_exposure_adjustment(batch_low_ratios, batch_high_ratios, e_rand):
    adjusted_exposures = []

    for low_ratio, high_ratio in zip(batch_low_ratios, batch_high_ratios):
        if low_ratio < high_ratio:
            # 고조도 비율이 더 높은 경우, 노출을 감소
            shifted_exp = exposure_shift(1, e_rand, high=True)
        else:
            # 저조도 비율이 더 높은 경우, 노출을 증가
            shifted_exp = exposure_shift(1, e_rand, high=False)
        adjusted_exposures.append(shifted_exp)

    # 배치 단위로 조정된 노출 값들을 텐서로 반환
    return torch.cat(adjusted_exposures, dim=0)


# & Show historgram by different grid_size
def show_histogram(histograms, grid_size):
    fig, axs = plt.subplots(grid_size, grid_size, figsize = (12,12))
    
    for i in range(grid_size):
        for j in range(grid_size):
            if grid_size == 1:
                axs.bar(range(256), histograms[i*grid_size + j].numpy())
                axs.set_title(f"Sub-image hist [{i},{j}]")
            else:
                axs[i,j].bar(range(256), histograms[i*grid_size + j].numpy())
                axs[i,j].set_title(f"Sub-image hist [{i},{j}]")
    
    plt.tight_layout()
    plt.show()

#^ Rule-based exposure control
def stereo_expsoure_control(batch_exp_f1, batch_exp_f2, batch_histograms_f1, batch_histograms_f2, alpha1, alpha2, max_exposure_gap=2.5):
    # print(f"=====Initial Exposure value F1 : {batch_exp_f1}, F2 : {batch_exp_f2}=====")
    shifted_exp_f1 = torch.zeros_like(batch_exp_f1)
    shifted_exp_f2 = torch.zeros_like(batch_exp_f2)
    
    #* Calculate Skewness and saturation
    skewness_f1 = calculate_batch_skewness(batch_histograms_f1)
    skewness_f2 = calculate_batch_skewness(batch_histograms_f2)
    batch_saturation_level_f1 = calculate_batch_histogram_exposure(skewness_f1, batch_histograms_f1)
    batch_saturation_level_f2 = calculate_batch_histogram_exposure(skewness_f2, batch_histograms_f2)
    
    #* Check HDR scene based on histogram shape
    bimodal_f1 = is_bimodal(batch_histograms_f1)
    bimodal_f2 = is_bimodal(batch_histograms_f2)
    
    for i in range(batch_exp_f1.size(0)):
        exp_f1, exp_f2 = batch_exp_f1[i], batch_exp_f2[i]
        sat_f1, sat_f2 = batch_saturation_level_f1[i], batch_saturation_level_f2[i]
        a1, a2 = alpha1[i], alpha2[i]
        
        # HDR scene - bimodal
        if bimodal_f1[i] or bimodal_f2[i]:
            print(f"Bimodal : F1 : {bimodal_f1} F2 : {bimodal_f2}")
            if bimodal_f1[i] and bimodal_f2[i]:
                # 둘 다 bimodal인 경우
                if exp_f1 > exp_f2:
                    new_exp_f1 = exposure_shift(exp_f1, alpha=a1, decrease=False)
                    new_exp_f2 = exposure_shift(exp_f2, alpha=a2, decrease=True)
                else:
                    new_exp_f1 = exposure_shift(exp_f1, alpha=a1, decrease=True)
                    new_exp_f2 = exposure_shift(exp_f2, alpha=a2, decrease=False)
            elif bimodal_f1[i]:
                # f1만 bimodal인 경우, f1만 조정
                if exp_f1 > exp_f2:
                    new_exp_f1 = exposure_shift(exp_f1, alpha=a1, decrease=False)
                else:
                    new_exp_f1 = exposure_shift(exp_f1, alpha=a1, decrease=True)
                new_exp_f2 = exp_f2  # f2는 유지
            elif bimodal_f2[i]:
                # f2만 bimodal인 경우, f2만 조정
                if exp_f1 > exp_f2:
                    new_exp_f2 = exposure_shift(exp_f2, alpha=a2, decrease=True)
                else:
                    new_exp_f2 = exposure_shift(exp_f2, alpha=a2, decrease=False)
                new_exp_f1 = exp_f1  # f1은 유지

        # LDR scene - unimodal
        else:
            new_exp_f1, new_exp_f2 = handle_unimodal_scene(exp_f1, exp_f2, sat_f1, sat_f2, a1, a2)

        # Control max exposure gap
        # Control max exposure ratio (instead of max exposure gap)
        exposure_ratio = max(new_exp_f1, new_exp_f2) / min(new_exp_f1, new_exp_f2)
        if exposure_ratio > max_exposure_gap:
            adjustment_ratio = (exposure_ratio - max_exposure_gap) / 2
            if new_exp_f1 > new_exp_f2:
                new_exp_f1 /= (1 + adjustment_ratio)
                new_exp_f2 *= (1 + adjustment_ratio)
            else:
                new_exp_f1 *= (1 + adjustment_ratio)
                new_exp_f2 /= (1 + adjustment_ratio)

        shifted_exp_f1[i] = new_exp_f1
        shifted_exp_f2[i] = new_exp_f2


    # print(f"=====Adjsuted Exposure value F1 : {shifted_exp_f1}, F2 : {shifted_exp_f2}=====")
    return shifted_exp_f1, shifted_exp_f2

def handle_unimodal_scene(exp_f1, exp_f2, sat_f1, sat_f2, a1, a2):
    if sat_f1 < 0 and sat_f2 < 0: # Both under saturation
        print("Both under saturation")
        return exposure_shift(exp_f1, alpha=a1, decrease=False), exposure_shift(exp_f2, alpha=a2, decrease=False)
    elif sat_f1 > 0 and sat_f2 < 0 and exp_f1 > exp_f2: # Frame1 over, Frame2 under
        print("Frame1 is over, Frame2 is under")
        return exposure_shift(exp_f1, alpha=a1, decrease=True), exposure_shift(exp_f2, alpha=a2, decrease=False)
    elif sat_f1 < 0 and sat_f2 > 0 and exp_f1 < exp_f2: # Frame1 under, Frame2 over
        print("Frame1 is under, Frame2 is over")
        return exposure_shift(exp_f1, alpha=a1, decrease=False), exposure_shift(exp_f2, alpha=a2, decrease=True)
    elif sat_f1 > 0 and sat_f2 > 0: # Both over saturation
        print("Both over saturation")
        return exposure_shift(exp_f1, alpha=a1, decrease=True), exposure_shift(exp_f2, alpha=a2, decrease=True)
    else: # Both symmetric or similar exposure
        print("Both appro saturation")
        return exposure_shift(exp_f1, alpha=a1 * 0.1, decrease=False), exposure_shift(exp_f2, alpha=a2 * 0.1, decrease=True)
        # print("Both appro saturation, reducing gap")
        # exp_gap = abs(exp_f1 - exp_f2)
        # # 노출 간격을 줄이기 위한 전략
        # reduce_factor = exp_gap * 0.1  # 10% 정도 간격을 줄임
        # if exp_f1 > exp_f2:
        #     return exp_f1 - reduce_factor, exp_f2 + reduce_factor
        # else:
        #     return exp_f1 + reduce_factor, exp_f2 - reduce_factor



def stereo_exposure_control2(batch_exp_f1, batch_exp_f2, batch_histograms_f1, batch_histograms_f2, alpha1, alpha2, max_exposure_gap=2.5, min_exposure_gap=1.0):
    shifted_exp_f1 = torch.zeros_like(batch_exp_f1)
    shifted_exp_f2 = torch.zeros_like(batch_exp_f2)
    
    #* Calculate Skewness and saturation
    skewness_f1 = calculate_batch_skewness(batch_histograms_f1)
    skewness_f2 = calculate_batch_skewness(batch_histograms_f2)
    batch_saturation_level_f1 = calculate_batch_histogram_exposure(skewness_f1, batch_histograms_f1)
    batch_saturation_level_f2 = calculate_batch_histogram_exposure(skewness_f2, batch_histograms_f2)
    
    #* Check HDR scene based on histogram shape
    bimodal_f1 = is_bimodal(batch_histograms_f1)
    bimodal_f2 = is_bimodal(batch_histograms_f2)
    
    for i in range(batch_exp_f1.size(0)):
        exp_f1, exp_f2 = batch_exp_f1[i], batch_exp_f2[i]
        sat_f1, sat_f2 = batch_saturation_level_f1[i], batch_saturation_level_f2[i]
        a1, a2 = alpha1[i], alpha2[i]
        
        # 1. **LDR 상황**: 노출값을 점진적으로 증가시켜 HDR 전환을 유도
        if not bimodal_f1[i] and not bimodal_f2[i]:
            skewness_diff = abs(skewness_f1[i] - skewness_f2[i])
            if skewness_diff < 1:
                # LDR 상황에서는 둘 다 under 또는 over이므로 노출을 조금씩 증가/감소
                if sat_f1 < 0 and sat_f2 < 0:  # 둘 다 under saturation
                    print("Both under saturation in LDR, increasing exposure")
                    new_exp_f1 = exposure_shift_smooth(exp_f1, alpha=a1, decrease=False)
                    new_exp_f2 = exposure_shift_smooth(exp_f2, alpha=a2, decrease=False)
                elif sat_f1 > 0 and sat_f2 > 0:  # 둘 다 over saturation
                    print("Both over saturation in LDR, decreasing exposure")
                    new_exp_f1 = exposure_shift_smooth(exp_f1, alpha=a1, decrease=True)
                    new_exp_f2 = exposure_shift_smooth(exp_f2, alpha=a2, decrease=True)
                else:
                    # 하나는 over, 하나는 under일 때
                    if sat_f1 < 0 and sat_f2 > 0:
                        new_exp_f1 = exposure_shift_smooth(exp_f1, alpha=a1, prev_exp=exp_f1, decrease=False)
                        new_exp_f2 = exposure_shift_smooth(exp_f2, alpha=a2, prev_exp=exp_f2,  decrease=True)
                    elif sat_f1 > 0 and sat_f2 < 0:
                        new_exp_f1 = exposure_shift_smooth(exp_f1, alpha=a1, prev_exp=exp_f1, decrease=True)
                        new_exp_f2 = exposure_shift_smooth(exp_f2, alpha=a2, prev_exp=exp_f2, decrease=False)

            else:
                # LDR이지만 노출값 차이가 이미 벌어졌을 경우 처리 (간격 줄이기)
                print("Both unimdal, but skewness_diff is larger than threshold")
                exposure_gap = abs(exp_f1 - exp_f2)
                if exposure_gap > min_exposure_gap:  # 최소 간격을 유지한 상태에서 줄이기
                    reduction_factor = 0.1 * exposure_gap  # 10%씩 간격을 줄임
                    if exp_f1 > exp_f2:
                        new_exp_f1 = exp_f1 - reduction_factor
                        new_exp_f2 = exp_f2 + reduction_factor
                    else:
                        new_exp_f1 = exp_f1 + reduction_factor
                        new_exp_f2 = exp_f2 - reduction_factor

        # 2. **HDR 상황 (bimodal)**: 노출 간격을 벌림
        elif bimodal_f1[i] or bimodal_f2[i]:
            print(f"Bimodal : F1 : {bimodal_f1} F2 : {bimodal_f2}")
            if bimodal_f1[i] and bimodal_f2[i]:
                if exp_f1 > exp_f2:
                    new_exp_f1 = exposure_shift_smooth(exp_f1, alpha=a1, prev_exp=exp_f1, decrease=False)
                    new_exp_f2 = exposure_shift_smooth(exp_f2, alpha=a2, prev_exp=exp_f2, decrease=True)
                else:
                    new_exp_f1 = exposure_shift_smooth(exp_f1,alpha=a1, prev_exp=exp_f1, decrease=True)
                    new_exp_f2 = exposure_shift_smooth(exp_f2, alpha=a2, prev_exp=exp_f2, decrease=False)
            elif bimodal_f1[i]:
                if exp_f1 > exp_f2:
                    new_exp_f1 = exposure_shift_smooth(exp_f1, alpha=a1, prev_exp=exp_f1, decrease=False)
                else:
                    new_exp_f1 = exposure_shift_smooth(exp_f1, alpha=a1, prev_exp=exp_f1, decrease=True)
                new_exp_f2 = exp_f2  # f2는 유지
            elif bimodal_f2[i]:
                if exp_f1 > exp_f2:
                    new_exp_f2 = exposure_shift_smooth(exp_f2, alpha=a2, prev_exp=exp_f2, decrease=True)
                else:
                    new_exp_f2 = exposure_shift_smooth(exp_f2, alpha=a2, prev_exp=exp_f2, decrease=False)
                new_exp_f1 = exp_f1  # f1은 유지

        # 3. max_exposure_gap 유지
        exposure_ratio = max(new_exp_f1, new_exp_f2) / min(new_exp_f1, new_exp_f2)
        if exposure_ratio > max_exposure_gap:
            adjustment_ratio = (exposure_ratio - max_exposure_gap) / 2
            if new_exp_f1 > new_exp_f2:
                new_exp_f1 /= (1 + adjustment_ratio)
                new_exp_f2 *= (1 + adjustment_ratio)
            else:
                new_exp_f1 *= (1 + adjustment_ratio)
                new_exp_f2 /= (1 + adjustment_ratio)

        shifted_exp_f1[i] = new_exp_f1
        shifted_exp_f2[i] = new_exp_f2

    return shifted_exp_f1, shifted_exp_f2


#^ Rule-based exposure control
def stereo_expsoure_control3(batch_exp_f1, batch_exp_f2, batch_histograms_f1, batch_histograms_f2, alpha1, alpha2, max_exposure_gap=2.5):
    # print(f"=====Initial Exposure value F1 : {batch_exp_f1}, F2 : {batch_exp_f2}=====")
    shifted_exp_f1 = torch.zeros_like(batch_exp_f1)
    shifted_exp_f2 = torch.zeros_like(batch_exp_f2)
    
    #* Calculate Skewness and saturation
    skewness_f1 = calculate_batch_skewness(batch_histograms_f1)
    skewness_f2 = calculate_batch_skewness(batch_histograms_f2)
    batch_saturation_level_f1 = calculate_batch_histogram_exposure(skewness_f1, batch_histograms_f1)
    batch_saturation_level_f2 = calculate_batch_histogram_exposure(skewness_f2, batch_histograms_f2)
    
    #* Check HDR scene based on histogram shape
    bimodal_f1 = is_bimodal(batch_histograms_f1)
    bimodal_f2 = is_bimodal(batch_histograms_f2)
    
    for i in range(batch_exp_f1.size(0)):
        exp_f1, exp_f2 = batch_exp_f1[i], batch_exp_f2[i]
        sat_f1, sat_f2 = batch_saturation_level_f1[i], batch_saturation_level_f2[i]
        a1, a2 = alpha1[i], alpha2[i]
        
    # HDR scene - bimodal
    if bimodal_f1[i] or bimodal_f2[i]:
        print(f"Bimodal : F1 : {bimodal_f1} F2 : {bimodal_f2}")
        # 둘 다 bimodal인 경우
        if exp_f1 > exp_f2:
            new_exp_f1 = exposure_shift(exp_f1, alpha=a1, decrease=False)
            new_exp_f2 = exposure_shift(exp_f2, alpha=a2, decrease=True)
        else:
            new_exp_f1 = exposure_shift(exp_f1, alpha=a1, decrease=True)
            new_exp_f2 = exposure_shift(exp_f2, alpha=a2, decrease=False)

    # LDR scene - unimodal
    else:
        new_exp_f1, new_exp_f2 = handle_unimodal_scene(exp_f1, exp_f2, sat_f1, sat_f2, a1, a2)

    # Control max exposure gap
    # Control max exposure ratio (instead of max exposure gap)
    exposure_ratio = max(new_exp_f1, new_exp_f2) / min(new_exp_f1, new_exp_f2)
    if exposure_ratio > max_exposure_gap:
        adjustment_ratio = (exposure_ratio - max_exposure_gap) / 2
        if new_exp_f1 > new_exp_f2:
            new_exp_f1 /= (1 + adjustment_ratio)
            new_exp_f2 *= (1 + adjustment_ratio)
        else:
            new_exp_f1 *= (1 + adjustment_ratio)
            new_exp_f2 /= (1 + adjustment_ratio)

    shifted_exp_f1[i] = new_exp_f1
    shifted_exp_f2[i] = new_exp_f2


    # print(f"=====Adjsuted Exposure value F1 : {shifted_exp_f1}, F2 : {shifted_exp_f2}=====")
    return shifted_exp_f1, shifted_exp_f2

def handle_unimodal_scene(exp_f1, exp_f2, sat_f1, sat_f2, a1, a2):
    if sat_f1 < 0 and sat_f2 < 0: # Both under saturation
        print("Both under saturation")
        return exposure_shift(exp_f1, alpha=a1, decrease=False), exposure_shift(exp_f2, alpha=a2, decrease=False)
    elif sat_f1 > 0 and sat_f2 < 0 and exp_f1 > exp_f2: # Frame1 over, Frame2 under
        print("Frame1 is over, Frame2 is under")
        return exposure_shift(exp_f1, alpha=a1, decrease=True), exposure_shift(exp_f2, alpha=a2, decrease=False)
    elif sat_f1 < 0 and sat_f2 > 0 and exp_f1 < exp_f2: # Frame1 under, Frame2 over
        print("Frame1 is under, Frame2 is over")
        return exposure_shift(exp_f1, alpha=a1, decrease=False), exposure_shift(exp_f2, alpha=a2, decrease=True)
    elif sat_f1 > 0 and sat_f2 > 0: # Both over saturation
        print("Both over saturation")
        return exposure_shift(exp_f1, alpha=a1, decrease=True), exposure_shift(exp_f2, alpha=a2, decrease=True)
    else: # Both symmetric or similar exposure
        print("Both appro saturation")
        return exposure_shift(exp_f1, alpha=a1 * 0.1, decrease=False), exposure_shift(exp_f2, alpha=a2 * 0.1, decrease=True)
        # print("Both appro saturation, reducing gap")
        # exp_gap = abs(exp_f1 - exp_f2)
        # # 노출 간격을 줄이기 위한 전략
        # reduce_factor = exp_gap * 0.1  # 10% 정도 간격을 줄임
        # if exp_f1 > exp_f2:
        #     return exp_f1 - reduce_factor, exp_f2 + reduce_factor
        # else:
        #     return exp_f1 + reduce_factor, exp_f2 - reduce_factor




def stereo_exposure_control4(batch_exp_f1, batch_exp_f2, batch_histograms_f1, batch_histograms_f2, alpha1, alpha2, max_exposure_gap=2.5):
    shifted_exp_f1 = torch.zeros_like(batch_exp_f1)
    shifted_exp_f2 = torch.zeros_like(batch_exp_f2)
    
    # Skewness 계산
    skewness_f1 = calculate_batch_skewness(batch_histograms_f1)
    skewness_f2 = calculate_batch_skewness(batch_histograms_f2)
    
    # 히스토그램 형태에 따른 HDR 장면 여부 확인
    bimodal_f1 = is_bimodal(batch_histograms_f1)
    bimodal_f2 = is_bimodal(batch_histograms_f2)
    
    for i in range(batch_exp_f1.size(0)):
        exp_f1, exp_f2 = batch_exp_f1[i], batch_exp_f2[i]
        a1, a2 = alpha1[i], alpha2[i]
        skewness_diff = abs(skewness_f1[i] - skewness_f2[i])
        
        new_exp_f1, new_exp_f2 = exp_f1, exp_f2
        
        # HDR 장면 - bimodal 판단
        if bimodal_f1[i] or bimodal_f2[i]:
            print(f"Bimodal : F1 : {bimodal_f1[i]} F2 : {bimodal_f2[i]}")
            
            if skewness_diff < 0.5:  # skewness_diff가 작은 경우
                # 두 노출값이 비슷할 때, 한쪽이 bimodal인 경우 (초기 LDR -> HDR 상황)
                print("Both frame exposure is preety same, increase exposure gap")
                if exp_f1 > exp_f2:
                    new_exp_f1 = exposure_shift(exp_f1, alpha=a1, decrease=False)
                    new_exp_f2 = exposure_shift(exp_f2, alpha=a2, decrease=True)
                else:
                    new_exp_f1 = exposure_shift(exp_f1, alpha=a1, decrease=True)
                    new_exp_f2 = exposure_shift(exp_f2, alpha=a2, decrease=False)
            if skewness_diff > 1.0:
                print("Both frame expsoure gap is large,  shift bimodal frame exposure slightly")
                # skewness_diff가 큰 경우, 노출값이 충분히 벌어진 상태에서 bimodality 체크
                if abs(skewness_f1[i]) > abs(skewness_f2[i]):
                    if skewness_f1[i] > 0:
                        new_exp_f1 = exposure_shift(exp_f1, alpha=a1*0.5, decrease=True)  # exp_f1을 낮춤
                    else:
                        new_exp_f1 = exposure_shift(exp_f1, alpha=a1*0.5, decrease=False)  # exp_f1을 높임
                    new_exp_f2 = exp_f2
                else:
                    if skewness_f2[i] > 0:
                        new_exp_f2 = exposure_shift(exp_f2, alpha=a2*0.5, decrease=True)  # exp_f2를 낮춤
                    else:
                        new_exp_f2 = exposure_shift(exp_f2, alpha=a2*0.5, decrease=False)  # exp_f2를 높임
                    new_exp_f1 = exp_f1
                        
        else:
            # LDR 장면 - skewness_diff에 따라 조정
            print(f"Both Frame is Unimodal")
            new_exp_f1, new_exp_f2 = handle_unimodal_scene_skewness(exp_f1, exp_f2, skewness_f1[i], skewness_f2[i], a1, a2, skewness_diff)

        # 최대 노출 비율 제어 (max_exposure_gap 대신 최대 노출 비율을 사용)
        exposure_ratio = max(new_exp_f1, new_exp_f2) / min(new_exp_f1, new_exp_f2)
        if exposure_ratio > max_exposure_gap:
            adjustment_ratio = (exposure_ratio - max_exposure_gap) / 2
            if new_exp_f1 > new_exp_f2:
                new_exp_f1 /= (1 + adjustment_ratio)
                new_exp_f2 *= (1 + adjustment_ratio)
            else:
                new_exp_f1 *= (1 + adjustment_ratio)
                new_exp_f2 /= (1 + adjustment_ratio)

        shifted_exp_f1[i] = new_exp_f1
        shifted_exp_f2[i] = new_exp_f2

    return shifted_exp_f1, shifted_exp_f2


def handle_unimodal_scene_skewness(exp_f1, exp_f2, skewness_f1, skewness_f2, a1, a2, skewness_diff):
     
    if skewness_f1 < 0 and skewness_f2 < 0:
        # 둘 다 낮은 skewness 값을 가질 경우 (under-exposure)
        print("Both have low skewness, increasing exposure")
        new_exp_f1 = exposure_shift(exp_f1, alpha=a1, decrease=False)
        new_exp_f2 = exposure_shift(exp_f2, alpha=a2, decrease=False)
    elif skewness_f1 > 0 and skewness_f2 < 0:
        print("Frame1 has high skewness, Frame2 has low skewness")
        new_exp_f1 = exposure_shift(exp_f1, alpha=a1, decrease=True)
        new_exp_f2 = exposure_shift(exp_f2, alpha=a2, decrease=False)
    elif skewness_f1 < 0 and skewness_f2 > 0:
        print("Frame1 has low skewness, Frame2 has high skewness")
        new_exp_f1 = exposure_shift(exp_f1, alpha=a1, decrease=False)
        new_exp_f2 = exposure_shift(exp_f2, alpha=a2, decrease=True)
    elif skewness_f1 > 0 and skewness_f2 > 0:
        print("Both have high skewness, reducing exposure")
        new_exp_f1 = exposure_shift(exp_f1, alpha=a1, decrease=True)
        new_exp_f2 = exposure_shift(exp_f2, alpha=a2, decrease=True)
    else:
        # skewness가 서로 비슷한 경우 (작거나 큰 차이가 없을 때)
        print("Both have similar skewness values, slight adjustment")
        new_exp_f1 = exposure_shift(exp_f1, alpha=a1 * 0.1, decrease=False)
        new_exp_f2 = exposure_shift(exp_f2, alpha=a2 * 0.1, decrease=True)

    return new_exp_f1, new_exp_f2


def stereo_exposure_control5(batch_exp_f1, batch_exp_f2, batch_histograms_f1, batch_histograms_f2, alpha1, alpha2, max_exposure_gap=2.5, stability_threshold=0.5, skewness_diff_threshold=1.0):
    shifted_exp_f1 = torch.zeros_like(batch_exp_f1)
    shifted_exp_f2 = torch.zeros_like(batch_exp_f2)
    
    # Skewness 계산
    skewness_f1 = calculate_batch_skewness(batch_histograms_f1)
    skewness_f2 = calculate_batch_skewness(batch_histograms_f2)
    
    # 히스토그램 형태에 따른 HDR 장면 여부 확인
    bimodal_f1 = is_bimodal(batch_histograms_f1)
    bimodal_f2 = is_bimodal(batch_histograms_f2)
    
    for i in range(batch_exp_f1.size(0)):
        exp_f1, exp_f2 = batch_exp_f1[i], batch_exp_f2[i]
        a1, a2 = alpha1[i], alpha2[i]
        skewness_diff = abs(skewness_f1[i] - skewness_f2[i])
        
        new_exp_f1, new_exp_f2 = exp_f1, exp_f2
        
        # HDR 장면 - bimodal 판단
        if bimodal_f1[i] or bimodal_f2[i]:
            print(f"Bimodal : F1 : {bimodal_f1[i]} F2 : {bimodal_f2[i]}")
            
            if skewness_diff < stability_threshold:  # skewness_diff가 stability_threshold보다 작은 경우
                # 두 노출값이 비슷할 때, 한쪽이 bimodal인 경우 (초기 LDR -> HDR 상황)
                print("Both frame exposure is pretty same, increase exposure gap")
                if exp_f1 > exp_f2:
                    new_exp_f1 = exposure_shift(exp_f1, alpha=a1, decrease=False)
                    new_exp_f2 = exposure_shift(exp_f2, alpha=a2, decrease=True)
                else:
                    new_exp_f1 = exposure_shift(exp_f1, alpha=a1, decrease=True)
                    new_exp_f2 = exposure_shift(exp_f2, alpha=a2, decrease=False)
            elif stability_threshold <= skewness_diff < skewness_diff_threshold:
                print("Exposure gap is moderate, stabilize exposure adjustment")
                # 중간 정도의 skewness_diff일 때, 노출 조정을 최소화하여 안정화
                if abs(skewness_f1[i]) > abs(skewness_f2[i]):
                    if skewness_f1[i] > 0:
                        new_exp_f1 = exposure_shift(exp_f1, alpha=a1 * 0.1, decrease=True)  # exp_f1을 약간 낮춤
                    else:
                        new_exp_f1 = exposure_shift(exp_f1, alpha=a1 * 0.1, decrease=False)  # exp_f1을 약간 높임
                    new_exp_f2 = exp_f2
                else:
                    if skewness_f2[i] > 0:
                        new_exp_f2 = exposure_shift(exp_f2, alpha=a2 * 0.1, decrease=True)  # exp_f2를 약간 낮춤
                    else:
                        new_exp_f2 = exposure_shift(exp_f2, alpha=a2 * 0.1, decrease=False)  # exp_f2를 약간 높임
                    new_exp_f1 = exp_f1
            else:
                print("Both frame exposure gap is large, shift bimodal frame exposure slightly")
                # skewness_diff가 큰 경우, 노출값이 충분히 벌어진 상태에서 bimodality 체크
                if abs(skewness_f1[i]) > abs(skewness_f2[i]):
                    if skewness_f1[i] > 0:
                        new_exp_f1 = exposure_shift(exp_f1, alpha=a1 * 0.25, decrease=True)  # exp_f1을 낮춤
                    else:
                        new_exp_f1 = exposure_shift(exp_f1, alpha=a1 * 0.25, decrease=False)  # exp_f1을 높임
                    new_exp_f2 = exp_f2
                else:
                    if skewness_f2[i] > 0:
                        new_exp_f2 = exposure_shift(exp_f2, alpha=a2 * 0.25, decrease=True)  # exp_f2를 낮춤
                    else:
                        new_exp_f2 = exposure_shift(exp_f2, alpha=a2 * 0.25, decrease=False)  # exp_f2를 높임
                    new_exp_f1 = exp_f1
        else:
            # LDR 장면 - skewness_diff에 따라 조정
            print(f"Both Frame is Unimodal")
            new_exp_f1, new_exp_f2 = handle_unimodal_scene_skewness(exp_f1, exp_f2, skewness_f1[i], skewness_f2[i], a1, a2, skewness_diff)
        
        # 추가: HDR -> LDR 상황 처리 (dynamic range가 줄어드는 경우)
        if not bimodal_f1[i] and not bimodal_f2[i] and skewness_diff < stability_threshold:
            print("Scene dynamic range is decreasing, reducing exposure gap")
            # 노출 격차를 줄여서 LDR 상황에 맞게 조정
            if new_exp_f1 > new_exp_f2:
                new_exp_f1 = exposure_shift(new_exp_f1, alpha=a1, decrease=True)
                new_exp_f2 = exposure_shift(new_exp_f2, alpha=a2, decrease=False)
            else:
                new_exp_f1 = exposure_shift(new_exp_f1, alpha=a1, decrease=False)
                new_exp_f2 = exposure_shift(new_exp_f2, alpha=a2, decrease=True)

        # 최대 노출 비율 제어 (max_exposure_gap 대신 최대 노출 비율을 사용)
        exposure_ratio = max(new_exp_f1, new_exp_f2) / min(new_exp_f1, new_exp_f2)
        if exposure_ratio > max_exposure_gap:
            adjustment_ratio = (exposure_ratio - max_exposure_gap) / 2
            if new_exp_f1 > new_exp_f2:
                new_exp_f1 /= (1 + adjustment_ratio)
                new_exp_f2 *= (1 + adjustment_ratio)
            else:
                new_exp_f1 *= (1 + adjustment_ratio)
                new_exp_f2 /= (1 + adjustment_ratio)

        shifted_exp_f1[i] = new_exp_f1
        shifted_exp_f2[i] = new_exp_f2

    return shifted_exp_f1, shifted_exp_f2


def stereo_exposure_control6(batch_exp_f1, batch_exp_f2, batch_histograms_f1, batch_histograms_f2, alpha1, alpha2, max_exposure_gap=2.5, stability_threshold=0.8):
    shifted_exp_f1 = torch.zeros_like(batch_exp_f1)
    shifted_exp_f2 = torch.zeros_like(batch_exp_f2)
    
    # Skewness 계산
    skewness_f1 = calculate_batch_skewness(batch_histograms_f1)
    skewness_f2 = calculate_batch_skewness(batch_histograms_f2)
    
    # 히스토그램 형태에 따른 HDR 장면 여부 확인
    bimodal_f1 = is_bimodal(batch_histograms_f1)
    bimodal_f2 = is_bimodal(batch_histograms_f2)
    
    for i in range(batch_exp_f1.size(0)):
        exp_f1, exp_f2 = batch_exp_f1[i], batch_exp_f2[i]
        a1, a2 = alpha1[i], alpha2[i]
        skewness_diff = abs(skewness_f1[i] - skewness_f2[i])
        
        new_exp_f1, new_exp_f2 = exp_f1, exp_f2
        
        # HDR 장면 - bimodal 판단
        if bimodal_f1[i] or bimodal_f2[i]:
            print(f"Bimodal : F1 : {bimodal_f1[i]} F2 : {bimodal_f2[i]}")
            
            if skewness_diff < stability_threshold:  # skewness_diff가 stability_threshold보다 작은 경우
                # 두 노출값이 비슷할 때, 한쪽이 bimodal인 경우 (초기 LDR -> HDR 상황)
                print("Both frame exposure is pretty same, increase exposure gap")
                if exp_f1 > exp_f2:
                    new_exp_f1 = exposure_shift(exp_f1, alpha=a1 * 0.5, decrease=False)
                    new_exp_f2 = exposure_shift(exp_f2, alpha=a2 * 0.5, decrease=True)
                else:
                    new_exp_f1 = exposure_shift(exp_f1, alpha=a1 * 0.5, decrease=True)
                    new_exp_f2 = exposure_shift(exp_f2, alpha=a2 * 0.5, decrease=False)
                    
            else:
                print("Both frame exposure gap is large, shift bimodal frame exposure slightly")
                # skewness_diff가 큰 경우, 노출값이 충분히 벌어진 상태에서 bimodality 체크
                if abs(skewness_f1[i]) > abs(skewness_f2[i]):
                    if skewness_f1[i] > 0:
                        new_exp_f1 = exposure_shift(exp_f1, alpha=a1 * 0.25, decrease=True)  # exp_f1을 낮춤
                    else:
                        new_exp_f1 = exposure_shift(exp_f1, alpha=a1 * 0.25, decrease=False)  # exp_f1을 높임
                    new_exp_f2 = exp_f2
                else:
                    if skewness_f2[i] > 0:
                        new_exp_f2 = exposure_shift(exp_f2, alpha=a2 * 0.25, decrease=True)  # exp_f2를 낮춤
                    else:
                        new_exp_f2 = exposure_shift(exp_f2, alpha=a2 * 0.25, decrease=False)  # exp_f2를 높임
                    new_exp_f1 = exp_f1
        else:
            # LDR 장면 - skewness_diff에 따라 조정
            print(f"Both Frame is Unimodal")
            new_exp_f1, new_exp_f2 = handle_unimodal_scene_skewness(exp_f1, exp_f2, skewness_f1[i], skewness_f2[i], a1, a2, skewness_diff)
        
        # 추가: HDR -> LDR 상황 처리 (dynamic range가 줄어드는 경우)
        if not bimodal_f1[i] and not bimodal_f2[i] and skewness_diff < stability_threshold:
            print("Scene dynamic range is decreasing, reducing exposure gap")
            # 노출 격차를 줄여서 LDR 상황에 맞게 조정
            if new_exp_f1 > new_exp_f2:
                new_exp_f1 = exposure_shift(new_exp_f1, alpha=a1, decrease=True)
                new_exp_f2 = exposure_shift(new_exp_f2, alpha=a2, decrease=False)
            else:
                new_exp_f1 = exposure_shift(new_exp_f1, alpha=a1, decrease=False)
                new_exp_f2 = exposure_shift(new_exp_f2, alpha=a2, decrease=True)

        # 최대 노출 비율 제어 (max_exposure_gap 대신 최대 노출 비율을 사용)
        exposure_ratio = max(new_exp_f1, new_exp_f2) / min(new_exp_f1, new_exp_f2)
        if exposure_ratio > max_exposure_gap:
            adjustment_ratio = (exposure_ratio - max_exposure_gap) / 2
            if new_exp_f1 > new_exp_f2:
                new_exp_f1 /= (1 + adjustment_ratio)
                new_exp_f2 *= (1 + adjustment_ratio)
            else:
                new_exp_f1 *= (1 + adjustment_ratio)
                new_exp_f2 /= (1 + adjustment_ratio)

        shifted_exp_f1[i] = new_exp_f1
        shifted_exp_f2[i] = new_exp_f2

    return shifted_exp_f1, shifted_exp_f2



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

# def calculate_batch_skewness(batch_histograms):
#     batch_skewness_level = []
#     for histogram in batch_histograms:
#         frequencies = histogram
    
#         pixel_values = torch.arange(len(frequencies), device=DEVICE)
        
#         fixed_mean = 128
#         skewness_numerator = torch.sum(frequencies * ((pixel_values - fixed_mean) **3))
#         total_pixels = torch.sum(frequencies)
#         skewness = skewness_numerator / total_pixels
        
#         skewness /= ((255 - 0) ** 1.5)
        
#         # Calculate mean skewness for combined multi-skewness value
#         batch_skewness_level.append(skewness/255)
               
#     return batch_skewness_level
def calculate_batch_skewness(batch_histograms):
    batch_skewness_level = []
    
    for histogram in batch_histograms:
        frequencies = histogram  # 각 히스토그램 빈도
        
        # 픽셀 값 범위 생성 (0~255)
        pixel_values = torch.arange(len(frequencies), device=frequencies.device)
        
        # 중간 회색을 기준으로 세 번째 중심 모멘트 계산
        fixed_mean = 128
        skewness_numerator = torch.sum(frequencies * ((pixel_values - fixed_mean) ** 3))
        total_pixels = torch.sum(frequencies)
        skewness = skewness_numerator / total_pixels
        
        # 중간 회색을 기준으로 한 표준편차 계산
        variance = torch.sum(frequencies * ((pixel_values - fixed_mean) ** 2)) / total_pixels
        std_dev = torch.sqrt(variance)
        
        # 표준편차의 세제곱으로 나누어 skewness 정규화
        if std_dev != 0:  # 표준편차가 0일 경우를 방지
            skewness /= std_dev ** 3
        else:
            skewness = torch.tensor(0.0, device=frequencies.device)  # 표준편차가 0인 경우 skewness를 0으로 설정
        
        batch_skewness_level.append(skewness)
    
    return batch_skewness_level


def adjust_exposure_gap(exp_f1, exp_f2, clamping_ratio_f1, clamping_ratio_f2, alpha1, alpha2):
    
    if exp_f1 > exp_f2:
        print("Getting wide exposure gap if exp_f1 > exp_f2")
        new_exp_f1 = exp_f1 + alpha1 * clamping_ratio_f1[0]
        new_exp_f2 = exp_f2 - alpha2 * clamping_ratio_f2[1]
        
        # new_exp_f1 = exposure_shift(exp_f1, alpha=alpha1, decrease=False)
        # new_exp_f2 = exposure_shift(exp_f2, alpha=alpha2, decrease=True) 
    else:
        print("Getting wide exposure gap if exp_f1 < exp_f2")
        new_exp_f1 = exp_f1 - alpha1 * clamping_ratio_f1[1]
        new_exp_f2 = exp_f2 + alpha2 * clamping_ratio_f2[0]
        # new_exp_f1 = exposure_shift(exp_f1, alpha=alpha1, decrease=True)
        # new_exp_f2 = exposure_shift(exp_f2, alpha=alpha2, decrease=False) 
    
    # # Absolute skewness difference is large, apply additional adjustment
    # skewness_abs_diff = torch.abs(skewness_f1 - skewness_f2)
    # if skewness_abs_diff > 0.2:  

    #     if abs(skewness_f1) > abs(skewness_f2):
    #         new_exp_f1 = exposure_shift_skewness(new_exp_f1, skewness_f1, alpha1)
    #         print("Apply additional adjustments with skewness_diff")
    #     else:
    #         new_exp_f2 = exposure_shift_skewness(new_exp_f2, skewness_f2, alpha2)
    #         print("Apply additional adjustments with skewness_diff")
               
   
    return (new_exp_f1, new_exp_f2)
    
def stereo_exposure_control7(batch_exp_f1, batch_exp_f2, batch_histograms_f1, batch_histograms_f2, alpha1, alpha2, exp_gap_threshold=2.0):
    shifted_exp_f1 = torch.zeros_like(batch_exp_f1)
    shifted_exp_f2 = torch.zeros_like(batch_exp_f2)
    
    # Skewness 
    skewness_f1 = calculate_batch_skewness(batch_histograms_f1)
    skewness_f2 = calculate_batch_skewness(batch_histograms_f2)
    
    # Check scene dynamic range
    hdr_f1 = clamping_ratio_flag(batch_histograms_f1)
    hdr_f2 = clamping_ratio_flag(batch_histograms_f2)
    
    clamping_ratio_f1 = clamping_ratio(batch_histograms_f1)
    clamping_ratio_f2 = clamping_ratio(batch_histograms_f2)
    
    # print(f"Clamping_ratio_f1 : {clamping_ratio_f1}, Clamping_ratio_f2 : {clamping_ratio_f2}")
    
    for i in range(batch_exp_f1.size(0)):
        exp_f1, exp_f2 = batch_exp_f1[i], batch_exp_f2[i]
        a1, a2 = alpha1[i], alpha2[i]
        skewness_diff = abs(skewness_f1[i] - skewness_f2[i])
        
        new_exp_f1 = exp_f1
        new_exp_f2 = exp_f2
        exp_diff = torch.abs(new_exp_f1 - new_exp_f2)
        
        if hdr_f1[i] or hdr_f2[i]: # Clamping on both side : HDR 
            # print(f"One of the Frame's clamping ratio exceed threshold : {hdr_f1[i]} {hdr_f2[i]}")
            if exp_diff < exp_gap_threshold:
                # print(f"Increase exposure gap until stability_threshold")
                new_exp_f1, new_exp_f2 = adjust_exposure_gap(exp_f1, exp_f2, clamping_ratio_f1[i], clamping_ratio_f1[i], a1, a2)
            
        else: # Clamping on one side: LDR ->  adjust exposure to bring skewness closer to zero
            # print(f"Both frames' clamping ratio not exceed threshold : {hdr_f1[i]} {hdr_f2[i]}")
            new_exp_f1, new_exp_f2 = handle_ldr_scene_skewness(exp_f1, exp_f2, skewness_f1[i], skewness_f2[i], alpha=0.1)

        shifted_exp_f1[i] = new_exp_f1
        shifted_exp_f2[i] = new_exp_f2
        
    return shifted_exp_f1, shifted_exp_f2


def stereo_exposure_control8(batch_exp_f1, batch_exp_f2, batch_histograms_f1, batch_histograms_f2, alpha1, alpha2, exp_gap_threshold=2.0):
    device = batch_exp_f1.device  # ✅ 현재 텐서의 device 확인 (e.g., cuda:2)
    
    shifted_exp_f1 = torch.zeros_like(batch_exp_f1, device=device)  # ✅ GPU로 이동
    shifted_exp_f2 = torch.zeros_like(batch_exp_f2, device=device)  # ✅ GPU로 이동
    
    # Skewness 
    skewness_f1 = calculate_batch_skewness(batch_histograms_f1)
    skewness_f2 = calculate_batch_skewness(batch_histograms_f2)
    
    # Check scene dynamic range
    hdr_f1 = clamping_ratio_flag(batch_histograms_f1).to(device)  # ✅ GPU로 이동
    hdr_f2 = clamping_ratio_flag(batch_histograms_f2).to(device)  # ✅ GPU로 이동
    
    clamping_ratio_f1 = clamping_ratio(batch_histograms_f1).to(device)  # ✅ GPU로 이동
    clamping_ratio_f2 = clamping_ratio(batch_histograms_f2).to(device)  # ✅ GPU로 이동
    
    for i in range(batch_exp_f1.size(0)):
        exp_f1, exp_f2 = batch_exp_f1[i].to(device), batch_exp_f2[i].to(device)  # ✅ GPU로 이동
        a1, a2 = alpha1[i].to(device), alpha2[i].to(device)  # ✅ GPU로 이동
        skewness_diff = abs(skewness_f1[i] - skewness_f2[i])
        
        new_exp_f1 = exp_f1
        new_exp_f2 = exp_f2
        exp_diff = torch.abs(new_exp_f1 - new_exp_f2)

        if hdr_f1[i] or hdr_f2[i]:  # HDR scene, clamping이 존재하는 경우
            if exp_diff < exp_gap_threshold:
                new_exp_f1, new_exp_f2 = adjust_exposure_gap(exp_f1, exp_f2, 
                                                             clamping_ratio_f1[i].to(device), 
                                                             clamping_ratio_f2[i].to(device), 
                                                             a1, a2)
        else:  # LDR scene, skewness 조정
            new_exp_f1, new_exp_f2 = handle_ldr_scene_skewness(exp_f1, exp_f2, 
                                                               skewness_f1[i], 
                                                               skewness_f2[i], 
                                                               alpha=0.1)

        shifted_exp_f1[i] = new_exp_f1
        shifted_exp_f2[i] = new_exp_f2
        
    return shifted_exp_f1, shifted_exp_f2




def adjust_exposure_gap_dynamic(exp_f1, exp_f2, skewness_f1, skewness_f2, skewness_diff, alpha1, alpha2, stability_threshold, max_exposure_gap, clamping_ratio_f1, clamping_ratio_f2):
    # HDR scene detected, if skewness_diff is large but we still need to control gap growth
    if skewness_diff < stability_threshold:
        if exp_f1 > exp_f2:
            print("Expanding exposure gap: exp_f1 > exp_f2")
            new_exp_f1 = exposure_shift(exp_f1, alpha=alpha1, decrease=False)
            new_exp_f2 = exposure_shift(exp_f2, alpha=alpha2, decrease=True)
        else:
            print("Expanding exposure gap: exp_f1 < exp_f2")
            new_exp_f1 = exposure_shift(exp_f1, alpha=alpha1, decrease=True)
            new_exp_f2 = exposure_shift(exp_f2, alpha=alpha2, decrease=False)
    else:
        # Reduce exposure gap if clamping ratios indicate decreasing dynamic range
        if abs(exp_f1 - exp_f2) > max_exposure_gap or (clamping_ratio_f1[1] < 0.25 and clamping_ratio_f2[0] < 0.5):
            print(f"Exposure gap exceeds max or dynamic range decreased, reducing gap")
            new_exp_f1 = exposure_shift(exp_f1, alpha=alpha1, decrease=True)
            new_exp_f2 = exposure_shift(exp_f2, alpha=alpha2, decrease=False)
        else:
            print("Exposure gap at threshold, maintaining current exposures")
            new_exp_f1 = exp_f1
            new_exp_f2 = exp_f2
    
    return new_exp_f1, new_exp_f2


def stereo_exposure_control_dynamic(batch_exp_f1, batch_exp_f2, batch_histograms_f1, batch_histograms_f2, alpha1, alpha2, stability_threshold=0.8, max_exposure_gap=2.5):
    shifted_exp_f1 = torch.zeros_like(batch_exp_f1)
    shifted_exp_f2 = torch.zeros_like(batch_exp_f2)
    
    skewness_f1 = calculate_batch_skewness(batch_histograms_f1)
    skewness_f2 = calculate_batch_skewness(batch_histograms_f2)
    
    hdr_f1 = clamping_ratio_flag(batch_histograms_f1)
    hdr_f2 = clamping_ratio_flag(batch_histograms_f2)
    
    clamping_ratio_f1 = clamping_ratio(batch_histograms_f1)
    clamping_ratio_f2 = clamping_ratio(batch_histograms_f2)
    
    print(f"Clamping_ratio_f1 : {clamping_ratio_f1}, Clamping_ratio_f2 : {clamping_ratio_f2}")
    
    for i in range(batch_exp_f1.size(0)):
        exp_f1, exp_f2 = batch_exp_f1[i], batch_exp_f2[i]
        a1, a2 = alpha1[i], alpha2[i]
        skewness_diff = abs(skewness_f1[i] - skewness_f2[i])
        
        if hdr_f1[i] or hdr_f2[i]:  
            print(f"HDR scene detected: Clamping ratio exceeded threshold")
            new_exp_f1, new_exp_f2 = adjust_exposure_gap_dynamic(
                exp_f1, exp_f2, skewness_f1[i], skewness_f2[i], 
                skewness_diff, a1, a2, stability_threshold, max_exposure_gap, 
                clamping_ratio_f1[i], clamping_ratio_f2[i]
            )
        else:
            print(f"LDR scene detected: Clamping ratio below threshold")
            if skewness_diff < 0.1:
                print("Exposure gap is small, adjusting exposure to close skewness values")
                new_exp_f1, new_exp_f2 = handle_unimodal_scene_skewness(exp_f1, exp_f2, skewness_f1[i], skewness_f2[i], alpha=a1)  # 더 작은 알파 값 적용
            else:
                print("Exposure gap at threshold, maintaining current exposures")
                new_exp_f1 = exp_f1
                new_exp_f2 = exp_f2
        
        # Adjust exposure gap if clamping indicates decreasing dynamic range or skewness diff is large
        if skewness_diff >= stability_threshold:
            print(f"Exposure gap large, checking clamping ratios: {clamping_ratio_f1[i]} {clamping_ratio_f2[i]}")
            if exp_f1 > exp_f2 and clamping_ratio_f1[i][1] <= 0.25 and clamping_ratio_f2[i][0] <= 0.25:
                new_exp_f1 = exposure_shift(new_exp_f1, alpha=a1, decrease=True)
                new_exp_f2 = exposure_shift(new_exp_f2, alpha=a2, decrease=False)
            elif exp_f2 > exp_f1 and clamping_ratio_f1[i][0] <= 0.25 and clamping_ratio_f2[i][1] <= 0.25:
                new_exp_f1 = exposure_shift(new_exp_f1, alpha=a1, decrease=False)
                new_exp_f2 = exposure_shift(new_exp_f2, alpha=a2, decrease=True)

        shifted_exp_f1[i] = new_exp_f1
        shifted_exp_f2[i] = new_exp_f2

    return shifted_exp_f1, shifted_exp_f2




#^ ##########################
#^ Nerual Exposure Netowrk
#^ ##########################
# * Alpha : exposure shift parameter
class AlphPredictionNet(nn.Module):
    def __init__(self):
        super(AlphPredictionNet,self).__init__()
        
        self.fc_layers = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, sat1, sat2, skewness1, skewness2):
        # Check if the inputs are already tensors, if not, convert them
        if isinstance(sat1, list):
            sat1 = torch.stack(sat1)
        if isinstance(sat2, list):
            sat2 = torch.stack(sat2)
        if isinstance(skewness1, list):
            skewness1 = torch.stack(skewness1)
        if isinstance(skewness2, list):
            skewness2 = torch.stack(skewness2)
        
        x = torch.cat((sat1, sat2, skewness1, skewness2), dim=-1)
        x = self.fc_layers(x)
        alpha = torch.sigmoid(x)
        return alpha
        
        
# & Global image feature branch network

# class GlobalFeatureNet(nn.Module):
#     def __init__(self):
#         super(GlobalFeatureNet, self).__init__()

#         self.conv_layers = nn.Sequential(
#             # Make sure the input channel is 59, or adjust
#             nn.Conv1d(59*2, 256, kernel_size=4, stride=4),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Conv1d(256, 512, kernel_size=4, stride=4),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Conv1d(512, 1024, kernel_size=4, stride=4),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#         )

#         self.fc_layers = nn.Sequential(
#             nn.Linear(4096, 1024), 
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(1024, 256),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(256, 64),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(64,1)
#         )

#     def forward(self, left_hist, right_hist):
#         x = torch.cat((left_hist, right_hist), dim=2)
#         x = x.permute(0, 2, 1)
        
#         x = self.conv_layers(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc_layers(x)
        
#         x = torch.sigmoid(x)
        
#         return x

class GlobalFeatureNet(nn.Module):
    def __init__(self):
        super(GlobalFeatureNet, self).__init__()

        # 2D CNN layers for image features
        self.image_conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
        )

        # 1D CNN layers for histogram features
        self.hist_conv_layers = nn.Sequential(
            nn.Conv1d(59, 256, kernel_size=4, stride=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=4, stride=4),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 1024, kernel_size=4, stride=4),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        # Fully connected layers for combined features
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8 + 1024 * 4, 1024),  # Adjust the input size accordingly
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid(),# Output 1 alpha values
        )

    def forward(self, left_image, right_image, left_hist, right_hist):
        # Extract features from images using 2D CNN
        img_feat1 = self.image_conv_layers(left_image)
        img_feat2 = self.image_conv_layers(right_image)
        img_feat1 = img_feat1.view(img_feat1.size(0), -1)
        img_feat2 = img_feat2.view(img_feat2.size(0), -1) # 1 x 16384
        # img_feat = torch.cat((img_feat1, img_feat2), dim=1)
        # img_feat = img_feat.view(img_feat.size(0), -1)
        # print(f"img_feat shape : {img_feat1.shape}")

        # Process histogram features using 1D CNN
        # hist_feat = torch.cat((left_hist, right_hist), dim=2)
        left_hist = left_hist.permute(0,2,1)
        left_hist_feat = self.hist_conv_layers(left_hist)
        left_hist_feat = left_hist_feat.view(left_hist_feat.size(0), -1) # 1 x 4096
        
        right_hist = right_hist.permute(0,2,1)
        right_hist_feat = self.hist_conv_layers(right_hist)
        right_hist_feat = right_hist_feat.view(right_hist_feat.size(0), -1)
        # print(f"left_hist_feat.shape : {left_hist_feat.shape}")
        
        combined_feat1 = torch.cat((img_feat1, left_hist_feat), dim = 1)
        combined_feat2 = torch.cat((img_feat2, right_hist_feat), dim = 1)
        print(f"combine_feat1.shape : {combined_feat1.shape}")
        
        alpha1 = self.fc_layers(combined_feat1)
        alpha2 = self.fc_layers(combined_feat2)
        
        # print(f"alpha1 : {alpha1}")
        # print(f"alpha2 : {alpha2}")
        
        # hist_feat = hist_feat.permute(0, 2, 1)
        # hist_feat = self.hist_conv_layers(hist_feat)
        # hist_feat = hist_feat.view(hist_feat.size(0), -1)
        # print(f"hist_feat shape : {hist_feat.shape}")

        # Combine image and histogram features
        # combined_feat = torch.cat((img_feat, hist_feat), dim=1)
        # alpha = torch.cat((alpha1, alpha2), dim=0)
        # alpha = torch.sigmoid(alpha)

        return alpha1, alpha2
    
# class ExposureAdjustmentPipeline(nn.Module):
#     def __init__(self):
#         super(ExposureAdjustmentPipeline, self).__init__()
#         self.alpha_net = GlobalFeatureNet()

#     def forward(self, left_image, right_image):
#         # Calculate histograms
#         left_hist, right_hist = calculate_histograms(left_image, right_image)
        
#         # Predict alpha values
#         alpha = self.alpha_net(left_hist, right_hist)
        
#         return alpha

class ExposureAdjustmentPipeline(nn.Module):
    def __init__(self):
        super(ExposureAdjustmentPipeline, self).__init__()
        self.alpha_net = GlobalFeatureNet()

    def forward(self, left_image, right_image):
        # Calculate histograms
        left_hist, right_hist = calculate_histograms(left_image, right_image)
        
        # Predict alpha values
        alpha = self.alpha_net(left_image, right_image, left_hist, right_hist)
        
        return alpha

class FeatureFusionNet(nn.Module):
    def __init__(self, M_exp=4):
        super(FeatureFusionNet, self).__init__()
        self.M_exp = M_exp
        self.exp_min = M_exp**-1  # 계산된 최소값
        self.exp_max = M_exp
        
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 128),
            nn.InstanceNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.InstanceNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        x = self.fc_layers(x)
        x = torch.sigmoid(x)
        x = x * (self.exp_max - self.exp_min) + self.exp_min
        
        return x


class ExposureNet(nn.Module):
    def __init__(self, M_exp=4):
        super(ExposureNet, self).__init__()
        
        # For LDR image histogram
        self.conv_layers = nn.Sequential(
            # Make sure the input channel is 59, or adjust
            nn.Conv1d(118, 256, kernel_size=4, stride=4),  
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=4, stride=4),
            nn.ReLU(),
        )
        
        # For HDR image dynamic range info, exposure value
        self.additional_fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
        )
        
        # fuse con_layer output and additional_fc output
        self.fc_layers = nn.Sequential(
            nn.Linear(2048 + 512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        
        self.M_exp = M_exp
        self.M_exp_log = torch.log(torch.tensor([self.M_exp], dtype=torch.float))

    def forward(self, ldr_histo, ldr_semantic, hdr_insto, rand_exp):
        
        print(f"ldr_histo shape : {ldr_histo}")
        
        ldr_histo = ldr_histo.permute(2, 0, 1)        
        ldr_histo = self.conv_layers(ldr_histo)
        ldr_histo = ldr_histo.view(ldr_histo.size(0), -1)
        
        print("Check shape in saec.py")
        print(ldr_histo.shape)
        
        # additional inpout : HDR dynamic range info, intput rand exposure 
        additional_inputs = self.additional_fc(additional_inputs)
        
        x = torch.cat((x, additional_inputs), dim=1)
        
        x = self.fc_layers(x)
        
        return x

class DualFrameExposureNet(nn.Module):
    def __init__(self):
        super(DualFrameExposureNet, self).__init__()

        # 각 프레임의 conv_layers 분리하여 독립적으로 학습
        self.conv_layers_f1 = nn.Sequential(
            nn.Conv1d(59, 128, kernel_size=4, stride=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=4, stride=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        self.conv_layers_f2 = nn.Sequential(
            nn.Conv1d(59, 128, kernel_size=4, stride=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=4, stride=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.cross_attention = nn.MultiheadAttention(embed_dim=4096, num_heads=4)
        
        # 차이 기반 모듈 수정 (입력 크기를 줄이도록 설정)
        self.difference_module = nn.Sequential(
            nn.Linear(4096, 1024),  # 차원이 커서 줄임
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # 최종 FC 레이어
        self.fc_layers = nn.Sequential(
            nn.Linear(128 + 4, 64),  # 추가 value(skewness, saturation) 포함
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # 두 프레임의 노출값
            nn.Sigmoid(),  # Sigmoid로 0~1 사이 값
        )
    
    def histogram_subimage(self, image, grid_size):
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
                    hist_tensor = torch.histc(sub_image_tensor[1], bins= 256, min =0, max =1).to(DEVICE)
                    histograms.append(hist_tensor)
            batch_histograms.append(histograms)
        
        return batch_histograms
    
    def stack_histogoram(self, list_of_histogram):
        
        # Transpose the structure to: [[histogram_image1_from_batch1, histogram_image1_from_batch2, histogram_image1_from_batch3], ...]
        transposed_list = list(zip(*list_of_histogram))

        stacked_histograms = []
        for histograms_per_image in transposed_list:
            # Flatten the histograms for a single image
            merged_list = [item for sublist in histograms_per_image for item in sublist]
            stacked_histo_tensor = torch.stack(merged_list, dim=1)
            stacked_histograms.append(stacked_histo_tensor)

        # Stack all histograms for the entire batch
        batch_stacked_histograms = torch.stack(stacked_histograms, dim=0).to(DEVICE)

        return batch_stacked_histograms
    
    
    def calculate_histograms(self, frame1, frame2):
        
        #^ Calculate histogram with multi scale
        histogram_coarest_l = self.histogram_subimage(frame1, 1)
        histogram_intermediate_l = self.histogram_subimage(frame1, 3)
        histogram_finest_l = self.histogram_subimage(frame1,7)
        
        histogram_coarest_r = self.histogram_subimage(frame2, 1)
        histogram_intermediate_r = self.histogram_subimage(frame2, 3)
        histogram_finest_r = self.histogram_subimage(frame2, 7)
        
        #^ Stack histogram [256,59]
        list_of_histograms_l = [histogram_coarest_l, histogram_intermediate_l, histogram_finest_l]
        stacked_histo_tensor_l = self.stack_histogoram(list_of_histograms_l)

        list_of_histograms_r = [histogram_coarest_r, histogram_intermediate_r, histogram_finest_r]
        stacked_histo_tensor_r = self.stack_histogoram(list_of_histograms_r)
        
        return stacked_histo_tensor_l, stacked_histo_tensor_r
        
    def calculate_histogram_global(self, image):
        histo_global = self.histogram_subimage(image, 1)
        return histo_global[0]
    
    def calculate_batch_skewness(self, batch_histograms):
        batch_skewness_level = []
        
        for histogram in batch_histograms:
            frequencies = histogram.to(DEVICE) 
            
            pixel_values = torch.arange(len(frequencies), device=frequencies.device)  
            
            fixed_mean = 128
            skewness_numerator = torch.sum(frequencies * ((pixel_values - fixed_mean) ** 3))
            total_pixels = torch.sum(frequencies)
            skewness = skewness_numerator / total_pixels
            
            skewness /= ((255 - 0) ** 1.5)
            # 배치 내 각 히스토그램에 대해 skewness 값을 텐서로 추가
            batch_skewness_level.append((skewness/255).unsqueeze(0))  # [1] 차원 추가
        
        # 결과 텐서 형태로 반환 [B, 1] 크기의 텐서로 변환
        return torch.stack(batch_skewness_level, dim=0)
    
    def is_bimodal(self, batch_histograms, low_threshold=0.1, high_threshold=0.9):
        bimodal_flags = []
        low_threshold = int(low_threshold*255)
        high_threshold = int(high_threshold*255)
        
        for histogram in batch_histograms:
            total_pixels = torch.sum(histogram)
            low_exposure_pixels = torch.sum(histogram[:low_threshold])
            low_exposure_ratio = low_exposure_pixels / total_pixels
            high_exposure_pixels = torch.sum(histogram[high_threshold:])
            high_exposure_ratio = high_exposure_pixels / total_pixels
            
            if low_exposure_ratio > 0.1 and high_exposure_ratio > 0.1:
                bimodal_flags.append(torch.tensor(1.0, device=histogram.device))  # True -> 1.0
            else:
                bimodal_flags.append(torch.tensor(0.0, device=histogram.device))  # False -> 0.0
        
        # 리스트를 [B, 1] 크기의 텐서로 변환
        return torch.stack(bimodal_flags, dim=0)
    
    def calculate_batch_histogram_exposure(self, skewness_level, batch_histograms, symmetric_threshold=0.1):
        batch_saturation_level = []
        
        for skewness, histogram in zip(skewness_level, batch_histograms):
            histogram = histogram[0]  # 해당 히스토그램을 이용
            
            # Under saturation
            if skewness < -symmetric_threshold:
                batch_saturation_level.append(torch.tensor(-1.0, device=histogram.device))
            # Over saturation
            elif skewness > symmetric_threshold:
                batch_saturation_level.append(torch.tensor(1.0, device=histogram.device))
            # Symmetric (unimodal)
            else:
                batch_saturation_level.append(torch.tensor(0.0, device=histogram.device))
        
        # 리스트를 [B, 1] 크기의 텐서로 변환
        return torch.stack(batch_saturation_level, dim=0).unsqueeze(1)

        
    def forward(self, frame1, frame2):
        
        f1_histo, f2_histo = self.calculate_histograms(frame1, frame2) # [B, 256, 59]
        f1_histo_global = self.calculate_histogram_global(frame1)
        f2_histo_global = self.calculate_histogram_global(frame2)
        
        f1_histo = f1_histo.permute(0, 2, 1)
        f2_histo = f2_histo.permute(0, 2, 1)
        
        f1_skewness = self.calculate_batch_skewness(f1_histo_global)
        f2_skewness = self.calculate_batch_skewness(f2_histo_global)
        
        # bimodal_f1 = self.is_bimodal(f1_histo)
        # bimodal_f2 = self.is_bimodal(f2_histo)
        
        sat_f1 = self.calculate_batch_histogram_exposure(f1_skewness, f1_histo)
        sat_f2 = self.calculate_batch_histogram_exposure(f2_skewness, f2_histo)
        
        # print(f"f1_skewness values : {f1_skewness}, {f2_skewness}")
        
        # 프레임 별로 독립적인 conv layers 사용
        f1_histo_feature = self.conv_layers_f1(f1_histo).view(f1_histo.size(0), -1)  # [B, 256*16]
        f2_histo_feature = self.conv_layers_f2(f2_histo).view(f2_histo.size(0), -1)  # [B, 256*16]
        
        # Cross-Attention 적용
        f1_cross_att, _ = self.cross_attention(f1_histo_feature.unsqueeze(0), f2_histo_feature.unsqueeze(0), f1_histo_feature.unsqueeze(0))
        f2_cross_att, _ = self.cross_attention(f2_histo_feature.unsqueeze(0), f1_histo_feature.unsqueeze(0), f2_histo_feature.unsqueeze(0))

        f1_cross_att = f1_cross_att.squeeze(0)
        f2_cross_att = f2_cross_att.squeeze(0)

        # 차이 계산 (차이 기반 모듈 통과)
        feature_difference = torch.abs(f1_cross_att - f2_cross_att)
        feature_diff_processed = self.difference_module(feature_difference)  # [B, 128]

        # 추가 feature(skewness, saturation) 포함
        combined_feature_f1 = torch.cat((f1_skewness, sat_f1), dim=1)
        combined_feature_f2 = torch.cat((f2_skewness, sat_f2), dim=1)

        # 노출값 예측
        combined_feature = torch.cat((feature_diff_processed, combined_feature_f1, combined_feature_f2), dim=1)
        exp_values = self.fc_layers(combined_feature)
        exp_values = torch.sigmoid(exp_values) * 4.0  # 0~4 범위로 제한
        exp1, exp2 = exp_values[:, 0], exp_values[:, 1]
        
        # print(f"Net output exp {exp1},{exp2}")
        
        return exp1, exp2
    
    
# Test 241003
# Global histgoram based simple network
class DualFrameExposureNet_Simple(nn.Module):
    def __init__(self):
        super(DualFrameExposureNet_Simple, self).__init__()

        # 차이 기반 모듈 (히스토그램 차이만을 기반으로 하는 단순 모델)
        self.difference_module = nn.Sequential(
            nn.Linear(256, 128),  # 256 = 128(histogram from two frames concatenated)
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # 최종 FC 레이어
        self.fc_layers = nn.Sequential(
            nn.Linear(32 + 4, 16),  # 추가 feature(skewness, saturation) 포함
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),  # 두 프레임의 노출값
            nn.Sigmoid(),
        )

    def histogram_subimage(self, image):
        # 단순한 전역 히스토그램 계산
        batch_size, _, height, width = image.shape
        histograms = []

        for b in range(batch_size):
            # Green channel based
            hist_tensor = torch.histc(image[b, 1], bins=256, min=0, max=1).to(DEVICE)
            histograms.append(hist_tensor)
        
        return torch.stack(histograms, dim=0)  # [B, 256]
    
    def calculate_batch_skewness(self, histograms):
        batch_skewness_level = []
        
        for histogram in histograms:
            frequencies = histogram.to(DEVICE) 
            pixel_values = torch.arange(len(frequencies), device=frequencies.device)  
            fixed_mean = 128
            skewness_numerator = torch.sum(frequencies * ((pixel_values - fixed_mean) ** 3))
            total_pixels = torch.sum(frequencies)
            skewness = skewness_numerator / total_pixels
            skewness /= ((255 - 0) ** 1.5)
            batch_skewness_level.append((skewness / 255).unsqueeze(0))  # [1] 차원 추가
        
        return torch.stack(batch_skewness_level, dim=0)

    def calculate_batch_histogram_exposure(self, skewness_level, histograms, symmetric_threshold=0.1):
        batch_saturation_level = []
        
        for skewness, histogram in zip(skewness_level, histograms):
            # Under saturation
            if skewness < -symmetric_threshold:
                batch_saturation_level.append(torch.tensor(-1.0, device=histogram.device))
            # Over saturation
            elif skewness > symmetric_threshold:
                batch_saturation_level.append(torch.tensor(1.0, device=histogram.device))
            # Symmetric (unimodal)
            else:
                batch_saturation_level.append(torch.tensor(0.0, device=histogram.device))
        
        return torch.stack(batch_saturation_level, dim=0).unsqueeze(1)


    def forward(self, frame1, frame2):
        
        # 1. 전역 히스토그램 계산
        f1_histo = self.histogram_subimage(frame1)  # [B, 256]
        f2_histo = self.histogram_subimage(frame2)  # [B, 256]
        
        # 2. 히스토그램의 skewness 및 saturation 계산
        f1_skewness = self.calculate_batch_skewness(f1_histo)
        f2_skewness = self.calculate_batch_skewness(f2_histo)
        sat_f1 = self.calculate_batch_histogram_exposure(f1_skewness, f1_histo)
        sat_f2 = self.calculate_batch_histogram_exposure(f2_skewness, f2_histo)

        # 3. 두 프레임의 히스토그램 차이 기반 특징 추출
        feature_difference = torch.abs(f1_histo - f2_histo)  # [B, 256]
        feature_diff_processed = self.difference_module(feature_difference)  # [B, 32]

        # 4. 추가 feature(skewness, saturation) 포함
        combined_feature_f1 = torch.cat((f1_skewness, sat_f1), dim=1)  # [B, 2]
        combined_feature_f2 = torch.cat((f2_skewness, sat_f2), dim=1)  # [B, 2]

        # 5. 최종 노출값 예측
        combined_feature = torch.cat((feature_diff_processed, combined_feature_f1, combined_feature_f2), dim=1)  # [B, 32 + 4]
        exp_values = self.fc_layers(combined_feature)  # [B, 2]
        # exp_values = torch.sigmoid(exp_values) *3.5 + 0.5 # 0.5 ~ 4.0
        exp_values = torch.sigmoid(exp_values) * 3.5 + 0.5 

        exp1, exp2 = exp_values[:, 0], exp_values[:, 1]
        return exp1, exp2


# Test 241003
# Histgoram + conv2d based simple network

class DualFrameExposureNet_Conv(nn.Module):
    def __init__(self):
        super(DualFrameExposureNet_Conv, self).__init__()
        
        self.conv_layers_f1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # 입력은 히스토그램, 채널 수를 증가시킴
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.conv_layers_f2 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # 입력은 히스토그램, 채널 수를 증가시킴
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # 차이 기반 모듈 (차원 축소)
        self.difference_module = nn.Sequential(
            nn.Linear(256 * 16 * 16, 1024),  # 16x16 크기의 feature 맵을 flatten하여 사용
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # 최종 FC 레이어
        self.fc_layers = nn.Sequential(
            nn.Linear(128 + 4, 64),  # 추가 value(skewness, saturation) 포함
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # 두 프레임의 노출값
            nn.Sigmoid(),  # Sigmoid로 0~1 사이 값
        )

    def histogram_subimage(self, image, grid_size):
        batch_size, _, height, width = image.shape
    
        grid_height, grid_width = height // grid_size, width // grid_size
        
        batch_histograms = []

        for b in range(batch_size):
            histograms = []
            for i in range(grid_size):
                for j in range(grid_size):
                    sub_image_tensor = image[b, :, i*grid_height:(i+1)*grid_height, j*grid_width:(j+1)*grid_width]
                    hist_tensor = torch.histc(sub_image_tensor[1], bins=256, min=0, max=1).to(image.device)
                    histograms.append(hist_tensor)
            batch_histograms.append(histograms)
        
        # 리스트를 Tensor로 변환
        batch_histograms = torch.stack([torch.stack(hist, dim=0) for hist in batch_histograms], dim=0)
        
        return batch_histograms

    def calculate_histograms(self, frame1, frame2):
        # 다양한 scale에서의 히스토그램을 계산
        histogram_coarest_f1 = self.histogram_subimage(frame1, 1)
        histogram_intermediate_f1 = self.histogram_subimage(frame1, 3)
        histogram_finest_f1 = self.histogram_subimage(frame1, 7)
        
        histogram_coarest_f2 = self.histogram_subimage(frame2, 1)
        histogram_intermediate_f2 = self.histogram_subimage(frame2, 3)
        histogram_finest_f2 = self.histogram_subimage(frame2, 7)
        
        return histogram_coarest_f1, histogram_intermediate_f1, histogram_finest_f1, \
               histogram_coarest_f2, histogram_intermediate_f2, histogram_finest_f2

    def calculate_batch_skewness(self, batch_histograms):
        batch_skewness_level = []
        
        for histogram in batch_histograms:
            frequencies = histogram.to(histogram.device) 
            pixel_values = torch.arange(len(frequencies), device=frequencies.device)  
            fixed_mean = 128
            skewness_numerator = torch.sum(frequencies * ((pixel_values - fixed_mean) ** 3))
            total_pixels = torch.sum(frequencies)
            skewness = skewness_numerator / total_pixels
            skewness /= ((255 - 0) ** 1.5)
            batch_skewness_level.append((skewness/255).unsqueeze(0))  # [1] 차원 추가
        
        return torch.stack(batch_skewness_level, dim=0)

    def calculate_batch_histogram_exposure(self, skewness_level, batch_histograms, symmetric_threshold=0.1):
        batch_saturation_level = []
        
        for skewness, histogram in zip(skewness_level, batch_histograms):
            histogram = histogram[0]
            
            if skewness < -symmetric_threshold:
                batch_saturation_level.append(torch.tensor(-1.0, device=histogram.device))
            elif skewness > symmetric_threshold:
                batch_saturation_level.append(torch.tensor(1.0, device=histogram.device))
            else:
                batch_saturation_level.append(torch.tensor(0.0, device=histogram.device))
        
        return torch.stack(batch_saturation_level, dim=0).unsqueeze(1)
    
    def forward(self, frame1, frame2):
        # frame1, frame2의 히스토그램 계산
        f1_histo_coarest, f1_histo_intermediate, f1_histo_finest, \
        f2_histo_coarest, f2_histo_intermediate, f2_histo_finest = self.calculate_histograms(frame1, frame2)
        
        # 각 프레임에 대해 히스토그램에서 skewness 계산
        f1_skewness = self.calculate_batch_skewness(f1_histo_coarest)
        f2_skewness = self.calculate_batch_skewness(f2_histo_coarest)
        
        # skewness 값을 기반으로 한 saturation 계산
        sat_f1 = self.calculate_batch_histogram_exposure(f1_skewness, f1_histo_coarest)
        sat_f2 = self.calculate_batch_histogram_exposure(f2_skewness, f2_histo_coarest)
        
        # frame1, frame2의 히스토그램을 Conv2D로 처리
        f1_histo_feature = self.conv_layers_f1(f1_histo_coarest.unsqueeze(1))  # [B, 1, H, W]로 변환
        f2_histo_feature = self.conv_layers_f2(f2_histo_coarest.unsqueeze(1))  # [B, 1, H, W]로 변환
        
        # Flatten하여 차이 계산
        f1_histo_feature_flat = f1_histo_feature.view(f1_histo_feature.size(0), -1)
        f2_histo_feature_flat = f2_histo_feature.view(f2_histo_feature.size(0), -1)
        
        # 차이 계산
        feature_difference = torch.abs(f1_histo_feature_flat - f2_histo_feature_flat)
        feature_diff_processed = self.difference_module(feature_difference)  # [B, 128]
        
        # skewness와 saturation 결합
        combined_feature_f1 = torch.cat((f1_skewness, sat_f1), dim=1)
        combined_feature_f2 = torch.cat((f2_skewness, sat_f2), dim=1)
        
        # 최종 feature 결합 및 노출값 예측
        combined_feature = torch.cat((feature_diff_processed, combined_feature_f1, combined_feature_f2), dim=1)
        exp_values = self.fc_layers(combined_feature)
        exp_values = torch.sigmoid(exp_values) * 4.0  # 0~4 범위로 제한
        exp1, exp2 = exp_values[:, 0], exp_values[:, 1]
        
        return exp1, exp2

## Test
## Semantic features + histogram + statistic value

class DualFrameExposureNet_ResNet(nn.Module):
    def __init__(self):
        super(DualFrameExposureNet_ResNet, self).__init__()
        
        # ResNet backbone에서 feature extraction을 위한 encoder
        resnet = models.resnet18(pretrained=True)  # Pretrained ResNet18 사용
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # 마지막 두 레이어 제거 (Global Avg Pooling, FC Layer)

        # 차이 기반 모듈 (히스토그램 차이 및 ResNet feature 차이 결합)
        self.difference_module = nn.Sequential(
            nn.Linear(512 + 256, 256),  # 512(ResNet feature 차이) + 256(histogram 차이)
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # 최종 FC 레이어
        self.fc_layers = nn.Sequential(
            nn.Linear(32 + 4, 16),  # 추가 feature(skewness, saturation) 포함
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),  # 두 프레임의 노출값
            nn.Sigmoid(),
        )

    def histogram_subimage(self, image):
        # 단순한 전역 히스토그램 계산
        batch_size, _, height, width = image.shape
        histograms = []

        for b in range(batch_size):
            # Green channel based
            hist_tensor = torch.histc(image[b, 1], bins=256, min=0, max=1).to(DEVICE)
            histograms.append(hist_tensor)
        
        return torch.stack(histograms, dim=0)  # [B, 256]
    
    def calculate_batch_skewness(self, histograms):
        batch_skewness_level = []
        
        for histogram in histograms:
            frequencies = histogram.to(DEVICE) 
            pixel_values = torch.arange(len(frequencies), device=frequencies.device)  
            fixed_mean = 128
            skewness_numerator = torch.sum(frequencies * ((pixel_values - fixed_mean) ** 3))
            total_pixels = torch.sum(frequencies)
            skewness = skewness_numerator / total_pixels
            skewness /= ((255 - 0) ** 1.5)
            batch_skewness_level.append((skewness / 255).unsqueeze(0))  # [1] 차원 추가
        
        return torch.stack(batch_skewness_level, dim=0)

    def calculate_batch_histogram_exposure(self, skewness_level, histograms, symmetric_threshold=0.1):
        batch_saturation_level = []
        
        for skewness, histogram in zip(skewness_level, histograms):
            # Under saturation
            if skewness < -symmetric_threshold:
                batch_saturation_level.append(torch.tensor(-1.0, device=histogram.device))
            # Over saturation
            elif skewness > symmetric_threshold:
                batch_saturation_level.append(torch.tensor(1.0, device=histogram.device))
            # Symmetric (unimodal)
            else:
                batch_saturation_level.append(torch.tensor(0.0, device=histogram.device))
        
        return torch.stack(batch_saturation_level, dim=0).unsqueeze(1)

    def forward(self, frame1, frame2):
        # 1. ResNet Encoder를 사용하여 두 프레임의 feature 추출
        f1_resnet = self.encoder(frame1)  # [B, 512, H/32, W/32]
        f2_resnet = self.encoder(frame2)  # [B, 512, H/32, W/32]

        # 2. ResNet feature에서 차이를 계산
        feature_diff_resnet = torch.abs(f1_resnet - f2_resnet)  # [B, 512, H/32, W/32]
        feature_diff_resnet = torch.mean(feature_diff_resnet, dim=[2, 3])  # [B, 512] (공간 평균)

        # 3. 전역 히스토그램 계산
        f1_histo = self.histogram_subimage(frame1)  # [B, 256]
        f2_histo = self.histogram_subimage(frame2)  # [B, 256]
        
        # 4. 히스토그램의 skewness 및 saturation 계산
        f1_skewness = self.calculate_batch_skewness(f1_histo)
        f2_skewness = self.calculate_batch_skewness(f2_histo)
        sat_f1 = self.calculate_batch_histogram_exposure(f1_skewness, f1_histo)
        sat_f2 = self.calculate_batch_histogram_exposure(f2_skewness, f2_histo)

        # 5. 히스토그램 차이 기반 특징 추출
        feature_difference_histo = torch.abs(f1_histo - f2_histo)  # [B, 256]

        # 6. ResNet feature 차이와 히스토그램 차이를 결합
        combined_difference_feature = torch.cat((feature_diff_resnet, feature_difference_histo), dim=1)  # [B, 512 + 256]

        # 7. 차이 기반 모듈을 통해 feature 처리
        feature_diff_processed = self.difference_module(combined_difference_feature)  # [B, 32]

        # 8. 추가 feature(skewness, saturation) 포함
        combined_feature_f1 = torch.cat((f1_skewness, sat_f1), dim=1)  # [B, 2]
        combined_feature_f2 = torch.cat((f2_skewness, sat_f2), dim=1)  # [B, 2]

        # 9. 최종 노출값 예측
        combined_feature = torch.cat((feature_diff_processed, combined_feature_f1, combined_feature_f2), dim=1)  # [B, 32 + 4]
        exp_values = self.fc_layers(combined_feature)  # [B, 2]
        exp_values = torch.sigmoid(exp_values) * 3.5 + 0.5  # 0.5 ~ 4.0로 scaling
        # exp_values = torch.clamp(exp_values, 0.25, 4.0)

        exp1, exp2 = exp_values[:, 0], exp_values[:, 1]
        return exp1, exp2


class HybridExposureAdjustmentNet(nn.Module):
    def __init__(self, feature_dim, exposure_dim):
        super(HybridExposureAdjustmentNet, self).__init__()

        # Feature map 처리 (두 프레임에서 추출된 feature map을 합침)
        self.feature_pool = nn.AdaptiveAvgPool2d((1, 1))  # feature map을 평균 풀링으로 차원 축소
        self.feature_fc1 = nn.Linear(feature_dim * 2, 128)  # 축소된 feature map을 사용
        self.feature_fc2 = nn.Linear(128, 64)

        # 초기 노출 값을 위한 경로
        self.exposure_fc1 = nn.Linear(exposure_dim * 2, 64)
        self.exposure_fc2 = nn.Linear(64, 32)

        # Feature와 노출 값을 합쳐 최종 노출 조정 값을 계산
        self.merge_fc = nn.Linear(96, 32)  # 64 (features) + 32 (exposure)
        self.output_fc = nn.Linear(32, 2)  # 2개의 출력 (프레임 1과 2의 노출 조정 값)

    def forward(self, fmap1, fmap2, exp1, exp2):
        # feature map을 평균 풀링으로 차원 축소
        fmap1 = self.feature_pool(fmap1).view(fmap1.size(0), -1)  # [batch_size, feature_dim]
        fmap2 = self.feature_pool(fmap2).view(fmap2.size(0), -1)

        # 두 프레임의 feature map을 합침
        concat_features = torch.cat((fmap1, fmap2), dim=1)

        # Feature map 처리
        x_features = F.relu(self.feature_fc1(concat_features))
        x_features = F.relu(self.feature_fc2(x_features))

        # 초기 노출 값을 처리 (차원 맞춰줌)
        # # exp1과 exp2가 1차원 또는 스칼라일 경우 차원을 맞춰줍니다.
        # exp1 = exp1.unsqueeze(0) if exp1.dim() == 0 else exp1
        # exp2 = exp2.unsqueeze(0) if exp2.dim() == 0 else exp2
        concat_exposures = torch.cat((exp1, exp2), dim=1)
        concat_exposures = concat_exposures.view(concat_exposures.size(0), -1)  # 1x2 -> 적절히 확장
        x_exposures = F.relu(self.exposure_fc1(concat_exposures))
        x_exposures = F.relu(self.exposure_fc2(x_exposures))

        # Feature와 노출 값을 합침
        x_merge = torch.cat((x_features, x_exposures), dim=1)
        x_merge = F.relu(self.merge_fc(x_merge))

        # 노출 조정 값 출력 (프레임 1과 2에 대한 노출 조정 값)
        exposure_adjustments = self.output_fc(x_merge)
        return exposure_adjustments
    
class HybridExposureAdjustmentNet_Spatial(nn.Module):
    def __init__(self, feature_dim, exposure_dim):
        super(HybridExposureAdjustmentNet_Spatial, self).__init__()

        # Feature map 처리 (맥스 풀링 추가)
        self.feature_pool = nn.AdaptiveAvgPool2d((6, 6))  # 평균 풀링
        self.feature_fc1 = nn.Linear(feature_dim * 4 * 36, 256)  # 맥스/평균 풀링 결합
        self.feature_fc2 = nn.Linear(256, 64)
        
        # BatchNorm 및 Dropout 추가
        self.bn1 = nn.InstanceNorm1d(256)
        self.bn2 = nn.InstanceNorm1d(64)

        # 초기 노출 값을 위한 경로
        self.exposure_fc1 = nn.Linear(exposure_dim * 2, 32)
        self.exposure_fc2 = nn.Linear(32, 16)

        # Feature와 노출 값을 합쳐 최종 노출 조정 값을 계산
        self.merge_fc = nn.Linear(80, 32)  # 64 (features) + 32 (exposure)
        self.output_fc = nn.Linear(32, 2)  # 2개의 출력 (프레임 1과 2의 노출 조정 값)

    def forward(self, fmap1, fmap2, exp1, exp2):
        # feature map을 평균 풀링과 맥스 풀링으로 차원 축소
        fmap1_avg = self.feature_pool(fmap1).view(fmap1.size(0), -1)
        fmap2_avg = self.feature_pool(fmap2).view(fmap2.size(0), -1)
        fmap1_max = F.adaptive_max_pool2d(fmap1, (6, 6)).view(fmap1.size(0), -1)
        fmap2_max = F.adaptive_max_pool2d(fmap2, (6, 6)).view(fmap2.size(0), -1)

        # 두 프레임의 feature map을 합침
        concat_features = torch.cat((fmap1_avg, fmap2_avg, fmap1_max, fmap2_max), dim=1)
        print(f"concat_features size: {concat_features.size()}")  # 크기 출력

        # Feature map 처리
        x_features = F.relu(self.bn1(self.feature_fc1(concat_features)))
        x_features = F.relu(self.bn2(self.feature_fc2(x_features)))
        print(f"x_features size: {x_features.size()}")  # 크기 출력

        # 초기 노출 값을 처리
        concat_exposures = torch.cat((exp1, exp2), dim=1).view(exp1.size(0), -1)
        x_exposures = F.relu(self.exposure_fc1(concat_exposures))
        x_exposures = F.relu(self.exposure_fc2(x_exposures))
        print(f"x_exposures size: {x_exposures.size()}")  # 크기 출력

        # Feature와 노출 값을 합침
        x_merge = torch.cat((x_features, x_exposures), dim=1)
        print(f"x_merge size: {x_merge.size()}")  # 크기 출력

        # 노출 조정 값 출력
        x_merge = F.relu(self.merge_fc(x_merge))
        exposure_adjustments = torch.tanh(self.output_fc(x_merge))  # [-1, 1] 범위
        return exposure_adjustments



class FiLM(nn.Module):
    def __init__(self, feature_dim):
        super(FiLM, self).__init__()
        self.gamma = nn.Linear(2, feature_dim)  # 노출 값에 따른 가중치 학습
        self.beta = nn.Linear(2, feature_dim)   # 노출 값에 따른 편향 학습

    def forward(self, feature, exposure):
        gamma = self.gamma(exposure)
        beta = self.beta(exposure)
        return gamma * feature + beta


class HybridExposureAdjustmentNet_FiLM(nn.Module):
    def __init__(self, feature_dim, exposure_dim):
        super(HybridExposureAdjustmentNet_FiLM, self).__init__()

        # Feature map 처리
        self.feature_fc1 = nn.Linear(feature_dim * 2, 128)
        self.feature_fc2 = nn.Linear(128, 64)

        # FiLM 기법을 사용하여 노출 값에 따라 feature 조정
        self.film = FiLM(64)

        # 초기 노출 값을 위한 경로
        self.exposure_fc1 = nn.Linear(exposure_dim * 2, 64)
        self.exposure_fc2 = nn.Linear(64, 32)

        # Feature와 노출 값을 합쳐 최종 노출 조정 값을 계산
        self.merge_fc = nn.Linear(96, 32)
        self.output_fc = nn.Linear(32, 2)

    def forward(self, fmap1, fmap2, exp1, exp2):
        # 두 프레임의 feature map을 결합
        fmap1 = fmap1.view(fmap1.size(0), -1)
        fmap2 = fmap2.view(fmap2.size(0), -1)
        concat_features = torch.cat((fmap1, fmap2), dim=1)

        # Feature map 처리
        x_features = F.relu(self.feature_fc1(concat_features))
        x_features = F.relu(self.feature_fc2(x_features))

        # FiLM 기법 적용
        concat_exposures = torch.cat((exp1, exp2), dim=1)
        x_features = self.film(x_features, concat_exposures)

        # 초기 노출 값을 처리
        x_exposures = F.relu(self.exposure_fc1(concat_exposures))
        x_exposures = F.relu(self.exposure_fc2(x_exposures))

        # Feature와 노출 값을 합침
        x_merge = torch.cat((x_features, x_exposures), dim=1)
        x_merge = F.relu(self.merge_fc(x_merge))

        # 노출 조정 값 출력
        exposure_adjustments = self.output_fc(x_merge)
        return exposure_adjustments

class HybridExposureAdjustmentNet_Attention(nn.Module):
    def __init__(self, feature_dim, exposure_dim):
        super(HybridExposureAdjustmentNet_Attention, self).__init__()

        # Self-Attention 기법을 사용한 feature map 처리
        self.feature_fc1 = nn.Linear(feature_dim * 2, 128)
        self.feature_fc2 = nn.Linear(128, 64)

        self.attention_layer = nn.MultiheadAttention(embed_dim=64, num_heads=8)

        # 초기 노출 값을 위한 경로
        self.exposure_fc1 = nn.Linear(exposure_dim * 2, 64)
        self.exposure_fc2 = nn.Linear(64, 32)

        # Feature와 노출 값을 합쳐 최종 노출 조정 값을 계산
        self.merge_fc = nn.Linear(96, 32)  # 64 (features) + 32 (exposure)
        self.output_fc = nn.Linear(32, 2)  # 2개의 출력 (프레임 1과 2의 노출 조정 값)

    def forward(self, fmap1, fmap2, exp1, exp2):
        # 두 프레임의 feature map을 결합
        fmap1 = fmap1.view(fmap1.size(0), -1)
        fmap2 = fmap2.view(fmap2.size(0), -1)
        concat_features = torch.cat((fmap1, fmap2), dim=1)
        
        # Feature map 처리
        x_features = F.relu(self.feature_fc1(concat_features))
        x_features = F.relu(self.feature_fc2(x_features))
        
        # Attention 적용
        attn_output, _ = self.attention_layer(x_features.unsqueeze(0), x_features.unsqueeze(0), x_features.unsqueeze(0))
        x_features = attn_output.squeeze(0)

        # 초기 노출 값을 처리
        concat_exposures = torch.cat((exp1, exp2), dim=1)
        x_exposures = F.relu(self.exposure_fc1(concat_exposures))
        x_exposures = F.relu(self.exposure_fc2(x_exposures))

        # Feature와 노출 값을 합침
        x_merge = torch.cat((x_features, x_exposures), dim=1)
        x_merge = F.relu(self.merge_fc(x_merge))

        # 노출 조정 값 출력
        exposure_adjustments = self.output_fc(x_merge)
        return exposure_adjustments


class AlphaAdjustmentNet(nn.Module):
    def __init__(self, feature_dim):
        super(AlphaAdjustmentNet, self).__init__()
        
        # feature map 크기 256 채널을 받음
        self.feature_pool = nn.AdaptiveAvgPool2d((4, 4))  # 4x4 크기로 줄임
        self.feature_fc1 = nn.Linear(feature_dim * 2 * 16, 128)  # 두 개의 feature map을 합쳐서 처리
        self.feature_fc2 = nn.Linear(128, 64)
        self.fc_output = nn.Linear(64, 2)  # Frame1과 Frame2의 alpha 값을 예측
        
    def forward(self, fmap1, fmap2):
        # feature map을 평균 풀링으로 차원 축소
        fmap1 = self.feature_pool(fmap1).view(fmap1.size(0), -1)  # [batch_size, feature_dim]
        fmap2 = self.feature_pool(fmap2).view(fmap2.size(0), -1)
        
        # 두 feature map을 concatenate하여 결합
        combined_features = torch.cat((fmap1, fmap2), dim=1)
        
        # Fully connected layer 통과 (alpha 값을 예측)
        x = F.relu(self.feature_fc1(combined_features))
        x = F.relu(self.feature_fc2(x))
        alphas = torch.sigmoid(self.fc_output(x))  # [0, 1] 사이로 제한
        
        return alphas  # Frame1과 Frame2에 대한 alpha 값
    
    
class BasicExposureCorrectionNet(nn.Module):
    def __init__(self):
        super(BasicExposureCorrectionNet, self).__init__()
        # FCC
        self.fc1 = nn.Linear(2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ExposureRefinementNet(nn.Module):
    def __init__(self, feature_dim=128):
        super(ExposureRefinementNet, self).__init__()

        # Feature Extraction for each frame
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, feature_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Frame-wise Feature Aggregation
        self.fc_aggregate = nn.Linear(2 * feature_dim + 4, 64)  # 두 프레임의 feature와 rule-based exposure 값을 합친 크기

        # Relationship Module for fine-tuning exposure adjustments
        self.fc_relationship = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # output exposure adjustments for two frames
        )

    def forward(self, rule_exp1, rule_exp2, histo_ldr1, histo_ldr2, skewness_level1, skewness_level2):
        # Convert rule-based exposures and skewness levels to torch tensors if they are lists
        if isinstance(rule_exp1, list):
            rule_exp1 = torch.tensor(rule_exp1, device=histo_ldr1.device)
        if isinstance(rule_exp2, list):
            rule_exp2 = torch.tensor(rule_exp2, device=histo_ldr1.device)
        if isinstance(skewness_level1, list):
            skewness_level1 = torch.tensor(skewness_level1, device=histo_ldr1.device)
        if isinstance(skewness_level2, list):
            skewness_level2 = torch.tensor(skewness_level2, device=histo_ldr1.device)

        # Ensure rule-based exposures and skewness levels have batch dimension
        rule_exp1 = rule_exp1.view(1, 1)  # (1, 1)
        rule_exp2 = rule_exp2.view(1, 1)  # (1, 1)
        skewness_level1 = skewness_level1.view(1, 1)  # (1, 1)
        skewness_level2 = skewness_level2.view(1, 1)  # (1, 1)
        
        # Extract features from histograms of each frame
        histo_ldr1 = histo_ldr1/histo_ldr1.sum()
        histo_ldr2 = histo_ldr2/histo_ldr2.sum()
        histo_ldr1 = histo_ldr1.unsqueeze(0)  # (1, 1, 256)
        histo_ldr2 = histo_ldr2.unsqueeze(0)  # (1, 1, 256)
        
        # Feature extraction and dimensionality reduction
        feature1 = self.feature_extractor(histo_ldr1).mean(dim=2)  # (1, feature_dim)
        feature2 = self.feature_extractor(histo_ldr2).mean(dim=2)  # (1, feature_dim)
        
        # Concatenate frame-wise features with skewness and rule-based exposures
        frame_relation_features = torch.cat([
            feature1, feature2, skewness_level1, skewness_level2, rule_exp1, rule_exp2
        ], dim=1)

        # Aggregate frame-wise features
        x = F.relu(self.fc_aggregate(frame_relation_features))
        
        # Relationship module to produce refined exposure adjustments
        output_exposures = self.fc_relationship(x)
        
        # Return exposure adjustments
        return output_exposures
