import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

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
        else:
            # Handle unimodal cases
            # e_high,under e_low_under case both increase
            if sat_f1 < 0 and sat_f2 < 0:
                shifted_exp_f1[i] = exposure_shift(exp_f1, alpha=a1, decrease=False)
                shifted_exp_f2[i] = exposure_shift(exp_f2, alpha=a2, decrease=False)
                
            # e_high,appro e_low,under both increase
            elif sat_f1 == 0 and sat_f2 < 0 and exp_f1 > exp_f2:
                shifted_exp_f1[i] = exposure_shift(exp_f1, alpha=a1, decrease=False)
                shifted_exp_f2[i] = exposure_shift(exp_f2, alpha=a2, decrease=False)
            elif sat_f1 < 0 and sat_f2 == 0 and exp_f1 < exp_f2:
                shifted_exp_f1[i] = exposure_shift(exp_f1, alpha=a1, decrease=False)
                shifted_exp_f2[i] = exposure_shift(exp_f2, alpha=a2, decrease=False)
            
            # e_high,appro e_low,appro case
            elif sat_f1 == 0 and sat_f2 == 0:
                # exp_f1 is high exposure case, increase high, decrease low
                if exp_f1 > exp_f2:
                    shifted_exp_f1[i] = exposure_shift(exp_f1, alpha=a1, decrease=False)
                    shifted_exp_f2[i] = exposure_shift(exp_f2, alpha=a2, decrease=True)
                # exp_f1 is low exposure case
                else:
                    shifted_exp_f1[i] = exposure_shift(exp_f1, alpha=a1, decrease=True)
                    shifted_exp_f2[i] = exposure_shift(exp_f2, alpha=a2, decrease=False)
                    
            # e_high,over  e_low,under case, increase low, decrease high
            elif sat_f1 > 0 and sat_f2 < 0 and exp_f1 > exp_f2:
                shifted_exp_f1[i] = exposure_shift(exp_f1, alpha=a1, decrease=True)
                shifted_exp_f2[i] = exposure_shift(exp_f2, alpha=a2, decrease=False)
            elif sat_f1 < 0 and sat_f2 > 0 and exp_f1 < exp_f2:
                shifted_exp_f1[i] = exposure_shift(exp_f1, alpha=a1, decrease=False)
                shifted_exp_f2[i] = exposure_shift(exp_f2, alpha=a2, decrease=True)
            
            # e_high,over  e_low,appro, both decrease
            elif sat_f1 > 0 and sat_f2 == 0 and exp_f1 > exp_f2:
                shifted_exp_f1[i] = exposure_shift(exp_f1, alpha=a1, decrease=True)
                shifted_exp_f2[i] = exposure_shift(exp_f2, alpha=a2, decrease=True)
            elif sat_f1 == 0 and sat_f2 > 0 and exp_f1 < exp_f2:
                shifted_exp_f1[i] = exposure_shift(exp_f1, alpha=a1, decrease=True)
                shifted_exp_f2[i] = exposure_shift(exp_f2, alpha=a2, decrease=True)
                
            # e_high,over  e_low,over, both decrease
            elif sat_f1 > 0 and sat_f2 > 0:
                shifted_exp_f1[i] = exposure_shift(exp_f1, decrease=True, alpha=a1)
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
            elif skew_diff > max_skew_diff:
                if new_skew_f1 > new_skew_f2:
                    print("Case 13")
                    new_exp_f1 = exposure_shift(new_exp_f1, alpha=a1, decrease=True)
                    new_exp_f2 = exposure_shift(new_exp_f2, alpha=a2, decrease=False)
                else:
                    print("Case 14")
                    new_exp_f1 = exposure_shift(new_exp_f1, alpha=a1, decrease=False)
                    new_exp_f2 = exposure_shift(new_exp_f2, alpha=a2, decrease=True)
            
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

def exposure_shift(exp, alpha, decrease=False):
    if decrease:
        shifted_exp = exp - (alpha * exp)
    else:
        shifted_exp = exp + (alpha * exp)
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

#^ Nerual Exposure Netowrk

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
        
        # Convolutional layers for feature extraction from each frame
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Fully connected layers for regression
        self.fc1 = nn.Linear(128 * 300 * 400 * 2, 512)  # Assuming input images are 32x32 and we have 2 frames
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)  # Predicting alpha for both frames
    
    def forward(self, frame1, frame2):
        # Feature extraction for frame1
        x1 = F.relu(self.conv1(frame1))
        x1 = F.max_pool2d(x1, 2)
        x1 = F.relu(self.conv2(x1))
        x1 = F.max_pool2d(x1, 2)
        x1 = F.relu(self.conv3(x1))
        x1 = F.max_pool2d(x1, 2)
        
        # Feature extraction for frame2
        x2 = F.relu(self.conv1(frame2))
        x2 = F.max_pool2d(x2, 2)
        x2 = F.relu(self.conv2(x2))
        x2 = F.max_pool2d(x2, 2)
        x2 = F.relu(self.conv3(x2))
        x2 = F.max_pool2d(x2, 2)
        
        # Flatten the tensors
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        
        # Concatenate features from both frames
        x = torch.cat((x1, x2), dim=1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        alpha = torch.sigmoid(self.fc3(x))  # Predicting two alpha values
        
        return alpha