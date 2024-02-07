import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from PIL import Image
import matplotlib.pyplot as plt


#  & Calculate histrogram by sub-images with different scale
def histogram_subimage(image, grid_size):
    """calculate histogram by Green Channel

    Args:
        image (tensor): batch of images [B, C, H, W]
        grid_size (int): grid size to divide original image

    Returns:
        list of tensors: histogram values based on green channel for each image in the batch
    """
    if isinstance(image, Image.Image):
        to_tensor = ToTensor()
        image = to_tensor(image)

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
    
# & Global image feature branch network

class GlobalFeatureNet(nn.Module):
    def __init__(self, M_exp=4):
        super(GlobalFeatureNet, self).__init__()

        self.conv_layers = nn.Sequential(
            # Make sure the input channel is 59, or adjust
            nn.Conv1d(118, 128, kernel_size=4, stride=4),  
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=4, stride=4),
            nn.ReLU(),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(2048, 512),  # Adjust this if needed
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        self.M_exp = M_exp
        self.M_exp_log = torch.log(torch.tensor([self.M_exp], dtype=torch.float))

    # def forward(self, x):
    #     # ! check batch size
    #     # Permute the dimensions to [batch, channel, histogram]
    #     x = x.permute(2, 0, 1)

    #     x = self.conv_layers(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.fc_layers(x)

    #     u_t= (2 * torch.sigmoid(x) - 0.5) * self.M_exp_log.to(x.device)
    #     # u_t = torch.exp(2 * torch.sigmoid(x) - 0.5) * self.M_exp_log.to(x.device)
    #     return u_t
    def forward(self, x):
        x = x.permute(2, 0, 1)
        
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        exp_t = (2 * torch.sigmoid(x) - 0.5) * self.M_exp_log.to(x.device)
        
        return exp_t


def exposure_shift(before_exposure, predicted_exposure, alpha = 0.2):
    difference = predicted_exposure - before_exposure
    adjusted_difference = alpha * difference
    shifted_exposure = before_exposure + adjusted_difference

    return shifted_exposure



# def exposure_shift(before_exposure, predicted_exposure, smoothing = 0.9):
#     shifted_exposure = before_exposure * predicted_exposure**(1-smoothing)
    
#     return shifted_exposure

def exposure_shift_by_threshold(before_exposure, predicted_exposure, smoothing = 0.9, threshold = 1):
    if before_exposure < threshold:
        shifted_exposure = before_exposure * predicted_exposure**(2-smoothing)
    else:
        shifted_exposure = before_exposure * predicted_exposure**(1-smoothing)
    
    return shifted_exposure