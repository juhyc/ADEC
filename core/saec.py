import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from PIL import Image
import matplotlib.pyplot as plt

# * Exposure control network

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
        
        self.exp_min = M_exp**-1
        self.exp_max = M_exp

        self.conv_layers = nn.Sequential(
            # Make sure the input channel is 59, or adjust
            nn.Conv1d(118, 256, kernel_size=4, stride=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=4, stride=4),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 1024, kernel_size=4, stride=4),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(4096, 1024), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64,1)
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
        
        # x = (2 * (torch.sigmoid(x) - 0.5)) * self.M_exp
        # x = (torch.sigmoid(x) * self.M_exp - 0.5)
        # x = x * (self.exp_max - self.exp_min) + self.exp_min
        # x = (2 * torch.sigmoid(x) - 0.5) * self.M_exp_log.to(x.device)
        x = torch.sigmoid(x)
        
        return x

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

# * Exposure shift functions

def exposure_shift(before_exposure, predicted_exposure, alpha = 0.2, M_exp = 4.0, device= DEVICE):
    H_exp = torch.tensor(M_exp).to(device)
    L_exp = torch.tensor(M_exp**-1).to(device)
    
    difference = predicted_exposure - before_exposure
    adjusted_difference = alpha * difference
    shifted_exposure = before_exposure + adjusted_difference

    # Set exposure under, upper value based on camera paramet M_exp.
    shifted_exposure_clamped = torch.max(torch.min(shifted_exposure, H_exp), L_exp)
    
    return shifted_exposure_clamped

def exposure_shift2(before_exposure, sigmoid_output, M_exp=4.0, alpha = 0.3, device= DEVICE):

    min_exposure, max_exposure = 1/M_exp, M_exp
    target_exposure = min_exposure + (max_exposure - min_exposure) * sigmoid_output
    
    adjusted_difference = (target_exposure - before_exposure) * alpha
    shifted_exposure = before_exposure + adjusted_difference

    shifted_exposure_clamped = torch.clamp(shifted_exposure, min_exposure, max_exposure)

    return shifted_exposure_clamped

def exposure_shift_by_threshold(before_exposure, predicted_exposure, smoothing = 0.9, threshold = 1):
    if before_exposure < threshold:
        shifted_exposure = before_exposure * predicted_exposure**(2-smoothing)
    else:
        shifted_exposure = before_exposure * predicted_exposure**(1-smoothing)
    
    return shifted_exposure