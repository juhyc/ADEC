import sys
sys.path.append('core')

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.transforms import ToTensor
from PIL import Image
import matplotlib.pyplot as plt
from utils.mask import soft_binary_threshold_batch
from core.saec import *
from core.utils.simulate import *
from core.raft_stereo import RAFTStereo
from core.utils.read_utils import prepare_inputs_custom
from core.utils.utils import InputPadder
from core.extractor import BasicEncoder
from core.disparity_fusion_ResUnet import DisparityFusion_ResUnet
import core.stereo_datasets as datasets

from core.saec import AlphPredictionNet

from ptlflow import get_model
from ptlflow.utils import flow_utils
from ptlflow.utils.io_adapter import IOAdapter
import cv2


DEVICE = 'cuda'
################################
# * End to end pipeline with exposure control not utilize network
# * Considering sequence image
# * Add warping
# * Modify the model to accept the initial exposure setting as an external input. 
################################

def load_image(imfile):
    img = np.array(imfile).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

# Quantize STE method
class QuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, n=8):
        # LDR image max value
        max_val = 2**n - 1
        
        # Quantize
        x_scaled = input * max_val
        x_clamped = torch.clamp(x_scaled, 0, max_val)   
        x_quantized = torch.round(x_clamped).to(torch.uint8)
        
        # Normalized to 0~1
        x_dequantized = x_quantized / max_val
        return x_dequantized

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

# * End to end pipeline using a simple algorithm instead of network.
# Todo) Changing the method of controlling exposure from using a alogorithm to using a network.

#^ Load the optical flow model
flow_model = get_model('raft_small', pretrained_ckpt='things')
flow_model.to(DEVICE)  # Assuming you are using a GPU
flow_model.eval()

############################################################
def adjust_exposure(image, current_exposure, target_exposure):
    """
    Adjusts the exposure of an image based on the current and target exposure values.
    """
    exposure_factor = target_exposure / current_exposure
    adjusted_image = image * exposure_factor
    return adjusted_image

def adjust_exposure_batch(image_batch, current_exposure, target_exposure):
    adjusted_images = []
    for img in image_batch:
        adjusted_img = adjust_exposure(img, current_exposure, target_exposure)
        adjusted_images.append(adjusted_img)
    return torch.stack(adjusted_images)
############################################################

class CombineModel_wo_net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.RAFTStereo = RAFTStereo(args)
        self.flow = flow_model
        self.alpha_net = ExposureAdjustmentPipeline()
    
    def convert_to_tensor(self, image):
        if isinstance(image, Image.Image):
            to_tensor = ToTensor()
            return to_tensor(image)
        return image
    
    def compute_optical_flow_batch(self, model, image_pairs, exposure1, exposure2):
        device = next(model.parameters()).device if next(model.parameters(), None) is not None else 'cuda'
        
        target_expousre = torch.sqrt(exposure1 * exposure2)

        all_flows = []
        for left_image, right_image in image_pairs:
            # Adjust exposures to a common value (e.g., exposure1)
            left_image = adjust_exposure(left_image, exposure1, target_expousre)
            right_image = adjust_exposure(right_image, exposure2, target_expousre)

            left_image = left_image.to(device)
            right_image = right_image.to(device)
            inputs = prepare_inputs_custom([left_image, right_image])
            inputs = {k: v.to(device) for k, v in inputs.items()} 

            with torch.no_grad():
                predictions = model(inputs)

            # Extract the flows
            flows = predictions['flows']
            all_flows.append(flows)
        
        return torch.cat(all_flows, dim=0).squeeze(1)
    
    @staticmethod
    def warp_image(image, flow):
        image = image.float()
        B, C, H, W = image.size()
        flow = flow.permute(0, 2, 3, 1)  # [B, H, W, 2]

        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=flow.device), torch.arange(W, device=flow.device))
        grid = torch.stack((grid_x, grid_y), 2).float()  # [H, W, 2]
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 2]

        flow = flow + grid  # [B, H, W, 2]

        flow = 2 * flow / torch.tensor([W-1, H-1], device=flow.device).float() - 1

        warped_image = F.grid_sample(image, flow, mode='bilinear', padding_mode='zeros', align_corners=False)

        return warped_image

    def forward(self, left_hdr, right_hdr, left_next_hdr, right_next_hdr, initial_exp_high, initial_exp_low, iters=32, valid_mode=False, test_mode=False, train_mode=False):
        
        #^ Capture simulator module
        phi_l_exph = ImageFormation(left_hdr, initial_exp_high, device=DEVICE)
        phi_r_exph = ImageFormation(right_hdr, initial_exp_high, device=DEVICE)
        phi_l_expl = ImageFormation(left_next_hdr, initial_exp_low, device=DEVICE)
        phi_r_expl = ImageFormation(right_next_hdr, initial_exp_low, device=DEVICE)
        
        #^ Captured LDR image pair for frame1,2
        ldr_left_exph_cap = QuantizeSTE.apply(phi_l_exph.noise_modeling(), 8)
        ldr_right_exph_cap = QuantizeSTE.apply(phi_r_exph.noise_modeling(), 8)
        ldr_left_expl_cap = QuantizeSTE.apply(phi_l_expl.noise_modeling(), 8)
        ldr_right_expl_cap = QuantizeSTE.apply(phi_r_expl.noise_modeling(), 8)

        #^ Caculate frame 1,2 histogram
        histo_ldr1 = calculate_histogram_global(ldr_left_exph_cap)
        histo_ldr2 = calculate_histogram_global(ldr_left_expl_cap)
        
        #^ Caculate frame 1,2 skewness
        skewness_level1 = calculate_batch_skewness(histo_ldr1)
        skewness_level2 = calculate_batch_skewness(histo_ldr2)

        #^ Calculate saturation level based on histo, skewness
        saturation_level_f1 = calculate_batch_histogram_exposure(skewness_level1, histo_ldr1)
        saturation_level_f2 = calculate_batch_histogram_exposure(skewness_level2, histo_ldr2) 
        
        # print("Check skewness, saturation shape")
        # print(skewness_level1)
        # print(saturation_level_f1)
        
        
        # Todo) Exposure Network
        # Todo. #1) Rule based + alpha prediction network
        # alpha = self.alpha_net(saturation_level_f1, saturation_level_f2, skewness_level1, skewness_level2) #&&&
        alpha1, alpha2 = self.alpha_net(ldr_left_exph_cap, ldr_right_expl_cap)
        
        #^ Exposure shift
        # * For logging
        print(f"ALPHA1 : {alpha1}")
        print(f"ALPHA2 : {alpha2}")
        # print(f"skewness_level1 : {skewness_level1}, skewness_level2 : {skewness_level2}")
        # print(f"initial exp high : {initial_exp_high}, initial exp low : {initial_exp_low}")
        # print(f"saturation level f1 : {saturation_level_f1}, saturation level f2 : {saturation_level_f2}")
        shifted_exp_f1, shifted_exp_f2 = batch_exp_adjustment(initial_exp_high, initial_exp_low, saturation_level_f1, saturation_level_f2, alpha1, alpha2)
        shifted_exp = [shifted_exp_f1, shifted_exp_f2]
        
        print(f"Shifted exposure : {shifted_exp_f1}")

        #^ Simulate capured LDR image with shifted exposure
        phi_hat_l_exph = ImageFormation(left_hdr, shifted_exp_f1, device=DEVICE)
        phi_hat_r_exph = ImageFormation(right_hdr, shifted_exp_f1, device=DEVICE)
        phi_hat_l_expl = ImageFormation(left_next_hdr, shifted_exp_f2, device=DEVICE)
        phi_hat_r_expl = ImageFormation(right_next_hdr, shifted_exp_f2, device=DEVICE)

        left_ldr_adj_exph = QuantizeSTE.apply(phi_hat_l_exph.noise_modeling(), 8)
        right_ldr_adj_exph = QuantizeSTE.apply(phi_hat_r_exph.noise_modeling(), 8)
        left_ldr_adj_expl = QuantizeSTE.apply(phi_hat_l_expl.noise_modeling(), 8)
        right_ldr_adj_expl = QuantizeSTE.apply(phi_hat_r_expl.noise_modeling(), 8)
        
        #* For logging 6/10
        # shifted_histo_level1 = calculate_histogram_global(left_ldr_adj_exph)
        # shifted_histo_level2 = calculate_histogram_global(left_ldr_adj_expl)
        # shifted_skewness_level1 = calculate_batch_skewness(shifted_histo_level1)
        # shifted_skewness_level2 = calculate_batch_skewness(shifted_histo_level2)
        # shifted_saturation_level1 = calculate_batch_histogram_exposure(shifted_skewness_level1, shifted_histo_level1)
        # shifted_saturation_level2 = calculate_batch_histogram_exposure(shifted_skewness_level2, shifted_histo_level2)
        # print(f"shifted_skewness f1 : {shifted_skewness_level1}, shifted_skewness f2 : {shifted_skewness_level2}")
        # print(f"shifted saturation f1 : {shifted_saturation_level1}, shifted saturation f2 : {shifted_saturation_level2}")

        mask_exph = soft_binary_threshold_batch(left_ldr_adj_exph)
        mask_expl = soft_binary_threshold_batch(left_ldr_adj_expl)
        
        #^ Add exposure normalization
        
        #^ Flow estimation
        image_pairs_left = [(left_ldr_adj_exph[i], left_ldr_adj_expl[i]) for i in range(left_ldr_adj_exph.size(0))] #batch size      
        left_flows_tensor = self.compute_optical_flow_batch(self.flow, image_pairs_left, shifted_exp_f1, shifted_exp_f2).to(DEVICE)

        image_pairs_right = [(right_ldr_adj_exph[i], right_ldr_adj_expl[i]) for i in range(right_ldr_adj_exph.size(0))]
        right_flows_tensor = self.compute_optical_flow_batch(self.flow, image_pairs_right, shifted_exp_f1, shifted_exp_f2).to(DEVICE)
        
        #^ Warping
        warped_left = []
        warped_right = []
        for i in range(left_ldr_adj_expl.size(0)):
            img1_tensor = left_ldr_adj_expl[i].unsqueeze(0)  # [1, C, H, W]
            flow_tensor = left_flows_tensor[i].unsqueeze(0)  # [1, 2, H, W]
            warped_image = self.warp_image(img1_tensor, flow_tensor)
            warped_left.append(warped_image)

        for i in range(right_ldr_adj_expl.size(0)):
            img1_tensor = right_ldr_adj_expl[i].unsqueeze(0)  # [1, C, H, W]
            flow_tensor = right_flows_tensor[i].unsqueeze(0)  # [1, 2, H, W]
            warped_image = self.warp_image(img1_tensor, flow_tensor)
            warped_right.append(warped_image)

        warped_left_tensor = torch.cat(warped_left, dim=0)  # [B, 3, H, W]
        warped_right_tensor = torch.cat(warped_right, dim=0)  # [B, 3, H, W]

        #^ Disparity estimation
        disparity_exph = self.RAFTStereo(left_ldr_adj_exph, right_ldr_adj_exph)
        disparity_expl = self.RAFTStereo(warped_left_tensor, warped_right_tensor)
        disparity_cap_exph = self.RAFTStereo(ldr_left_exph_cap, ldr_right_exph_cap)
        disparity_cap_expl = self.RAFTStereo(ldr_left_expl_cap, ldr_right_expl_cap)
        
        # For logging
        original_img_list = [left_hdr, left_next_hdr]
        captured_rand_img_list = [ldr_left_exph_cap, ldr_left_expl_cap]
        captured_adj_img_list = [left_ldr_adj_exph, left_ldr_adj_expl, right_ldr_adj_exph, right_ldr_adj_expl]
        warped_img_list = [warped_left_tensor, warped_right_tensor]
        # mask_list = [mask_exph, mask_expl]
        disparity_list = [disparity_cap_exph[-1], disparity_cap_expl[-1]]

        #* disparity fusion
        epsilon = 1e-8
        mask_exph = mask_exph + epsilon
        mask_expl = mask_expl + epsilon
        disparity_exph_mul = disparity_exph[-1] * mask_exph
        disparity_expl_mul = disparity_expl[-1] * mask_expl

        fused_disparity_mul = (disparity_exph_mul + disparity_expl_mul) / (mask_exph + mask_expl)
        fused_disparity = torch.nan_to_num(fused_disparity_mul, nan=0.0, posinf=0.0, neginf=0.0)

        mask_mul_list = [disparity_exph_mul, disparity_expl_mul]

        if valid_mode:
            return fused_disparity, disparity_exph[-1], disparity_expl[-1], captured_adj_img_list, shifted_exp

        if test_mode:
            return fused_disparity, disparity_exph[-1], disparity_expl[-1], original_img_list, captured_rand_img_list, captured_adj_img_list, disparity_list

        return fused_disparity, disparity_exph[-1], disparity_expl[-1], original_img_list, captured_rand_img_list, captured_adj_img_list, warped_img_list, mask_mul_list, disparity_list, shifted_exp
    
    
