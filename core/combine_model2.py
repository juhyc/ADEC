import sys
sys.path.append('core')

import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.transforms import ToTensor
from PIL import Image
import matplotlib.pyplot as plt
from utils.mask import soft_binary_threshold_batch
from core.saec import *
from core.utils.simulate import *
from core.raft_stereo import RAFTStereo
from core.utils.utils import InputPadder
from core.extractor import BasicEncoder
from core.disparity_fusion_ResUnet import DisparityFusion_ResUnet
import core.stereo_datasets as datasets

import cv2

DEVICE = 'cuda'
################################
# * End to end pipeline with exposure control network
# * Exposure network output is alpha that decide how to control exposure gap.
# * Simulator -> Exposure Net -> Disparity Estimation
################################

# Todo) 240625, Test network output value alpha affects overall loss.

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
class CombineModel_wo_net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.RAFTStereo = RAFTStereo(args)
        self.alpha_net = ExposureAdjustmentPipeline()
        
    def convert_to_tensor(self, image):
        if isinstance(image, Image.Image):
            to_tensor = ToTensor()
            return to_tensor(image)
        return image
    
    # Todo) Edit combine model input
    # * To calculate overall loss depends on manual alpha.
    def forward(self, left_hdr, right_hdr, initial_exp_high, initial_exp_low, alpha, iters=32, valid_mode = False, test_mode=False, train_mode=False):
        
        #^ Simulator Module HDR -> LDR
        #* HDR scene
        phi_l_exph = ImageFormation(left_hdr, initial_exp_high, device=DEVICE)
        phi_r_exph = ImageFormation(right_hdr, initial_exp_high, device=DEVICE)
        phi_l_expl = ImageFormation(left_hdr, initial_exp_low, device=DEVICE)
        phi_r_expl = ImageFormation(right_hdr, initial_exp_low, device=DEVICE)
        
        #* Captured LDR image pair
        ldr_left_exph_cap = QuantizeSTE.apply(phi_l_exph.noise_modeling(), 8)
        ldr_right_exph_cap = QuantizeSTE.apply(phi_r_exph.noise_modeling(), 8)
        ldr_left_expl_cap = QuantizeSTE.apply(phi_l_expl.noise_modeling(), 8)
        ldr_right_expl_cap = QuantizeSTE.apply(phi_r_expl.noise_modeling(), 8)
        
        # * Calculate frame 1,2 histogram 
        histo_ldr1 = calculate_histogram_global(ldr_left_exph_cap)
        histo_ldr2 = calculate_histogram_global(ldr_left_expl_cap)   
        
        # * Check batch skewness calculation
        # ! Edit : Calculate global skewness to local skewness
        # skewness_level1 = calculate_batch_skewness(histo_ldr1)
        # skewness_level2 = calculate_batch_skewness(histo_ldr2)
        
        # _,_,height,width = left_hdr.shape
        # roi_mask = create_mask(height, width).to(DEVICE) # torch size [300,400]
        skewness_level1, skewness_f1_g1, skewness_f1_g3, skewness_f1_g7 = calculate_combined_skewness(ldr_left_exph_cap)
        skewness_level2, skewness_f2_g1, skewness_f2_g3, skewness_f2_g7 = calculate_combined_skewness(ldr_left_expl_cap)
        # skewness_level1 = skewness_f1_g1
        # skewness_level2 = skewness_f2_g1
        
        
        #? For logging skewness level shape
        skewness_level1 = torch.tensor([[skewness_level1]], dtype=torch.float32) # torch size [1,1]
        skewness_level2 = torch.tensor([[skewness_level2]], dtype=torch.float32)
        
        saturation_level_f1 = calculate_batch_histogram_exposure(skewness_level1, histo_ldr1)
        saturation_level_f2 = calculate_batch_histogram_exposure(skewness_level2, histo_ldr2)
        
        
        # ^ Expsoure alpha network
        alpha1, alpha2 = self.alpha_net(ldr_left_exph_cap, ldr_right_exph_cap)
        # alpha1, alpha2 = alpha, alpha
        
        #? Check Exposure values
        # print("=====Random exp values=====")
        # print(f"Random exp_l : {e_rand_high_pair[0].item():4f}, Random exp_r : {e_rand_low_pair[0].item():4f}")
        # print("=====Shifted exp values=====")
        # print(f"shifted_exp_l : {shifted_exp_f1[0].item():4f}, shifted_exp_r : {shifted_exp_f2[0].item():4f}")
        
        # * Exposure shift
        # shifted_exp_f1, shifted_exp_f2 = batch_exp_adjustment(initial_exp_high, initial_exp_low, saturation_level_f1, saturation_level_f2, histo_ldr1, histo_ldr2, alpha1, alpha2)
        shifted_exp_f1, shifted_exp_f2 = batch_exp_adjustment2(initial_exp_high, initial_exp_low, histo_ldr1, histo_ldr2, alpha1, alpha2)
        shifted_exp = [shifted_exp_f1, shifted_exp_f2]
        
        #* Simulate Image LDR with shifted exposure value
        phi_hat_l_exph = ImageFormation(left_hdr, shifted_exp_f1, device=DEVICE)
        phi_hat_r_exph = ImageFormation(right_hdr, shifted_exp_f1, device=DEVICE)
        phi_hat_l_expl = ImageFormation(left_hdr, shifted_exp_f2, device=DEVICE)
        phi_hat_r_expl = ImageFormation(right_hdr, shifted_exp_f2, device=DEVICE)
        
        left_ldr_adj_exph = QuantizeSTE.apply(phi_hat_l_exph.noise_modeling(), 8)
        right_ldr_adj_exph = QuantizeSTE.apply(phi_hat_r_exph.noise_modeling(), 8)
        left_ldr_adj_expl = QuantizeSTE.apply(phi_hat_l_expl.noise_modeling(), 8)
        right_ldr_adj_expl = QuantizeSTE.apply(phi_hat_r_expl.noise_modeling(), 8)
       
        skewness_level1_shifted, skewness_f1_g1, skewness_f1_g3, skewness_f1_g7 = calculate_combined_skewness(left_ldr_adj_exph)
        skewness_level2_shifted, skewness_f2_g1, skewness_f2_g3, skewness_f2_g7 = calculate_combined_skewness(left_ldr_adj_expl)
        
         #? Check Saturation value
        histo_ldr1_shifted = calculate_histogram_global(left_ldr_adj_exph)
        histo_ldr2_shifted = calculate_histogram_global(left_ldr_adj_expl)
        
        print(f"First Frame : initial_exp_high : {initial_exp_high[0][0].item():2f}, shifted_exp_high : {shifted_exp_f1[0][0].item():2f}, skewness_F1 : {skewness_level1.item():2f}, is_bimodal : {is_bimodal_print(histo_ldr1).item()}")
        # print(f"First Frame : Combine Skewness : {skewness_level1[0].item():2f}, Grid1 Skewness : {skewness_f1_g1[0].item():2f}, Grid3 Skewness : {skewness_f1_g3[0].item():2f}, Grid7 Skewness : {skewness_f1_g7[0].item():2f}")
        print(f"Second Frame : initial_exp_low : {initial_exp_low[0][0].item():2f}, shifted_exp_low : {shifted_exp_f2[0][0].item():2f}, skewness_F2 : {skewness_level2.item():2f}, is_bimodal : {is_bimodal_print(histo_ldr2).item()}")
        # print(f"Second Frame : Combine Skewness : {skewness_level2[0].item():2f}, Grid1 Skewness : {skewness_f2_g1[0].item():2f}, Grid3 Skewness : {skewness_f2_g3[0].item():2f}, Grid7 Skewness : {skewness_f2_g7[0].item():2f}")
       
       
        #* Create exposure mask 
        mask_exph = soft_binary_threshold_batch(left_ldr_adj_exph)
        mask_expl = soft_binary_threshold_batch(left_ldr_adj_expl)
        
        #^ Disparity Estimation 
        disparity_exph = self.RAFTStereo(left_ldr_adj_exph, right_ldr_adj_exph) # list, list[0]= [B, 1, H, W]
        disparity_expl = self.RAFTStereo(left_ldr_adj_expl, right_ldr_adj_expl)
        #? Disparity on not adjusted exposure images
        disparity_cap_exph = self.RAFTStereo(ldr_left_exph_cap, ldr_right_exph_cap)
        disparity_cap_expl = self.RAFTStereo(ldr_left_expl_cap, ldr_right_expl_cap)

        #? For exposure shift check [HDR, LDR_rand, LDR_adjust]
        original_img_list = [left_hdr]
        captured_rand_img_list = [ldr_left_exph_cap, ldr_left_expl_cap]
        captured_adj_img_list = [left_ldr_adj_exph, left_ldr_adj_expl, right_ldr_adj_exph, right_ldr_adj_expl, phi_hat_l_exph.ldr_denom, phi_hat_l_expl.ldr_denom]
        mask_list = [mask_exph, mask_expl]
        disparity_list = [disparity_cap_exph[-1], disparity_cap_expl[-1]]
        
        #^ Disparity fusion using exposure mask multiplication
        epsilon = 1e-8
        mask_exph = mask_exph + epsilon
        mask_expl = mask_expl + epsilon
        disparity_exph_mul = disparity_exph[-1] * mask_exph 
        disparity_expl_mul = disparity_expl[-1] * mask_expl 

        fused_disparity_mul = (disparity_exph_mul + disparity_expl_mul)/(mask_exph + mask_expl)
        # adjsut nan, inf value 
        fused_disparity = torch.nan_to_num(fused_disparity_mul, nan = 0.0, posinf=0.0, neginf=0.0)       
        
        mask_mul_list = [disparity_exph_mul, disparity_expl_mul]
        
        # ^ Run Raft-stereo model on random captured pair
        if valid_mode:
            
            disparity_exprand1 = self.RAFTStereo(ldr_left_exph_cap, ldr_right_exph_cap) # list, list[0]= [B, 1, H, W]
            disparity_exprand2 = self.RAFTStereo(ldr_left_expl_cap, ldr_right_expl_cap)
            
            # print("@@@@@@Validation@@@@@@")
            # print("=====Random exp values=====")
            # print(f"Random exp_l : {e_rand_high_pair[0].item():4f}, Random exp_r : {e_rand_low_pair[0].item():4f}")
            # print("=====Before shifted exp values=====")
            # print(f"output_exp_l : {e_exp_high2[0].item()}, output_exp_r : {e_exp_low2[0].item()}")
            # print("=====Shifted exp values=====")
            # print(f"shifted_exp_l : {e_shifted_high[0].item():4f}, shifted_exp_r : {e_shifted_low[0].item():4f}")
            return fused_disparity, disparity_exph[-1], disparity_expl[-1], original_img_list, captured_rand_img_list, captured_adj_img_list, disparity_exprand1[-1], disparity_exprand2[-1], mask_list, mask_mul_list
        
        if test_mode:
            return fused_disparity, disparity_exph[-1], disparity_expl[-1], original_img_list, captured_rand_img_list, captured_adj_img_list, disparity_list
          
        return fused_disparity, disparity_exph[-1], disparity_expl[-1], captured_rand_img_list, captured_adj_img_list,mask_mul_list, disparity_list, shifted_exp

    
        