import sys
sys.path.append('core')

import argparse
import numpy as np
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.transforms import ToTensor
from PIL import Image
import matplotlib.pyplot as plt
from utils.mask import soft_binary_threshold_batch
from core.adec import *
from core.utils.simulate import *
from core.utils.read_utils import prepare_inputs_custom

import time
import core.stereo_datasets as datasets
from core.disp_recon_model_dual import RAFTStereoFusion_refine
import cv2


DEVICE = 'cuda'

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

class CombineModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device(args.device)
        self.alpha_values = torch.full((self.args.batch_size,), 0.1, device= self.device)
        self.disp_recon_net = RAFTStereoFusion_refine(args).to(self.device)
        
        for name, module in self.disp_recon_net.named_modules():
            module.to(self.device)
            
        # print(f"[DEBUG] disp_recon_net is on {next(self.disp_recon_net.parameters()).device}")
    
    def forward(self, left_hdr, right_hdr, left_next_hdr, right_next_hdr, 
                initial_exp_high, initial_exp_low, test_mode = False, fixed_range_middle = None):
        
        device = self.device
        
        left_hdr, right_hdr = left_hdr.to(device), right_hdr.to(device)
        left_next_hdr, right_next_hdr = left_next_hdr.to(device), right_next_hdr.to(device)
        initial_exp_high, initial_exp_low = initial_exp_high.to(device), initial_exp_low.to(device)

        # print(f"[DEBUG] disp_recon_net is on {next(self.disp_recon_net.parameters()).device}")

        
        # Test mode, simulation param is fixed 
        if test_mode and fixed_range_middle is None:
            phi_l_exph = ImageFormation(left_hdr, initial_exp_high, device= device)
            fixed_range_middle = phi_l_exph.range_middle.view(-1)
        else:
            phi_l_exph = ImageFormation(left_hdr, initial_exp_high, device= device, fixed_range_middle=fixed_range_middle)
            
        phi_r_exph = ImageFormation(right_hdr, initial_exp_high, device= device, fixed_range_middle=fixed_range_middle)
        phi_l_expl = ImageFormation(left_next_hdr, initial_exp_low, device= device, fixed_range_middle=fixed_range_middle)
        phi_r_expl = ImageFormation(right_next_hdr, initial_exp_low, device= device, fixed_range_middle=fixed_range_middle)
        
        
        #^ Simulated LDR image pair for frame1,2        
        ldr_left_exph_cap = QuantizeSTE.apply(phi_l_exph.noise_modeling(), 8)
        ldr_right_exph_cap = QuantizeSTE.apply(phi_r_exph.noise_modeling(), 8)
        ldr_left_expl_cap = QuantizeSTE.apply(phi_l_expl.noise_modeling(), 8)
        ldr_right_expl_cap = QuantizeSTE.apply(phi_r_expl.noise_modeling(), 8)
        
        
        #* Exposure module
        start_time = time.time()
        #^ Caculate frame 1,2 histogram
        histo_ldr1 = calculate_histogram_global(ldr_left_exph_cap)
        histo_ldr2 = calculate_histogram_global(ldr_left_expl_cap)

        #^ Exposure control
        exp1, exp2 = stereo_exposure_control(initial_exp_high, initial_exp_low, histo_ldr1, histo_ldr2, alpha1=self.alpha_values, alpha2=self.alpha_values, exp_gap_threshold=2.0)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0 
       
        #^ Simulate capured LDR image with shifted exposure
        phi_hat_l_exph = ImageFormation(left_hdr, exp1, device=DEVICE)
        phi_hat_r_exph = ImageFormation(right_hdr, exp1, device=DEVICE)
        phi_hat_l_expl = ImageFormation(left_next_hdr, exp2, device=DEVICE)
        phi_hat_r_expl = ImageFormation(right_next_hdr, exp2, device=DEVICE)    

        left_ldr_adj_exph = QuantizeSTE.apply(phi_hat_l_exph.noise_modeling(), 8)
        right_ldr_adj_exph = QuantizeSTE.apply(phi_hat_r_exph.noise_modeling(), 8)
        left_ldr_adj_expl = QuantizeSTE.apply(phi_hat_l_expl.noise_modeling(), 8)
        right_ldr_adj_expl = QuantizeSTE.apply(phi_hat_r_expl.noise_modeling(), 8)
        
        #*Disparity reconstuction
        disp_predictions, fmap1, fmap1_next, warped_fmap_left, flow_L, fmap_list, _, mask_list = self.disp_recon_net(
                left_ldr_adj_exph, right_ldr_adj_exph, left_ldr_adj_expl, right_ldr_adj_expl
            )
        # For logging denormalized ldr images
        rmap_ldr1 = phi_hat_l_exph.ldr_denom
        rmap_ldr2 = phi_hat_l_expl.ldr_denom
         
        # For logging
        original_img_list = [left_hdr, left_next_hdr, rmap_ldr1, rmap_ldr2]
        captured_rand_img_list = [ldr_left_exph_cap, ldr_left_expl_cap]
        captured_adj_img_list = [left_ldr_adj_exph, left_ldr_adj_expl, right_ldr_adj_exph, right_ldr_adj_expl]
        # disparity_list = [disparity_cap_exph[-1], disparity_cap_expl[-1]]

        # if valid_mode:
        #     return fused_disparity, disparity_exph[-1], disparity_expl[-1], captured_adj_img_list, shifted_exp

        # if test_mode:
        #     return fused_disparity, disparity_exph[-1], disparity_expl[-1], original_img_list, captured_rand_img_list, captured_adj_img_list, disparity_list

        return disp_predictions, original_img_list, captured_rand_img_list, captured_adj_img_list, exp1, exp2, fmap_list, mask_list, flow_L, fixed_range_middle
    