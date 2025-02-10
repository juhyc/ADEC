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
from types import SimpleNamespace

import core.stereo_datasets as datasets

from core.saec import HybridExposureAdjustmentNet, HybridExposureAdjustmentNet_Spatial, BasicExposureCorrectionNet
from core.disp_recon_model_refine import RAFTStereoFusion_refine

from core.nae_reference import NeuralExposureControl

import cv2


DEVICE = 'cuda'

###############################
# * NAE + RAFT_stereo
###############################


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

# * End to end pipeline NAE + RAFT-stereo for comparison


class CombineModel_w_nae(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.nae = NeuralExposureControl()
        self.raft_stereo = RAFTStereo(args)
        
    
    def forward(self, left_hdr, right_hdr, initial_exp, iters=32):
        
        #* 1. Capture Simulation
        #^ Capture simulator module
        phi_l_exph = ImageFormation(left_hdr, initial_exp, device=DEVICE)
        phi_r_exph = ImageFormation(right_hdr, initial_exp, device=DEVICE)

        
        #^ Simulated LDR image pair      
        ldr_left_exph_cap = QuantizeSTE.apply(phi_l_exph.noise_modeling(), 8)
        ldr_right_exph_cap = QuantizeSTE.apply(phi_r_exph.noise_modeling(), 8)
  
        #* 2. NAE
        ldr_left_stacked = torch.stack([ldr_left_exph_cap[0][1]*0.5, ldr_left_exph_cap[0][1], ldr_left_exph_cap[0][1]*2.0], dim = 0)
        # print(f"In combine_mode_nae.py, ldr_left_stacked shape : {ldr_left_stacked.shape}")
        
        alpha = self.nae(ldr_left_stacked.unsqueeze(0)) # Network output, predicted exposure
        

        #* 3. Simulate with shifted exposure 
        #^ Simulate capured LDR image with shifted exposure
        phi_hat_l_exph = ImageFormation(left_hdr, alpha, device=DEVICE)
        phi_hat_r_exph = ImageFormation(right_hdr, alpha, device=DEVICE)

        left_ldr_adj_exph = QuantizeSTE.apply(phi_hat_l_exph.noise_modeling(), 8)
        right_ldr_adj_exph = QuantizeSTE.apply(phi_hat_r_exph.noise_modeling(), 8)

        
        #* 4. Disparity reconstuction
        disp_predictions = self.raft_stereo(left_ldr_adj_exph*255, right_ldr_adj_exph*255)
        
        # For logging denormalized ldr images
        rmap_ldr1 = phi_hat_l_exph.ldr_denom
        
        
        # For logging
        original_img_list = [left_hdr, rmap_ldr1]
        captured_rand_img_list = [ldr_left_exph_cap]
        captured_adj_img_list = [left_ldr_adj_exph, right_ldr_adj_exph]
        # disparity_list = [disparity_cap_exph[-1], disparity_cap_expl[-1]]

        # if valid_mode:
        #     return fused_disparity, disparity_exph[-1], disparity_expl[-1], captured_adj_img_list, shifted_exp

        # if test_mode:
        #     return fused_disparity, disparity_exph[-1], disparity_expl[-1], original_img_list, captured_rand_img_list, captured_adj_img_list, disparity_list

        return disp_predictions, alpha, left_hdr, captured_rand_img_list, captured_adj_img_list