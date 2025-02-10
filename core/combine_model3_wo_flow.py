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
from core.saec import *
from core.utils.simulate import *
from core.raft_stereo import RAFTStereo
from core.utils.read_utils import prepare_inputs_custom
from core.utils.utils import InputPadder
from core.extractor import BasicEncoder

import time

import core.stereo_datasets as datasets

from core.saec import HybridExposureAdjustmentNet, HybridExposureAdjustmentNet_Spatial, BasicExposureCorrectionNet
from core.disp_recon_model_refine_wo_flow import RAFTStereoFusion_refine_wo_flow



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

class CombineModel_wo_net_wo_flow(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.disp_recon_net = RAFTStereoFusion_refine_wo_flow(args)
        
    
    def forward(self, left_hdr, right_hdr, left_next_hdr, right_next_hdr, initial_exp_high, initial_exp_low, iters=32):
        
        #* 1. Capture Simulation
        #^ Capture simulator module
        phi_l_exph = ImageFormation(left_hdr, initial_exp_high, device=DEVICE)
        phi_r_exph = ImageFormation(right_hdr, initial_exp_high, device=DEVICE)
        phi_l_expl = ImageFormation(left_next_hdr, initial_exp_low, device=DEVICE)
        phi_r_expl = ImageFormation(right_next_hdr, initial_exp_low, device=DEVICE)
        
        #^ Simulated LDR image pair for frame1,2        
        ldr_left_exph_cap = QuantizeSTE.apply(phi_l_exph.noise_modeling(), 8)
        ldr_right_exph_cap = QuantizeSTE.apply(phi_r_exph.noise_modeling(), 8)
        ldr_left_expl_cap = QuantizeSTE.apply(phi_l_expl.noise_modeling(), 8)
        ldr_right_expl_cap = QuantizeSTE.apply(phi_r_expl.noise_modeling(), 8)
 
        
        #* 2. Exposure module
        start_time = time.time()
        #^ Caculate frame 1,2 histogram
        histo_ldr1 = calculate_histogram_global(ldr_left_exph_cap)
        histo_ldr2 = calculate_histogram_global(ldr_left_expl_cap)
        
        #^ Caculate frame 1,2 skewness
        skewness_level1 = calculate_batch_skewness(histo_ldr1)
        skewness_level2 = calculate_batch_skewness(histo_ldr2)

        #^ Calculate saturation level based on histo, skewness
        saturation_level_f1 = calculate_batch_histogram_exposure(skewness_level1, histo_ldr1)
        saturation_level_f2 = calculate_batch_histogram_exposure(skewness_level2, histo_ldr2) 
        
        # logging
        # print(f"============Check skewness, saturation shape {skewness_level1}, {saturation_level_f1}, {skewness_level2}, {saturation_level_f2}")       

        a1 = torch.full((self.args.batch_size,), 0.9, device=DEVICE)
        a2 = torch.full((self.args.batch_size,), 0.9, device=DEVICE)

        #^ Rule-based exposure control
        exp1, exp2 = stereo_exposure_control7(initial_exp_high, initial_exp_low, histo_ldr1, histo_ldr2, alpha1=a1, alpha2=a2, exp_gap_threshold=2.5)
        rule_exp1, rule_exp2 = [exp1, exp2]
        
       
        #* 3. Simulate with shifted exposure 
        #^ Simulate capured LDR image with shifted exposure
        # print(f"Check in combine_model3.py adjusted exposure : {exp1} {exp2}")
        phi_hat_l_exph = ImageFormation(left_hdr, exp1, device=DEVICE)
        phi_hat_r_exph = ImageFormation(right_hdr, exp1, device=DEVICE)
        phi_hat_l_expl = ImageFormation(left_next_hdr, exp2, device=DEVICE)
        phi_hat_r_expl = ImageFormation(right_next_hdr, exp2, device=DEVICE)    

        left_ldr_adj_exph = QuantizeSTE.apply(phi_hat_l_exph.noise_modeling(), 8)
        right_ldr_adj_exph = QuantizeSTE.apply(phi_hat_r_exph.noise_modeling(), 8)
        left_ldr_adj_expl = QuantizeSTE.apply(phi_hat_l_expl.noise_modeling(), 8)
        right_ldr_adj_expl = QuantizeSTE.apply(phi_hat_r_expl.noise_modeling(), 8)
        
        #* 4. Disparity reconstuction
        disp_predictions = self.disp_recon_net(
                left_ldr_adj_exph, right_ldr_adj_exph, left_ldr_adj_expl, right_ldr_adj_expl
            )

        
        return disp_predictions