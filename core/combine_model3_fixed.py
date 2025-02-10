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
from core.disp_recon_model_refine import RAFTStereoFusion_refine



import cv2


DEVICE = 'cuda'
################################
# * End to end pipeline with exposure control, not utilize network
# * Considering sequence image
# * Exposure module + Disparity reconstruction module
# * Modify the model to accept the initial exposure setting as an external input. 
################################

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

class CombineModel_wo_net_fixed(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.disp_recon_net = RAFTStereoFusion_refine(args)
        
    
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
        
            
        #* 4. Disparity reconstuction
        disp_predictions, _, _, _, _, _, _, _ = self.disp_recon_net(
                ldr_left_exph_cap, ldr_right_exph_cap, ldr_left_expl_cap, ldr_right_expl_cap
            )
    
        return disp_predictions, ldr_left_exph_cap, ldr_left_expl_cap