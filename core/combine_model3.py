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

# * End to end pipeline using a simple algorithm instead of network.
# Todo) Changing the method of controlling exposure from using a alogorithm to using a network.


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
        self.alpha_values = torch.full((self.args.batch_size,), 0.5, device=DEVICE)
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
        
        # Compute initial disparity maps using the unadjusted exposures
        # disp_pred_initial, _, _, _, _, _, _, _ = self.disp_recon_net(
        #     ldr_left_exph_cap, ldr_right_exph_cap, ldr_left_expl_cap, ldr_right_expl_cap
        # )
        
        #* 2. Exposure module
        start_time = time.time()
        #^ Caculate frame 1,2 histogram
        histo_ldr1 = calculate_histogram_global(ldr_left_exph_cap)
        histo_ldr2 = calculate_histogram_global(ldr_left_expl_cap)
        
        #^ Caculate frame 1,2 skewness
        # skewness_level1 = calculate_batch_skewness(histo_ldr1)
        # skewness_level2 = calculate_batch_skewness(histo_ldr2)

        #^ Calculate saturation level based on histo, skewness
        # saturation_level_f1 = calculate_batch_histogram_exposure(skewness_level1, histo_ldr1)
        # saturation_level_f2 = calculate_batch_histogram_exposure(skewness_level2, histo_ldr2) 
        
        # logging
        # print(f"============Check skewness, saturation shape {skewness_level1}, {saturation_level_f1}, {skewness_level2}, {saturation_level_f2}")       

        # a1 = torch.full((self.args.batch_size,), 0.1, device=DEVICE)
        # a1 = torch.full((self.args.batch_size,), 0.5, device=DEVICE)

        #^ Rule-based exposure control
        # print(f"@@@Before Rule-based, initial expsoure : {initial_exp_high} {initial_exp_low}")
        exp1, exp2 = stereo_exposure_control7(initial_exp_high, initial_exp_low, histo_ldr1, histo_ldr2, alpha1=self.alpha_values, alpha2=self.alpha_values, exp_gap_threshold=2.2)
        # print(f"@@@After Rule-based expsoure : {exp1} {exp2}")
        rule_exp1, rule_exp2 = [exp1, exp2]
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0 

        # 실험 결과를 저장할 폴더 경로에 FPS 값을 기록
        experiment_folder = 'test_results_real_final/FPS/DualAE'  # 실제 파일 경로로 변경
        with open(os.path.join(experiment_folder, "exposure_control_fps.txt"), 'a') as f:
            f.write(f"Exposure control FPS: {fps:.2f}\n")

        # ##########*Rule-based exposure shift###############
        # phi_hat_l_exph_rule = ImageFormation(left_hdr, rule_exp1, device=DEVICE)
        # phi_hat_r_exph_rule = ImageFormation(right_hdr, rule_exp1, device=DEVICE)
        # phi_hat_l_expl_rule = ImageFormation(left_next_hdr, rule_exp2, device=DEVICE)
        # phi_hat_r_expl_rule  = ImageFormation(right_next_hdr, rule_exp2, device=DEVICE)
        
        # left_ldr_adj_exph_rule, rmap_ldr_left1 = phi_hat_l_exph_rule.noise_modeling()
        # right_ldr_adj_exph_rule, _ = phi_hat_r_exph_rule.noise_modeling()
        # left_ldr_adj_expl_rule, rmap_ldr_left2 = phi_hat_l_expl_rule.noise_modeling()
        # right_ldr_adj_expl_rule, _ = phi_hat_r_expl_rule .noise_modeling
        
        # left_ldr_adj_exph_rule = QuantizeSTE.apply(left_ldr_adj_exph_rule, 8)
        # right_ldr_adj_exph_rule = QuantizeSTE.apply(right_ldr_adj_exph_rule, 8)
        # left_ldr_adj_expl_rule = QuantizeSTE.apply(left_ldr_adj_expl_rule, 8)
        # right_ldr_adj_expl_rule = QuantizeSTE.apply(right_ldr_adj_expl_rule, 8)
        

        # disp_predictions_rule, _, _, _, _, _, _ = self.disp_recon_net(
        #         left_ldr_adj_exph_rule, right_ldr_adj_exph_rule, left_ldr_adj_expl_rule, right_ldr_adj_expl_rule
        #     )
        
        # ############### * ######################
        
        # #logging
        # print("@@@@@@@@@In combine_model3.py, check each frame value@@@@@@@@@@@@")
        # print(f"skewness_level1 : {skewness_level1[0]}, skewness_level2 : {skewness_level2[0]}")
        # print(f"skewness_diff : {torch.abs(skewness_level1[0]-skewness_level2[0])}")
       
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
        disp_predictions, fmap1, fmap1_next, warped_fmap_left, flow_L, fmap_list, _, mask_list = self.disp_recon_net(
                left_ldr_adj_exph, right_ldr_adj_exph, left_ldr_adj_expl, right_ldr_adj_expl
            )
        # For logging denormalized ldr images
        rmap_ldr1 = phi_hat_l_exph.ldr_denom
        rmap_ldr2 = phi_hat_l_expl.ldr_denom
        
        # print(f"@@@@@@@@@@@@@@rmap_ldr1 : {rmap_ldr1.max()}")
        
        # #* For exp1 disparity
        # phi_hat_l_exph_exp1 = ImageFormation(left_hdr, exp1, device=DEVICE)
        # phi_hat_r_exph_exp1 = ImageFormation(right_hdr, exp1, device=DEVICE)
        # phi_hat_l_expl_exp1 = ImageFormation(left_next_hdr, exp1, device=DEVICE)
        # phi_hat_r_expl_exp1 = ImageFormation(right_next_hdr, exp1, device=DEVICE)

        # left_ldr_adj_exph_exp1 = QuantizeSTE.apply(phi_hat_l_exph_exp1.noise_modeling(), 8)
        # right_ldr_adj_exph_exp1 = QuantizeSTE.apply(phi_hat_r_exph_exp1.noise_modeling(), 8)
        # left_ldr_adj_expl_exp1 = QuantizeSTE.apply(phi_hat_l_expl_exp1.noise_modeling(), 8)
        # right_ldr_adj_expl_exp1 = QuantizeSTE.apply(phi_hat_r_expl_exp1.noise_modeling(), 8)
        
        # #* 4. Disparity reconstuction
        # disp_predictions_exp1, _, _, _, _, _, _, _ = self.disp_recon_net(
        #         left_ldr_adj_exph_exp1, right_ldr_adj_exph_exp1, left_ldr_adj_expl_exp1, right_ldr_adj_expl_exp1
        #     )
        
        # #* For exp2 disparity
        # phi_hat_l_exph_exp2 = ImageFormation(left_hdr, exp2, device=DEVICE)
        # phi_hat_r_exph_exp2 = ImageFormation(right_hdr, exp2, device=DEVICE)
        # phi_hat_l_expl_exp2 = ImageFormation(left_next_hdr, exp2, device=DEVICE)
        # phi_hat_r_expl_exp2 = ImageFormation(right_next_hdr, exp2, device=DEVICE)

        # left_ldr_adj_exph_exp2 = QuantizeSTE.apply(phi_hat_l_exph_exp2.noise_modeling(), 8)
        # right_ldr_adj_exph_exp2 = QuantizeSTE.apply(phi_hat_r_exph_exp2.noise_modeling(), 8)
        # left_ldr_adj_expl_exp2 = QuantizeSTE.apply(phi_hat_l_expl_exp2.noise_modeling(), 8)
        # right_ldr_adj_expl_exp2 = QuantizeSTE.apply(phi_hat_r_expl_exp2.noise_modeling(), 8)
        
        # #* 4. Disparity reconstuction
        # disp_predictions_exp2, _, _, _, _, _, _, _= self.disp_recon_net(
        #         left_ldr_adj_exph_exp2, right_ldr_adj_exph_exp2, left_ldr_adj_expl_exp2, right_ldr_adj_expl_exp2
        #     )
        
        # For logging
        original_img_list = [left_hdr, left_next_hdr, rmap_ldr1, rmap_ldr2]
        captured_rand_img_list = [ldr_left_exph_cap, ldr_left_expl_cap]
        captured_adj_img_list = [left_ldr_adj_exph, left_ldr_adj_expl, right_ldr_adj_exph, right_ldr_adj_expl]
        # disparity_list = [disparity_cap_exph[-1], disparity_cap_expl[-1]]

        # if valid_mode:
        #     return fused_disparity, disparity_exph[-1], disparity_expl[-1], captured_adj_img_list, shifted_exp

        # if test_mode:
        #     return fused_disparity, disparity_exph[-1], disparity_expl[-1], original_img_list, captured_rand_img_list, captured_adj_img_list, disparity_list

        return disp_predictions, original_img_list, captured_rand_img_list, captured_adj_img_list, exp1, exp2, fmap_list, mask_list, flow_L
    