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
# * End to end pipeline with exposure control not utilize network
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
# Todo) Edit model input and intermediate processes to consider not just 1 stereo pairs but 2 stereo pairs in sequence.
class CombineModel_wo_net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.RAFTStereo = RAFTStereo(args)
        self.initial_exp_high = torch.tensor([1.3], dtype=torch.float32)
        self.initial_exp_low = torch.tensor([1.3], dtype=torch.float32)
        
    def convert_to_tensor(self, image):
        if isinstance(image, Image.Image):
            to_tensor = ToTensor()
            return to_tensor(image)
        return image
    
    # ! 04.23 just test modified method
    def forward(self, left_hdr, right_hdr, iters=32, flow_init=None, valid_mode = False, test_mode=False, train_mode=False):
        
        #^ Simulator Module HDR -> LDR
        # initial random exposure for training
        if train_mode:
            e_rand_high_pair = generate_random_exposures(left_hdr.shape[0], valid_mode=True, value=1.3)
            e_rand_low_pair = generate_random_exposures(right_hdr.shape[0], valid_mode=True, value=1.3)
        # For valid mode, not random exposure but default 1.0 exp value
        if valid_mode:
            e_rand_high_pair = generate_random_exposures(left_hdr.shape[0], valid_mode=True)
            e_rand_low_pair = generate_random_exposures(right_hdr.shape[0], valid_mode=True)
        if test_mode:
            e_rand_high_pair = self.initial_exp_high
            e_rand_low_pair = self.initial_exp_low

        # HDR scene
        phi_l_exph = ImageFormation(left_hdr, e_rand_high_pair, device=DEVICE)
        phi_r_exph = ImageFormation(right_hdr, e_rand_high_pair, device=DEVICE)
        phi_l_expl = ImageFormation(left_hdr, e_rand_low_pair, device=DEVICE)
        phi_r_expl = ImageFormation(right_hdr, e_rand_low_pair, device=DEVICE)
        
        #^ Captured LDR image pair
        ldr_left_exph_cap = QuantizeSTE.apply(phi_l_exph.noise_modeling(), 8)
        ldr_right_exph_cap = QuantizeSTE.apply(phi_r_exph.noise_modeling(), 8)
        ldr_left_expl_cap = QuantizeSTE.apply(phi_l_expl.noise_modeling(), 8)
        ldr_right_expl_cap = QuantizeSTE.apply(phi_r_expl.noise_modeling(), 8)
        
        # * Calculate frame 1,2 histogram 
        histo_ldr1 = calculate_histogram_global(ldr_left_exph_cap)
        histo_ldr2 = calculate_histogram_global(ldr_left_expl_cap)   
        
        # * Check batch skewness calculation
        skewness_level1 = calculate_batch_skewness(histo_ldr1)
        skewness_level2 = calculate_batch_skewness(histo_ldr2)
        
        saturation_level_f1 = calculate_batch_histogram_exposure(skewness_level1, histo_ldr1)
        saturation_level_f2 = calculate_batch_histogram_exposure(skewness_level2, histo_ldr2)
        
        print(f"e_rand_high_pair : {e_rand_high_pair}")
        print(f"saturation_level_f1 : {saturation_level_f1}")
        print(f"e_rand_low_pair : {e_rand_low_pair}")
        print(f"saturation_level_f2 : {saturation_level_f2}")

        shifted_exp_f1, shifted_exp_f2 = batch_exp_adjustment(e_rand_high_pair, e_rand_low_pair, saturation_level_f1, saturation_level_f2)
        print(f"shifted_exp_f1 : {shifted_exp_f1}")
        print(f"shifted_exp_f2 : {shifted_exp_f2}")
        
        # # * Save shifted exposure
        # self.initial_exp_high = shifted_exp_f1
        # self.initial_exp_low = shifted_exp_f2

        # * For fixed Expsoure
        # shifted_exp_f1, shifted_exp_f2 = self.initial_exp_high, self.initial_exp_low
        
        #^ Check Exposure values
        print("=====Random exp values=====")
        print(f"Random exp_l : {e_rand_high_pair[0].item():4f}, Random exp_r : {e_rand_low_pair[0].item():4f}")
        print("=====Shifted exp values=====")
        print(f"shifted_exp_l : {shifted_exp_f1[0].item():4f}, shifted_exp_r : {shifted_exp_f2[0].item():4f}")
        
        #^ Simulate Image LDR with shifted exposure value
        phi_hat_l_exph = ImageFormation(left_hdr, shifted_exp_f1, device=DEVICE)
        phi_hat_r_exph = ImageFormation(right_hdr, shifted_exp_f1, device=DEVICE)
        phi_hat_l_expl = ImageFormation(left_hdr, shifted_exp_f2, device=DEVICE)
        phi_hat_r_expl = ImageFormation(right_hdr, shifted_exp_f2, device=DEVICE)
        
        left_ldr_adj_exph = QuantizeSTE.apply(phi_hat_l_exph.noise_modeling(), 8)
        right_ldr_adj_exph = QuantizeSTE.apply(phi_hat_r_exph.noise_modeling(), 8)
        left_ldr_adj_expl = QuantizeSTE.apply(phi_hat_l_expl.noise_modeling(), 8)
        right_ldr_adj_expl = QuantizeSTE.apply(phi_hat_r_expl.noise_modeling(), 8)
       
        # ^ Create exposure mask 
        mask_exph = soft_binary_threshold_batch(left_ldr_adj_exph)
        mask_expl = soft_binary_threshold_batch(left_ldr_adj_expl)
        
        # ^ Run Raft-stereo model on 2 pair
        disparity_exph = self.RAFTStereo(left_ldr_adj_exph, right_ldr_adj_exph) # list, list[0]= [B, 1, H, W]
        disparity_expl = self.RAFTStereo(left_ldr_adj_expl, right_ldr_adj_expl)
        # disparity on not adjusted exposure images
        disparity_cap_exph = self.RAFTStereo(ldr_left_exph_cap, ldr_right_exph_cap)
        disparity_cap_expl = self.RAFTStereo(ldr_left_expl_cap, ldr_right_expl_cap)

        # * For exposure shift check [HDR, LDR_rand, LDR_adjust]
        original_img_list = [left_hdr]
        captured_rand_img_list = [ldr_left_exph_cap, ldr_left_expl_cap]
        captured_adj_img_list = [left_ldr_adj_exph, left_ldr_adj_expl, right_ldr_adj_exph, right_ldr_adj_expl]
        mask_list = [mask_exph, mask_expl]
        disparity_list = [disparity_cap_exph[-1], disparity_cap_expl[-1]]
        
        # ^ disparity fusion using exposure mask multiplication
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
          
        return fused_disparity, disparity_exph[-1], disparity_expl[-1], original_img_list, captured_rand_img_list, captured_adj_img_list, mask_list, mask_mul_list, disparity_list

    
        
    # def forward(self, left_hdr, right_hdr, iters=32, flow_init=None, valid_mode = False, test_mode=False):
        
    #     #^ Simulator Module HDR -> LDR
    #     # initial random exposure
    #     # ! For checking qunatization result, set genereated exposure value to 0.6
    #     e_rand_high_pair = generate_random_exposures(left_hdr.shape[0], valid_mode=False)
    #     e_rand_low_pair = generate_random_exposures(right_hdr.shape[0], valid_mode=False)
        
    #     # For valid mode, not random exposure but default 1.0 exp value
    #     if valid_mode:
    #         e_rand_high_pair = generate_random_exposures(left_hdr.shape[0], valid_mode=True)
    #         e_rand_low_pair = generate_random_exposures(right_hdr.shape[0], valid_mode=True)

    #     # HDR scene
    #     phi_l_exph = ImageFormation(left_hdr, e_rand_high_pair, device=DEVICE)
    #     phi_l_expl = ImageFormation(left_hdr, e_rand_low_pair, device=DEVICE)
    #     phi_r_exph = ImageFormation(right_hdr, e_rand_high_pair, device=DEVICE)
    #     phi_r_expl = ImageFormation(right_hdr, e_rand_low_pair, device=DEVICE)
        
    #     #^ Captured LDR image pair
    #     # if valid_mode:
    #     #     ldr_left_exph_cap = left_hdr
    #     #     ldr_right_exph_cap = right_hdr
    #     #     ldr_left_expl_cap = left_hdr
    #     #     ldr_right_expl_cap = right_hdr
    #     # else:
    #     ldr_left_exph_cap = QuantizeSTE.apply(phi_l_exph.noise_modeling(), 8)
    #     ldr_right_exph_cap = QuantizeSTE.apply(phi_r_exph.noise_modeling(), 8)
    #     ldr_left_expl_cap = QuantizeSTE.apply(phi_l_expl.noise_modeling(), 8)
    #     ldr_right_expl_cap = QuantizeSTE.apply(phi_r_expl.noise_modeling(), 8)
        
    #     ###########################################################################
        
    #     # Todo) Calculate the global histogram to check which direction i should move.
    #     # ***************************************************************************
        
        
    #     histo_ldr1 = calculate_histogram_global(ldr_left_exph_cap)
    #     histo_ldr2 = calculate_histogram_global(ldr_left_expl_cap)   
        
    #     # ! To check batch skewness calculation
    #     skewness_level1 = calculate_batch_skewness(histo_ldr1)
    #     skewness_level2 = calculate_batch_skewness(histo_ldr2)
    #     #! #########################################
        
    #     # print(skewness_level1)
    #     # print(histo_ldr1[0])
    #     saturation_level_f1 = calculate_batch_histogram_exposure(skewness_level1, histo_ldr1)
    #     saturation_level_f2 = calculate_batch_histogram_exposure(skewness_level2, histo_ldr2)
    #     # saturation_level_f1 = calculate_batch_exposure_adjustment(histo_ldr1)
    #     # saturation_level_f2 = calculate_batch_exposure_adjustment(histo_ldr2)
        
    #     print(f"e_rand_high_pair : {e_rand_high_pair}")
    #     print(f"saturation_level_f1 : {saturation_level_f1}")
    #     print(f"e_rand_low_pair : {e_rand_low_pair}")
    #     print(f"saturation_level_f2 : {saturation_level_f2}")

    #     shifted_exp_f1, shifted_exp_f2 = batch_exp_adjustment(e_rand_high_pair, e_rand_low_pair, saturation_level_f1, saturation_level_f2)
    #     print(f"shifted_exp_f1 : {shifted_exp_f1}")
    #     print(f"shifted_exp_f2 : {shifted_exp_f2}")
        
    #     # tensorboard logging histogram
        
        
    #     # Todo) End. 2024.04.04.
    #     # ****************************************************************************
        
    #     #################################################CHANGED###################################################
        
    #     #^ Check Exposure values
    #     print("=====Random exp values=====")
    #     print(f"Random exp_l : {e_rand_high_pair[0].item():4f}, Random exp_r : {e_rand_low_pair[0].item():4f}")
    #     print("=====Shifted exp values=====")
    #     print(f"shifted_exp_l : {shifted_exp_f1[0].item():4f}, shifted_exp_r : {shifted_exp_f2[0].item():4f}")
        
    #     ##############################################################################################################
        
    #     #^ Simulate Image LDR with shifted exposure value
    #     phi_hat_l_exph = ImageFormation(left_hdr, shifted_exp_f1, device=DEVICE)
    #     left_ldr_adj_exph = QuantizeSTE.apply(phi_hat_l_exph.noise_modeling(), 8)
    #     phi_hat_r_exph = ImageFormation(right_hdr, shifted_exp_f1, device=DEVICE)
    #     right_ldr_adj_exph = QuantizeSTE.apply(phi_hat_r_exph.noise_modeling(), 8)
    #     phi_hat_l_expl = ImageFormation(left_hdr, shifted_exp_f2, device=DEVICE)
    #     left_ldr_adj_expl = QuantizeSTE.apply(phi_hat_l_expl.noise_modeling(), 8)
    #     phi_hat_r_expl = ImageFormation(right_hdr, shifted_exp_f2, device=DEVICE)
    #     right_ldr_adj_expl = QuantizeSTE.apply(phi_hat_r_expl.noise_modeling(), 8)
              
    #     # ^ Create exposure mask 
    #     mask_exph = soft_binary_threshold_batch(left_ldr_adj_exph)
    #     mask_expl = soft_binary_threshold_batch(left_ldr_adj_expl)
        
    #     # ^ Run Raft-stereo model on 2 pair
    #     disparity_exph = self.RAFTStereo(left_ldr_adj_exph, right_ldr_adj_exph) # list, list[0]= [B, 1, H, W]
    #     disparity_expl = self.RAFTStereo(left_ldr_adj_expl, right_ldr_adj_expl)

    #     # * For exposure shift check [HDR, LDR_rand, LDR_adjust]
    #     original_img_list = [left_hdr]
    #     captured_rand_img_list = [ldr_left_exph_cap, ldr_left_expl_cap]
    #     captured_adj_img_list = [left_ldr_adj_exph, left_ldr_adj_expl]
    #     mask_list = [mask_exph, mask_expl]
        
    #     # Todo) Run Disparity_fusion module using disparity1,2 and its mask.
    #     # Todo) Check : disparity1,2, mask type. shape
        
    #     # ^ disparity fusion using exposure mask multiplication
    #     epsilon = 1e-8
    #     mask_exph = mask_exph + epsilon
    #     mask_expl = mask_expl + epsilon
    #     disparity_exph_mul = disparity_exph[-1] * mask_exph 
    #     disparity_expl_mul = disparity_expl[-1] * mask_expl 
        
    #     fused_disparity_mul = (disparity_exph_mul + disparity_expl_mul)/(mask_exph + mask_expl)
    #     # adjsut nan, inf value 
    #     fused_disparity_mul = torch.nan_to_num(fused_disparity_mul, nan = 0.0, posinf=0.0, neginf=0.0)
    #     fused_disparity = fused_disparity_mul        
        
    #     mask_mul_list = [disparity_exph_mul, disparity_expl_mul]
        
    #     # ^ Run Raft-stereo model on random captured pair
    #     if valid_mode:
            
    #         disparity_exprand1 = self.RAFTStereo(ldr_left_exph_cap, ldr_right_exph_cap) # list, list[0]= [B, 1, H, W]
    #         disparity_exprand2 = self.RAFTStereo(ldr_left_expl_cap, ldr_right_expl_cap)
            
    #         # print("@@@@@@Validation@@@@@@")
    #         # print("=====Random exp values=====")
    #         # print(f"Random exp_l : {e_rand_high_pair[0].item():4f}, Random exp_r : {e_rand_low_pair[0].item():4f}")
    #         # print("=====Before shifted exp values=====")
    #         # print(f"output_exp_l : {e_exp_high2[0].item()}, output_exp_r : {e_exp_low2[0].item()}")
    #         # print("=====Shifted exp values=====")
    #         # print(f"shifted_exp_l : {e_shifted_high[0].item():4f}, shifted_exp_r : {e_shifted_low[0].item():4f}")
    #         return fused_disparity, disparity_exph[-1], disparity_expl[-1], original_img_list, captured_rand_img_list, captured_adj_img_list, disparity_exprand1[-1], disparity_exprand2[-1], mask_list, mask_mul_list 
          
        # return fused_disparity, disparity_exph[-1], disparity_expl[-1], original_img_list, captured_rand_img_list, captured_adj_img_list, mask_list, mask_mul_list

    

