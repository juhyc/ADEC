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

# def evaluate_dynamic_range(batch_images, bins=256, bright_threshold=0.75, dark_threshold=0.25):
#     """
#     evaluate dynamic range based on batch images
#     """
#     batch_size = batch_images.size(0)
#     dynamic_range_scores = torch.zeros(batch_size)
#     avg_brightness_scores = torch.zeros(batch_size)
    
#     for i in range(batch_size):
#         image = batch_images[i].mean(dim=0)  # 채널별 평균을 계산하여 그레이스케일 이미지처럼 처리
#         hist = torch.histc(image.flatten(), bins=bins, min=0, max=1)
        
#         # 밝은 영역과 어두운 영역의 픽셀 수 계산
#         bright_pixels = hist[int(bright_threshold*bins):].sum()
#         dark_pixels = hist[:int(dark_threshold*bins)].sum()
        
#         # 동적 범위 평가
#         dynamic_range_scores[i] = bright_pixels / (dark_pixels + 1e-5)
#         # 평균 밝기
#         avg_brightness_scores[i] = image.mean()
    
#     return dynamic_range_scores, avg_brightness_scores

# def adjust_exposure_based_on_dynamic_range(dynamic_range_scores, avg_brightness_scores, base_exposure_gap=0.5):
#     """
#     동적 범위 점수에 기반하여 노출 갭을 조정합니다.
    
#     :param dynamic_range_scores: 동적 범위를 나타내는 점수의 배치 ([B] 형태)
#     :param base_exposure_gap: 기본 노출 갭 값
#     :return: 조정된 노출 갭의 배치 ([B] 형태)
#     """
#     # 최소 노출 갭과 최대 노출 갭 계산
#     min_exposure_gap = 1/4
#     max_exposure_gap = M_exp - min_exposure_gap
    
#     # 동적 범위 점수에 따라 기본 노출 갭에 스케일 조정을 적용
#     scaled_exposure_gaps = base_exposure_gap * (1 + (dynamic_range_scores - 1) / 10)
    
#     # 계산된 노출 갭이 최소값과 최대값 사이에 있도록 조정
#     adjusted_exposure_gaps = torch.clamp(scaled_exposure_gaps, min_exposure_gap, 0.75)
    
#     return adjusted_exposure_gaps

# def adjust_exposure_based_on_dynamic_range(dynamic_range_scores, avg_brightness_scores, base_exposure_gap=0.5):
#     """
#     동적 범위 및 평균 밝기 점수에 기반하여 노출 갭을 조정합니다.
#     """
#     # 조정된 노출 갭 초기화
#     adjusted_exposure_gaps = torch.zeros_like(dynamic_range_scores)
#     # 조정 전략 결정
#     for i in range(len(dynamic_range_scores)):
#         if dynamic_range_scores[i] > 1.0 and 0.4 < avg_brightness_scores[i] < 0.6:
#             # 동적 범위가 넓고 평균 밝기가 중간
#             adjusted_exposure_gaps[i] = base_exposure_gap * 2  # 큰 노출 갭
#         elif avg_brightness_scores[i] <= 0.3:
#             # 평균 밝기가 어두운 경우
#             adjusted_exposure_gaps[i] = base_exposure_gap * 0.5  # 작은 노출 갭, 둘다 증가
#         elif avg_brightness_scores[i] >= 0.7:
#             # 평균 밝기가 밝은 경우
#             adjusted_exposure_gaps[i] = base_exposure_gap * 0.5  # 작은 노출 갭, 둘다 감소
#         elif dynamic_range_scores[i] <= 1.0 and 0.3 < avg_brightness_scores[i] < 0.7:
#             # 동적 범위가 좁고 평균 밝기가 중간
#             adjusted_exposure_gaps[i] = base_exposure_gap * 0.75  # 작은 노출 갭, 하나는 증가 하나는 감소
            
#     return adjusted_exposure_gaps

# def adjust_frame_exposures(base_exposure_1, base_exposure_2, exposure_gaps, avg_brightness_scores, scaling_factor = 1.0):
#     # 평균 밝기가 낮은 이미지에 대해서는 노출값을 증가
#     for i in range(len(exposure_gaps)):
#         if avg_brightness_scores[i] <= 0.25:
#             # 노출 갭이 줄어든 만큼 기본 노출값을 증가시킴
#             adjusted_exposure_1 = base_exposure_1 + (0.5 * exposure_gaps[i]) * scaling_factor
#             adjusted_exposure_2 = base_exposure_2 + (0.5 * exposure_gaps[i]) * scaling_factor
#         elif avg_brightness_scores[i] >= 0.75:
#             # 기본 노출값 조정 로직
#             adjusted_exposure_1 = base_exposure_1 - (0.5 * exposure_gaps[i]) * scaling_factor
#             adjusted_exposure_2 = base_exposure_2 - (0.5 * exposure_gaps[i]) * scaling_factor
#         elif 0.25 < avg_brightness_scores[i] < 0.75:
#             adjusted_exposure_1 = base_exposure_1 - (0.3 * exposure_gaps[i]) 
#             adjusted_exposure_2 = base_exposure_2 + (0.7 * exposure_gaps[i])
            
#     return adjusted_exposure_1, adjusted_exposure_2

# Quantize STE method
class QuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, n=8):
        # LDR image max value
        max_val = 2**n - 1
        
        # Quantize
        x_scaled = input * max_val
        x_clamped = torch.clamp(x_scaled, 0, max_val)
        x_quantized = torch.round(x_clamped)
        
        # Normalized to 0~1
        x_dequantized = x_quantized / max_val
        return x_dequantized

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


# ! Not use exposure network but algorithm
class CombineModel_wo_net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.RAFTStereo = RAFTStereo(args)
    
    def convert_to_tensor(self, image):
        if isinstance(image, Image.Image):
            to_tensor = ToTensor()
            return to_tensor(image)
        return image
        
    def forward(self, left_hdr, right_hdr, iters=32, flow_init=None, valid_mode = False, test_mode=False):
        
        #^ Simulator Module HDR -> LDR
        # initial random exposure
        e_rand_high_pair = generate_random_exposures(left_hdr.shape[0])
        e_rand_low_pair = generate_random_exposures(right_hdr.shape[0])
        
        # For valid mode, not random exposure but default 1.0 exp value
        if valid_mode:
            e_rand_high_pair = generate_random_exposures(left_hdr.shape[0], valid_mode=True)
            e_rand_low_pair = generate_random_exposures(right_hdr.shape[0], valid_mode=True)

        # HDR scene
        phi_l_exph = ImageFormation(left_hdr, e_rand_high_pair, device=DEVICE)
        phi_l_expl = ImageFormation(left_hdr, e_rand_low_pair, device=DEVICE)
        phi_r_exph = ImageFormation(right_hdr, e_rand_high_pair, device=DEVICE)
        phi_r_expl = ImageFormation(right_hdr, e_rand_low_pair, device=DEVICE)
        
        #^ Captured LDR image pair
        # if valid_mode:
        #     ldr_left_exph_cap = left_hdr
        #     ldr_right_exph_cap = right_hdr
        #     ldr_left_expl_cap = left_hdr
        #     ldr_right_expl_cap = right_hdr
        # else:
        ldr_left_exph_cap = QuantizeSTE.apply(phi_l_exph.noise_modeling(), 8)
        ldr_right_exph_cap = QuantizeSTE.apply(phi_r_exph.noise_modeling(), 8)
        ldr_left_expl_cap = QuantizeSTE.apply(phi_l_expl.noise_modeling(), 8)
        ldr_right_expl_cap = QuantizeSTE.apply(phi_r_expl.noise_modeling(), 8)
        
        # Todo) Calculate the global histogram to check which direction i should move.
        # ***************************************************************************
        histo_ldr1 = calculate_histogram_global(ldr_left_exph_cap)
        histo_ldr2 = calculate_histogram_global(ldr_left_expl_cap)
        saturation_level_f1 = calculate_batch_exposure_adjustment(histo_ldr1)
        saturation_level_f2 = calculate_batch_exposure_adjustment(histo_ldr2)
        
        print(f"e_rand_high_pair : {e_rand_high_pair}")
        print(f"saturation_level_f1 : {saturation_level_f1}")
        print(f"e_rand_low_pair : {e_rand_low_pair}")
        print(f"saturation_level_f2 : {saturation_level_f2}")

        shifted_exp_f1, shifted_exp_f2 = batch_exp_adjustment(e_rand_high_pair, e_rand_low_pair, saturation_level_f1, saturation_level_f2)
        print(f"shifted_exp_f1 : {shifted_exp_f1}")
        print(f"shifted_exp_f2 : {shifted_exp_f2}")
        
        # tensorboard logging histogram
        
        
        # Todo) End. 2024.04.04.
        # ****************************************************************************
        
        #################################################CHANGED###################################################
        
        #^ Check Exposure values
        print("=====Random exp values=====")
        print(f"Random exp_l : {e_rand_high_pair[0].item():4f}, Random exp_r : {e_rand_low_pair[0].item():4f}")
        print("=====Shifted exp values=====")
        print(f"shifted_exp_l : {shifted_exp_f1[0].item():4f}, shifted_exp_r : {shifted_exp_f2[0].item():4f}")
        
        ##############################################################################################################
        
        #^ Simulate Image LDR with shifted exposure value
        phi_hat_l_exph = ImageFormation(left_hdr, shifted_exp_f1, device=DEVICE)
        left_ldr_adj_exph = QuantizeSTE.apply(phi_hat_l_exph.noise_modeling(), 8)
        phi_hat_r_exph = ImageFormation(right_hdr, shifted_exp_f1, device=DEVICE)
        right_ldr_adj_exph = QuantizeSTE.apply(phi_hat_r_exph.noise_modeling(), 8)
        phi_hat_l_expl = ImageFormation(left_hdr, shifted_exp_f2, device=DEVICE)
        left_ldr_adj_expl = QuantizeSTE.apply(phi_hat_l_expl.noise_modeling(), 8)
        phi_hat_r_expl = ImageFormation(right_hdr, shifted_exp_f2, device=DEVICE)
        right_ldr_adj_expl = QuantizeSTE.apply(phi_hat_r_expl.noise_modeling(), 8)
              
        # ^ Create exposure mask 
        mask_exph = soft_binary_threshold_batch(left_ldr_adj_exph)
        mask_expl = soft_binary_threshold_batch(left_ldr_adj_expl)
        
        # ^ Run Raft-stereo model on 2 pair
        disparity_exph = self.RAFTStereo(left_ldr_adj_exph, right_ldr_adj_exph) # list, list[0]= [B, 1, H, W]
        disparity_expl = self.RAFTStereo(left_ldr_adj_expl, right_ldr_adj_expl)

        # * For exposure shift check [HDR, LDR_rand, LDR_adjust]
        original_img_list = [left_hdr]
        captured_rand_img_list = [ldr_left_exph_cap, ldr_left_expl_cap]
        captured_adj_img_list = [left_ldr_adj_exph, left_ldr_adj_expl]
        mask_list = [mask_exph, mask_expl]
        
        # Todo) Run Disparity_fusion module using disparity1,2 and its mask.
        # Todo) Check : disparity1,2, mask type. shape
        
        # ^ disparity fusion using exposure mask multiplication
        epsilon = 1e-8
        mask_exph = mask_exph + epsilon
        mask_expl = mask_expl + epsilon
        disparity_exph_mul = disparity_exph[-1] * mask_exph 
        disparity_expl_mul = disparity_expl[-1] * mask_expl 
        
        fused_disparity_mul = (disparity_exph_mul + disparity_expl_mul)/(mask_exph + mask_expl)
        # adjsut nan, inf value 
        fused_disparity_mul = torch.nan_to_num(fused_disparity_mul, nan = 0.0, posinf=0.0, neginf=0.0)
        fused_disparity = fused_disparity_mul        
        
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
          
        return fused_disparity, disparity_exph[-1], disparity_expl[-1], original_img_list, captured_rand_img_list, captured_adj_img_list, mask_list, mask_mul_list

    

