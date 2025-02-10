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
from torch.cuda.amp import autocast

from ptlflow import get_model
from ptlflow.utils import flow_utils
from ptlflow.utils.io_adapter import IOAdapter


import time

import core.stereo_datasets as datasets

from core.saec import HybridExposureAdjustmentNet, HybridExposureAdjustmentNet_Spatial, BasicExposureCorrectionNet
from core.disp_recon_model_refine_wo_flow import RAFTStereoFusion_refine_wo_flow



import cv2


################################
# * disparity fusion (Post-fusion) without motion compensation
# * Independently estimation disparity map and fusion
################################


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

class CombineModel_dispfusion_w_flow(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.disp_recon_net = RAFTStereo(args)
        
        self.flow_model = get_model('rapidflow_it6', pretrained_ckpt='kitti').to(DEVICE)
    
    
    
    # Compute optical flow on consecutive frame
    def compute_optical_flow_batch(self, model, image_pairs):
        device = next(model.parameters()).device if next(model.parameters(), None) is not None else 'cuda'
        
        all_flows = []
        for left_image, right_image in image_pairs:

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
    
    # Resize flow to the feature map size
    def resize_flow(self, flow, target_size):
        resized_flow = F.interpolate(flow, size=target_size, mode='bilinear', align_corners=True)
        
        scale_factor_h = target_size[0] / flow.size(2)
        scale_factor_w = target_size[1] / flow.size(3)
        
        resized_flow[:, 0, :, :] *= scale_factor_w 
        resized_flow[:, 1, :, :] *= scale_factor_h  
        
        return resized_flow
    
    def warp_image(self, image, flow):
        n, c, h, w = image.size()
        resized_flow = self.resize_flow(flow, target_size=(h, w))
        
        # 기존 grid 생성 방식에서 float16을 사용하여 메모리 사용을 줄임
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h, dtype=torch.float16, device=image.device), 
            torch.arange(0, w, dtype=torch.float16, device=image.device), 
            indexing='ij'
        )
        grid = torch.stack((grid_x, grid_y), dim=0)  # [2, h, w]
        grid = grid.unsqueeze(0).repeat(n, 1, 1, 1)  # [n, 2, h, w]
        
        # float32로 변환하여 flow와 연산 (float32로 변환해도 메모리 사용이 크게 늘지 않음)
        grid = grid.to(torch.float32) + resized_flow

        # 그리드 값을 -1에서 1 범위로 정규화 (경계 조건 개선을 위해 0.5 추가)
        grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :].clone() / (w - 0.5) - 1.0
        grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :].clone() / (h - 0.5) - 1.0
        
        # grid의 차원을 재배열하여 F.grid_sample에 전달
        grid = grid.permute(0, 2, 3, 1)  # [n, h, w, 2]
        
        # padding_mode를 'border'로 변경하여 경계 조건 개선, align_corners는 True로 설정
        warped_image = F.grid_sample(image, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        return warped_image
    
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
        
        #Compute flow between Frame1 and Frame2
        with autocast(enabled=self.args.mixed_precision):
            img_pair_left = [(ldr_left_exph_cap[i], ldr_left_expl_cap[i]) for i in range(ldr_left_exph_cap.shape[0])]
            # img_pair_right = [(ldr_right_exph_cap[i], ldr_right_expl_cap[i]) for i in range(ldr_right_exph_cap.shape[0])]
            flow_left = self.compute_optical_flow_batch(self.flow_model, img_pair_left)
            # flow_right = self.compute_optical_flow_batch(self.flow_model, img_pair_right)
        
        #* 2. Estimate disparity map independently
        disp1 = self.disp_recon_net(ldr_left_exph_cap, ldr_right_exph_cap)
        disp2 = self.disp_recon_net(ldr_left_expl_cap, ldr_right_expl_cap)
        
        warped_disp2 = self.warp_image(disp2[-1], flow_left)
        
        #* 4. Mask 생성 및 hole 처리
        mask_disp2 = (warped_disp2 > 0).float()  # 유효한 영역은 1, hole 영역은 0으로 설정

        # Warped disparity에서 hole 처리
        valid_disp2 = mask_disp2 * warped_disp2 + (1 - mask_disp2) * disp2[-1]

        #* 5. 가중 평균 계산
        fused_disp = mask_disp2 * (disp1[-1] + warped_disp2) / 2.0 + (1 - mask_disp2) * disp1[-1]

            
        return fused_disp, disp1, warped_disp2