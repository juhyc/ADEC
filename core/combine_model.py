import sys
sys.path.append('core')

import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from PIL import Image
import matplotlib.pyplot as plt
from utils.mask import soft_binary_threshold_batch
from core.saec import *
from core.utils.simulate import *
from core.raft_stereo import RAFTStereo
from core.utils.utils import InputPadder
from core.disparity_fusion_ResUnet import DisparityFusion_ResUnet
import core.stereo_datasets as datasets

DEVICE = 'cuda'
  
def load_image(imfile):
    img = np.array(imfile).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

class CombineModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.GlobalFeatureNet = GlobalFeatureNet()
        self.RAFTStereo = RAFTStereo(args)
        self.DisparityFusion = DisparityFusion_ResUnet()
    
    def convert_to_tensor(self, image):
        if isinstance(image, Image.Image):
            to_tensor = ToTensor()
            return to_tensor(image)
        return image  
        
    def forward(self, left_hdr, right_hdr, iters=12, flow_init=None, test_mode=False):

        #^ Simulator Module HDR -> LDR
        # Todo) 초기 Captured image에 필요한 exposure 값을 random 한 값이 아니라 HDR scene에 기반한 값으로 제시.
        e_rand_high_pair = generate_random_exposures(left_hdr.shape[0])
        e_rand_low_pair = generate_random_exposures(right_hdr.shape[0])
        
        # e_rand_l_pair = torch.tensor([[1.3]])
        # e_rand_r_pair = torch.tensor([[0.7]])

        # HDR scene
        phi_l_exph = ImageFormation(left_hdr, e_rand_high_pair, device=DEVICE)
        phi_l_expl = ImageFormation(left_hdr, e_rand_low_pair, device=DEVICE)
        phi_r_exph = ImageFormation(right_hdr, e_rand_high_pair, device=DEVICE)
        phi_r_expl = ImageFormation(right_hdr, e_rand_low_pair, device=DEVICE)
              
        #^ Captured LDR image pair
        ldr_left_exph_cap = phi_l_exph.noise_modeling()
        ldr_right_exph_cap = phi_r_exph.noise_modeling()
        ldr_left_expl_cap = phi_l_expl.noise_modeling()
        ldr_right_expl_cap = phi_r_expl.noise_modeling()

        stacked_histo_exph_pair = calculate_histograms2(ldr_left_exph_cap, ldr_right_exph_cap)
        stacked_histo_expl_pair = calculate_histograms2(ldr_left_expl_cap, ldr_right_expl_cap)
        
        #^ Global FeatureNetwork
        # e_exp_l_pair = self.GlobalFeatureNet(stacked_histo_tensor_l.T)
        # e_exp_r_pair = self.GlobalFeatureNet(stacked_histo_tensor_r.T)
        
        # ! Tensor with 2 elements cannot be converted to Scalar
        # ! output_l, output_r shape [batch_size, estimated value]
    
        e_exp_high = self.GlobalFeatureNet(stacked_histo_exph_pair.T)
        e_exp_low = self.GlobalFeatureNet(stacked_histo_expl_pair.T)
        
        e_shifted_high = exposure_shift(e_rand_high_pair.to(device=DEVICE), e_exp_high)
        e_shifted_low = exposure_shift(e_rand_low_pair.to(device=DEVICE), e_exp_low)
        
        #^ Check Exposure values
        print("=====Random exp values=====")
        print(f"Random exp_l : {e_rand_high_pair[0].item():4f}, Random exp_r : {e_rand_low_pair[0].item():4f}")
        print("=====Before shifted exp values=====")
        print(f"output_exp_l : {e_exp_high[0].item()}, output_exp_r : {e_exp_low[0].item()}")
        print("=====Shifted exp values=====")
        print(f"shifted_exp_l : {e_shifted_high[0].item():4f}, shifted_exp_r : {e_shifted_low[0].item():4f}")
        
        #^ Simulate Image LDR with shifted exposure value
        
        # Todo) 메모리 사용량 고려해서 순차적 연산으로 수정
        phi_hat_l_exph = ImageFormation(left_hdr, e_shifted_high, device=DEVICE)
        left_ldr_adj_exph = phi_hat_l_exph.noise_modeling()
        phi_hat_r_exph = ImageFormation(right_hdr, e_shifted_high, device=DEVICE)
        right_ldr_adj_exph = phi_hat_r_exph.noise_modeling()
        
        phi_hat_l_expl = ImageFormation(left_hdr, e_shifted_low, device=DEVICE)
        left_ldr_adj_expl = phi_hat_l_expl.noise_modeling()
        phi_hat_r_expl = ImageFormation(right_hdr, e_shifted_low, device=DEVICE)
        right_ldr_adj_expl = phi_hat_r_expl.noise_modeling()
        
        # Todo) Saturation Mask calculation on left image
        mask_exph = soft_binary_threshold_batch(left_ldr_adj_exph)
        mask_expl = soft_binary_threshold_batch(left_ldr_adj_expl)
        
        # ^ Run Raft-stereo model on 2 pair
        disparity_exph = self.RAFTStereo(left_ldr_adj_exph, right_ldr_adj_exph) # list, list[0]= [B, 1, H, W]
        disparity_expl = self.RAFTStereo(left_ldr_adj_expl, right_ldr_adj_expl)

        # * For exposure shift check
        captured_rand_img_list = [ldr_left_exph_cap, ldr_right_exph_cap, ldr_left_expl_cap, ldr_right_expl_cap]
        captured_adj_img_list = [left_ldr_adj_exph, right_ldr_adj_exph, left_ldr_adj_expl, right_ldr_adj_expl]
        
        # Todo) Run Disparity_fusion module using disparity1,2 and its mask.
        # Todo) Check : disparity1,2, mask type. shape
        fused_disparity = self.DisparityFusion(disparity_exph[-1], disparity_expl[-1], mask_exph, mask_expl)
        
        # exposure_dict = {'e_rand_l': e_rand_l_pair[0].item(),'e_rand_r':e_rand_r_pair[0].item(),
        #                  'e_exp_l': e_exp_l_pair[0].item(), 'e_exp_r': e_exp_r_pair[0].item(),
        #                  'e_shifted_l' : e_shifted_l[0].item(), 'e_shifted_r' : e_shifted_r[0].item()
        #                  }
        
        return fused_disparity, disparity_exph[-1], disparity_expl[-1], captured_rand_img_list, captured_adj_img_list

    
