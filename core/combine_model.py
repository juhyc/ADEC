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
        
    def exposure_change(self, image, exposure_factor):
        image = image * exposure_factor
        image = torch.clamp(image, min = 0, max = 1)
        return image
    
    # Todo) 기존 노출값과 예측 노출값을 이용해서 노출값을 조정
    def exposure_shift(self, before_exposure, predicted_exposure, alpha = 0.3):
        difference = predicted_exposure - before_exposure
        adjusted_difference = alpha * difference
        shifted_exposure = before_exposure + adjusted_difference
    
        return shifted_exposure
    
    def convert_to_tensor(self, image):
        if isinstance(image, Image.Image):
            to_tensor = ToTensor()
            return to_tensor(image)
        return image

    def calculate_histograms(self, left_ldr_image, right_ldr_image):
        
        #^ Calculate histogram with multi scale
        histogram_coarest_l = histogram_subimage(left_ldr_image, 1)
        histogram_intermediate_l = histogram_subimage(left_ldr_image, 3)
        histogram_finest_l = histogram_subimage(left_ldr_image,7)
        
        histogram_coarest_r = histogram_subimage(right_ldr_image, 1)
        histogram_intermediate_r = histogram_subimage(right_ldr_image, 3)
        histogram_finest_r = histogram_subimage(right_ldr_image, 7)
        
        #^ Stack histogram [256,59]
        list_of_histograms_l = [histogram_coarest_l, histogram_intermediate_l, histogram_finest_l]
        stacked_histo_tensor_l = stack_histogram(list_of_histograms_l)

        list_of_histograms_r = [histogram_coarest_r, histogram_intermediate_r, histogram_finest_r]
        stacked_histo_tensor_r = stack_histogram(list_of_histograms_r)
        
        return stacked_histo_tensor_l, stacked_histo_tensor_r
    
        
    def forward(self, left_hdr, right_hdr, iters=12, flow_init=None, test_mode=False):

        #^ Simulator Module HDR -> LDR
        # Todo) 초기 Captured image에 필요한 exposure 값을 random 한 값이 아니라 HDR scene에 기반한 값으로 제시.
        # exp_rand_l, exp_rand_r = generate_random_exposure()
        e_rand_l, e_rand_r = 0.75, 1.25
        # HDR scene
        phi_l = ImageFormation(left_hdr, e_rand_l, device=DEVICE)
        phi_r = ImageFormation(right_hdr, e_rand_r, device=DEVICE)
              
        #^ Captured LDR image pair
        left_ldr_cap = phi_l.noise_modeling()
        right_ldr_cap = phi_r.noise_modeling()
        
        stacked_histo_tensor_l, stacked_histo_tensor_r = self.calculate_histograms(left_ldr_cap, right_ldr_cap)
        
        #^ Global FeatureNetwork
        e_exp_l = self.GlobalFeatureNet(stacked_histo_tensor_l.T)
        e_exp_r = self.GlobalFeatureNet(stacked_histo_tensor_r.T)
        
        # ! Tensor with 2 elements cannot be converted to Scalar
        # ! output_l, output_r shape [batch_size, estimated value]
  
        e_shifted_l = self.exposure_shift(e_rand_l, e_exp_l.mean().item())
        e_shifted_r = self.exposure_shift(e_rand_r, e_exp_r.mean().item())
        
        #^ Check Exposure values
        print("=====Random exp values=====")
        print(f"Random exp_l : {e_rand_l}, Random exp_r : {e_rand_r}")
        print("=====Before shifted exp values=====")
        print(f"output_exp_l : {e_exp_l.mean().item():4f}, output_exp_r : {e_exp_r.mean().item():4f}")
        print("=====Shifted exp values=====")
        print(f"shifted_exp_l : {e_shifted_l:4f}, shifted_exp_r : {e_shifted_r:4f}")
        
        #^ Simulate Image LDR with shifted exposure value
        
        # Todo) 기존에 1 pair 이미지에서 번갈아서 찍은 형태인 2 pair로 : O
        phi_hat_l1 = ImageFormation(left_hdr, e_shifted_l, device=DEVICE)
        phi_hat_r1 = ImageFormation(right_hdr, e_shifted_l, device=DEVICE)
        phi_hat_l2 = ImageFormation(left_hdr, e_shifted_r, device=DEVICE)
        phi_hat_r2 = ImageFormation(right_hdr, e_shifted_r, device=DEVICE)
        
        left_ldr_adj1 = phi_hat_l1.noise_modeling()
        right_ldr_adj1 = phi_hat_r1.noise_modeling()
        left_ldr_adj2 = phi_hat_l2.noise_modeling()
        right_ldr_adj2 = phi_hat_r2.noise_modeling()
        
        # Todo) Saturation Mask calculation on left image
        mask1 = soft_binary_threshold_batch(left_ldr_adj1)
        mask2 = soft_binary_threshold_batch(left_ldr_adj2)
        
        # ^ Run Raft-stereo model on 2 pair
        disparity1 = self.RAFTStereo(left_ldr_adj1, right_ldr_adj1) # list, list[0]= [B, 1, H, W]
        disparity2 = self.RAFTStereo(left_ldr_adj2, right_ldr_adj2)

        # * For exposure shift check    
        captured_img_list = [left_ldr_adj1, right_ldr_adj1, left_ldr_adj2, right_ldr_adj2]
        
        # Todo) Run Disparity_fusion module using disparity1,2 and its mask.
        # Todo) Check : disparity1,2, mask type. shape
        fused_disparity = self.DisparityFusion(disparity1[-1], disparity2[-1], mask1, mask2)
        
        # exposure_dict = {'e_exp_l': e_exp_l, 'e_exp_r':e_exp_r, 'e_shifted_l': e_shifted_l, 'shifted_exp_r': e_shifted_r}
        
        return fused_disparity, disparity1[-1], disparity2[-1], captured_img_list

    
