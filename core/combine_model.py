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

def check_hdr_image(image):
    print(type(image))
    print(image.shape)
    if isinstance(image, torch.Tensor):
        temp = image.clone().detach()
        temp = temp[0].cpu().permute(1,2,0)
        temp = temp.numpy().astype(np.uint8)
    plt.imshow(temp)
    plt.show()

def check_ldr_image(image):
    if isinstance(image, torch.Tensor):
        temp = image.clone().detach()
        temp = temp[0].cpu().permute(1,2,0)
        temp = torch.clamp(temp*255, 0, 255)
        temp = temp.numpy().astype(np.uint8)
    plt.imshow(temp)
    plt.show()
    
def check_ldr_image2(image):
    if isinstance(image, torch.Tensor):
        temp = image.clone().detach()
        temp = temp[1].cpu().permute(1,2,0)
        temp = torch.clamp(temp*255, 0, 255)
        temp = temp.numpy().astype(np.uint8)
    plt.imshow(temp)
    plt.show()
    

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
    
    # ^ SAEC로 통과하기 전 hdr image를 ldr captureed image로 simulation
    # ^ 목적 : Random exposure로 capture 한 이미지와, random exposure value 획득
    def simulator_before_saec(self, left_hdr_image, right_hdr_image):
        
        left_image_torch = self.convert_to_tensor(left_hdr_image)
        right_image_torch = self.convert_to_tensor(right_hdr_image)
        
        # ! For sanity check, not random exposure but left low right high    
        # exp_rand_l, exp_rand_r = generate_random_exposure()
        exp_rand_l, exp_rand_r = 0.5, 1.5
        
        a_l, b_l = cal_dynamic_range(left_image_torch, exp_rand_l)
        a_r, b_r = cal_dynamic_range(right_image_torch, exp_rand_r)
        
        left_ldr_image = adjust_dr(left_hdr_image, exp_rand_l, (a_l,b_l))
        right_ldr_image = adjust_dr(right_hdr_image, exp_rand_r, (a_r, b_r))

        left_ldr_image = poisson_gauss_noise(left_ldr_image, iso=100)
        right_ldr_image = poisson_gauss_noise(right_ldr_image, iso=100)
        
        return left_ldr_image, right_ldr_image, exp_rand_l, exp_rand_r
    
    # ^ SAEC 통과후 예측한 exposure 값을 이용해서 adjusted captured image simulation
    def simulator_after_saec(self, left_ldr_image, right_ldr_image, left_hdr_image, right_hdr_image, shifted_exp_l, shifted_exp_r):
        
        a_l_s, b_l_s = cal_dynamic_range(left_ldr_image*255.0, shifted_exp_l)
        a_r_s, b_r_s = cal_dynamic_range(right_ldr_image*255.0, shifted_exp_r)
        
        # Todo) (기존) : 서로 다른 노출값의 1 pair image -> (변경) : 서로 다른 노출값으로 번갈아서 capture한 2 pair image
        sim_left_ldr_image1 = adjust_dr(left_hdr_image, shifted_exp_l, (a_l_s, b_l_s))
        sim_right_ldr_image1 = adjust_dr(right_hdr_image, shifted_exp_r, (a_r_s, b_r_s))
        sim_left_ldr_image2 = adjust_dr(left_hdr_image, shifted_exp_r, (a_r_s, b_r_s))
        sim_right_ldr_image2 = adjust_dr(right_hdr_image, shifted_exp_l, (a_l_s, b_l_s))

        sim_left_ldr_image1 = poisson_gauss_noise(sim_left_ldr_image1, iso = 100)
        sim_right_ldr_image1 = poisson_gauss_noise(sim_right_ldr_image1, iso = 100)
        sim_left_ldr_image2 = poisson_gauss_noise(sim_left_ldr_image2, iso = 100)
        sim_right_ldr_image2 = poisson_gauss_noise(sim_right_ldr_image2, iso = 100)

        return sim_left_ldr_image1, sim_right_ldr_image1, sim_left_ldr_image2, sim_right_ldr_image2
    
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
        # exp_rand_l, exp_rand_r = generate_random_exposure()
        e_rand_l, e_rand_r = 0.75, 1.25
        phi_l = ImageFormation(left_hdr, e_rand_l, device=DEVICE)
        phi_r = ImageFormation(right_hdr, e_rand_r, device=DEVICE)
              
        # Captured LDR image pair
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
        phi_hat_r1 = ImageFormation(right_hdr, e_shifted_r, device=DEVICE)
        phi_hat_l2 = ImageFormation(left_hdr, e_shifted_r, device=DEVICE)
        phi_hat_r2 = ImageFormation(right_hdr, e_shifted_l, device=DEVICE)
        
        left_ldr_adj1 = phi_hat_l1.noise_modeling()
        right_ldr_adj1 = phi_hat_r1.noise_modeling()
        left_ldr_adj2 = phi_hat_l2.noise_modeling()
        right_ldr_adj2 = phi_hat_r2.noise_modeling()
        
        # Todo) Saturation Mask calculation on left image
        mask1 = soft_binary_threshold_batch(left_ldr_adj1)
        mask2 = soft_binary_threshold_batch(left_ldr_adj2)
        
        # ^ Run Raft-stereo model on 2 pair
        disparity1 = self.RAFTStereo(left_ldr_adj1, right_ldr_adj2) # list, list[0]= [B, 1, H, W]
        disparity2 = self.RAFTStereo(left_ldr_adj2, right_ldr_adj1)
              
        # Todo) Run Disparity_fusion module using disparity1,2 and its mask.
        # Todo) Check : disparity1,2, mask type. shape
        fused_disparity = self.DisparityFusion(disparity1[-1], disparity2[-1], mask1, mask2)
        
        # exposure_dict = {'e_exp_l': e_exp_l, 'e_exp_r':e_exp_r, 'e_shifted_l': e_shifted_l, 'shifted_exp_r': e_shifted_r}
        
        return fused_disparity, disparity1[-1], disparity2[-1]

    
