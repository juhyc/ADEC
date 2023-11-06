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
from core.saec import *
from core.utils.simulate import *
from core.raft_stereo import RAFTStereo
from core.utils.utils import InputPadder
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
        
    def exposure_change(self, image, exposure_factor):
        image = image * exposure_factor
        image = torch.clamp(image, min = 0, max = 1)
        return image
    
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
    
    def simulator_after_saec(self, left_ldr_image, right_ldr_image, left_hdr_image, right_hdr_image, shifted_exp_l, shifted_exp_r):
        
        a_l_s, b_l_s = cal_dynamic_range(left_ldr_image*255.0, shifted_exp_l)
        a_r_s, b_r_s = cal_dynamic_range(right_ldr_image*255.0, shifted_exp_r)
        
        # ! 문제
        sim_left_ldr_image = adjust_dr(left_hdr_image, shifted_exp_l, (a_l_s, b_l_s))
        sim_right_ldr_image = adjust_dr(right_hdr_image, shifted_exp_r, (a_r_s, b_r_s))

        sim_left_ldr_image = poisson_gauss_noise(sim_left_ldr_image, iso = 100)
        sim_right_ldr_image = poisson_gauss_noise(sim_right_ldr_image, iso = 100)
        
        # before denormalized
        sim_left_before_norm = sim_left_ldr_image
        sim_right_before_norm = sim_right_ldr_image
        
        sim_left_ldr_image = denormalized_image(sim_left_ldr_image, shifted_exp_l, (a_l_s, b_l_s))
        sim_right_ldr_image = denormalized_image(sim_right_ldr_image, shifted_exp_r, (a_r_s, b_r_s))

        return sim_left_ldr_image, sim_right_ldr_image, sim_left_before_norm, sim_right_before_norm
    
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

        #* Simulator Module HDR -> LDR
        # exp_rand_l, exp_rand_r = generate_random_exposure()
        e_rand_l, e_rand_r = 0.5, 1.5
        phi_l = ImageFormation(left_hdr, e_rand_l, device=DEVICE)
        phi_r = ImageFormation(right_hdr, e_rand_r, device=DEVICE)
        
        left_ldr_cap = phi_l.noise_modeling()
        right_ldr_cap = phi_r.noise_modeling()
        
        stacked_histo_tensor_l, stacked_histo_tensor_r = self.calculate_histograms(left_ldr_cap, right_ldr_cap)
        
        #* Global FeatureNetwork    
        e_exp_l = self.GlobalFeatureNet(stacked_histo_tensor_l.T)
        e_exp_r = self.GlobalFeatureNet(stacked_histo_tensor_r.T)
        
        # ! Tensor with 2 elements cannot be converted to Scalar
        # ! output_l, output_r shape [batch_size, estimated value]
  
        e_shifted_l = self.exposure_shift(e_rand_l, e_exp_l.mean().item())
        e_shifted_r = self.exposure_shift(e_rand_r, e_exp_r.mean().item())
        
        # * Check Exposure values
        print("====Random exp values====")
        print(f"Random exp_l : {e_rand_l}, Random exp_r : {e_rand_r}")
        print("====before shifted exp values====")
        print(f"output_exp_l : {e_exp_l.mean().item()}, output_exp_r : {e_exp_r.mean().item()}")
        print("====shifted exp values====")
        print(f"shifted_exp_l : {e_shifted_l}, shifted_exp_r : {e_shifted_r}")
        
        # * Simulate Image LDR with shifted exposure value
        
        phi_hat_l = ImageFormation(left_hdr, e_shifted_l, device=DEVICE)
        phi_hat_r = ImageFormation(right_hdr, e_shifted_r, device=DEVICE)
        
        left_ldr_adj = phi_hat_l.noise_modeling()
        right_ldr_adj = phi_hat_r.noise_modeling()
        
        left_ldr_adj_denom = phi_hat_l.denormalized_image(left_ldr_adj, e_shifted_l)
        right_ldr_adj_denom = phi_hat_r.denormalized_image(right_ldr_adj, e_shifted_r)
        
        flow_predictions = self.RAFTStereo(left_ldr_adj_denom, right_ldr_adj_denom, iters = 12) # list, list[0]= [B, 1, H, W]
        
        exposure_dict = {'e_exp_l': e_exp_l, 'e_exp_r':e_exp_r, 'e_shifted_l': e_shifted_l, 'shifted_exp_r': e_shifted_r}
        return flow_predictions, left_ldr_cap, right_ldr_cap, left_ldr_adj_denom, right_ldr_adj_denom, exposure_dict , left_ldr_adj, right_ldr_adj

    
