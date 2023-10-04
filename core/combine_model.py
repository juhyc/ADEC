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
    
    def convert_to_tensor(self, image):
        if isinstance(image, Image.Image):
            to_tensor = ToTensor()
            return to_tensor(image)
        return image    
     
    def simulator_before_saec(self, left_hdr_image, right_hdr_image):
        
        left_image_torch = self.convert_to_tensor(left_hdr_image)
        right_image_torch = self.convert_to_tensor(right_hdr_image)
            
        exp_rand_l, exp_rand_r = generate_random_exposure()
        a_l, b_l = cal_dynamic_range(left_image_torch, exp_rand_l)
        a_r, b_r = cal_dynamic_range(right_image_torch, exp_rand_r)
        
        left_ldr_image = adjust_dr(left_hdr_image, exp_rand_l, (a_l,b_l))
        right_ldr_image = adjust_dr(right_hdr_image, exp_rand_r, (a_r, b_r))

        left_ldr_image = poisson_gauss_noise(left_ldr_image, iso=100)
        right_ldr_image = poisson_gauss_noise(right_ldr_image, iso=100)
        
        return left_ldr_image, right_ldr_image, exp_rand_l, exp_rand_r
    
    def simulator_after_saec(self, left_ldr_image, right_ldr_image, left_hdr_image, right_hdr_image, shifted_exp_l, shifted_exp_r):
        
        a_l_s, b_l_s = cal_dynamic_range(left_ldr_image, shifted_exp_l)
        a_r_s, b_r_s = cal_dynamic_range(right_ldr_image, shifted_exp_r)

        sim_left_ldr_image = adjust_dr(left_hdr_image, shifted_exp_l, (a_l_s, b_l_s))
        sim_right_ldr_image = adjust_dr(right_hdr_image, shifted_exp_r, (a_r_s, b_r_s))

        sim_left_ldr_image = poisson_gauss_noise(sim_left_ldr_image, iso = 100)
        sim_right_ldr_image = poisson_gauss_noise(sim_right_ldr_image, iso = 100)
        
        return sim_left_ldr_image, sim_right_ldr_image
    
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
    
    
        
    def forward(self, left_hdr_image, right_hdr_image, iters=12, flow_init=None, test_mode=False):
        
        #* Simulator Module HDR -> LDR
        left_ldr_image, right_ldr_image, exp_rand_l, exp_rand_r  = self.simulator_before_saec(left_hdr_image, right_hdr_image)

        stacked_histo_tensor_l, stacked_histo_tensor_r = self.calculate_histograms(left_ldr_image, right_ldr_image)
        
        #* Global FeatureNetwork    
        output_l = self.GlobalFeatureNet(stacked_histo_tensor_l.T)
        output_r = self.GlobalFeatureNet(stacked_histo_tensor_r.T)
        
        # ! Tensor with 2 elements cannot be converted to Scalar
        # ! output_l, output_r shape [batch_size, estimated value]
        shifted_exp_l = exposure_shift(exp_rand_l, output_l.mean().item())
        shifted_exp_r = exposure_shift(exp_rand_r, output_r.mean().item())
        
        # * Simulate Image LDR with shited exposure value
        sim_left_ldr_image, sim_right_ldr_image = self.simulator_after_saec(left_ldr_image, right_ldr_image, left_hdr_image, right_hdr_image, shifted_exp_l, shifted_exp_r)
        
        # RAFT
        # padder = InputPadder(image1.shape, divis_by=32)
        # image1, image2 = padder.pad(image1, image2)
        
        # ! If input images are batch
        # _, flow_up = self.RAFTStereo(sim_left_ldr_image, sim_right_ldr_image)
        flow_up = self.RAFTStereo(sim_left_ldr_image, sim_right_ldr_image)
        flow_up = flow_up[0]
        
        # print(type(flow_up))
        # print(flow_up)
        # print(flow_up.shape)
        
        flow_up = flow_up.squeeze()
        
        return flow_up


    
