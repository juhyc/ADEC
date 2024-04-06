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

DEVICE = 'cuda'

###############################
# * End to end pipeline with exposure control module using network
###############################

def load_image(imfile):
    img = np.array(imfile).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

class FeatureDimReducer(nn.Module):
    def __init__ (self, input_dim = 1000, output_dim = 64):
        super(FeatureDimReducer, self).__init__()
        self.reducer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
    def forward(self, x):
        return self.reducer(x)

def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class CombineModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.GlobalFeatureNet1 = GlobalFeatureNet()
        self.GlobalFeatureNet2 = GlobalFeatureNet()
        self.RAFTStereo = RAFTStereo(args)
        
        self.GlobalFeatureNet1.apply(weights_init)
        self.GlobalFeatureNet2.apply(weights_init)
        # self.DisparityFusion = DisparityFusion_ResUnet()
        # * Feature encoder
        # self.feature_encoder = models.resnet50(pretrained=True)
        # self.feature_encoder.eval()
        # self.feature_encoder.fc = nn.Identity()
        # self.dim_reducer = FeatureDimReducer(input_dim=2048, output_dim = 128)
        # * Feature FusionNet for incorporate globalfeature and semantic feature
        # self.feature_fusion = FeatureFusionNet()
    
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
        ldr_left_exph_cap = phi_l_exph.noise_modeling()
        ldr_right_exph_cap = phi_r_exph.noise_modeling()
        ldr_left_expl_cap = phi_l_expl.noise_modeling()
        ldr_right_expl_cap = phi_r_expl.noise_modeling()
        
        # Calculate image histogram for image statistic feature
        stacked_histo_exph_pair = calculate_histograms2(ldr_left_exph_cap, ldr_right_exph_cap)
        stacked_histo_expl_pair = calculate_histograms2(ldr_left_expl_cap, ldr_right_expl_cap)
        
        #^ Feature encoder for semantic information
        # with torch.no_grad():
        #     ldr_left_exph_feature = self.feature_encoder(ldr_left_exph_cap)
        #     ldr_right_exph_feature = self.feature_encoder(ldr_right_exph_cap)
        #     ldr_left_expl_feature = self.feature_encoder(ldr_left_expl_cap)
        #     ldr_right_expl_feature = self.feature_encoder(ldr_right_expl_cap)
            
        # ldr_left_exph_featured_reduced = self.dim_reducer(ldr_left_exph_feature)
        # ldr_right_exph_featured_reduced = self.dim_reducer(ldr_right_exph_feature)
        # ldr_left_expl_featured_reduced = self.dim_reducer(ldr_left_expl_feature)
        # ldr_right_expl_featured_reduced = self.dim_reducer(ldr_right_expl_feature)
        
        #^ Global FeatureNetwork        
        # ! Tensor with 2 elements cannot be converted to Scalar
        # ! output_l, output_r shape [batch_size, estimated value]
    
        e_exp_high = self.GlobalFeatureNet1(stacked_histo_exph_pair.T)
        e_exp_low = self.GlobalFeatureNet2(stacked_histo_expl_pair.T)
        
        #^ Concat statistic, semantic inform
        # combined_feature_high = torch.cat((ldr_left_exph_featured_reduced, e_exp_high, ldr_right_exph_featured_reduced), dim = 1)
        # combined_feature_low = torch.cat((ldr_left_expl_featured_reduced, e_exp_low, ldr_right_expl_featured_reduced), dim = 1) #[batch, 512]
        
        #^ Feature Fusion Network [statistic, semantic]
        # e_exp_high2 = self.feature_fusion(combined_feature_high) # [batch, 1]
        # e_exp_low2 = self.feature_fusion(combined_feature_low)
        
        #^ Exposure shift
        e_shifted_high = exposure_shift2(e_rand_high_pair.to(device=DEVICE), e_exp_high)
        e_shifted_low = exposure_shift2(e_rand_low_pair.to(device=DEVICE), e_exp_low)
        
        #^ Check Exposure values
        print("=====Random exp values=====")
        print(f"Random exp_l : {e_rand_high_pair[0].item():4f}, Random exp_r : {e_rand_low_pair[0].item():4f}")
        print("=====Before shifted exp values=====")
        print(f"output_exp_l : {e_exp_high[0].item()}, output_exp_r : {e_exp_low[0].item()}")
        print("=====Shifted exp values=====")
        print(f"shifted_exp_l : {e_shifted_high[0].item():4f}, shifted_exp_r : {e_shifted_low[0].item():4f}")
        
        #^ Simulate Image LDR with shifted exposure value
        phi_hat_l_exph = ImageFormation(left_hdr, e_shifted_high, device=DEVICE)
        left_ldr_adj_exph = phi_hat_l_exph.noise_modeling()
        phi_hat_r_exph = ImageFormation(right_hdr, e_shifted_high, device=DEVICE)
        right_ldr_adj_exph = phi_hat_r_exph.noise_modeling()
        phi_hat_l_expl = ImageFormation(left_hdr, e_shifted_low, device=DEVICE)
        left_ldr_adj_expl = phi_hat_l_expl.noise_modeling()
        phi_hat_r_expl = ImageFormation(right_hdr, e_shifted_low, device=DEVICE)
        right_ldr_adj_expl = phi_hat_r_expl.noise_modeling()
              
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
        
        # * Existed. disaprity fusion using network.
        # fused_disparity = self.DisparityFusion(disparity_exph[-1], disparity_expl[-1], mask_exph, mask_expl)
        # exposure_dict = {'e_rand_l': e_rand_l_pair[0].item(),'e_rand_r':e_rand_r_pair[0].item(),
        #                  'e_exp_l': e_exp_l_pair[0].item(), 'e_exp_r': e_exp_r_pair[0].item(),
        #                  'e_shifted_l' : e_shifted_l[0].item(), 'e_shifted_r' : e_shifted_r[0].item()
        #                  }
        
        # ^ disparity fusion using exposure mask multiplication
        disparity_exph_mul = disparity_exph[-1] * mask_exph
        disparity_expl_mul = disparity_expl[-1] * mask_expl
        
        epsilon = 1e-8
        fused_disparity_mul = (disparity_exph_mul + disparity_expl_mul)/(mask_exph + mask_expl + epsilon)
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

    

