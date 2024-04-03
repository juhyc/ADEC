import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import os
import cv2
import glob
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from core.utils.simulate import *
from core.raft_stereo import RAFTStereo
from core.utils.utils import InputPadder
from PIL import Image
from core.saec import *
from core.combine_model2 import CombineModel
from core.utils.display import *

from core.stereo_datasets2 import fetch_dataloader

# * Test code

writer = SummaryWriter('runs/demo')

DEVICE = 'cuda'

def load_image(file_name):
    ext = os.path.splitext(file_name)[-1]
    
    if ext == '.hdr':
        img = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        min_val = np.min(img)
        max_val = np.max(img)
        normalized_img = (img - min_val) / (max_val - min_val)
        
        normalized_img = torch.from_numpy(normalized_img).permute(2,0,1).float()
        return normalized_img[None].to(DEVICE)
    
    elif ext == '.npy':
        return np.load(file_name)
    
    return []


def demo(args):
    
    model = torch.nn.DataParallel(CombineModel(args))
    
    checkpoint = torch.load(args.restore_ckpt)
    
    model.load_state_dict(checkpoint)
    
    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        
        print(f"Found {len(left_images)} images./")
        num_cnt = 0
        
        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            # padder = InputPadder(image1.shape, divis_by=32)
            # image1, image2 = padder.pad(image1, image2)

            fused_disp, disp1, disp2, origin_img_list, cap_rand_img_list, cap_adj_img_list, disp1_r, disp2_r = model(image1, image2, test_mode=True) 
            
            # * Visualize to check intermediate stage
            writer.add_image('Demo/Fused_disparity', visualize_flow_cmap(fused_disp), num_cnt)
            writer.add_image('Demo/disp1', visualize_flow_cmap(disp1),num_cnt)
            writer.add_image('Demo/disp2', visualize_flow_cmap(disp2),num_cnt)  
            writer.add_image('Demo/disp1_r', visualize_flow_cmap(disp1_r),num_cnt)
            writer.add_image('Demo/disp2_r', visualize_flow_cmap(disp2_r),num_cnt)
            writer.add_image('Demo/hdr_left', origin_img_list[0][0], num_cnt)
            writer.add_image('Demo/img1_rand_left', cap_rand_img_list[0][0], num_cnt)
            writer.add_image('Demo/img2_rand_left', cap_rand_img_list[1][0], num_cnt)
            writer.add_image('Demo/img1_adj_left', cap_adj_img_list[0][0], num_cnt)
            writer.add_image('Demo/img2_adj_left', cap_adj_img_list[1][0], num_cnt)
            
            num_cnt += 1
           

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="datasets/Middlebury/MiddEval3/testH/*/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="datasets/Middlebury/MiddEval3/testH/*/im1.png")
    parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    
    args = parser.parse_args()

    demo(args)