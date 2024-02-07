import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from core.raft_stereo import RAFTStereo
from core.combine_model import CombineModel

from evaluate_stereo import *
from core.stereo_datasets2 import fetch_dataloader
import matplotlib.pyplot as plt
from PIL import Image

from core.loss import BerHuLoss
from core.depth_datasets import DepthDataset_stereo
from core.utils.display import *

# Tensorboard를 위한 Writer 초기화
writer = SummaryWriter('runs/combine_pipeline_demo')

# CUDA
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    
    model = torch.nn.DataParallel(CombineModel(args), device_ids=[0])
    print("Parameter Count : %d" % count_parameters(model))
    
    #^Load dataloader
    train_loader = fetch_dataloader(args)
    criterion = nn.SmoothL1Loss().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr = 0.0002)
    total_steps = 0
    
    #^ Load RAFT module checkpoint
    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        raft_checkpoint = torch.load(args.restore_ckpt)
        
        # * downsampling = 3 인 경우
        if args.n_downsample == 3:
            del raft_checkpoint['module.update_block.mask.2.weight'], raft_checkpoint['module.update_block.mask.2.bias']
        
        new_raft_state_dict = {}
        for k, v in raft_checkpoint.items():
            if k.startswith('module.'):
                new_k = k[7:]
            else:
                new_k = k
            new_raft_state_dict[new_k] = v
        
        combined_state_dict = model.state_dict()
        count = 0
        
        for k in new_raft_state_dict.keys():
            combined_keys = "module.RAFTStereo." + k
            if combined_keys in combined_state_dict:
                combined_state_dict[combined_keys] = new_raft_state_dict[k] 
                count += 1
        
        model.load_state_dict(combined_state_dict)
                
        logging.info(f"Done loading checkpoint")
    
    #^ Load SAEC checkpoint
    # saec_checkpoint = torch.load('/home/juhyung/SAEC/checkpoints/SAEC_200_epoch.pth')
    # model.load_state_dict(saec_checkpoint)
    # logging.info(f"Done loading saec checkpoint")
    
    model.cuda()
    model.train()
    
    #^ Freeze RAFTstereo module
    for param in model.module.RAFTStereo.parameters():
        param.requires_grad = False
    # for param in model.module.DisparityFusion.parameters():
    #     param.requires_grad = False
    
    model.module.RAFTStereo.freeze_bn()
    
    validation_frequency = 10
    
    should_keep_training = True
    global_batch_num = 0
    
    while should_keep_training:
    
        for i_batch, (_, *data_blob) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            left_hdr, right_hdr, disparity, valid = [x.cuda() for x in data_blob]
            
            print(left_hdr.shape)
            
            fused_disparity, disparity1, disparity2, captured_rand_img_list, captured_img_list = model(left_hdr, right_hdr, iters=args.train_iters)
            
            assert model.training
            
            
            


if __name__  == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='SAEC', help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=6, help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['carla'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=100000, help="length of training schedule.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[320, 720], help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=16, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

    # RAFT Architecure choices
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    
    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=None, help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[0, 0], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')

    args = parser.parse_args()
    
    train(args)