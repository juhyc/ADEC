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
# * 수정
from core.combine_model2 import CombineModel_wo_net

from evaluate_stereo import *
from core.stereo_datasets2 import fetch_dataloader
import matplotlib.pyplot as plt
from PIL import Image

import core.loss as loss
from core.depth_datasets import DepthDataset_stereo
from core.utils.display import *

# ! Training code without exposure control network

# Initialize writer for tensorboard logging
writer = SummaryWriter('runs/combine_pipeline_carla')

# CUDA
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    
    model = torch.nn.DataParallel(CombineModel_wo_net(args))
    print("Parameter Count : %d" % count_parameters(model))
    
    #^Load dataloader
    train_loader = fetch_dataloader(args)
    # criterion = nn.MSELoss().to(DEVICE)
    criterion = nn.L1Loss().to(DEVICE)
    # criterion = BerHuLoss().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr = 0.0001)
    total_steps = 0
    
    #^ Load RAFT module checkpoint
    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        raft_checkpoint = torch.load(args.restore_ckpt)
        
        # * downsampling = 3 case
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
    
    # #^ Load SAEC checkpoint
    # saec_checkpoint = torch.load('/home/user/juhyung/SAEC/checkpoints/SAEC.pth')
    # model.load_state_dict(saec_checkpoint)
    # logging.info(f"Done loading saec checkpoint")
    
    model.cuda()
    # model.train()
    
    #^ Freeze RAFTstereo module
    for param in model.module.RAFTStereo.parameters():
        param.requires_grad = False
    
    model.module.RAFTStereo.freeze_bn()
    
    validation_frequency = 10
    
    should_keep_training = True
    global_batch_num = 0
    
    # ^ Training
    while should_keep_training:
    
        for i_batch, (_, *data_blob) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            left_hdr, right_hdr, disparity, valid = [x.cuda() for x in data_blob]
            
            assert model.training
            
            fused_disparity, disparity1, disparity2, original_img_list, captured_rand_img_list, captured_adj_img_list, mask_list, mask_mul_list = model(left_hdr, right_hdr, iters=args.train_iters)
            
            # ^ Visualize during training
            # visualize disparity 
            writer.add_image('Train/Fused_disparity', visualize_flow_cmap(fused_disparity), global_batch_num)
            writer.add_image('Train/disparity1', visualize_flow_cmap(disparity1), global_batch_num)
            writer.add_image('Train/disparity2', visualize_flow_cmap(disparity2), global_batch_num)
            writer.add_image('Train/disparity_gt', visualize_flow_cmap(disparity), global_batch_num)
            
            # ! Add mask multiplication result
            writer.add_image('Train/disaprity1_mul', visualize_flow_cmap(mask_mul_list[0]), global_batch_num)
            writer.add_image('Train/disaprity2_mul', visualize_flow_cmap(mask_mul_list[1]), global_batch_num)
            
            # visualize captured image
            writer.add_image('Captured(T)/hdr_left', original_img_list[0][0], global_batch_num)
            writer.add_image('Captured(T)/img1_rand_left', captured_rand_img_list[0][0], global_batch_num)
            writer.add_image('Captured(T)/img2_rand_left', captured_rand_img_list[1][0], global_batch_num)
            
            # visualize dynamic range
            writer.add_image('Captured(T)/hdr_dynamic_range', visualize_dynamic_range(original_img_list[0]), global_batch_num)
            writer.add_image('Histogram(T)/img1_dynamic_range(rand)', visualize_dynamic_range(captured_rand_img_list[0],HDR=False), global_batch_num)
            writer.add_image('Histogram(T)/img2_dynamic_range(rand)', visualize_dynamic_range(captured_rand_img_list[1],HDR=False), global_batch_num)
            
            # visualize adjusted image
            writer.add_image('Captured(T)/img1_adj_left', captured_adj_img_list[0][0], global_batch_num)
            writer.add_image('Captured(T)/img2_adj_left', captured_adj_img_list[1][0], global_batch_num)
            
            writer.add_image('Captured(T)/img1_adj_mask', visualize_mask(mask_list[0]), global_batch_num)
            writer.add_image('Captured(T)/img2_adj_mask', visualize_mask(mask_list[1]), global_batch_num)
            
            writer.add_image('Histogram(T)/img1_dynamic_range(adj)', visualize_dynamic_range(captured_adj_img_list[0], HDR=False), global_batch_num)
            writer.add_image('Histogram(T)/img2_dynamic_range(adj)', visualize_dynamic_range(captured_adj_img_list[1], HDR=False), global_batch_num)
    
            assert model.training
            
            #^ Exist valid mask calculation
            valid_mask = (valid >= 0.5)
            valid_mask = valid_mask.unsqueeze(1)
            
            #^mask visualize to check loss calculation
            
            disparity_loss = criterion(fused_disparity[valid_mask], disparity[valid_mask])
            total_loss = disparity_loss
            
            writer.add_scalar("Training_total_loss", total_loss.item(), global_batch_num)
            global_batch_num += 1        
            # total_loss.backward()
            # optimizer.step()
            
            # Todo) Validation code 수정
            if total_steps % validation_frequency == validation_frequency - 1:
                print("=====Validation=====")
                valid_num = (total_steps / validation_frequency) * 10 + 1
                # * Save validation checkpoint
                # save_path = Path('checkpoints/%d_%s.pth' % (total_steps + 1, args.name))
                # logging.info(f"Saving file {save_path.absolute()}")
                # torch.save(model.state_dict(), save_path)

                valid_loss, valid_d1, valid_fused_disparity, valid_disparity1, valid_disparity2, valid_origin_list, valid_rand_img_list, valid_captured_img_list, valid_disparity_gt, disp_rand1, disp_rand2, mask_list_val, mask_mul_list_val = validate_carla(model.module, iters=32)
                
                model.train()
                model.module.RAFTStereo.freeze_bn() # 수정
                
                # Valid disparity map logging
                writer.add_image('Valid/Fused_disparity', visualize_flow_cmap(valid_fused_disparity), valid_num)
                writer.add_image('Valid/disparity1', visualize_flow_cmap(valid_disparity1), valid_num)
                writer.add_image('Valid/disparity2', visualize_flow_cmap(valid_disparity2), valid_num)
                
                writer.add_image('Valid/disparity1_rand', visualize_flow_cmap(disp_rand1), valid_num)
                writer.add_image('Valid/disparity2_rand', visualize_flow_cmap(disp_rand2), valid_num)
                
                writer.add_image('Valid/disparity1_mul', visualize_flow_cmap(mask_mul_list_val[0]), valid_num)
                writer.add_image('Valid/disparity2_mul', visualize_flow_cmap(mask_mul_list_val[1]), valid_num)
                
                writer.add_image('Valid/disparity_gt',visualize_flow_cmap(valid_disparity_gt), valid_num)
                writer.add_scalar('Valid_loss', valid_loss.item(), valid_num)
                writer.add_scalar('Valid_d1',valid_d1.item(), valid_num)
                
                # Simulated Captured image logging
                writer.add_image('Captured(V)/hdr_left', valid_origin_list[0][0], global_batch_num)
                writer.add_image('Captured(V)/img1_rand_left', valid_rand_img_list[0][0], global_batch_num)
                writer.add_image('Captured(V)/img2_rand_left', valid_rand_img_list[1][0], global_batch_num)
                writer.add_image('Captured(V)/img1_adj_left', valid_captured_img_list[0][0], global_batch_num)
                writer.add_image('Captured(V)/img2_adj_left', valid_captured_img_list[1][0], global_batch_num)
                
                # writer.add_image('Histogram(V)/hdr_dynamic_range', visualize_dynamic_range(valid_origin_list[0]), global_batch_num)
                # writer.add_image('Histogram(V)/img1_dynamic_range(rand)', visualize_dynamic_range(valid_rand_img_list[0], HDR=False), global_batch_num)
                
                # writer.add_image('Histogram(V)/img1_dynamic_range(adj)', visualize_dynamic_range(valid_captured_img_list[0],HDR=False), global_batch_num)
                # writer.add_image('Histogram(V)/img2_dynamic_range(adj)', visualize_dynamic_range(valid_captured_img_list[1],HDR=False), global_batch_num)
                
            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break
            
            #^ save intermediate checkpoint file to display
            if total_steps%100 == 0:
                save_path = Path('checkpoints/%s_%d_epoch.pth' % (args.name, total_steps))
                logging.info(f"Saving intermediate file {save_path}")
                torch.save(model.state_dict(), save_path)

        if len(train_loader) >= 10000:
            save_path = Path('checkpoints/%d_epoch_%s.pth.gz' % (total_steps + 1, args.name))
            logging.info(f"Saving file {save_path}")
            torch.save(model.state_dict(), save_path)

        print("FINISHED TRAINING")
        PATH = 'checkpoints/%s.pth' % args.name
        torch.save(model.state_dict(), PATH)

    return PATH
            


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