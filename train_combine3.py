import argparse
import logging
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from core.raft_stereo import RAFTStereo
from core.combine_model3 import CombineModel_wo_net

from evaluate_stereo import *
from core.stereo_datasets3 import fetch_dataloader, CARLASequenceDataset
import matplotlib.pyplot as plt
from PIL import Image

from core.loss import gradient_loss, smoothness_loss
from core.depth_datasets import DepthDataset_stereo
from core.utils.display import *

import torch.nn.utils as nn_utils

# from torchviz import make_dot

# ###############################################
# # * Training code without exposure control network
# # * Considering sequence
# # * Add flow estimate model and freeze
# ###############################################

# # Initialize writer for tensorboard logging
writer = SummaryWriter('runs/combine_pipeline_carla')

# CUDA
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def adjust_exposure(image, current_exposure, target_exposure):
    """
    Adjusts the exposure of an image based on the current and target exposure values.
    """
    device = image.device()
    exposure_factor = target_exposure / current_exposure
    adjusted_image = image * exposure_factor
    return adjusted_image.to(device)

def train(args):
    model = torch.nn.DataParallel(CombineModel_wo_net(args))
    print("Parameter Count : %d" % count_parameters(model))

    #^ Load dataloader
    train_loader = fetch_dataloader(args)
    # criterion = nn.L1Loss().to(DEVICE)
    criterion = nn.SmoothL1Loss().to(DEVICE)
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    optimizer = optim.AdamW(model.module.alpha_net.parameters(), lr=args.lr, weight_decay=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)
    total_steps = 0
    

    #^ Load RAFT module checkpoint
    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        raft_checkpoint = torch.load(args.restore_ckpt)
        
        # Downsampling = 3 case
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

    model.to(DEVICE)
    model.train()
    
    
    for param in model.module.alpha_net.parameters():
        param.requires_grad = True
    # Freeze RAFTStereo module
    for param in model.module.RAFTStereo.parameters():
        param.requires_grad = False
    # Freeze Flowmodel
    for param in model.module.flow.parameters():
        param.requires_grad = False

    model.module.RAFTStereo.freeze_bn()

    validation_frequency = 50

    should_keep_training = True
    global_batch_num = 0

    #^ Training
    while should_keep_training:

        for i_batch, (_, *data_blob) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
                   
            left_hdr, right_hdr, left_next_hdr, right_next_hdr, disparity, valid = [x.cuda() for x in data_blob]
            
            print(left_hdr.shape)
            
            #* Generate random expsoures considering the batch size
            initial_exp_high = generate_random_exposures(left_hdr.shape[0])
            initial_exp_low = generate_random_exposures(left_hdr.shape[0])
            
            # print(f"initial_exp_high_batch : {initial_exp_high}")
            # print(f"initial_exp_low_batch : {initial_exp_low}")
            
            assert model.training

            # Model forward pass
            fused_disparity, disparity1, disparity2, original_img_list, captured_rand_img_list, captured_adj_img_list, warped_img_list, mask_mul_list, disparity_list, shifted_exp = model(
                left_hdr, right_hdr, left_next_hdr, right_next_hdr, initial_exp_high, initial_exp_low, train_mode=True)

            shifted_exp_high = shifted_exp[0]
            shifted_exp_low = shifted_exp[1]
            # print(f"shifted_exp_high_batch : {shifted_exp_high}")
            # print(f"shifted_exp_low_batch :{shifted_exp_low}")

            #^ Visualize during training
            writer.add_image('Train/Fused_disparity', visualize_flow_cmap(fused_disparity), global_batch_num)
            writer.add_image('Train/disparity_frame1', visualize_flow_cmap(disparity1), global_batch_num)
            writer.add_image('Train/disparity_frame2_warped', visualize_flow_cmap(disparity2), global_batch_num)
            writer.add_image('Train/disparity_GT', visualize_flow_cmap(disparity), global_batch_num)
            
            writer.add_image('Train/disparity_cap_frame1', visualize_flow_cmap(disparity_list[0]), global_batch_num)
            writer.add_image('Train/disparity_cap_frame2', visualize_flow_cmap(disparity_list[1]), global_batch_num)

            # Add mask multiplication result
            writer.add_image('Train/disparity1_mul', visualize_flow_cmap(mask_mul_list[0]), global_batch_num)
            writer.add_image('Train/disparity2_mul', visualize_flow_cmap(mask_mul_list[1]), global_batch_num)

            # Visualize captured image
            writer.add_image('Captured(T)/hdr_left_frame1', original_img_list[0][0], global_batch_num)
            writer.add_image('Captured(T)/hdr_left_frame2', original_img_list[1][0], global_batch_num)
            writer.add_image('Captured(T)/img1_rand_left', captured_rand_img_list[0][0], global_batch_num)
            writer.add_image('Captured(T)/img2_rand_left', captured_rand_img_list[1][0], global_batch_num)
            
            # # Normalized image
            # target_exp = (initial_exp_high[0] + initial_exp_low[0])/2
            # print(f"target_exp : {target_exp}")
            # exposure_factor1 = (target_exp / initial_exp_high[0]).to(DEVICE)
            # exposure_factor2 = (target_exp / initial_exp_low[0]).to(DEVICE)
            # adjusted_img1 = captured_rand_img_list[0][0]*exposure_factor1
            # adjusted_img2 = captured_rand_img_list[1][0]*exposure_factor2
            # writer.add_image('Captured(T)/img1_rand_normalize', adjusted_img1, global_batch_num)
            # writer.add_image('Captured(T)/img2_rand_normalize', adjusted_img2, global_batch_num)
            
            # Visualize warped image
            writer.add_image('Captured(T)/warped_left', warped_img_list[0][0], global_batch_num)
            writer.add_image('Captured(T)/warped_right', warped_img_list[1][0], global_batch_num)

            # Visualize dynamic range
            # writer.add_image('Captured(T)/hdr_dynamic_range', visualize_dynamic_range(original_img_list[0]), global_batch_num)
            # writer.add_image('Histogram(T)/img1_dynamic_range(rand)', visualize_dynamic_range(captured_rand_img_list[0], HDR=False), global_batch_num)
            # writer.add_image('Histogram(T)/img2_dynamic_range(rand)', visualize_dynamic_range(captured_rand_img_list[1], HDR=False), global_batch_num)

            # Visualize adjusted image
            writer.add_image('Captured(T)/img1_adj_left', captured_adj_img_list[0][0], global_batch_num)
            writer.add_image('Captured(T)/img2_adj_left', captured_adj_img_list[1][0], global_batch_num)

            # writer.add_image('Captured(T)/img1_adj_mask', visualize_mask(mask_list[0]), global_batch_num)
            # writer.add_image('Captured(T)/img2_adj_mask', visualize_mask(mask_list[1]), global_batch_num)

            # writer.add_image('Histogram(T)/img1_dynamic_range(adj)', visualize_dynamic_range(captured_adj_img_list[0], HDR=False), global_batch_num)
            # writer.add_image('Histogram(T)/img2_dynamic_range(adj)', visualize_dynamic_range(captured_adj_img_list[1], HDR=False), global_batch_num)

            assert model.training

            #^ Exist valid mask calculation
            valid_mask = (valid >= 0.5)
            valid_mask = valid_mask.unsqueeze(1)

            # Mask visualize to check loss calculation
            disparity_loss = criterion(fused_disparity[valid_mask], disparity[valid_mask])
            # grad_loss = gradient_loss(fused_disparity, disparity)
            # smooth_loss = smoothness_loss(fused_disparity, left_hdr)

            # total_loss = disparity_loss + 0.1 * grad_loss + 0.1 * smooth_loss
            total_loss = disparity_loss

            writer.add_scalar("Training_total_loss", total_loss.item(), global_batch_num)
            global_batch_num += 1
            #^ Loss calculation
            total_loss.backward()
            # * gradient clipping
            nn_utils.clip_grad_norm_(model.module.alpha_net.parameters(), max_norm = 5.0)
            
            #! Check gradient
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    print(f"{name} gradient norm: {param.grad.norm().item()}")
            
            optimizer.step()
            scheduler.step()

            #^ Validation code
            if total_steps % validation_frequency == validation_frequency - 1:
                print("=====Validation=====")
                valid_num = (total_steps / validation_frequency) * 10 + 1
                valid_loss, valid_d1, valid_fused_disparity  = validate_carla(model.module, iters=32)

                model.train()
                model.module.RAFTStereo.freeze_bn()

                writer.add_scalar('Valid_loss', valid_loss.item(), valid_num)
                writer.add_scalar('Valid_d1', valid_d1.item(), valid_num)

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break
      
            #^ Save intermediate checkpoint file to display
            if total_steps % 100 == 0:
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
    parser.add_argument('--batch_size', type=int, default=4, help="batch size used during training.")
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



