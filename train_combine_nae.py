import argparse
import logging
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.optim as optim
from torch.utils.data import DataLoader
from core.raft_stereo import RAFTStereo

# from core.combine_model3 import CombineModel_wo_net
from core.combine_model_nae import CombineModel_w_nae
from core.stereo_datasets3 import fetch_dataloader, CARLASequenceDataset
import matplotlib.pyplot as plt

from PIL import Image
from evaluate_stereo import *
from core.loss import gradient_loss, smoothness_loss
from core.utils.display import *

# ###############################################
# # * Training code for overall pipeline
# # * Considering sequence
# # * Model code : combine_model3.py (NAE module + Disparity recon)
# # ! Finetuning with nae module
# ###############################################

# Initialize writer for tensorboard logging
writer = SummaryWriter('runs/combine_pipeline_carla_nae')

# CUDA
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_random_exposure(batch_size, min_gap=0.5, max_gap=3.0, base_value=None, valid_mode=False):
    """
    노출값을 두 개 생성하는 함수.
    min_gap과 max_gap을 이용해 두 노출값의 차이를 제한.
    base_value가 None인 경우 0.8 ~ 2.0 사이의 랜덤 값을 사용.
    """
    exp_list1 = []
    exp_list2 = []
    
    # base_value가 None이면 0.8 ~ 2.5 범위의 랜덤 값으로 설정
    if base_value is None:
        base_value = random.uniform(0.8, 2.5)
    
    if valid_mode:
        for _ in range(batch_size):
            exp1 = base_value
            exp2 = base_value
            exp_list1.append([exp1])
            exp_list2.append([exp2])
    else:
        for _ in range(batch_size):
            exp1 = random.uniform(2**(-base_value), 2**(base_value))  # 첫 번째 노출값 랜덤 생성
            exp2 = exp1 * random.uniform(min_gap, max_gap)  # 첫 번째 노출값을 기준으로 두 번째 노출값 생성
            
            # 두 번째 노출값이 너무 크거나 작지 않도록 클리핑
            exp2 = max(2**(-2.0), min(exp2, 2**(2.0)))  # 클리핑 범위를 약간 넓게 조정
            
            exp_list1.append([exp1])
            exp_list2.append([exp2])
    
    return torch.tensor(exp_list1), torch.tensor(exp_list2)


try:
    from torch.cuda.amp import GradScaler
except:
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

# Existed RAFT-stereo disparity loss
def sequence_loss(flow_preds, flow_gt, valid, loss_gamma=0.9, max_flow=700):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    mag = torch.sum(flow_gt**2, dim=1).sqrt()

    valid = ((valid >= 0.5) & (mag < max_flow)).unsqueeze(1)
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    for i in range(n_predictions):
        assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
        flow_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

def l2_loss(pred_disparity, gt_disparity, valid_mask=None):
    """ Compute simple L2 loss between predicted disparity and ground truth disparity. """

    # Calculate squared difference between prediction and ground truth
    l2_loss = (pred_disparity - gt_disparity) ** 2

    # If valid mask is provided, apply it
    if valid_mask is not None:
        valid_mask = valid_mask.unsqueeze(1)
        l2_loss = l2_loss[valid_mask.bool()]

    # Take the mean of the valid loss values
    loss = l2_loss.mean()

    return loss

def smooth_l1_loss(pred_disparity, gt_disparity, valid_mask=None, beta=1.0):
    """ Compute Smooth L1 loss between predicted disparity and ground truth disparity. 
    Args:
        pred_disparity: predicted disparity map (tensor)
        gt_disparity: ground truth disparity map (tensor)
        valid_mask: optional mask to apply (tensor)
        beta: defines the threshold between L1 and L2 loss behavior (default=1.0)
    """
    
    # Calculate the absolute difference
    abs_diff = torch.abs(pred_disparity - gt_disparity)
    
    # Smooth L1 Loss: use L2 for small differences and L1 for large differences
    smooth_l1 = torch.where(abs_diff < beta, 
                            0.5 * (abs_diff ** 2) / beta,  # L2 region
                            abs_diff - 0.5 * beta)  # L1 region
    
    # If valid mask is provided, apply it
    if valid_mask is not None:
        valid_mask = valid_mask.unsqueeze(1)
        smooth_l1 = smooth_l1[valid_mask.bool()]
    
    # Take the mean of the valid loss values
    loss = smooth_l1.mean()

    return loss

# def dual_exposure_fusion_loss(pred_disparity_fused, pred_disparity_exp1, pred_disparity_exp2, gt_disparity, valid_mask=None):
#     """ Compute Dual-exposure Fusion Loss based on L2 losses for fused, exp1, and exp2 disparity maps. """
    
#     # Compute L2 loss for fused disparity map
#     loss_fused = l2_loss(pred_disparity_fused, gt_disparity, valid_mask)
    
#     # Compute L2 loss for exp1 disparity map
#     loss_exp1 = l2_loss(pred_disparity_exp1, gt_disparity, valid_mask)
    
#     # Compute L2 loss for exp2 disparity map
#     loss_exp2 = l2_loss(pred_disparity_exp2, gt_disparity, valid_mask)
    
#     # Calculate the dual-exposure fusion loss
#     fusion_loss = torch.max(torch.tensor(0.0).to(loss_fused.device), loss_fused - torch.min(loss_exp1, loss_exp2))
    
#     return fusion_loss

# def dual_exposure_fusion_sequenceloss(pred_disparity_fused, pred_disparity_exp1, pred_disparity_exp2, gt_disparity, valid_mask=None):
#     """ Compute Dual-exposure Fusion Loss based on smooth l1 loss for fused, exp1, and exp2 disparity maps. """
    
#     # Compute smooth l1 loss for fused disparity map
#     loss_fused, _  = sequence_loss(pred_disparity_fused, gt_disparity, valid_mask)
    
#     # Compute smooth l1 loss for exp1 disparity map
#     loss_exp1, _ = sequence_loss(pred_disparity_exp1, gt_disparity, valid_mask)
    
#     # Compute smooth l1 loss for exp2 disparity map
#     loss_exp2, _ = sequence_loss(pred_disparity_exp2, gt_disparity, valid_mask)
    
#     # Calculate the dual-exposure fusion loss
#     fusion_loss = torch.max(torch.tensor(0.0).to(loss_fused.device), loss_fused - torch.min(loss_exp1, loss_exp2))
    
#     return fusion_loss, loss_exp1, loss_exp2

def fetch_optimizer(args, model):
    all_params = model.parameters()
    
    optimizer = optim.AdamW(all_params, lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

#* Training code
def train(args):
    model = torch.nn.DataParallel(CombineModel_w_nae(args), device_ids=[0,1])
    print("Parameter Count : %d" % count_parameters(model))

    #^ Load dataloader, optimizer
    train_loader = fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0
    
    # Unfreeze only gru network + nae
    for name, param in model.module.named_parameters():
        if "update_block.gru" in name:
            param.requires_grad = True
        
        elif "nae" in name:
            param.requires_grad = True
        
        else:
            param.requires_grad = False
    
    # For debugging, Check module freezed
    for name, param in model.module.named_parameters():
        print(f"{name}: requires_grad = {param.requires_grad}")
    
    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info(f"Loading checkpoint... : {args.restore_ckpt}")
        checkpoint = torch.load(args.restore_ckpt)
        
        # eliminate 'module.' prefix 
        new_checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        
        # Debugging checkpoint keys
        # disp_recon_net_keys = model.module.disp_recon_net.state_dict().keys()
        # logging.info(f"disp_recon_net keys: {list(disp_recon_net_keys)}")
        # common_keys = set(new_checkpoint.keys()).intersection(set(disp_recon_net_keys))
        # logging.info(f"Common keys between checkpoint and disp_recon_net: {list(common_keys)}")
        
        # Checkpoint load to disp_recon_net
        model.module.raft_stereo.load_state_dict(new_checkpoint, strict=False)
        logging.info(f"Done loading checkpoint")

    model.to(DEVICE)
    model.train()
    
    training_frequency = 20
    validation_frequency = 100
    
    scaler = GradScaler(enabled=args.mixed_precision)

    should_keep_training = True
    global_batch_num = 0

    #^ Training
    while should_keep_training:

        for i_batch, (_, *data_blob) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
                   
            left_hdr, right_hdr, _, _, disp, valid = [x.cuda() for x in data_blob]
            
            #* Generate random expsoures considering the batch size
            initial_exp_high, initial_exp_low = generate_random_exposure(args.batch_size)
            initial_random_exp = (initial_exp_high + initial_exp_low)/2
            
            assert model.training
            # Model forward pass
            fused_disparity, exp_alpha, left_hdr, captured_rand_img_list, captured_adj_img_list= model(
                left_hdr, right_hdr, initial_random_exp)
            assert model.training

            # disparity_loss= l2_loss(fused_disparity[-1], disp, valid)
            disparity_loss, _ = sequence_loss(fused_disparity, disp, valid)
 
            
            global_batch_num += 1
            # * Graident update
            scaler.scale(disparity_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            if total_steps % training_frequency == training_frequency - 1:
                #* Tensorboard logging
                vmin, vmax = -disp.max(), -disp.min()
                # print(f"=====Initial Exposure value F1 : {initial_exp_high}, F2 : {initial_exp_low}=====")
                # print(f"=====Adjsuted Exposure value F1 : {exp1}, F2 : {exp2}=====")
                # print(f"=========Initial_random_disparity_loss : {initial_disparity_loss.item()}, training_frequency : {total_steps}//{args.num_steps}")
                # print(f"=========Rule_disparity_loss : {initial_disparity_loss.item()}, training_frequency : {total_steps}//{args.num_steps}")
                print(f"=========NAE_disparity_loss : {disparity_loss.item()}, training_frequency : {total_steps}//{args.num_steps}")
                print(f"=========Initial_exposure : {initial_random_exp}, training_frequency : {total_steps}//{args.num_steps}")
                print(f"=========NAE_exposure_output : {exp_alpha}, training_frequency : {total_steps}//{args.num_steps}")
                
                # total_loss, disparity map
                # writer.add_scalar("Initial_disparity_loss", initial_disparity_loss.item(), global_batch_num)
                # writer.add_scalar("Rule_disparity_loss", rule_disparity_loss.item(), global_batch_num)
                writer.add_scalar("Total loss", disparity_loss.item(), global_batch_num)

                writer.add_image('Train/Fused_disparity', visualize_disparity_with_colorbar(fused_disparity[-1], vmin, vmax), global_batch_num)

                # writer.add_image('Train/rule_disparity', visualize_disparity_with_colorbar(rule_disparity[-1], vmin, vmax), global_batch_num)
                # writer.add_image('Train/exp1_disparity', visualize_disparity_with_colorbar(disp_exp1[-1], vmin, vmax), global_batch_num)
                # writer.add_image('Train/exp2_disparity', visualize_disparity_with_colorbar(disp_exp2[-1], vmin, vmax), global_batch_num)
                writer.add_image('Train/disparity_GT', visualize_disparity_with_colorbar(disp, vmin, vmax), global_batch_num)

                # Visualize captured image
                # writer.add_image('Captured(T)/hdr_left_frame1', original_img_list[0][0], global_batch_num)
                # writer.add_image('Captured(T)/hdr_left_frame2', original_img_list[1][0], global_batch_num)
                writer.add_image('Captured(T)/img1_rand_left', captured_rand_img_list[0][0], global_batch_num)
                writer.add_image('Captured(T)/img1_adj_left', captured_adj_img_list[0][0], global_batch_num)
                
                # log_multiple_feature_map_with_colorbar(writer, fmap_list[0], "Train/Unwarped_Fmap1_L", global_batch_num, num_channels=1)
                # log_multiple_feature_map_with_colorbar(writer, fmap_list[1], "Train/Unwarped_Fmap2_L", global_batch_num, num_channels=1)
                # log_multiple_feature_map_with_colorbar(writer, fmap_list[2], "Train/Fused_Fmap", global_batch_num, num_channels=1)
                
            global_batch_num += 1
            
            # # Todo) Modify validation code for overall pipeline
            # #^ Validation code
            # if total_steps % validation_frequency == validation_frequency - 1:
            #     print("=====Validation=====")
            #     valid_num = (total_steps / validation_frequency) * 10 + 1
            #     valid_loss, valid_fused_disparity  = validate_carla_nae(model, iters=16)

            #     model.train()
            #     # model.module.raft_stereo.freeze_bn()

            #     writer.add_scalar('Valid_loss', valid_loss.item(), valid_num)
            #     writer.add_image('Valid_Fused_disparity', visualize_disparity_with_colorbar(valid_fused_disparity[-1], vmin, vmax), valid_num)

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
    parser.add_argument('--name', default='combine_model_nae', help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4, help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['carla'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.0001, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=10000, help="length of training schedule.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[800, 600], help="size of the random image crops used during training.")
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
    
    # # Data augmentation
    # parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    # parser.add_argument('--saturation_range', type=float, nargs='+', default=None, help='color saturation')
    # parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
    # parser.add_argument('--spatial_scale', type=float, nargs='+', default=[0, 0], help='re-scale the images randomly')
    # parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')

    args = parser.parse_args()
    
    torch.manual_seed(1234)
    np.random.seed(1234)
    
    train(args)



