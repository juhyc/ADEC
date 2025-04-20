from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import time
import logging
import numpy as np
import torch
from tqdm import tqdm
from core.raft_stereo import RAFTStereo, autocast
import core.stereo_datasets as datasets
from core.utils.utils import InputPadder
from core.utils.simulate import *
from core.loss import *
from core.utils.display import *
from torch.utils.tensorboard import SummaryWriter

import core.stereo_datasets as datasets

###############################################
# * For validate pipeline
###############################################

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def validate_eth3d(model, iters=32, mixed_prec=False):
    """ Peform validation using the ETH3D (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.ETH3D(aug_params)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr.float()).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = valid_gt.flatten() >= 0.5
        out = (epe_flattened > 1.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(f"ETH3D {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print("Validation ETH3D: EPE %f, D1 %f" % (epe, d1))
    return {'eth3d-epe': epe, 'eth3d-d1': d1}

def sequence_loss_valid(flow_preds, flow_gt, valid, loss_gamma=0.9, max_flow=700):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()

    # exclude extremly large displacements
    valid = ((valid >= 0.5) & (mag < max_flow)).unsqueeze(1)
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    for i in range(n_predictions):
        assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
        flow_loss += i_weight * i_loss[valid.bool()].mean()
        
    return flow_loss

def disparity_consistancy_loss(disp_t1, disp_t2):
    return torch.mean(torch.abs(disp_t1 - disp_t2))



# ^ Raft sequence loss
def sequence_loss(flow_preds, flow_gt, valid, loss_gamma=0.9, max_flow=700):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()

    # exclude extremly large displacements
    valid = ((valid >= 0.5) & (mag < max_flow)).unsqueeze(1)
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    for i in range(n_predictions):
        assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
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

        
# * Validation Writer for logging
eval_disp_writer = SummaryWriter('runs/eval_disp_model')


# * Validation Writer for logging
# eval_disp_writer = SummaryWriter('runs/eval_disp_model_blur')
@torch.no_grad()
def validate_carla_blur(model, valid_cnt, valid_case, iters=16, mixed_prec=False):
    """Perform validation using the CARLA synthetic dataset"""
    exp1 = torch.tensor([1.5]).cuda()
    exp2 = torch.tensor([3.0]).cuda()
    
    # Freeze BatchNorm Layer
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.InstanceNorm2d):
            module.train()
    
    val_dataset = datasets.CARLASequenceDataset(image_set='test', valid_mode=True, valid_case=valid_case)
    torch.backends.cudnn.benchmark = True
    
    valid_loss_list = []
    consistancy_loss_list = []
    
    print(len(val_dataset))
    
    for val_id in range(len(val_dataset)): # 100 sequence
        
        _, image1, image2, image1_next, image2_next, flow_gt, valid_gt = val_dataset[val_id]
        
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        image1_next = image1_next[None].cuda()
        image2_next = image2_next[None].cuda()
        flow_gt = flow_gt[None].cuda()
        valid_gt = valid_gt[None].cuda()

        with autocast(enabled=mixed_prec):
            disp_predictions, fmap1, fmap1_next, fused_fmap1, flow_L, fmap_list, cap_img_list, _v = model(image1=image1, image2=image2, image1_next=image1_next, image2_next=image2_next, exp_h=exp1, exp_l=exp2)

        valid_loss = sequence_loss_valid(disp_predictions, flow_gt, valid_gt)
        valid_loss_list.append(valid_loss.cpu().item())
        
    #* Validation logging
    print(f"In validation exposure value : {exp1} {exp2}")
    
    eval_disp_writer.add_image(f'Valid{valid_case}/gt_disp', visualize_flow_cmap(flow_gt), valid_cnt)
    eval_disp_writer.add_image(f'Valid{valid_case}/refined_disparity', visualize_flow_cmap(disp_predictions[-1]), valid_cnt)

    eval_disp_writer.add_scalar(f'Valid{valid_case}/valid_loss', np.mean(valid_loss_list), valid_cnt)
    eval_disp_writer.add_scalar(f'Valid{valid_case}/consistancy_loss', np.mean(consistancy_loss_list), valid_cnt)
    
    eval_disp_writer.add_image(f'Valid{valid_case}/Left F1', image1[0]**(1/2.2), valid_cnt)
    eval_disp_writer.add_image(f'Valid{valid_case}/Left F2', image1_next[0]**(1/2.2), valid_cnt)
    eval_disp_writer.add_image(f'Valid{valid_case}/Left cap', cap_img_list[0][0]**(1/2.2), valid_cnt)
    eval_disp_writer.add_image(f'Valid{valid_case}/Left cap_next', cap_img_list[2][0]**(1/2.2), valid_cnt)
    # logging feature map
    visualize_flow(eval_disp_writer, flow_L, f'Valid{valid_case}/Flow', valid_cnt)
    log_multiple_feature_map_with_colorbar(eval_disp_writer, fmap1, f'Valid{valid_case}/Unwarped_Fmap1_L', valid_cnt, num_channels=1)
    log_multiple_feature_map_with_colorbar(eval_disp_writer, fmap1_next, f'Valid{valid_case}/Unwarped_Fmap2_L', valid_cnt, num_channels=1)
    log_multiple_feature_map_with_colorbar(eval_disp_writer, fused_fmap1, f'Valid{valid_case}/Warped_Fmap2', valid_cnt, num_channels=1)
    log_multiple_feature_map_with_colorbar(eval_disp_writer, fused_fmap1, f'Valid{valid_case}/Fused_Fmap', valid_cnt, num_channels=1)
        
    return np.mean(valid_loss_list)

# * Validation Writer for logging
@torch.no_grad()
def validate_carla_exp(model, valid_cnt, valid_case, exp1, exp2, iters=16, mixed_prec=False):
    """Perform validation using the CARLA synthetic dataset"""
        
    # Freeze BatchNorm Layer
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.InstanceNorm2d):
            module.train()
    
    val_dataset = datasets.CARLASequenceDataset(image_set='test', valid_mode=True, valid_case=valid_case)
    torch.backends.cudnn.benchmark = True
    
    valid_loss_list = []
    consistancy_loss_list = []
    
    print(len(val_dataset))
    
    for val_id in range(len(val_dataset)): # 100 sequence
        
        _, image1, image2, image1_next, image2_next, flow_gt, valid_gt = val_dataset[val_id]
        
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        image1_next = image1_next[None].cuda()
        image2_next = image2_next[None].cuda()
        flow_gt = flow_gt[None].cuda()
        valid_gt = valid_gt[None].cuda()
        
        with autocast(enabled=mixed_prec):
            disp_predictions, fmap1, fmap1_next, fused_fmap1, flow_L, fmap_list, cap_img_list, _, disp_predictions_rev = model(image1=image1, image2=image2, image1_next=image1_next, image2_next=image2_next, exp_h=exp1, exp_l=exp2)

        valid_loss = sequence_loss_valid(disp_predictions, flow_gt, valid_gt)
        consistancy_loss = disparity_consistancy_loss(disp_predictions[-1], disp_predictions_rev[-1])
        valid_loss_list.append(valid_loss.cpu().item())
        consistancy_loss_list.append(consistancy_loss.cpu().item())
        
    #* Validation logging
    print(f"In validation exposure value : {exp1} {exp2}")
    
    eval_disp_writer.add_image(f'Valid{exp1, exp2}/gt_disp', visualize_flow_cmap(flow_gt), valid_cnt)
    eval_disp_writer.add_image(f'Valid{exp1, exp2}/refined_disparity', visualize_flow_cmap(disp_predictions[-1]), valid_cnt)
    eval_disp_writer.add_image(f'Valid{exp1, exp2}/refined_disparity_rev', visualize_flow_cmap(disp_predictions_rev[-1]), valid_cnt)
    eval_disp_writer.add_scalar(f'Valid{exp1, exp2}/valid_loss', np.mean(valid_loss_list), valid_cnt)
    eval_disp_writer.add_scalar(f'Valid{exp1, exp2}/consistancy_loss', np.mean(consistancy_loss_list), valid_cnt)
    
    eval_disp_writer.add_image(f'Valid{exp1, exp2}/Left F1', image1[0]**(1/2.2), valid_cnt)
    eval_disp_writer.add_image(f'Valid{exp1, exp2}/Left F2', image1_next[0]**(1/2.2), valid_cnt)
    eval_disp_writer.add_image(f'Valid{exp1, exp2}/Left cap', cap_img_list[0][0]**(1/2.2), valid_cnt)
    eval_disp_writer.add_image(f'Valid{exp1, exp2}/Left cap_next', cap_img_list[2][0]**(1/2.2), valid_cnt)
    # logging feature map
    visualize_flow(eval_disp_writer, flow_L, f'Valid{exp1, exp2}/Flow', valid_cnt)
    log_multiple_feature_map_with_colorbar(eval_disp_writer, fmap1, f'Valid{exp1, exp2}/Unwarped_Fmap1_L', valid_cnt, num_channels=1)
    log_multiple_feature_map_with_colorbar(eval_disp_writer, fmap1_next, f'Valid{exp1, exp2}/Unwarped_Fmap2_L', valid_cnt, num_channels=1)
    log_multiple_feature_map_with_colorbar(eval_disp_writer, fused_fmap1, f'Valid{exp1, exp2}/Warped_Fmap2', valid_cnt, num_channels=1)
    log_multiple_feature_map_with_colorbar(eval_disp_writer, fused_fmap1, f'Valid{exp1, exp2}/Fused_Fmap', valid_cnt, num_channels=1)
        
    return np.mean(valid_loss_list)

