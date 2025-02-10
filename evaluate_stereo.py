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
import core.stereo_datasets2 as datasets2
import core.stereo_datasets3 as datasets3

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


@torch.no_grad()
def validate_kitti(model, iters=32, mixed_prec=False):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.KITTI(aug_params, image_set='validate')
    torch.backends.cudnn.benchmark = True
    
    out_list, epe_list, elapsed_list = [], [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()


        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        # ! 수정, validation visualtion, model이 raft가 아닌 Combine model 형태로 반환값이 다름.
        with autocast(enabled=mixed_prec):
            start = time.time()
            # 수정
            flow_pr, left_ldr_cap, right_ldr_cap, left_ldr_adj_denom, right_ldr_adj_denom, exposure_dict, left_ldr_adj, right_ldr_adj = model(image1, image2, iters=iters, test_mode=True)
            end = time.time()
        
        # 수정
        flow_pr = flow_pr[-1]
        
        if val_id > 50:
            elapsed_list.append(end-start)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        # flow_pr= flow_pr.cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = valid_gt.flatten() >= 0.5

        out = (epe_flattened > 3.0)
        image_out = out[val].float().mean().item()
        
        image_epe = epe_flattened[val].mean().item()
        if val_id < 9 or (val_id+1)%10 == 0:
            logging.info(f"KITTI Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}. Runtime: {format(end-start, '.3f')}s ({format(1/(end-start), '.2f')}-FPS)")
        epe_list.append(epe_flattened[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    avg_runtime = np.mean(elapsed_list)

    print(f"Validation KITTI: EPE {epe}, D1 {d1}, {format(1/avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")
    return {'kitti-epe-Valid': epe, 'kitti-d1-Valid': d1}, -flow_pr, -flow_gt, left_ldr_cap, right_ldr_cap, left_ldr_adj_denom, right_ldr_adj_denom, exposure_dict, left_ldr_adj, right_ldr_adj

@torch.no_grad()
def validate_kitti2(model, iters=32, mixed_prec=False):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.KITTI(aug_params, image_set='validate')
    torch.backends.cudnn.benchmark = True
    criterion = nn.SmoothL1Loss().cuda()
    # criterion = BerHuLoss().cuda()
    
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()


        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        # ! 수정, validation visualtion, model이 raft가 아닌 Combine model 형태로 반환값이 다름.
        with autocast(enabled=mixed_prec):
            start = time.time()
            # 수정
            fused_disparity, disparity1, disparity2, captured_rand_img_list, captured_img_list = model(image1, image2, iters = iters)
            end = time.time()
        
        fused_disparity = padder.unpad(fused_disparity).cpu().squeeze(0)
        valid_mask = (valid_gt >=0.5)
        valid_mask = valid_mask.unsqueeze(0)  

        loss = criterion(fused_disparity[valid_mask], flow_gt[valid_mask])
        
        print(f"Validation loss : {loss}")
          
    return loss, fused_disparity, disparity1, disparity2, captured_rand_img_list, captured_img_list

# @torch.no_grad()
# def validate_carla(model, iters=32, mixed_prec=False):
#     """Perform validation using the CARLA synthetic dataset"""
#     model.eval()
#     val_dataset = datasets2.CARLA(image_set='validate')
#     torch.backends.cudnn.benchmark = True
#     # criterion = nn.MSELoss().cuda()
#     criterion = nn.L1Loss().cuda()
    
#     out_list, epe_list, loss_list = [], [], []
#     for val_id in range(len(val_dataset)):
#         _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
#         image1 = image1[None].cuda()
#         image2 = image2[None].cuda()
        
#         padder = InputPadder(image1.shape, divis_by = 32)
#         image1, image2 = padder.pad(image1, image2)
        
#         with autocast(enabled=mixed_prec):
#             start = time.time()
#             fused_disparity, disparity1, disparity2, origin, captured_rand_img_list, captured_img_list, disp_rand1, disp_rand2, mask_list, mask_mul_list = model(image1, image2, iters=iters, valid_mode=True)
#             end = time.time()
            
#         fused_disparity= padder.unpad(fused_disparity).cpu().squeeze(0)
#         valid_mask = (valid_gt >= 0.5)
#         valid_mask = valid_mask.unsqueeze(0)
        
#         # Validation loss
#         loss = criterion(fused_disparity[valid_mask], flow_gt[valid_mask])
#         loss_list.append(loss.item())
        
#         # Calculate epe(End-point-error)
#         epe = torch.sum((fused_disparity - flow_gt)**2, dim=0).sqrt()
#         epe_flattened = epe.flatten()
#         val = valid_gt.flatten() >= 0.5
        
#         # pixel threshold 3.0 pixel
#         out = (epe_flattened > 3.0)
        
#         epe_list.append(epe_flattened[val].mean().item())
#         out_list.append(out[val].cpu().numpy())
                
#         # print(f"====Validation loss : {loss}====")
    
#     epe = np.mean(epe_list)
#     d1 = 100 * np.mean(out_list)
#     loss_mean = np.mean(loss_list)
    
#     print(f"Validation CARLA : Loss {loss_mean}, D1 {d1}, EPE {epe}")
    
#     return loss_mean, d1, fused_disparity, disparity1, disparity2, origin, captured_rand_img_list, captured_img_list, flow_gt, disp_rand1, disp_rand2, mask_list, mask_mul_list

# eval_writer = SummaryWriter('runs/evaluation_longsequence')

@torch.no_grad()
def validate_carla(model, iters=32, mixed_prec=False):
    """Perform validation using the CARLA synthetic dataset"""
    model.eval()
    val_dataset = datasets3.CARLASequenceDataset(image_set='validate')
    torch.backends.cudnn.benchmark = True
    criterion = nn.SmoothL1Loss().cuda()
    
    out_list, epe_list, loss_list = [], [], []
    # For save image
    num_cnt = 0
    
    for val_id in range(len(val_dataset)): # 100 sequence
        
        _, image1, image2, image1_next, image2_next, flow_gt, valid_gt = val_dataset[val_id]
        
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        image1_next = image1_next[None].cuda()
        image2_next = image2_next[None].cuda()
        flow_gt = flow_gt.cuda()
        valid_gt = valid_gt.cuda()
        
        if val_id == 0:
            initial_exp1 = generate_random_exposures(image1.shape[0], valid_mode=True).cuda()
            initial_exp2 = generate_random_exposures(image1.shape[0], valid_mode=True).cuda()
        
        padder = InputPadder(image1.shape, divis_by = 32)
        image1, image2 = padder.pad(image1, image2)
        image1_next, image2_next = padder.pad(image1_next, image2_next)
        
        # Todo) model validation mode일 때로 수정
        with autocast(enabled=mixed_prec):
            start = time.time()
            fused_disparity, disp1, disp2, cap_adj_img_list, shifted_exp = model(image1, image2, image1_next, image2_next, initial_exp1, initial_exp2, iters=iters, valid_mode=True)
            end = time.time()
        
        # #* Validation logging
        # eval_writer.add_image('gt_disp', visualize_flow_cmap(flow_gt), val_id)
        # eval_writer.add_image('disp1', visualize_flow_cmap(disp1), val_id)
        # eval_writer.add_image('disp2', visualize_flow_cmap(disp2), val_id)
        # eval_writer.add_image('Fused_disparity', visualize_flow_cmap(fused_disparity), val_id)
        # eval_writer.add_image('f1_adj_left', cap_adj_img_list[0][0], val_id)
        # eval_writer.add_image('f2_adj_left', cap_adj_img_list[1][0], val_id)
        
        save_image(visualize_flow_cmap(fused_disparity), f'Demo/cap_fused_disp03/fused_disp_{num_cnt}.png')
        save_image(visualize_flow_cmap(disp1[0]), f'Demo/cap_disp1/disp1_{num_cnt}.png')
        save_image(visualize_flow_cmap(disp2[0]), f'Demo/cap_disp2/disp2_{num_cnt}.png')
        save_image(visualize_flow_cmap(flow_gt), f'Demo/cap_disp_gt/gt_{num_cnt}.png')
        
        # Save captured imag
        save_image_255(cap_adj_img_list[0][0], f'Demo/cap_f1/cap_f1_{num_cnt}.png')
        save_image_255(cap_adj_img_list[0][0], f'Demo/cap_f1_f2/cap_{num_cnt}.png')
        num_cnt += 1
        save_image_255(cap_adj_img_list[1][0], f'Demo/cap_f1_f2/cap_{num_cnt}.png')
        save_image_255(cap_adj_img_list[1][0], f'Demo/cap_f2/cap_f2_{num_cnt}.png')
        num_cnt += 1
        
        fused_disparity= padder.unpad(fused_disparity).cpu().squeeze(0)
        valid_mask = (valid_gt >= 0.5)
        valid_mask = valid_mask.unsqueeze(0)
        
        # Update exposure value
        initial_exp1 = shifted_exp[0]
        initial_exp2 = shifted_exp[1]
        
        # Validation loss
        loss = criterion(fused_disparity[valid_mask], flow_gt[valid_mask])
        loss_list.append(loss.item())
        
        # Calculate epe(End-point-error)
        epe = torch.sum((fused_disparity - flow_gt)**2, dim=0).sqrt()
        epe_flattened = epe.flatten()
        val = valid_gt.flatten() >= 0.5
        
        # pixel threshold 3.0 pixel
        out = (epe_flattened > 3.0)
        
        epe_list.append(epe_flattened[val].mean().item())
        out_list.append(out[val].cpu().numpy())
        
        print(f"Validation images : [{val_id}/{len(val_dataset)}]")
        print(f"Shifted exp : [even_frame : {shifted_exp[0]}, odd_frame : {shifted_exp[1]}]")
        # print(f"====Validation loss : {loss}====")
    
    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)
    loss_mean = np.mean(loss_list)
    
    print(f"Validation CARLA : Loss {loss_mean}, D1 {d1}, EPE {epe}")
    
    return loss_mean, d1, fused_disparity

@torch.no_grad()
def validate_carla_nae(model, iters=16, mixed_prec=False):
    """Perform validation using the CARLA synthetic dataset"""
    val_dataset = datasets3.CARLASequenceDataset(image_set='test')
    torch.backends.cudnn.benchmark = True

    loss_list = []
    # For save image
    num_cnt = 0
    
    # Freeze BatchNorm Layer
    model.eval()
    # for module in model.modules():
    #     if isinstance(module, torch.nn.BatchNorm2d):
    #         module.train()

    _, image1, image2, _ , _, flow_gt, valid_gt = val_dataset[0]

    initial_exp = generate_random_exposures(image1.shape[0], valid_mode=True, value=1.0).cuda()
    
    image1 = image1[None].cuda()
    image2 = image2[None].cuda()
    flow_gt = flow_gt[None].cuda()
    valid_gt = valid_gt[None].cuda()

    with autocast(enabled=mixed_prec):
        flow_predictions, alpha, hdr_left, intial_left, adjusted_left = model(image1, image2, initial_exp) 

    loss, _ = sequence_loss(flow_predictions, flow_gt, valid_gt)

    print(f"Validation loss : {loss}")
    
    return loss, flow_predictions[-1]

# * Validation Writer for logging
eval_disp_writer = SummaryWriter('runs/eval_disp_model')
@torch.no_grad()
def validate_carla_warp(model, valid_cnt, iters=16, mixed_prec=False):
    """Perform validation using the CARLA synthetic dataset"""
    exp1 = torch.tensor([1.5]).cuda()
    exp2 = torch.tensor([4.0]).cuda()
    
    # Freeze BatchNorm Layer
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.train()
    
    val_dataset = datasets3.CARLASequenceDataset(image_set='test')
    torch.backends.cudnn.benchmark = True
    
    valid_loss_list = []
    
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
            disp_predictions, fmap1, fmap1_next, fused_fmap1, flow_L, fmap_list, cap_img_list, _ = model(image1=image1, image2=image2, image1_next=image1_next, image2_next=image2_next, exp_h=exp1, exp_l=exp2)

        valid_loss = sequence_loss_valid(disp_predictions, flow_gt, valid_gt)
        valid_loss_list.append(valid_loss.cpu().item())
        
    #* Validation logging
    print(f"In validation exposure value : {exp1} {exp2}")
    
    eval_disp_writer.add_image('Valid/gt_disp', visualize_flow_cmap(flow_gt), valid_cnt)
    eval_disp_writer.add_image('Valid/refined_disparity', visualize_flow_cmap(disp_predictions[-1]), valid_cnt)
    eval_disp_writer.add_scalar('Valid/valid_loss', np.mean(valid_loss_list), valid_cnt)
    
    eval_disp_writer.add_image('Valid/Left F1', image1[0]**(1/2.2), valid_cnt)
    eval_disp_writer.add_image('Valid/Left F2', image1_next[0]**(1/2.2), valid_cnt)
    eval_disp_writer.add_image('Valid/Left cap', cap_img_list[0][0]**(1/2.2), valid_cnt)
    eval_disp_writer.add_image('Valid/Left cap_next', cap_img_list[2][0]**(1/2.2), valid_cnt)
    # logging feature map
    visualize_flow(eval_disp_writer, flow_L, 'Valid/Flow', valid_cnt)
    log_multiple_feature_map_with_colorbar(eval_disp_writer, fmap1, 'Valid/Unwarped_Fmap1_L', valid_cnt, num_channels=1)
    log_multiple_feature_map_with_colorbar(eval_disp_writer, fmap1_next, 'Valid/Unwarped_Fmap2_L', valid_cnt, num_channels=1)
    log_multiple_feature_map_with_colorbar(eval_disp_writer, fused_fmap1, 'Valid/Warped_Fmap2', valid_cnt, num_channels=1)
    log_multiple_feature_map_with_colorbar(eval_disp_writer, fused_fmap1, 'Valid/Fused_Fmap', valid_cnt, num_channels=1)
    
    # Feature map consine similarity
    # print("@@@@@@@@@@@@Validation@@@@@@@@@@@@@@@")
    # print(f"Consine similarity between first left and second left frame : {normalized_cosine_similarity(fmap1, fmap1_next)}")
    # print(f"Consine similarity between first left and warped left frame : {normalized_cosine_similarity(fmap1, warped_fmap_left)}")
    # print(f"Consine similarity between first left and fused left frame : {normalized_cosine_similarity(fmap1, fused_fmap1)}")
    # print(f"Consine similarity between warped fmap left and fused left frame : {normalized_cosine_similarity(warped_fmap_left, fused_fmap1)}")
    
    # # logging atten score
    # eval_disp_writer.add_image('Valid/Attention_score1', attn_scores1[0], valid_cnt)
        
    return np.mean(valid_loss_list)

@torch.no_grad()
def validate_carla_longsequence(model, iters=32, mixed_prec=False):
    """Perform validation using the CARLA synthetic dataset"""
    model.eval()
    val_dataset = datasets2.CARLA(image_set='validate')
    torch.backends.cudnn.benchmark = True
    num_cnt = 0
    
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        
        padder = InputPadder(image1.shape, divis_by = 32)
        image1, image2 = padder.pad(image1, image2)
        
        
        # ! For testing hdr intensity 
        # image1 = image1 * 10
        # image2 = image2 * 10
        
        with autocast(enabled=mixed_prec):
            start = time.time()
            fused_disp, disp1, disp2, origin_img_list, cap_rand_img_list, cap_adj_img_list, disp_cap= model(image1, image2, iters=iters, test_mode=True)
            end = time.time()
            
        fused_disp= padder.unpad(fused_disp).cpu().squeeze(0)

        # eval_writer.add_image('Demo/Fused_disparity', visualize_flow_cmap(fused_disp), num_cnt)
        # eval_writer.add_image('Demo/disp1', visualize_flow_cmap(disp1),num_cnt)
        # eval_writer.add_image('Demo/disp2', visualize_flow_cmap(disp2),num_cnt)  
        # eval_writer.add_image('Demo/disp1_r', visualize_flow_cmap(disp_cap[0]),num_cnt)
        # eval_writer.add_image('Demo/disp2_r', visualize_flow_cmap(disp_cap[1]),num_cnt)
        # eval_writer.add_image('Demo/gt', visualize_flow_cmap(flow_gt), num_cnt)
        # eval_writer.add_image('Demo/hdr_left', origin_img_list[0][0], num_cnt)
        # eval_writer.add_image('Demo/img1_rand_left', cap_rand_img_list[0][0], num_cnt)
        # eval_writer.add_image('Demo/img2_rand_left', cap_rand_img_list[1][0], num_cnt)
        # eval_writer.add_image('Demo/img1_adj_left', cap_adj_img_list[0][0], num_cnt)
        # eval_writer.add_image('Demo/img2_adj_left', cap_adj_img_list[1][0], num_cnt)
        
        # print(cap_adj_img_list[1][0].shape)
        # print(cap_adj_img_list[1][0])
        
        save_image(visualize_flow_cmap(fused_disp), f'Demo/fused_disp/fused_disp_{num_cnt}.png')
        save_image(visualize_flow_cmap(disp_cap[1]), f'Demo/disp_cap2/disp_cap2_{num_cnt}.png')
        save_image(visualize_flow_cmap(flow_gt), f'Demo/gt/gt_{num_cnt}.png')
        
        save_image_255(cap_adj_img_list[0][0], f'Demo/cap_rand1/cap_rand1_{num_cnt}.png')
        save_image_255(cap_adj_img_list[1][0], f'Demo/cap_rand2/cap_rand2_{num_cnt}.png')
        save_image_255(cap_adj_img_list[2][0], f'Demo/cap_rand1_R/cap_rand1_R_{num_cnt}.png')
        save_image_255(cap_adj_img_list[3][0], f'Demo/cap_rand2_R/cap_rand2_R_{num_cnt}.png')
        
        num_cnt += 1
        
# * Validation Writer for logging
eval_disp_writer = SummaryWriter('runs/eval_disp_model')
@torch.no_grad()
def validate_real(model, valid_cnt, iters=32, mixed_prec=False):
    """Perform validation using the CARLA synthetic dataset"""
    exp1 = torch.tensor([1.0]).cuda()
    exp2 = torch.tensor([4.0]).cuda()
    
    # Freeze BatchNorm Layer
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.train()
    
    val_dataset = datasets3.CARLASequenceDataset(image_set='validate')
    torch.backends.cudnn.benchmark = True
    
    valid_loss_list = []
    
    for val_id in range(len(val_dataset)): # 100 sequence
        
        _, image1, image2, image1_next, image2_next, flow_gt, valid_gt = val_dataset[val_id]
        
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        image1_next = image1_next[None].cuda()
        image2_next = image2_next[None].cuda()
        flow_gt = flow_gt[None].cuda()
        valid_gt = valid_gt[None].cuda()
        
        with autocast(enabled=mixed_prec):
            disp_predictions, fmap1, fmap1_next, warped_fmap_left, flow_L,fused_fmap1, cap_img_list = model(image1=image1, image2=image2, image1_next=image1_next, image2_next=image2_next, exp_h=exp1, exp_l=exp2)

        valid_loss = sequence_loss_valid(disp_predictions, flow_gt, valid_gt)
        valid_loss_list.append(valid_loss.cpu().item())
        
    #* Validation logging
    print(f"In validation exposure value : {exp1} {exp2}")
    
    eval_disp_writer.add_image('Valid/gt_disp', visualize_flow_cmap(flow_gt), valid_cnt)
    eval_disp_writer.add_image('Valid/refined_disparity', visualize_flow_cmap(disp_predictions[-1]), valid_cnt)
    eval_disp_writer.add_scalar('Valid/valid_loss', np.mean(valid_loss_list), valid_cnt)
    
    eval_disp_writer.add_image('Valid/Left F1', image1[0]**(1/2.2), valid_cnt)
    eval_disp_writer.add_image('Valid/Left F2', image1_next[0]**(1/2.2), valid_cnt)
    eval_disp_writer.add_image('Valid/Left cap', cap_img_list[0][0]**(1/2.2), valid_cnt)
    eval_disp_writer.add_image('Valid/Left cap_next', cap_img_list[2][0]**(1/2.2), valid_cnt)
    # logging feature map
    visualize_flow(eval_disp_writer, flow_L, 'Valid/Flow', valid_cnt)
    log_multiple_feature_map_with_colorbar(eval_disp_writer, fmap1, 'Valid/Unwarped_Fmap1_L', valid_cnt, num_channels=1)
    log_multiple_feature_map_with_colorbar(eval_disp_writer, fmap1_next, 'Valid/Unwarped_Fmap2_L', valid_cnt, num_channels=1)
    log_multiple_feature_map_with_colorbar(eval_disp_writer, warped_fmap_left, 'Valid/Warped_Fmap2', valid_cnt, num_channels=1)
    log_multiple_feature_map_with_colorbar(eval_disp_writer, fused_fmap1, 'Valid/Fused_Fmap', valid_cnt, num_channels=1)
    
    # Feature map consine similarity
    # print("@@@@@@@@@@@@Validation@@@@@@@@@@@@@@@")
    # print(f"Consine similarity between first left and second left frame : {normalized_cosine_similarity(fmap1, fmap1_next)}")
    # print(f"Consine similarity between first left and warped left frame : {normalized_cosine_similarity(fmap1, warped_fmap_left)}")
    # print(f"Consine similarity between first left and fused left frame : {normalized_cosine_similarity(fmap1, fused_fmap1)}")
    # print(f"Consine similarity between warped fmap left and fused left frame : {normalized_cosine_similarity(warped_fmap_left, fused_fmap1)}")
    
    # # logging atten score
    # eval_disp_writer.add_image('Valid/Attention_score1', attn_scores1[0], valid_cnt)
        
    return np.mean(valid_loss_list)


@torch.no_grad()
def validate_things(model, iters=32, mixed_prec=False):
    """ Peform validation using the FlyingThings3D (TEST) split """
    model.eval()
    val_dataset = datasets.SceneFlowDatasets(dstype='frames_finalpass', things_test=True)

    out_list, epe_list = [], []
    for val_id in tqdm(range(len(val_dataset))):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe = epe.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)

        out = (epe > 1.0)
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print("Validation FlyingThings: %f, %f" % (epe, d1))
    return {'things-epe': epe, 'things-d1': d1}


@torch.no_grad()
def validate_middlebury(model, iters=32, split='F', mixed_prec=False):
    """ Peform validation using the Middlebury-V3 dataset """
    model.eval()
    aug_params = {}
    val_dataset = datasets.Middlebury(aug_params, split=split)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, _, _), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = (valid_gt.reshape(-1) >= -0.5) & (flow_gt[0].reshape(-1) > -1000)

        out = (epe_flattened > 2.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(f"Middlebury Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print(f"Validation Middlebury{split}: EPE {epe}, D1 {d1}")
    return {f'middlebury{split}-epe': epe, f'middlebury{split}-d1': d1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default=None)
    parser.add_argument('--dataset', help="dataset for evaluation", required=True, choices=["eth3d", "kitti", "things"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecure choices
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

    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")

    model.cuda()
    model.eval()

    print(f"The model has {format(count_parameters(model)/1e6, '.2f')}M learnable parameters.")

    # The CUDA implementations of the correlation volume prevent half-precision
    # rounding errors in the correlation lookup. This allows us to use mixed precision
    # in the entire forward pass, not just in the GRUs & feature extractors. 
    use_mixed_precision = args.corr_implementation.endswith("_cuda")

    if args.dataset == 'eth3d':
        validate_eth3d(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)

    elif args.dataset == 'kitti':
        validate_kitti(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)

    elif args.dataset in [f"middlebury_{s}" for s in 'FHQ']:
        validate_middlebury(model, iters=args.valid_iters, split=args.dataset[-1], mixed_prec=use_mixed_precision)

    elif args.dataset == 'things':
        validate_things(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)


