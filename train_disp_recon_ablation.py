import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
# from core.disp_recon_model import Disp_recon_model 
from core.disp_recon_model_ablation import RAFTStereoFusion
from core.stereo_datasets3 import fetch_dataloader, CARLASequenceDataset
from types import SimpleNamespace
from evaluate_stereo import *
import torchvision.utils as vutils

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

# For Raftstereo inference
args_raft = {
    'restore_ckpt': '/home/user/juhyung/SAEC/models/raftstereo_carla_4000.pth',
    'save_numpy': False,
    'output_directory': "demo_output",
    'mixed_precision': False,
    'valid_iters': 32,
    'hidden_dims': [128]*3,
    'corr_implementation': "reg",
    'shared_backbone': False,
    'corr_levels': 4,
    'corr_radius': 4,
    'n_downsample': 2,
    'context_norm': "batch",
    'slow_fast_gru': False,
    'n_gru_layers': 3,
}
args_raft = SimpleNamespace(**args_raft)

writer = SummaryWriter('runs/train_disp_recon_ablation')

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

def balance_loss(fmap1, fmap2):
    info_diff = torch.abs(fmap1.var(dim=[2, 3]) - fmap2.var(dim=[2, 3]))
    return torch.mean(info_diff)



def fetch_optimizer(args, model):
    # flow_params = list(filter(lambda p: p.requires_grad, model.module.flow_model.parameters()))
    # other_params = list(filter(lambda p: p.requires_grad, model.module.raft_warp_stereo.parameters()))
    # all_params = flow_params + other_params
    all_params = model.parameters()
    
    optimizer = optim.AdamW(all_params, lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

def train(args):
    model = nn.DataParallel(RAFTStereoFusion(args), device_ids=[0,1])
    print("Parameter Count: %d" % count_parameters(model))
    
    train_loader = fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        
        model.load_state_dict(checkpoint, strict = False)
        logging.info(f"Done loading checkpoint")

    model.cuda()
    model.train()

    validation_frequency = 1
    training_frequency = 1

    scaler = GradScaler(enabled=args.mixed_precision)

    should_keep_training = True
    global_batch_num = 0
    while should_keep_training:

        for i_batch, (img_list, *data_blob) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            img1_left, img1_right, img2_left, img2_right, disp, valid = [x.cuda() for x in data_blob]
            
            # exp1 = generate_random_exposures(args.batch_size, value=1.5)
            # exp2 = generate_random_exposures(args.batch_size, value=1.5)
            exp1 = generate_random_exposures(args.batch_size, valid_mode=True, value=3.0)
            exp2 = generate_random_exposures(args.batch_size, valid_mode=True, value=0.6)
            

            assert model.training
            disp_predictions, fmap1, fmap1_next, warped_fmap_left, flow_L, fused_fmap1, cap_img_list, mask_list = model(img1_left, img1_right, img2_left, img2_right, exp1, exp2)
            print(f"exposure : {exp1} {exp2}")
            # print(f"Shape in training, img1_left : {img1_left.shape}, img2_left : {img2_left.shape}, fmap1 : {fmap1.shape}, fmap1_next : {fmap1_next.shape}, warped_fmap_next : {warped_fmap_left.shape}, fused_fmap : {fused_fmap1.shape}")
            assert model.training
            
            print(f"In train_disp_ablation.py : {len(disp_predictions)}")
            flow_predictions_fused, flow_predictions_fmap1, flow_predictions_fmap1_next = disp_predictions
            img1_mask, img1_next_mask, img1_fused_mask = mask_list[0], mask_list[1], mask_list[2]
            
            loss, metrics = sequence_loss(flow_predictions_fused, disp, valid)
            
            
            global_batch_num += 1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            
            
            if total_steps % training_frequency == training_frequency - 1:
                #* For tensorboard logging
                writer.add_scalar("live_loss", loss.item(), total_steps/training_frequency)
                writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], total_steps/training_frequency)
                writer.add_image('Train/Left_F1', img1_left[0]**(1/2.2), total_steps/training_frequency)
                # writer.add_image('Train/Right_F1', img1_right[0]**(1/2.2), total_steps/training_frequency)
                writer.add_image('Train/Left_F2', img2_left[0]**(1/2.2), total_steps/training_frequency)
                writer.add_image('Train/Left Cap1', cap_img_list[0][0], total_steps/training_frequency)
                writer.add_image('Train/Left Cap1_next', cap_img_list[2][0], total_steps/training_frequency)
                # writer.add_image('Train/Right_F2', img2_right[0]**(1/2.2), total_steps/training_frequency)
                # writer.add_image('Train/Left_Warped_img', img2_left_warped[0]**(1/2.2), total_steps/training_frequency)
                
                vmin, vmax = -disp[0].min(), -disp[0].max()
                
                writer.add_image('Train/GT_disparity', visualize_disparity_with_colorbar(disp, vmin, vmax), total_steps/training_frequency)
                writer.add_image('Train/GT_disparity2', visualize_flow_cmap(disp), total_steps/training_frequency)
                writer.add_image('Train/Refined_disparity', visualize_disparity_with_colorbar(flow_predictions_fused[-1], vmin, vmax), total_steps/training_frequency)
                
                writer.add_image('Train/LDR_disparity1', visualize_disparity_with_colorbar(flow_predictions_fmap1[-1], vmin, vmax), total_steps/training_frequency)
                writer.add_image('Train/LDR_disparity2', visualize_disparity_with_colorbar(flow_predictions_fmap1_next[-1], vmin, vmax), total_steps/training_frequency)
                
                # Feature map logging
                # visualize_flow(writer, flow_L, "Train/Flow_L", total_steps/training_frequency)
                log_multiple_feature_map_with_colorbar(writer, fmap1, "Train/Unwarped_Fmap1_L", total_steps/training_frequency, num_channels=1)
                log_multiple_feature_map_with_colorbar(writer, fmap1_next, "Train/Unwarped_Fmap2_L", total_steps/training_frequency, num_channels=1)
                log_multiple_feature_map_with_colorbar(writer, warped_fmap_left, "Train/Warped_Fmap2", total_steps/training_frequency, num_channels=1)
                log_multiple_feature_map_with_colorbar(writer, fused_fmap1, "Train/Fused_Fmap", total_steps/training_frequency, num_channels=1)
                
                # Mask logging
                log_mask_to_tensorboard(writer, img1_mask, "Train/F1_mask", total_steps/training_frequency)
                log_mask_to_tensorboard(writer, img1_next_mask, "Train/F2_mask", total_steps/training_frequency)
                log_mask_to_tensorboard(writer, img1_fused_mask, "Train/F1_fused_mask", total_steps/training_frequency)
                
                # log_difference_map(writer, fmap1, warped_fmap_left, fused_fmap1, total_steps/training_frequency)
                
                # print("fmap1 min/max:", fmap1.min().item(), fmap1.max().item())
                # print("fmap2_warped min/max:", warped_fmap_left.min().item(), warped_fmap_left.max().item())
                # print("fused_fmap1 min/max:", fused_fmap1.min().item(), fused_fmap1.max().item())

                # # Feature map consine similarity
                # print(f"Consine similarity between first left and second left frame : {normalized_cosine_similarity(fmap1, fmap1_next)}")
                # print(f"Consine similarity between first left and warped left frame : {normalized_cosine_similarity(fmap1, warped_fmap_left)}")
                # print(f"Consine similarity between first left and fused left frame : {normalized_cosine_similarity(fmap1, fused_fmap1)}")
                # print(f"Consine similarity between warped fmap left and fused left frame : {normalized_cosine_similarity(warped_fmap_left, fused_fmap1)}")

            # # Validation
            # if total_steps % validation_frequency == validation_frequency - 1:
            #     save_path = Path('checkpoints/%d_%s.pth' % (total_steps + 1, args.name))
            #     logging.info(f"Saving file {save_path.absolute()}")
            #     torch.save(model.state_dict(), save_path)

            #     results = validate_carla_warp(model, total_steps/validation_frequency, iters=args.valid_iters)
            #     model.train()

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

        if len(train_loader) >= 10000:
            save_path = Path('checkpoints/%d_epoch_%s.pth.gz' % (total_steps + 1, args.name))
            logging.info(f"Saving file {save_path}")
            torch.save(model.state_dict(), save_path)

    print("FINISHED TRAINING")
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='disp_fusion_mask_ablation', help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1, help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['carla'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=1000, help="length of training schedule.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[800, 600], help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=16, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

    # Validation parameters
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during validation forward pass')

    # Architecture choices
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

    Path("checkpoints").mkdir(exist_ok=True, parents=True)

    train(args)