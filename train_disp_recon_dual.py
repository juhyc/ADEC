import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
import math
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from core.disp_recon_model_dual_finetune import RAFTStereoFusion_refine

from core.stereo_datasets import fetch_dataloader, CARLASequenceDataset
from types import SimpleNamespace
from evaluate_stereo_dual import *
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

writer = SummaryWriter('runs/train_disp_recon')

def get_blur_parameters(epoch, total_epochs, initial_mean=5, final_mean=15, initial_std=2, final_std=6):

    progress = epoch / total_epochs  

    mean_degree = initial_mean + (final_mean - initial_mean) * (progress) 
    std_degree = initial_std + (final_std - initial_std) * (progress)
    
    return mean_degree, std_degree

def generate_random_exposure_epoch(batch_size, epoch, total_epochs, base_range=(1.0, 2.0), min_gap=0.5, max_gap=3.0, valid_mode=False):
    exp_list1 = []
    exp_list2 = []
    current_gap = min_gap + (max_gap - min_gap) * ((epoch - 1) / (total_epochs - 1))

    if valid_mode:
        for _ in range(batch_size):
            base_value = random.uniform(*base_range)
            exp_list1.append([base_value])
            exp_list2.append([base_value])
    else:
        for _ in range(batch_size):
            base_value = random.uniform(0.8, 2.5)  # base exposure range
            exp1 = base_value
            exp2 = exp1 + current_gap + random.uniform(-0.1, 0.1) 
            exp1 = max(0.5, min(exp1, 5.0))
            exp2 = max(0.5, min(exp2, 5.0))
            exp_list1.append([exp1])
            exp_list2.append([exp2])

    return torch.tensor(exp_list1), torch.tensor(exp_list2)

def generate_random_exposure_epoch_improved(batch_size, epoch, total_epochs,
                                            base_range=(1.0, 2.0), min_gap=0.5,
                                            max_gap=3.0, valid_mode=False,
                                            prev_base_value=None):
    
    exp_list1 = []
    exp_list2 = []

    current_gap = min_gap + (max_gap - min_gap) * 0.5 * \
                  (1 - math.cos(math.pi * (epoch - 1) / (total_epochs - 1)))

    if valid_mode:
        for _ in range(batch_size):
            base_value = random.uniform(*base_range)
            exp_list1.append([base_value])
            exp_list2.append([base_value])
    else:
        for _ in range(batch_size):

            if prev_base_value is not None:
                base_value = prev_base_value + random.uniform(-0.05, 0.05)
            else:
                base_value = random.uniform(*base_range)

            base_value = max(0.8, min(base_value, 2.5))  # clipping
            exp1 = base_value

            exp2 = exp1 + current_gap + random.uniform(-0.05, 0.05)
            exp2 = max(0.8, min(exp2, 2.5))

            exp_list1.append([exp1])
            exp_list2.append([exp2])

    return torch.tensor(exp_list1), torch.tensor(exp_list2)


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

def fetch_optimizer(args, model):
    # flow_params = list(filter(lambda p: p.requires_grad, model.module.flow_model.parameters()))
    # other_params = list(filter(lambda p: p.requires_grad, model.module.raft_warp_stereo.parameters()))
    # all_params = flow_params + other_params
    all_params = model.parameters()
    
    optimizer = optim.AdamW(all_params, lr=args.lr, weight_decay=args.wdecay, eps=1e-8)
    

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

def parse_device_ids(device_arg: str):
    """
    "cuda:0,1"  →  [0, 1]
    "0,1,2"     →  [0, 1, 2]
    """
    cleaned = device_arg.replace('cuda:', '')
    return [int(x) for x in cleaned.split(',') if x != '']


def train(args):
    device_ids = parse_device_ids(args.device)  # [0, 1]
    main_device_idx = device_ids[0]
    args.device = f'cuda:{main_device_idx}'
    
    print(f"device_ids : {device_ids}")
    model = nn.DataParallel(RAFTStereoFusion_refine(args), device_ids=device_ids, output_device=main_device_idx)
    print("Parameter Count: %d" % count_parameters(model))
    
    # Freeze network layer 
    for name, param in model.module.named_parameters():
        if "update_block.gru" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    # for name, param in model.module.named_parameters():
    #     print(f"{name}: requires_grad = {param.requires_grad}")

    train_loader = fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=False)
        logging.info(f"Done loading checkpoint")

    model.cuda()
    model.train()

    # Training, Validation Frequency
    validation_frequency = 5
    training_frequency = 2

    scaler = GradScaler(enabled=args.mixed_precision)

    should_keep_training = True
    global_batch_num = 0

    # Total epochs
    total_epochs = args.num_steps // (len(train_loader))
    current_epoch = 0
    
    print(f"num_steps, len(train_loader) : {args.num_steps} {len(train_loader)}")

    while should_keep_training:
        print(f"Epoch {current_epoch}/{total_epochs}")

        # Blurness update
        mean_degree, std_degree = get_blur_parameters(current_epoch, total_epochs)
        train_loader.dataset.update_blur_parameters(mean_degree, std_degree)
        print(f"Applied Blur Mean Degree: {mean_degree:.2f}, Std Dev: {std_degree:.2f}")

        for i_batch, (img_list, *data_blob) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            img1_left, img1_right, img2_left, img2_right, disp, valid = [x.cuda() for x in data_blob]
            
            exp1, exp2 = generate_random_exposure_epoch_improved(args.batch_size, current_epoch, total_epochs)
            
            assert model.training

            # Model forward pass
            disp_predictions, fmap1, fmap1_next, fused_fmap1, flow_L, fmap_list, cap_img_list, mask_list = model(
                img1_left, img1_right, img2_left, img2_right, exp1, exp2
            )

            # Loss
            loss, metrics = sequence_loss(disp_predictions, disp, valid)
            total_loss = loss

            global_batch_num += 1
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            
            #Tensorboard logging
            if total_steps % training_frequency == training_frequency - 1:
                print(f"=========total_loss : {total_loss.item()}, training_frequency : {total_steps}//{args.num_steps}")
                
                #? Blur intensity logging per epoch
                writer.add_scalar("live_total_loss", total_loss.item(), total_steps / training_frequency)
                writer.add_scalar(f'Blur/Mean_Degree', mean_degree, total_steps / training_frequency)
                writer.add_scalar(f'Blur/Std_Dev', std_degree, total_steps / training_frequency)
                writer.add_scalar(f"Exposure_gap", (exp2[0]/exp1[0]), total_steps / training_frequency)
                
                #? Image logging
                writer.add_image('Train/Left_F1', img1_left[0]**(1/2.2), total_steps/training_frequency)
                writer.add_image('Train/Right_F1', img1_right[0]**(1/2.2), total_steps/training_frequency)
                writer.add_image('Train/Left_F2', img2_left[0]**(1/2.2), total_steps/training_frequency)
                writer.add_image('Train/Right_F2', img2_right[0]**(1/2.2), total_steps/training_frequency)
                writer.add_image('Train/Left Cap1', cap_img_list[0][0], total_steps/training_frequency)
                writer.add_image('Train/Left Cap1_next', cap_img_list[2][0], total_steps/training_frequency)
                
                #? Fusion stage feature map, flow logging
                visualize_flow(writer, flow_L, 'Train/Flow', total_steps/training_frequency)
                log_multiple_feature_map_with_colorbar(writer, fmap1, 'Fmap/Unwarped_Fmap1_L', total_steps/training_frequency, num_channels=1)
                log_multiple_feature_map_with_colorbar(writer, fmap1_next, 'Fmap/Unwarped_Fmap2_L', total_steps/training_frequency, num_channels=1)
                log_multiple_feature_map_with_colorbar(writer, fmap_list[2], 'Fmap/Warped_Fmap2', total_steps/training_frequency, num_channels=1)
                log_multiple_feature_map_with_colorbar(writer, fmap_list[4], 'Fmap/Mask_fused_Fmap', total_steps/training_frequency, num_channels=1)
                log_multiple_feature_map_with_colorbar(writer, fused_fmap1, 'Fmap/New_mask_fused_Fmap', total_steps/training_frequency, num_channels=1)

                #? Fusion stage mask logging
                log_mask_to_tensorboard(writer, mask_list[0], 'Mask/Image1_mask', total_steps/training_frequency)
                log_mask_to_tensorboard(writer, mask_list[1], 'Mask/Image2_mask', total_steps/training_frequency)
                log_mask_to_tensorboard(writer, mask_list[2], 'Mask/Warped_image2_mask', total_steps/training_frequency)
                log_multiple_feature_map_with_colorbar(writer, mask_list[3], 'Mask/Image1_mask_mul_fmap1', total_steps/training_frequency, num_channels=1)
                log_multiple_feature_map_with_colorbar(writer, mask_list[4], 'Mask/Image2_mask_mul_fmap2', total_steps/training_frequency, num_channels=1)
                log_mask_to_tensorboard2(writer, mask_list[5], 'Mask/near_zero_mask', total_steps/training_frequency)
                log_mask_to_tensorboard2(writer, mask_list[6].float(), 'Mask/warping_failuer_mask', total_steps/training_frequency)
                log_mask_to_tensorboard2(writer, mask_list[7], 'Mask/update_mask', total_steps/training_frequency)
                
                #? Disparity logging
                writer.add_image('Train/GT_disparity', visualize_flow_cmap(disp), total_steps / training_frequency)
                writer.add_image('Train/Refined_disparity', visualize_flow_cmap(disp_predictions[-1]), total_steps / training_frequency)

            #* Save validation checkpoints
            if total_steps % validation_frequency == validation_frequency - 1:
                save_path = Path('checkpoints/%d_%s.pth' % (total_steps + 1, args.name))
                logging.info(f"Saving file {save_path.absolute()}")
                torch.save(model.state_dict(), save_path)

                # results = validate_carla_warp(model, total_steps / validation_frequency, iters=args.valid_iters)
                
                #? Validation blur case by case logging
                validate_carla_blur(model, total_steps / validation_frequency, iters=args.valid_iters, valid_case=0)
                validate_carla_blur(model, total_steps / validation_frequency, iters=args.valid_iters, valid_case=1)
                validate_carla_blur(model, total_steps / validation_frequency, iters=args.valid_iters, valid_case=2)
                validate_carla_blur(model, total_steps / validation_frequency, iters=args.valid_iters, valid_case=3)
                
                #? Validation exposure case by case logging
                exp00, exp01 = torch.tensor([0.8]).cuda(), torch.tensor([1.6]).cuda()
                exp02, exp03 = torch.tensor([0.8]).cuda(), torch.tensor([2.4]).cuda()
                exp10, exp11 = torch.tensor([1.2]).cuda(), torch.tensor([2.4]).cuda()
                exp12, exp13 = torch.tensor([1.2]).cuda(), torch.tensor([3.6]).cuda()
                exp20, exp21 = torch.tensor([1.6]).cuda(), torch.tensor([3.2]).cuda()
                exp22, exp23 = torch.tensor([1.6]).cuda(), torch.tensor([4.8]).cuda()
                
                validate_carla_exp(model, total_steps / validation_frequency, iters=args.valid_iters, valid_case=0, exp1=exp00, exp2=exp01)
                validate_carla_exp(model, total_steps / validation_frequency, iters=args.valid_iters, valid_case=0, exp1=exp02, exp2=exp03)
                validate_carla_exp(model, total_steps / validation_frequency, iters=args.valid_iters, valid_case=0, exp1=exp10, exp2=exp11)
                validate_carla_exp(model, total_steps / validation_frequency, iters=args.valid_iters, valid_case=0, exp1=exp12, exp2=exp13)
                validate_carla_exp(model, total_steps / validation_frequency, iters=args.valid_iters, valid_case=0, exp1=exp20, exp2=exp21)
                validate_carla_exp(model, total_steps / validation_frequency, iters=args.valid_iters, valid_case=0, exp1=exp22, exp2=exp23)           
                
                model.train()

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

        current_epoch += 1

    print("FINISHED TRAINING")
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='disp_recon_dual', help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1, help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['carla'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.0001, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=10000, help="length of training schedule.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[800, 600], help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=16, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")
    parser.add_argument('--device', type=str, default='cuda:0,2')

    # Validation parameters
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during validation forward pass')

    # Architecture choices
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="instance", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")

    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    Path("checkpoints").mkdir(exist_ok=True, parents=True)

    train(args)