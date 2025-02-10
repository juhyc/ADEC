import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from core.disp_recon_model import RAFTStereoFusion
# from core.real_datasets import fetch_real_dataloader, RealDataset
# from core.real_datasets_lidar import fetch_real_dataloader
from core.real_datasets_lidar_downsample import fetch_real_dataloader
from core.disp_recon_model_refine import RAFTStereoFusion_refine
# from core.real_datasets_crop import fetch_real_dataloader, RealDataset
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

writer = SummaryWriter('runs/train_diso_recon_real')


def mae_loss_lidar(disp, points, focal_length, baseline, thres = 15000):
    # 조건에 맞는 좌표와 깊이값 필터링
    mask = points[..., 2] < thres
    u = points[..., 0][mask].long()
    v = points[..., 1][mask].long()
    z = points[..., 2][mask]  # LiDAR depth 값

    # Disparity map을 depth로 변환
    disp2depth = baseline * focal_length / disp  # [B, 1, W, H]
    depth = -disp2depth[0][0]  # [W, H], 단일 이미지

    # u, v 좌표가 depth 맵 크기를 벗어나지 않도록 마스킹
    valid_mask = (u >= 0) & (u < depth.shape[1]) & (v >= 0) & (v < depth.shape[0])
    u = u[valid_mask]
    v = v[valid_mask]
    z = z[valid_mask]

    # 유효한 u, v 좌표가 있는지 확인
    if u.shape[0] == 0 or v.shape[0] == 0:
        print("Not valid lidar point")
        return torch.tensor(0.0)

    # u, v 좌표를 사용해 depth 맵에서 샘플링
    sampled_depth = depth[v, u]

    # MAE 손실 계산
    mae_loss = torch.mean(torch.abs(sampled_depth - z))

    # u, v 좌표를 사용해 depth 맵에서 샘플링
    sampled_depth = depth[v, u]

    # MAE 손실 계산
    mae_loss = torch.mean(torch.abs(sampled_depth - z))/1000 # m
    
    return mae_loss

def generate_random_exposure(batch_size, min_gap=0.5, max_gap=3.0, base_value=None, valid_mode=False):
    """
    노출값을 두 개 생성하는 함수.
    min_gap과 max_gap을 이용해 두 노출값의 차이를 제한.
    base_value가 None인 경우 0.8 ~ 2.0 사이의 랜덤 값을 사용.
    """
    exp_list1 = []
    exp_list2 = []
    
    # base_value가 None이면 0.8 ~ 2.0 범위의 랜덤 값으로 설정
    if base_value is None:
        base_value = random.uniform(0.8, 2.0)
    
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
    model = nn.DataParallel(RAFTStereoFusion_refine(args), device_ids=[0,1])
    print("Parameter Count: %d" % count_parameters(model))

    # Except FeatureRefiner, freeze other layers
    for name, param in model.module.named_parameters():
        # if "fnet" in name and ("layer2" in name) or "cnet" in name and ("layer5" in name ):
        # if "update_block.gru" in name or "cnet.layer5" in name or "fnet.layer3" in name:
        if "update_block.gru32" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    #For debugging, Check layer freezed
    for name, param in model.module.named_parameters():
        print(f"{name}: requires_grad = {param.requires_grad}")

    train_loader = fetch_real_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        
        model.load_state_dict(checkpoint, strict = False)
        logging.info(f"Done loading checkpoint")
        
    model.train()
    model.cuda()

    validation_frequency = 50
    training_frequency = 10

    scaler = GradScaler(enabled=args.mixed_precision)

    should_keep_training = True
    global_batch_num = 0
    while should_keep_training:

        for i_batch, batch in enumerate(tqdm(train_loader)):
            
            if batch is None:
                continue
            
            img_list, *data_blob = batch
        
            img1_left, img1_right, img2_left, img2_right, f, b, points  = [x.cuda() for x in data_blob]
            
            exp1, exp2 = generate_random_exposure(args.batch_size)      

            assert model.training
            disp_predictions, fmap1, fmap1_next, fused_fmap1, flow_L, fmap_list, cap_img_list, _ = model(img1_left, img1_right, img2_left, img2_right, exp1, exp2)
            assert model.training
            
            model.train()
            
            print(len(disp_predictions))
            
            total_loss = mae_loss_lidar(disp_predictions[-1], points, f, b)
            
            global_batch_num += 1
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()            
            
            if total_steps % training_frequency == training_frequency - 1:
                #* For tensorboard logging
                print(f"=========total_loss : {total_loss.item()}, training_frequency : {total_steps}//{args.num_steps}")
                writer.add_scalar("live_total_loss", total_loss.item(), total_steps/training_frequency)
                writer.add_scalar("MAE_LiDAR_loss", total_loss.item(), total_steps/training_frequency)
                
                writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], total_steps/training_frequency)
                writer.add_image('Train/Left_F1', img1_left[0], total_steps/training_frequency)
                # writer.add_image('Train/Right_F1', img1_right[0]**(1/2.2), total_steps/training_frequency)
                writer.add_image('Train/Left_F2', img2_left[0], total_steps/training_frequency)
                writer.add_image('Train/Left Cap1', cap_img_list[0][0], total_steps/training_frequency)
                writer.add_image('Train/Left Cap1_next', cap_img_list[2][0], total_steps/training_frequency)
                # writer.add_image('Train/Right_F2', img2_right[0]**(1/2.2), total_steps/training_frequency)
                # writer.add_image('Train/Left_Warped_img', img2_left_warped[0]**(1/2.2), total_steps/training_frequency)
                writer.add_image('Train/Refined_disparity', visualize_flow_cmap(disp_predictions[-1]), total_steps/training_frequency)
                
                # LiDAR logging
                fig = plot_lidar_points(points[0], img1_left[0])
                writer.add_figure('Test/LiDAR points', fig, total_steps/training_frequency)
                
                # Feature map logging
                # visualize_flow(writer, flow_L, "Train/Flow_L", total_steps/training_frequency)
                log_multiple_feature_map_with_colorbar(writer, fmap1, "Train/Unwarped_Fmap1_L", total_steps/training_frequency, num_channels=1)
                log_multiple_feature_map_with_colorbar(writer, fmap1_next, "Train/Unwarped_Fmap2_L", total_steps/training_frequency, num_channels=1)
                log_multiple_feature_map_with_colorbar(writer, fused_fmap1_simple, "Train/Fused_Fmap_simple", total_steps/training_frequency, num_channels=1)
                log_multiple_feature_map_with_colorbar(writer, fused_fmap1, "Train/Fused_Fmap", total_steps/training_frequency, num_channels=1)
                

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
    parser.add_argument('--name', default='disp_fusion_finetuned_real_gru', help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1, help="batch size used during training.")
    # parser.add_argument('--train_datasets', nargs='+', default=['carla'], help="training datasets.")
    parser.add_argument('--train_datasets', type=str, nargs='+', default=['test_real'])
    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=1000, help="length of training schedule.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[800, 600], help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=32, help="number of updates to the disparity field in each forward pass.")
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

    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    Path("checkpoints").mkdir(exist_ok=True, parents=True)

    train(args)