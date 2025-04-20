import torch
from core.combine_model_dual import CombineModel
# from core.combine_model3_blur import CombineModel_wo_net
from core.real_datasets_lidar import fetch_real_dataloader
from core.utils.display import *
from core.raft_stereo import *
from core.utils.simulate import *
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import logging
import argparse
import torchvision.models as models
# import torchmetrics
from tqdm import tqdm
from types import SimpleNamespace

import imageio
import time

import torch.backends.cudnn as cudnn
from collections import OrderedDict
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True  

# DEVICE = 'cuda'

writer = SummaryWriter('runs/test_sequence_real')

def sort_key_func(file):
    numbers = re.findall(r'\d+', os.path.basename(file))
    return int(numbers[0]) if numbers else 0

def save_disp_as_image(image, folder, frame_idx, label):
    os.makedirs(folder, exist_ok=True)

    # RGBA -> RGB
    if image.shape[0] == 4:  # Check if image has 4 channels (RGBA)
        image = image[:3, :, :]  # Remove alpha channel (convert to RGB)
    
    # Transpose the image if it's in (C, H, W) format to (H, W, C) for saving
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))  # Convert (C, H, W) to (H, W, C)

    image_path = os.path.join(folder, f"{label}_step{frame_idx}.png")
    imageio.imwrite(image_path, image)
    
def save_image(image, folder, frame_idx, label):
    os.makedirs(folder, exist_ok=True)

    if image.dtype == torch.float32 or image.dtype == np.float32:
        image = (image * 255).detach().cpu().numpy().astype(np.uint8)

    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))  # Convert (C, H, W) to (H, W, C)

    image_path = os.path.join(folder, f"{label}_step{frame_idx}.png")
    imageio.imwrite(image_path, image)
    

def save_disp_as_svg(disp_map, folder, frame_idx, label, vmin, vmax):
    """Save disparity map as SVG image with colorbar."""
    os.makedirs(folder, exist_ok=True)
    
    # Generate disparity map visualization with colorbar
    fig = visualize_disparity_with_colorbar_svg(disp_map, vmin, vmax, title=label)
    
    # Save figure as SVG
    svg_path = os.path.join(folder, f"{label}_step{frame_idx}.svg")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)  # Close the figure to free memory
    
def disparity_to_depth(disparity_map, baseline, focal_length):
    """Disparity map to depth map(mm)."""
    disparity_map = disparity_map.detach().cpu()
    baseline = baseline.detach().cpu()
    focal_length = focal_length.detach().cpu()

    
    depth_map = (baseline * focal_length) / (disparity_map + 1e-6)
    return depth_map
    
def save_depth_as_svg(depth_map, folder, frame_idx, label, vmin, vmax):
    """Save depth map to SVG image."""
    os.makedirs(folder, exist_ok=True)
    
    # Generate depth map visualization with colorbar
    fig = visualize_disparity_with_colorbar_svg(depth_map, vmin, vmax, title=label)
    
    # Save figure as SVG
    svg_path = os.path.join(folder, f"{label}_depth_step{frame_idx}.svg")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig) 

# MAE, RMSE loss lidar
def mae_rmse_loss_lidar(disp, points, focal_length, baseline, thres = 15000):

    # Filtering
    mask = points[..., 2] < thres
    u = points[..., 0][mask].long()
    v = points[..., 1][mask].long()
    z = points[..., 2][mask]  # LiDAR depth

    # Disparity map -> depth
    disp2depth = baseline * focal_length / disp  # [B, 1, W, H]
    depth = -disp2depth[0][0]  # [W, H] 

    # Masking
    valid_mask = (u >= 0) & (u < depth.shape[1]) & (v >= 0) & (v < depth.shape[0])
    u = u[valid_mask]
    v = v[valid_mask]
    z = z[valid_mask]

    if u.shape[0] == 0 or v.shape[0] == 0:
        print("Not valid lidar point")
        return torch.tensor(0.0)

    # Sampling
    sampled_depth = depth[v, u]

    # MAE, RMSE loss
    mae_loss = torch.mean(torch.abs(sampled_depth - z))/1000 # m
    rmse_loss = torch.sqrt(torch.mean((sampled_depth - z) ** 2)) / 1000 #m    
    
    return mae_loss, rmse_loss

# ^ Test sequence image with overall-pipeline
def test_sequence(args):
    
    device = torch.device(args.device)  
    
    model = CombineModel(args).to(device)
            
    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info(f"Loading checkpoint... : {args.restore_ckpt}")
        
        checkpoint = torch.load(args.restore_ckpt, map_location=device)     
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]  

        new_checkpoint = {k.replace('module.', 'disp_recon_net.'): v for k, v in checkpoint.items()}

        missing_keys, unexpected_keys = model.load_state_dict(new_checkpoint, strict=False)
        
        logging.info(f"Done loading checkpoint")
        
    
    test_loader = fetch_real_dataloader(args)
    # model.cuda()
    model.eval()

    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            module.eval()
            module.affine = False
    
    # mae loss list
    mae_loss_dual_list = []

    # rmse loss list
    rmse_loss_dual_list = []

    test_step = 0
    base_output_folder = 'test_results_real'
    os.makedirs(base_output_folder, exist_ok=True)
    
    #* Initial exposure value
    initial_exp1 = torch.tensor([2.0], dtype=torch.float32).unsqueeze(0).to(device)
    initial_exp2 = torch.tensor([2.0], dtype=torch.float32).unsqueeze(0).to(device)
    
    fixed_range_middle = torch.tensor([0.5], dtype=torch.float32).to(device)
    
    for i_batch, batch in enumerate(tqdm(test_loader)):
        
        if batch is None:
            continue
        
        img_list, *data_blob = batch
        
        # Folder name
        experiment_name = test_loader.dataset.experiment_name = test_loader.dataset.get_experiment_name(i_batch)
        experiment_folder = os.path.join(base_output_folder, experiment_name)
        os.makedirs(experiment_folder, exist_ok=True)
        
        print(f"===== Saving results in {experiment_folder} for step {test_step} =====")
        

        left_hdr, right_hdr, left_next_hdr, right_next_hdr, focal_length, base_line, points, hdr_denom1, hdr_denom2  = [x.to(device) for x in data_blob]
                
        #^ Our method inference
        with torch.no_grad():
            
            if fixed_range_middle is None:
                fused_disparity, original_img_list, captured_rand_img_list, captured_adj_img_list, exp1, exp2, fmap_list, mask_list, flow_L, fixed_range_middle = model(
                    left_hdr, right_hdr, left_next_hdr, right_next_hdr, initial_exp1, initial_exp2, test_mode=True
                )
            
            else:
                fused_disparity, original_img_list, captured_rand_img_list, captured_adj_img_list, exp1, exp2, fmap_list, mask_list, flow_L, _= model(
                    left_hdr, right_hdr, left_next_hdr, right_next_hdr, initial_exp1, initial_exp2,
                    test_mode=True, fixed_range_middle=fixed_range_middle
                )
        
        
        # Dual exposure with 2 frame
        # with torch.no_grad():
        #     # With stereo exposure control module
            
        #     fused_disparity, original_img_list, captured_rand_img_list, captured_adj_img_list, exp1, exp2, fmap_list, mask_list, flow_L = model(
        #         left_hdr, right_hdr, left_next_hdr, right_next_hdr, initial_exp1, initial_exp2, test_mode=True)
        
        
        # Update exposure value
        initial_exp1 , initial_exp2 = exp1, exp2
        
        
        # Exposure logging
        writer.add_scalars("Exposures", {'Expsoure1' : initial_exp1.item(), 
                                         'Exposure2' : initial_exp2.item(), 
                                         }, test_step)
        
        exposure_log_path = os.path.join(experiment_folder, f"exposure_values_step{test_step}.txt")
        with open(exposure_log_path, 'w') as f:
            f.write(f"Exposure1: {initial_exp1.item()}\n")
            f.write(f"Exposure2: {initial_exp2.item()}\n")

        # MAE loss
        mae_loss_dual_exp, rmse_loss_dual_exp = mae_rmse_loss_lidar(fused_disparity[-1], points, focal_length, base_line)

        # MAE loss save as txt
        experiment_folder = os.path.join(base_output_folder, experiment_name)
        depth_mae_path = os.path.join(experiment_folder, f"depth_loss_step{test_step}.txt")
        with open(depth_mae_path, 'w') as f:
            f.write(f"DualAE_depth_mae: {mae_loss_dual_exp.item()}\n")

        # Logging
        
        vmin, vmax = 0, 30

        writer.add_scalars("MAE", {'ADEC' : mae_loss_dual_exp, 
        }, test_step)
        
        situations = {
            "DualAE_fused_disp": fused_disparity[-1],
        }
        experiment_folder = os.path.join(base_output_folder, experiment_name)
        # depth_rmse_path = os.path.join(experiment_folder, f"depth_loss_rmse_step{test_step}.txt")

        for label, disp_map in situations.items():
            folder = os.path.join(base_output_folder, experiment_name, label)
            # Save as disparity map
            save_disp_as_svg(disp_map, folder, test_step, label, vmin, vmax)
            # Save as depth map
            depth_map = disparity_to_depth(disp_map, base_line, focal_length)
            save_depth_as_svg(depth_map, folder, test_step, label, vmin, 50000)

        # LiDAR point Save
        lidar_path = os.path.join(base_output_folder, experiment_name, 'LiDAR')
        os.makedirs(lidar_path, exist_ok=True)

        fig_path = os.path.join(lidar_path, f"lidar_points_step{test_step}.svg")

        fig = plot_lidar_points(points[0], original_img_list[0][0]**(1/2.2), vmax=50000)
        fig.savefig(fig_path, format='svg', bbox_inches="tight")
        plt.close(fig)

        # Save images into different subfolders for each situation
        cap_situations = {
            "HDR_img" : original_img_list[0][0]**(1/2.2),
            "DualAE1_img" : captured_adj_img_list[0][0],
            "DualAE2_img" : captured_adj_img_list[1][0],
            "DualAE1_img_r" : captured_adj_img_list[2][0],
            "DualAE2_img_r" : captured_adj_img_list[3][0],
        }
        for label, cap_img in cap_situations.items():
            folder = os.path.join(base_output_folder, experiment_name, label)
            save_image(cap_img, folder, test_step, label)
            
        #* Logging
        writer.add_image('Disp/ADEC', visualize_disparity_with_colorbar(fused_disparity[-1], vmin, vmax), test_step)
     
        # Visualize captured image
        writer.add_image('Captured(T)/hdr_left_frame1', original_img_list[0][0], test_step)
        writer.add_image('Captured(T)/hdr_left_frame2', original_img_list[1][0], test_step)
        writer.add_image('Captured(T)/hdr_left_frame1_tonemapped', original_img_list[0][0]**(1/2.2), test_step)
        writer.add_image('Captured(T)/hdr_left_frame2_tonemapped', original_img_list[1][0]**(1/2.2), test_step)
        writer.add_image('Captured(T)/img1_adj_left', captured_adj_img_list[0][0], test_step)
        writer.add_image('Captured(T)/img2_adj_left', captured_adj_img_list[1][0], test_step)

        test_step += 1    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing script")
    parser.add_argument('--name', default='ADEC_test_real', help="name your experiment")
    parser.add_argument('--restore_ckpt', default='checkpoints/5000_disp_fusion_mask_finetuned_gru.pth', help="restore checkpoint")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--train_datasets', type=str, nargs='+', default=['test_real'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--device', type=str, default='cuda:0')

    
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
    
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)
    
    test_sequence(args)