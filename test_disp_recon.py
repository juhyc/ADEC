import torch
from core.disp_recon_model import RAFTStereoFusion
from core.disp_recon_model_refine import RAFTStereoFusion_refine
from core.stereo_datasets3 import fetch_dataloader
from core.utils.read_utils import *
import argparse
import logging
from pathlib import Path
from evaluate_stereo import visualize_flow_cmap
import numpy as np
from pathlib import Path
from tqdm import tqdm
from types import SimpleNamespace
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim

from evaluate_stereo import *


writer = SummaryWriter('runs/test_disp_recon_carla')

# For Raftstereo inference
args_raft = {
    'restore_ckpt': '/home/user/juhyung/SAEC/models/raftstereo-middlebury.pth',
    # 'restore_ckpt': '/home/user/juhyung/SAEC/models/raftstereo_carla_4000.pth',
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

# For RaftstereoFusion inference
args_sf_refine = {
    # 'restore_ckpt': '/home/user/juhyung/SAEC/checkpoints/1000_disp_fusion_mask_refinenet.pth',
    'restore_ckpt': '/home/user/juhyung/SAEC/models/raftstereo-middlebury.pth',
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
args_sf_refine = SimpleNamespace(**args_sf_refine)

# Test disparity reconstruction pipeline
def test(args):
    # Load model
    model = nn.DataParallel(RAFTStereoFusion(args), device_ids=[0])
    
    # Pretrained raft-stereo
    raft_stereo = nn.DataParallel(RAFTStereo(args_raft), device_ids=[0])
    raft_stereo_ckpt = torch.load(args_raft.restore_ckpt)
    logging.info(f"Loading RAFT_stereo checkpoint... {args_raft.restore_ckpt}")
    raft_stereo.load_state_dict(raft_stereo_ckpt, strict=False)
    logging.info(f"Done RAFT_stereo loading checkpoint...{args.restore_ckpt}")
    raft_stereo.cuda()
    raft_stereo.eval()
    
    # RAFTstereoFusion_refine
    model_fusion_refine = nn.DataParallel(RAFTStereoFusion_refine(args_sf_refine), device_ids=[0])
    model_fusion_refine_ckpt = torch.load(args_sf_refine.restore_ckpt)
    logging.info(f"Loading Stereo_fusion_refine checkpoint... {args_sf_refine.restore_ckpt}")
    model_fusion_refine.load_state_dict(model_fusion_refine_ckpt, strict=False)
    logging.info(f"Done Stereo_fusion_refine loading checkpoint...{args_sf_refine.restore_ckpt}")
    model_fusion_refine.cuda()
    model_fusion_refine.eval()
    
    train_loader = fetch_dataloader(args)
    total_steps = 0
    
    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info(f"Loading checkpoint...{args.restore_ckpt}")
        checkpoint = torch.load(args.restore_ckpt)
        
        model.load_state_dict(checkpoint, strict = False)
        logging.info(f"Done loading checkpoint...{args.restore_ckpt}")
        
    model.cuda()
    model.eval()  # Set to evaluation mode
    
    num_samples = 0
    
    should_keep_testing = True
    global_batch_num = 0
    while should_keep_testing:
        with torch.no_grad(): 
            
            for i_batch, (img_list, *data_blob) in enumerate(tqdm(train_loader)):
                img1_left, img1_right, img2_left, img2_right, disp, valid = [x.cuda() for x in data_blob]

                exp1 = generate_random_exposures(args.batch_size, valid_mode=True, value=1.0)
                exp2 = generate_random_exposures(args.batch_size, valid_mode=True, value=2.0)

                # print(f"exposure : {exp1} {exp2}")
                
                # Forward pass
                disp_predictions, fmap1, fmap1_next, warped_fmap_left, flow_L, fused_fmap1, cap_img_list  = model(
                    img1_left, img1_right, img2_left, img2_right, exp1, exp2
                )
                
                # stereofusion with refine network infernece
                disp_refine, _, _, _, _, fused_fmap1_refine, _ = model_fusion_refine(
                    img1_left, img1_right, img2_left, img2_right, exp1, exp2
                )
                
                # Raftstereo inference
                disp_raft1 = raft_stereo(cap_img_list[0]*255, cap_img_list[1]*255)
                disp_raft2 = raft_stereo(cap_img_list[2]*255, cap_img_list[3]*255)
                # print(len(disp_raft1))
                
                # Calculate L2 loss (MSE) between GT and predicted disparities
                loss_fusion = F.mse_loss(disp_predictions[-1], disp)
                loss_refine = F.mse_loss(disp_refine[-1], disp)
                loss_raft1 = F.mse_loss(disp_raft1[-1], disp)
                loss_raft2 = F.mse_loss(disp_raft2[-1], disp)
                
                # flow_predictions_fused, flow_predictions_fmap1, flow_predictions_fmap1_next = disp_predictions
                # img1_mask, img1_next_mask, img1_fused_mask = mask_list[0], mask_list[1], mask_list[2]
                # img1_mask, img1_next_mask= mask_list[0], mask_list[1]
                
                #* Logging
                if num_samples%2 ==0:
                
                    # * Image logging
                    writer.add_image('Test/Left_F1_HDR', img1_left[0]**(1/2.2), num_samples)
                    writer.add_image('Test/Left_F2_HDR', img2_left[0]**(1/2.2), num_samples)
                    writer.add_image('Test/Left_Cap1', cap_img_list[0][0], num_samples)
                    writer.add_image('Test/Left_Cap1_next', cap_img_list[2][0], num_samples)
                    
                    # print(disp.min(), disp.max())
                    vmin, vmax = -disp.max(), -disp.min()
                    
                    # * Disparity logging
                    writer.add_image('Test/GT_disparity', visualize_disparity_with_colorbar(disp, vmin, vmax), num_samples)
                    writer.add_image('Test/Fusion_disparity', visualize_disparity_with_colorbar(disp_predictions[-1], vmin, vmax), num_samples)
                    writer.add_image('Test/Refine_fusion_disparity', visualize_disparity_with_colorbar(disp_refine[-1], vmin, vmax), num_samples)
                    
                    writer.add_image('Test/RAFT_F1_disparity', visualize_disparity_with_colorbar(disp_raft1[-1], vmin, vmax), num_samples)
                    writer.add_image('Test/RAFT_F2_disparity', visualize_disparity_with_colorbar(disp_raft2[-1], vmin, vmax), num_samples)
                    # writer.add_image('Test/LDR_disparity1', visualize_disparity_with_colorbar(flow_predictions_fmap1[-1], vmin, vmax), num_samples)
                    # writer.add_image('Test/LDR_disparity2', visualize_disparity_with_colorbar(flow_predictions_fmap1_next[-1], vmin, vmax), num_samples)  
                    
                    # You can also add these losses to tensorboard
                    print(f"======Loss betweeen GT , Fusion : {loss_fusion.item()}, Refined_Fusion : {loss_refine.item()}, LDR1 : {loss_raft1.item()}, LDR2 : {loss_raft2.item()}")
                    writer.add_scalar('Loss/Fusion_Loss', loss_fusion.item(), num_samples)
                    writer.add_scalar('Loss/Refine_Loss', loss_refine.item(), num_samples)
                    writer.add_scalar('Loss/RAFT_F1_Loss', loss_raft1.item(), num_samples)
                    writer.add_scalar('Loss/RAFT_F2_Loss', loss_raft2.item(), num_samples)
                    
                    
                    #* Feature map logging
                    # visualize_flow(writer, flow_L, "Test/Flow_L", num_samples)
                    log_multiple_feature_map_with_colorbar(writer, fmap1, "Test/Unwarped_Fmap1_L", num_samples, num_channels=1)
                    log_multiple_feature_map_with_colorbar(writer, fmap1_next, "Test/Unwarped_Fmap2_L", num_samples, num_channels=1)
                    log_multiple_feature_map_with_colorbar(writer, warped_fmap_left, "Test/Warped_Fmap2", num_samples, num_channels=1)
                    log_multiple_feature_map_with_colorbar(writer, fused_fmap1, "Test/Fused_Fmap", num_samples, num_channels=1)
                    
                    log_multiple_feature_map_with_colorbar(writer, fused_fmap1_refine, "Test/Fused_fmap_refine", num_samples, num_channels=1)
                
                # Mask Logging
                # log_mask_to_tensorboard(writer, img1_mask, "Train/F1_mask", num_samples)
                # log_mask_to_tensorboard(writer, img1_next_mask, "Train/F2_mask", num_samples)
                # log_mask_to_tensorboard(writer, img1_fused_mask, "Train/F1_fused_mask", num_samples)
                
                num_samples += 1

                # Save or visualize predictions
                # save_results(img1_left, disp_gt, disp_predictions[-1], i_batch, args.results_dir)

            # Average metrics over all samples
            # avg_metrics = {k: v / num_samples for k, v in total_metrics.items()}
            # print(f"Average Test Results: EPE={avg_metrics['epe']}, 1px={avg_metrics['1px']}, 3px={avg_metrics['3px']}, 5px={avg_metrics['5px']}")


def save_results(img1_left, disp_gt, disp_pred, batch_idx, results_dir):
    """ Saves the images and disparity maps for inspection. """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    img_path = results_dir / f"img_left_{batch_idx}.png"
    gt_disp_path = results_dir / f"gt_disp_{batch_idx}.png"
    pred_disp_path = results_dir / f"pred_disp_{batch_idx}.png"

    vutils.save_image(img1_left[0], img_path)
    vutils.save_image(visualize_flow_cmap(disp_gt[0]), gt_disp_path)
    vutils.save_image(visualize_flow_cmap(disp_pred[0]), pred_disp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='disp_recon_feature_fusion_test', help="name your experiment")
    parser.add_argument('--restore_ckpt', required=True, help="path to the checkpoint to restore")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--batch_size', type=int, default=1, help="batch size used during testing.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[700, 500], help="size of the test images.")
    parser.add_argument('--results_dir', default='test_results', help="directory to save test results")
    parser.add_argument('--train_datasets', nargs='+', default=['carla'], help="training datasets.")
    
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

    Path(args.results_dir).mkdir(exist_ok=True, parents=True)

    test(args)
