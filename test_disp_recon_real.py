import torch
from core.disp_recon_model import RAFTStereoFusion
from core.real_datasets import fetch_real_dataloader, RealDataset
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
# from core.disp_recon_model_ablation import RAFTStereoFusion
import torch
import torch.nn as nn
import torch.optim as optim

from evaluate_stereo import *


writer = SummaryWriter('runs/test_disp_recon_real')

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

# Test disparity reconstruction pipeline
def test(args):
    # Load model
    model = nn.DataParallel(RAFTStereoFusion(args), device_ids=[0])
    
    # Load pretrained raft-stereo
    raft_stereo = nn.DataParallel(RAFTStereo(args_raft), device_ids=[0])
    raft_stereo_ckpt = torch.load(args_raft.restore_ckpt)
    logging.info(f"Loading RAFT_stereo checkpoint... {args_raft.restore_ckpt}")
    raft_stereo.load_state_dict(raft_stereo_ckpt, strict=False)
    logging.info(f"Done RAFT_stereo loading checkpoint...{args.restore_ckpt}")
    raft_stereo.cuda()
    raft_stereo.eval()
    
    train_loader = fetch_real_dataloader(args)
    
    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        
        model.load_state_dict(checkpoint, strict = False)
        logging.info(f"Done loading checkpoint")
        
    model.cuda()
    model.eval()
    
    num_samples = 0
    
    
    
    with torch.no_grad(): 
        
        for i_batch, batch in enumerate(tqdm(train_loader)):
        
            if batch is None:
                continue
            
            img_list, *data_blob = batch
            
            img1_left, img1_right, img2_left, img2_right, lidar_disp, u, v  = [x.cuda() for x in data_blob]

            exp1 = generate_random_exposures(args.batch_size, valid_mode=True, value=0.75)
            exp2 = generate_random_exposures(args.batch_size, valid_mode=True, value=3.0)

            # print(f"exposure : {exp1} {exp2}")
            
            # Forward pass        
            disp_predictions, fmap1, fmap1_next, warped_fmap_left, flow_L, fused_fmap1, cap_img_list = model(
                img1_left, img1_right, img2_left, img2_right, exp1, exp2
            )
            
            # Raftstereo inference
            disp_raft1 = raft_stereo(cap_img_list[0]*255, cap_img_list[1]*255)
            disp_raft2 = raft_stereo(cap_img_list[2]*255, cap_img_list[3]*255)
            disp_raft_hdr = raft_stereo(img1_left*255, img1_right*255)
            
            #* Logging
            if num_samples%2 == 0:
                
                # * Image logging
                writer.add_image('Test/Left_F1_HDR', img1_left[0], num_samples)
                writer.add_image('Test/Left_F2_HDR', img2_left[0], num_samples)
                writer.add_image('Test/Left Cap1', cap_img_list[0][0], num_samples)
                writer.add_image('Test/Left Cap1_next', cap_img_list[2][0], num_samples)
                
                print(lidar_disp.shape, lidar_disp.min(), lidar_disp.max())
                # print(flow_predictions_fused[-1].shape, flow_predictions_fused[-1].min(), flow_predictions_fused[-1].max())
                # vmin, vmax = -disp.max(), -disp.min()
                
                # * Disparity logging
                writer.add_image('Test/Refined_disparity', visualize_disparity_with_colorbar(disp_predictions[-1], 0, 25), num_samples)
                writer.add_image('Test/RAFT_hdr', visualize_disparity_with_colorbar(disp_raft_hdr[-1], 0, 25), num_samples)
                writer.add_image('Test/RAFT_F1_disparity', visualize_disparity_with_colorbar(disp_raft1[-1], 0, 25), num_samples)
                writer.add_image('Test/RAFT_F2_disparity', visualize_disparity_with_colorbar(disp_raft2[-1], 0, 25), num_samples)
                
                #*  Feature map logging
                # visualize_flow(writer, flow_L, "Test/Flow_L", num_samples)
                log_multiple_feature_map_with_colorbar(writer, fmap1, "Test/Unwarped_Fmap1_L", num_samples, num_channels=1)
                log_multiple_feature_map_with_colorbar(writer, fmap1_next, "Test/Unwarped_Fmap2_L", num_samples, num_channels=1)
                # log_multiple_feature_map_with_colorbar(writer, warped_fmap_left, "Test/Warped_Fmap2", num_samples, num_channels=1)
                log_multiple_feature_map_with_colorbar(writer, fused_fmap1, "Test/Fused_Fmap", num_samples, num_channels=1)
                
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
    
    # parser.add_argument('--train_datasets', nargs='+', default=['carla'], help="training datasets.")
    parser.add_argument('--num_steps', type=int, default=1000, help="length of training schedule.")
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
