import torch
from core.disp_recon_model import RAFTStereoFusion
from core.stereo_datasets import fetch_dataloader, KITTI_Sequence
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


writer = SummaryWriter('runs/test_disp_recon_kitti')

# Test disparity reconstruction pipeline
def test(args):
    # Load model
    model = nn.DataParallel(RAFTStereoFusion(args), device_ids=[0])
    train_loader = fetch_dataloader(args)
    
    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        
        model.load_state_dict(checkpoint, strict = False)
        logging.info(f"Done loading checkpoint")
        
    model = model.cuda()
    model.eval()  # Set to evaluation mode
    

    # Fetch test dataloader
    # test_loader = fetch_dataloader(args, split='test')  # Load test data
    # total_metrics = {'epe': 0.0, '1px': 0.0, '3px': 0.0, '5px': 0.0}
    num_samples = 0
    
    with torch.no_grad():  # No need to track gradients in test
        
        for i_batch, (img_list, *data_blob) in enumerate(tqdm(train_loader)):
        
            img1_left, img1_right, img2_left, img2_right, disp, valid = [x.cuda() for x in data_blob]

            exp1 = generate_random_exposures(args.batch_size, valid_mode=True, value=3.0)
            exp2 = generate_random_exposures(args.batch_size, valid_mode=True, value=0.5)

            print(f"exposure : {exp1} {exp2}")
            
            # Forward pass        
            disp_predictions, fmap1, fmap1_next, warped_fmap_left, flow_L, fused_fmap1, cap_img_list = model(
                img1_left, img1_right, img2_left, img2_right, exp1, exp2
            )
            # flow_predictions_fused, flow_predictions_fmap1, flow_predictions_fmap1_next = disp_predictions
            # img1_mask, img1_next_mask, img1_fused_mask = mask_list[0], mask_list[1], mask_list[2]
            # img1_mask, img1_next_mask= mask_list[0], mask_list[1]

            # writer.add_image('rgb_left1', torch.tensor(rgb_left1).permute(2,0,1))
            # writer.add_image('rgb_left1_8bit', torch.tensor(rgb_left1_8bit/255.0).permute(2,0,1))
            # writer.add_image('rgb_left1_filtered', torch.tensor(bilateralFilter(rgb_left1)).permute(2,0,1))
            # writer.add_image('rgb_rec_left1', torch.tensor(rgb_rec_left1).permute(2,0,1))
            # writer.add_image('left1_cropped', torch.tensor(left1_cropped).permute(2,0,1))
            
            writer.add_image('Test/Left_F1', img1_left[0]/255.0, num_samples)
            writer.add_image('Test/Left_F2', img2_left[0]/255.0, num_samples)
            writer.add_image('Test/Left Cap1', cap_img_list[0][0], num_samples)
            writer.add_image('Test/Left Cap1_next', cap_img_list[2][0], num_samples)
            
            print("In test_disp_recon_kitti.py")
            print(disp.shape, -disp[0].min(), -disp[0].max())
            # print(flow_predictions_fused[-1].shape, flow_predictions_fused[-1].min(), flow_predictions_fused[-1].max())
            vmin, vmax = -disp.max(), -disp.min()
            
            writer.add_image('Test/Refined_disparity', visualize_disparity_with_colorbar(disp_predictions[-1], vmin, vmax), num_samples)
            # writer.add_image('Test/LDR_disparity1', visualize_disparity_with_colorbar(flow_predictions_fmap1[-1], vmin, vmax), num_samples)
            # writer.add_image('Test/LDR_disparity2', visualize_disparity_with_colorbar(flow_predictions_fmap1_next[-1], vmin, vmax), num_samples)
            writer.add_image('Test/GT_disparity', visualize_disparity_with_colorbar(disp, vmin, vmax), num_samples)
            writer.add_image('Test/GT_disparity2', visualize_flow_cmap(disp), num_samples)

            
            # Feature map logging
            # visualize_flow(writer, flow_L, "Test/Flow_L", num_samples)
            log_multiple_feature_map_with_colorbar(writer, fmap1, "Test/Unwarped_Fmap1_L", num_samples, num_channels=1)
            log_multiple_feature_map_with_colorbar(writer, fmap1_next, "Test/Unwarped_Fmap2_L", num_samples, num_channels=1)
            log_multiple_feature_map_with_colorbar(writer, warped_fmap_left, "Test/Warped_Fmap2", num_samples, num_channels=1)
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


def compute_metrics(disp_predictions, disp_gt, valid_gt):
    """ Computes the metrics like EPE, 1px, 3px, 5px for disparity predictions. """
    epe = torch.sum((disp_predictions[-1] - disp_gt) ** 2, dim=1).sqrt()
    valid = valid_gt >= 0.5
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return epe, metrics


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
