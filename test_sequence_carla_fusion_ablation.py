import torch
from core.combine_model3 import CombineModel_wo_net
from core.combine_model3_expfusion_w_flow import CombineModel_expfusion_w_flow
from core.combine_model3_expfusion_wo_flow import CombineModel_expfusion_wo_flow
from core.combine_model3_dispfusion_w_flow import CombineModel_dispfusion_w_flow
from core.combine_model3_dispfusion_wo_flow import CombineModel_dispfusion_wo_flow
from core.raft_stereo import RAFTStereo
# from core.combine_model3_fixed import CombineModel_wo_net_fixed
# from core.combine_model3_wo_mask import CombineModel_wo_net_wo_mask
# from core.combine_model3_wo_flow import CombineModel_wo_net_wo_flow
from core.stereo_datasets3 import CARLASequenceDataset, fetch_dataloader
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

# For comparison import reference method
from core.averaged_ae_reference import AverageBasedAutoExposure
from core.shim_et_al_reference import GradientExposureControl
from core.nae_reference import NeuralExposureControl


DEVICE = 'cuda'

###################################################
#####* Fusion strategy comparison ####
###################################################

writer = SummaryWriter('runs/test_sequence_carla_fusion_ablation2')

def sort_key_func(file):
    numbers = re.findall(r'\d+', os.path.basename(file))
    return int(numbers[0]) if numbers else 0

def save_disp_as_image(image, folder, frame_idx, label):
    os.makedirs(folder, exist_ok=True)

    # RGBA 형식인 경우, 알파 채널을 제거하여 RGB로 변환
    if image.shape[0] == 4:  # Check if image has 4 channels (RGBA)
        image = image[:3, :, :]  # Remove alpha channel (convert to RGB)
    
    # Transpose the image if it's in (C, H, W) format to (H, W, C) for saving
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))  # Convert (C, H, W) to (H, W, C)

    image_path = os.path.join(folder, f"{label}_step{frame_idx}.png")
    imageio.imwrite(image_path, image)
    
def save_image(image, folder, frame_idx, label):
    os.makedirs(folder, exist_ok=True)

    # 이미지가 float32일 경우, uint8로 변환
    if image.dtype == torch.float32 or image.dtype == np.float32:
        image = (image * 255).detach().cpu().numpy().astype(np.uint8)

    # 이미지가 C x H x W 형식이면 H x W x C 형식으로 변환
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
    
def save_error_map_as_svg(error_map, folder, frame_idx, label, vmin=0, vmax=5):
    """Save error map as SVG image with colorbar."""
    os.makedirs(folder, exist_ok=True)
    
    # Generate error map visualization with colorbar
    fig = visualize_error_map_svg(error_map, vmin=vmin, vmax=vmax, title=label)
    
    # Save figure as SVG
    svg_path = os.path.join(folder, f"{label}_step{frame_idx}.svg")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig) 


def create_video_from_frames(folder, output_path, fps=10, target_size=(1280, 384)):
    frames = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")], key=sort_key_func)
    writer = imageio.get_writer(output_path, fps=fps)
    
    for frame in frames:
        img = imageio.imread(frame)

        # 이미지를 고정된 크기로 리사이즈
        img_resized = Image.fromarray(img).resize(target_size)
        writer.append_data(np.array(img_resized))

    writer.close()


# Existed raft-stereo loss
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
        valid_mask = valid_mask.unsqueeze(0)
        l2_loss = l2_loss[valid_mask.bool()]

    # Take the mean of the valid loss values
    loss = l2_loss.mean()

    return loss

def mae_loss(pred_disparity, gt_disparity, valid_mask=None):
    """ Compute MAE (Mean Absolute Error) between predicted disparity and ground truth disparity. """

    # Calculate absolute difference between prediction and ground truth
    mae_loss = torch.abs(pred_disparity - gt_disparity)

    # If valid mask is provided, apply it
    if valid_mask is not None:
        valid_mask = valid_mask.unsqueeze(0)
        mae_loss = mae_loss[valid_mask.bool()]

    # Take the mean of the valid loss values to obtain MAE
    loss = mae_loss.mean()

    return loss


def compute_error_map(pred_disparity, gt_disparity, valid_mask=None):
    """Compute absolute error map between predicted and ground truth disparity."""
    error_map = torch.abs(pred_disparity - gt_disparity)
    if valid_mask is not None:
        error_map = error_map * valid_mask
    return error_map

#* For Raftstereo inference
args_raft = {
    'restore_ckpt': '/home/user/juhyung/SAEC/checkpoints/2000_finetuned_raftstereo_gru_carla.pth',
    # 'restore_ckpt': '/home/user/juhyung/SAEC/models/raftstereo_carla_4000.pth',
    'save_numpy': False,
    'output_directory': "demo_output",
    'mixed_precision': False,
    'valid_iters': 10,
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

# ^ Test sequence image with overall-pipeline
# ^ Comparison between existed Raft-stereo and our overall-pipelien
def test_sequence(args):
    model = nn.DataParallel(CombineModel_wo_net(args), device_ids=[0])
    model_expfusion_w_flow = nn.DataParallel(CombineModel_expfusion_w_flow(args_raft), device_ids=[0])
    model_expfusion_wo_flow = nn.DataParallel(CombineModel_expfusion_wo_flow(args_raft), device_ids=[0])
    model_dispfusion_w_flow = nn.DataParallel(CombineModel_dispfusion_w_flow(args_raft), device_ids=[0])
    model_dispfusion_wo_flow = nn.DataParallel(CombineModel_dispfusion_wo_flow(args_raft), device_ids=[0])
    
    model_raftstereo = nn.DataParallel(RAFTStereo(args_raft), device_ids=[0])
    
    # Load disp_recon finetuning model
    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info(f"Loading checkpoint... : {args.restore_ckpt}")
        checkpoint = torch.load(args.restore_ckpt)
        
        # eliminate 'module.' prefix 
        new_checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
               
        # Checkpoint load to disp_recon_net
        model.module.disp_recon_net.load_state_dict(new_checkpoint, strict=False)
        logging.info(f"Done loading checkpoint")
    
    test_loader = fetch_dataloader(args)
    model.cuda()
    model.eval()
    
    
    # Fixed exposure control model
    if args_raft is not None:
        logging.info(f"Loading checkpoint... : {args_raft}")
        checkpoint = torch.load(args_raft.restore_ckpt)
        # eliminate 'module.' prefix 
        new_checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        # Checkpoint load to disp_recon_net
        model_expfusion_w_flow.module.disp_recon_net.load_state_dict(new_checkpoint, strict=False)
        model_expfusion_wo_flow.module.disp_recon_net.load_state_dict(new_checkpoint, strict=False)
        model_dispfusion_w_flow.module.disp_recon_net.load_state_dict(new_checkpoint, strict=False)
        model_dispfusion_wo_flow.module.disp_recon_net.load_state_dict(new_checkpoint, strict=False)
        model_raftstereo.module.load_state_dict(new_checkpoint, strict=False)
        
        logging.info(f"Done loading checkpoint")
    
    
    model_expfusion_w_flow.cuda()
    model_expfusion_w_flow.eval()
    
    model_expfusion_wo_flow.cuda()
    model_expfusion_wo_flow.eval()

    model_dispfusion_w_flow.cuda()
    model_dispfusion_w_flow.eval()
    
    model_dispfusion_wo_flow.cuda()
    model_dispfusion_wo_flow.eval()
    
    model_raftstereo.cuda()
    model_raftstereo.eval()
       
    test_step = 0
    base_output_folder = 'test_results_carla_fusion_ablation2'
    os.makedirs(base_output_folder, exist_ok=True)
    
    
    # Initial exposure value
    initial_exp1 = torch.tensor([1.5], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    initial_exp2 = torch.tensor([3.5], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    # For ablation Fixed exposure value
    # fixed_explow = torch.tensor([1.0], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    # fixed_exphigh = torch.tensor([3.0], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    for i_batch, (_, *data_blob) in enumerate(tqdm(test_loader)):
        
        # Folder name
        experiment_name = test_loader.dataset.experiment_name = test_loader.dataset.get_experiment_name(i_batch)
        experiment_folder = os.path.join(base_output_folder, experiment_name)
        os.makedirs(experiment_folder, exist_ok=True)
        
        print(f"===== Saving results in {experiment_folder} for step {test_step} =====")
        
                   
        left_hdr, right_hdr, left_next_hdr, right_next_hdr, disp, valid  = [x.cuda() for x in data_blob]
        
        #^ Capture simulator module
        phi_l_exph = ImageFormation(left_hdr, initial_exp1, device=DEVICE)
        phi_r_exph = ImageFormation(right_hdr, initial_exp1, device=DEVICE)
        phi_l_expl = ImageFormation(left_next_hdr, initial_exp2, device=DEVICE)
        phi_r_expl = ImageFormation(right_next_hdr, initial_exp2, device=DEVICE)
        
        #^ Simulated LDR image pair for frame1,2        
        ldr_left_exph_cap = QuantizeSTE.apply(phi_l_exph.noise_modeling(), 8)
        ldr_right_exph_cap = QuantizeSTE.apply(phi_r_exph.noise_modeling(), 8)
        ldr_left_expl_cap = QuantizeSTE.apply(phi_l_expl.noise_modeling(), 8)
        ldr_right_expl_cap = QuantizeSTE.apply(phi_r_expl.noise_modeling(), 8)
        
        
             
        #^ Our method inference
        # Dual exposure with 2 frame
        with torch.no_grad():
            # With stereo exposure control module
            fused_disparity, original_img_list, captured_rand_img_list, captured_adj_img_list, exp1, exp2, fmap_list, mask_list, flow_L = model(
                left_hdr, right_hdr, left_next_hdr, right_next_hdr, initial_exp1, initial_exp2)
        
        # #^ Update exposure value
        # initial_exp1 , initial_exp2 = exp1, exp2
        
        with torch.no_grad():
            disp_raft_exp1 = model_raftstereo(ldr_left_exph_cap, ldr_right_exph_cap)
            disp_raft_exp2 = model_raftstereo(ldr_left_expl_cap, ldr_right_expl_cap)
        
        
        #^  Ablation exposure fusion with flow -> disparity
        with torch.no_grad():
            disp_expfusion_w_flow, _= model_expfusion_w_flow(left_hdr, right_hdr, left_next_hdr, right_next_hdr, initial_exp1, initial_exp2)
        
        #^  Ablation exposure fusion without flow -> disparity
        with torch.no_grad():
            disp_expfusion_wo_flow, _= model_expfusion_wo_flow(left_hdr, right_hdr, left_next_hdr, right_next_hdr, initial_exp1, initial_exp2)
            
        #^  Ablation disparity fusion with flow -> disparity
        with torch.no_grad():
            disp_dispfusion_w_flow, _, _= model_dispfusion_w_flow(left_hdr, right_hdr, left_next_hdr, right_next_hdr, initial_exp1, initial_exp2)
                    
        #^  Ablation disparity fusion without flow -> disparity
        with torch.no_grad():
            disp_dispfusion_wo_flow, _, _= model_dispfusion_wo_flow(left_hdr, right_hdr, left_next_hdr, right_next_hdr, initial_exp1, initial_exp2)

        
        # # disparity_loss, _ = sequence_loss(fused_disparity, disp, valid)
        # fused_disparity_l2loss = l2_loss(fused_disparity[-1], disp, valid)
        # AverageAE_disparity_l2loss = l2_loss(AverageAE_disparity[-1], disp, valid)
        # GradientAE_disparity_l2loss = l2_loss(GradientAE_dispariy[-1], disp, valid)
        # NeuralAE_disparity_l2loss = l2_loss(NeuralAE_disparity[-1], disp, valid)

        disp_raft_exp1_mae = mae_loss(disp_raft_exp1[-1], disp, valid)
        disp_raft_exp2_mae = mae_loss(disp_raft_exp2[-1], disp, valid)
        fused_disparity_mae = mae_loss(fused_disparity[-1], disp, valid)
        disp_expfusion_w_flow_mae = mae_loss(disp_expfusion_w_flow[-1], disp, valid)
        disp_expfusion_wo_flow_mae = mae_loss(disp_expfusion_wo_flow[-1], disp, valid)
        disp_dispfusion_w_flow_mae = mae_loss(disp_dispfusion_w_flow[-1], disp, valid)
        disp_dispfusion_wo_flow_mae = mae_loss(disp_dispfusion_wo_flow[-1], disp, valid)
    


        # # Error map save
        # error_maps = {
        #     "Fused_disparity_error": compute_error_map(fused_disparity[-1], disp, valid),
        #     "Fixed_exp_disparity_error": compute_error_map(fused_disp_fixed[-1], disp ,valid),
        #     "Fused_disp_no_mask_error" : compute_error_map(fused_disp_nomask[-1], disp, valid),
        #     "Fused_disp_no_flow_error" : compute_error_map(fused_disp_noflow[-1], disp, valid),
        # }
        # for label, error_map in error_maps.items():
        #         folder = os.path.join(base_output_folder, experiment_name, label)
        #         save_error_map_as_svg(error_map, folder, test_step, label)   
        
        #* Logging
        vmin, vmax = -disp.max(), -disp.min()
        # writer.add_scalar("Fused disparity loss", disparity_loss.item(), test_step)

        # MAE loss 텍스트 파일에 저장
        disparity_mae_path = os.path.join(experiment_folder, f"disparity_loss_step{test_step}.txt")
        with open(disparity_mae_path, 'w') as f:
            f.write(f"disp_raft_exp1_mae: {disp_raft_exp1_mae.item()}\n")
            f.write(f"disp_raft_exp2_mae: {disp_raft_exp2_mae.item()}\n")
            f.write(f"Dual_disparity_mae: {fused_disparity_mae.item()}\n")
            f.write(f"disp_expfusion_w_flow_mae: {disp_expfusion_w_flow_mae.item()}\n")
            f.write(f"disp_expfusion_wo_flow_mae: {disp_expfusion_wo_flow_mae.item()}\n")
            f.write(f"disp_dispfusion_w_flow_mae: {disp_dispfusion_w_flow_mae.item()}\n")
            f.write(f"disp_dispfusion_wo_flow_mae: {disp_dispfusion_wo_flow_mae.item()}\n")

        
        # Save disparity svg into different subfolders for each situation
        situations = {
            "disp_raft_exp1":disp_raft_exp1[-1],
            "disp_raft_exp2":disp_raft_exp2[-1],            
            "Dual_disp": fused_disparity[-1],
            "disp_expfusion_w_flow" : disp_expfusion_w_flow[-1],
            "disp_expfusion_wo_flow" : disp_expfusion_wo_flow[-1],
            "disp_dispfusion_w_flow" : disp_dispfusion_w_flow[-1],
            "disp_dispfusion_wo_flow" : disp_dispfusion_wo_flow[-1],
            "GT_disp": disp
        }
        
        for label, disp_map in situations.items():
            folder = os.path.join(base_output_folder, experiment_name, label)
            save_disp_as_svg(disp_map, folder, test_step, label, vmin, vmax)
        
        # Save images into different subfolders for each situation
        cap_situations = {
            "HDR_img" : original_img_list[0][0],
            "Dual1_img" : captured_adj_img_list[0][0],
            "Dual2_img" : captured_adj_img_list[1][0],
            # "Fixed_exp1_img" : cap_exp_fixed_low[0],
            # "Fixed_exp2_img" : cap_exp_fixed_high[0],
        }
        for label, cap_img in cap_situations.items():
            folder = os.path.join(base_output_folder, experiment_name, label)
            save_image(cap_img, folder, test_step, label)
            
            
        #! Logging
        writer.add_scalars("L2loss comparison", {'ADEC' : fused_disparity_mae.item(),
                                                 'disp_expfusion_w_flow_mae' : disp_expfusion_w_flow_mae.item(),
                                                 'disp_expfusion_wo_flow_mae' : disp_expfusion_wo_flow_mae.item(),
                                                 'disp_dispfusion_w_flow_mae' : disp_dispfusion_w_flow_mae.item(),
                                                 'disp_dispfusion_wo_flow_mae' : disp_dispfusion_wo_flow_mae.item(),
                                                 'disp_raft_exp1_mae' : disp_raft_exp1_mae.item(),
                                                 'disp_raft_exp2_mae' : disp_raft_exp2_mae.item(),
                                                 }, test_step)
        
        writer.add_image('Disp/ADEC', visualize_disparity_with_colorbar(fused_disparity[-1], vmin, vmax), test_step)
        writer.add_image('Disp/disp_expfusion_w_flow', visualize_disparity_with_colorbar(disp_expfusion_w_flow[-1], vmin, vmax), test_step)
        writer.add_image('Disp/disp_expfusion_wo_flow', visualize_disparity_with_colorbar(disp_expfusion_wo_flow[-1], vmin, vmax), test_step)
        writer.add_image('Disp/disp_dispfusion_w_flow', visualize_disparity_with_colorbar(disp_dispfusion_w_flow[-1], vmin, vmax), test_step)
        writer.add_image('Disp/disp_dispfusion_wo_flow', visualize_disparity_with_colorbar(disp_dispfusion_wo_flow[-1], vmin, vmax), test_step)
        writer.add_image('Disp/disp_raft_exp1', visualize_disparity_with_colorbar(disp_raft_exp1[-1], vmin, vmax), test_step)
        writer.add_image('Disp/disp_raft_exp2', visualize_disparity_with_colorbar(disp_raft_exp2[-1], vmin, vmax), test_step)
        writer.add_image('Disp/GT', visualize_disparity_with_colorbar(disp, vmin, vmax), test_step)
        
        writer.add_image('Captured(T)/img1_adj_left', captured_adj_img_list[0][0]**(1/2.2), test_step)
        writer.add_image('Captured(T)/img2_adj_left', captured_adj_img_list[1][0]**(1/2.2), test_step)
        
        
        
        test_step += 1
        
    

# Example usage of test_sequence function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing script")
    parser.add_argument('--name', default='test_sequence_carla_fusion_ablation', help="name your experiment")
    parser.add_argument('--restore_ckpt', default='checkpoints/5000_disp_fusion_mask_finetuned_gru_eth3d_ba.pth', help="restore checkpoint")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--train_datasets', type=str, nargs='+', default=['test_carla'])
    parser.add_argument('--batch_size', type=int, default=1)
    
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