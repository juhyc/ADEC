import torch
from core.combine_model3 import CombineModel_wo_net
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
##### Test long sequence for synthetic dataset#####
###################################################

writer = SummaryWriter('runs/test_sequence_calra')

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
    
    
def save_histograms_for_exposure_simulations(hdr_image, simulated_images, base_folder, target_dynamic_range=48):
    """
    HDR 이미지를 입력으로 하여 여러 노출로 시뮬레이션된 이미지들의 히스토그램을 각각 SVG로 저장하고 커버리지 정보를 텍스트 파일로 기록합니다.
    
    Parameters:
    - hdr_image (torch.Tensor): HDR 이미지 텐서
    - simulated_images (dict): 시뮬레이션된 이미지 딕셔너리, 예: {"Dual1": img1, "Dual2": img2, "AverageAE": img3, ...}
    - base_folder (str): 결과를 저장할 기본 폴더 경로
    - target_dynamic_range (float): 목표 동적 범위, 기본값 48 dB
    """
    os.makedirs(base_folder, exist_ok=True)
    
    # 텍스트 파일로 커버리지 정보 저장
    coverage_log_path = os.path.join(base_folder, "coverage_info.txt")
    with open(coverage_log_path, 'w') as f:
        for label, sim_image in simulated_images.items():
            save_path = os.path.join(base_folder, f"{label}_histogram.svg")
            
            # 시뮬레이션된 이미지가 단일 LDR인 경우와 두 개의 LDR 이미지인 경우에 따라 적절한 함수 사용
            if isinstance(sim_image, tuple) and len(sim_image) == 2:
                # 두 개의 LDR 이미지를 포함하는 경우 (Dual exposure)
                coverage_info = visualize_2img_with_DR(hdr_image, sim_image[0], sim_image[1], target_dynamic_range, save_path=save_path, y_max=2**24)
            else:
                # 단일 LDR 이미지를 포함하는 경우 (Single exposure)
                coverage_info = visualize_1img_with_DR(hdr_image, sim_image, target_dynamic_range, save_path=save_path, y_max=2**24)
            
            # 커버리지 정보를 텍스트 파일에 기록
            f.write(f"===== {label} Coverage Info =====\n")
            for key, value in coverage_info.items():
                f.write(f"{key}: {value:.2f}\n")
            f.write("\n")  # 각 이미지의 정보 사이에 빈 줄 추가
    
    print(f"Histogram SVG files and coverage info saved in {base_folder}")


def visualize_disparity_without_extras(disp_map, vmin=None, vmax=None, norm=False):
    """Visualize disparity map without title, grid, or colorbar."""
    # Clone and detach the image tensor, then move it to CPU
    image_tensor = -disp_map[0].clone().detach()
    image = image_tensor.cpu().numpy().squeeze()
    
    # Normalize if requested
    if norm:
        image = (image - image.min()) / (image.max() - image.min())
    
    # Set vmin and vmax if they are not provided
    if vmin is None:
        vmin = image.min()
    if vmax is None:
        vmax = image.max()
        
    # Create a figure for the disparity map without extras
    fig, ax = plt.subplots(figsize=(6, 3), dpi=200)
    cmap = plt.cm.magma
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')  # Remove axes for clean output

    # Adjust layout to remove padding and extra whitespace
    plt.tight_layout(pad=0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    return fig


def save_disp_as_png(disp_map, folder, frame_idx, label, vmin, vmax):
    """Save disparity map as PNG image without extras."""
    os.makedirs(folder, exist_ok=True)
    
    # Generate disparity map visualization without extras
    fig = visualize_disparity_without_extras(disp_map, vmin, vmax)
    
    # Save figure as PNG
    png_path = os.path.join(folder, f"{label}_step{frame_idx}.png")
    fig.savefig(png_path, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)  # Close the figure to free memory


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


def hdr_dynamic_range(img_tensor):
    img_np = img_tensor[0].cpu().numpy()
    img_np_max = img_np.max()
    img_np_min = img_np[img_np>0].min()
    dr = 20*np.log10(img_np_max/img_np_min)
    return dr
    
# For Raftstereo inference
args_raft = {
    # 'restore_ckpt': '/home/user/juhyung/SAEC/models/raftstereo-eth3d.pth',
    'restore_ckpt': '/home/user/juhyung/SAEC/models/raftstereo_carla_4000.pth',
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
    
    # Load pretrained raft-stereo
    raft_stereo = nn.DataParallel(RAFTStereo(args_raft), device_ids=[0])
    raft_stereo_ckpt = torch.load(args_raft.restore_ckpt)
    logging.info(f"Loading RAFT_stereo checkpoint... {args_raft.restore_ckpt}")
    raft_stereo.load_state_dict(raft_stereo_ckpt, strict=False)
    logging.info(f"Done RAFT_stereo loading checkpoint...{args.restore_ckpt}")
    raft_stereo.cuda()
    raft_stereo.eval()
    
    # # Load NAE parameter
    # nae_ckpt = torch.load('/home/user/juhyung/SAEC/checkpoints/NAE_finetuned_ckpt.pth')
    # NueralAEC = NeuralExposureControl()
    # nae_ckpt = {k.replace('module.', ''): v for k, v in nae_ckpt.items()}
    # # Extract NAE weight from combine model (NAE + RAFT-stereo)
    # nae_module_weight = {k : v for k, v in nae_ckpt.items() if k.startswith("nae.")}
    # nae_module_weight = {k.replace('nae.',''):v for k,v in nae_module_weight.items()}
    # NueralAEC.load_state_dict(nae_module_weight)
    # print(NueralAEC.load_state_dict(nae_module_weight))
    # NueralAEC.cuda()
    # NueralAEC.eval()
    
    # Load comparison AEC model
    AverageAE = AverageBasedAutoExposure(Mwhite=255) # input 8-bit image
    # GradientAEC = GradientExposureControl(scale=1.0, Lambda=10, delta=0.01, n_points=61, default_K_p=0.2) # Input 0~1 image

    
    test_step = 0
    base_output_folder = 'test_results_carla_w_histo'
    os.makedirs(base_output_folder, exist_ok=True)
    
    
    # Initial exposure value
    initial_exp1 = torch.tensor([2.0], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    initial_exp2 = torch.tensor([2.0], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    # Initial exposure value for comparison
    exp_averageAE = torch.tensor([2.0], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    # exp_gradientAE = torch.tensor([2.0], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    # exp_neuralAE = torch.tensor([2.0], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    for i_batch, (_, *data_blob) in enumerate(tqdm(test_loader)):
        
        # Folder name
        experiment_name = test_loader.dataset.experiment_name = test_loader.dataset.get_experiment_name(i_batch)
        experiment_folder = os.path.join(base_output_folder, experiment_name)
        os.makedirs(experiment_folder, exist_ok=True)
        
        print(f"===== Saving results in {experiment_folder} for step {test_step} =====")
        
        # folder_step = 0 
                   
        left_hdr, right_hdr, left_next_hdr, right_next_hdr, disp, valid, hdr_denom1, hdr_denom2 = [x.cuda() for x in data_blob]
        
        # Image simulation with initial exposure (For comparison AEC method)
        phi_l_ldr_averageAE = ImageFormation(left_hdr, exp_averageAE, device=DEVICE)
        # phi_l_ldr_gradientAE = ImageFormation(left_hdr, exp_gradientAE, device=DEVICE)
        # phi_l_ldr_neuralAE = ImageFormation(left_hdr, exp_neuralAE, device=DEVICE)

        ldr_left_cap_averageAE = QuantizeSTE.apply(phi_l_ldr_averageAE.noise_modeling(), 8)
        # ldr_left_cap_gradientAE = QuantizeSTE.apply(phi_l_ldr_gradientAE.noise_modeling(), 8)
        # ldr_left_cap_NeuralAE = QuantizeSTE.apply(phi_l_ldr_neuralAE.noise_modeling(), 8)
        
        
        ######### For histogram visualization#########
        phi_dualAE1  = ImageFormation(hdr_denom1, initial_exp1, device=DEVICE)
        _ = phi_dualAE1.noise_modeling()
        phi_dualAE1_ldr = phi_dualAE1.ldr_denom
        
        phi_dualAE2 = ImageFormation(hdr_denom2, initial_exp2, device=DEVICE)
        _ = phi_dualAE2.noise_modeling()
        phi_dualAE2_ldr = phi_dualAE2.ldr_denom
        
        phi_averageAE1 = ImageFormation(hdr_denom1, exp_averageAE, device=DEVICE)
        _ = phi_averageAE1.noise_modeling()
        phi_averageAE1_ldr = phi_averageAE1.ldr_denom
        
        # phi_gradientAE1 = ImageFormation(hdr_denom1, exp_gradientAE, device=DEVICE)
        # _ = phi_gradientAE1.noise_modeling()
        # phi_gradientAE1_ldr = phi_gradientAE1.ldr_denom
        
        # phi_neuralAE1 = ImageFormation(hdr_denom1, exp_neuralAE, device=DEVICE)
        # _ = phi_neuralAE1.noise_modeling()
        # phi_neuralAE1_ldr = phi_neuralAE1.ldr_denom
        
        # print(hdr_denom1.max(), phi_dualAE1_ldr.max(), phi_dualAE2_ldr.max(), phi_averageAE1_ldr.max(), phi_gradientAE1_ldr.max())
        
        ##############################################
        
        # Histogram, coverage info 저장
        hdr_image = hdr_denom1
        simulated_images = {
            "DualAE": (phi_dualAE1_ldr, phi_dualAE2_ldr),
            "AverageAE": phi_averageAE1_ldr,
            # "GradientAE": phi_gradientAE1_ldr,
            # "NeuralAE": phi_neuralAE1_ldr
        }
        histogram_log_path = os.path.join(experiment_folder, f"histogram_results_step{test_step}")
        save_histograms_for_exposure_simulations(hdr_image, simulated_images, histogram_log_path)
        
        

        #^ Our method inference
        # Dual exposure with 2 frame
        with torch.no_grad():
            # With stereo exposure control module
            fused_disparity, original_img_list, captured_rand_img_list, captured_adj_img_list, exp1, exp2, fmap_list, mask_list, flow_L = model(
                left_hdr, right_hdr, left_next_hdr, right_next_hdr, initial_exp1, initial_exp2)
        
        
        # Save images into different subfolders for each situation
        cap_situations = {
            "HDR_img" : original_img_list[0][0],
            "DualAE1_img" : captured_adj_img_list[0][0],
            "DualAE2_img" : captured_adj_img_list[1][0],
            "AverageAE_img" : ldr_left_cap_averageAE[0],
            # "GradientAE_img" : ldr_left_cap_gradientAE[0],
            # "NeuralAE_img" : ldr_left_cap_neuralAE[0]
        }
        for label, cap_img in cap_situations.items():
            folder = os.path.join(base_output_folder, experiment_name, label)
            save_image(cap_img, folder, test_step, label)
        
        
        # Exposure 텍스트 파일에 저장
        exposure_log_path = os.path.join(experiment_folder, f"exposure_values_step{test_step}.txt")
        with open(exposure_log_path, 'w') as f:
            f.write(f"Exposure1: {initial_exp1.item()}\n")
            f.write(f"Exposure2: {initial_exp2.item()}\n")
            f.write(f"Exposure_average: {exp_averageAE.item()}\n")
            # f.write(f"Exposure_gradient: {exp_gradientAE.item()}\n")
            # f.write(f"Exposure_nae: {exp_neuralAE.item()}\n")

        #^ Update exposure value
        initial_exp1 , initial_exp2 = exp1, exp2
        
        # Image simulation with AverageAE
        _, gamma1 = AverageAE.adjust_exposure(ldr_left_cap_averageAE)
        print(f"Gamma Value by AverageAE : {gamma1, gamma1/255.0}")
        exp_averageAE = exp_averageAE*0.5 + (gamma1/255.0)*0.5
        # exp_averageAE = (gamma1/255.0)
        
        phi_l_ldr_averageAE = ImageFormation(left_hdr, exp_averageAE, device=DEVICE)
        phi_r_ldr_averageAE = ImageFormation(right_hdr, exp_averageAE, device=DEVICE)
        ldr_left_cap_averageAE = QuantizeSTE.apply(phi_l_ldr_averageAE.noise_modeling(), 8)
        ldr_right_cap_averageAE = QuantizeSTE.apply(phi_r_ldr_averageAE.noise_modeling(), 8)
    
       
        # # Image simulation with GradientAE
        # exp_gradientAE = GradientAEC.gradient_exposure_control(ldr_left_cap_gradientAE)
        
        # phi_l_ldr_gradientAE = ImageFormation(left_hdr, exp_gradientAE, device=DEVICE)
        # phi_r_ldr_gradientAE = ImageFormation(right_hdr, exp_gradientAE, device=DEVICE)
        # ldr_left_cap_gradientAE = QuantizeSTE.apply(phi_l_ldr_gradientAE.noise_modeling(), 8)
        # ldr_right_cap_gradientAE = QuantizeSTE.apply(phi_r_ldr_gradientAE.noise_modeling(), 8)
        
        # # Image simulation with NeuralAE
        # ldr_left_stacked = torch.stack([ldr_left_cap_NeuralAE[0][0], ldr_left_cap_NeuralAE[0][1], ldr_left_cap_NeuralAE[0][2]], dim = 0)
        # exp_neural_alpha = NueralAEC(ldr_left_stacked.unsqueeze(0))
        # # exp_neuralAE = exp_neuralAE*0.5 + exp_neural_alpha*0.5
        # exp_neuralAE = exp_neural_alpha
        # print(f"NeuralAE expsoure : {exp_neuralAE}")

        # phi_l_ldr_neuralAE = ImageFormation(left_hdr, exp_neuralAE, device=DEVICE)
        # phi_r_ldr_neuralAE = ImageFormation(right_hdr, exp_neuralAE, device=DEVICE)
        # ldr_left_cap_neuralAE = QuantizeSTE.apply(phi_l_ldr_neuralAE.noise_modeling(), 8)
        # ldr_right_cap_neuralAE = QuantizeSTE.apply(phi_r_ldr_neuralAE.noise_modeling(), 8)
        
        
        
        #^  Inference with existed raft-stereo
        # Single exposure with 1 frame
        with torch.no_grad():
            AverageAE_disparity = raft_stereo(ldr_left_cap_averageAE*255, ldr_right_cap_averageAE*255)
            # GradientAE_dispariy = raft_stereo(ldr_left_cap_gradientAE*255, ldr_right_cap_gradientAE*255)
            # NeuralAE_disparity = raft_stereo(ldr_left_cap_neuralAE*255, ldr_right_cap_neuralAE*255)
        
        
        # # disparity_loss, _ = sequence_loss(fused_disparity, disp, valid)
        # fused_disparity_l2loss = l2_loss(fused_disparity[-1], disp, valid)
        # AverageAE_disparity_l2loss = l2_loss(AverageAE_disparity[-1], disp, valid)
        # GradientAE_disparity_l2loss = l2_loss(GradientAE_dispariy[-1], disp, valid)
        # NeuralAE_disparity_l2loss = l2_loss(NeuralAE_disparity[-1], disp, valid)

        fused_disparity_mae = mae_loss(fused_disparity[-1], disp, valid)
        AverageAE_disparity_mae = mae_loss(AverageAE_disparity[-1], disp, valid)
        # GradientAE_disparity_mae = mae_loss(GradientAE_dispariy[-1], disp, valid)
        # NeuralAE_disparity_mae = mae_loss(NeuralAE_disparity[-1], disp, valid)

        # # Error map save
        # error_maps = {
        #     "Fused_disparity_error": compute_error_map(fused_disparity[-1], disp, valid),
        #     "AverageAE_disparity_error": compute_error_map(AverageAE_disparity[-1], disp, valid),
        #     # "GradientAE_disparity_error": compute_error_map(GradientAE_dispariy[-1], disp, valid),
        #     # "NeuralAE_disparity_error": compute_error_map(NeuralAE_disparity[-1], disp, valid),
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
            f.write(f"DualAE_disparity_mae: {fused_disparity_mae.item()}\n")
            f.write(f"AverageAE_disparity_mae: {AverageAE_disparity_mae.item()}\n")
            # f.write(f"GradientAE_disparity_mae: {GradientAE_disparity_mae.item()}\n")
            # f.write(f"NeuralAE_disparity_mae: {NeuralAE_disparity_mae.item()}\n")
        
        
        # Save disparity svg into different subfolders for each situation
        situations = {
            "DualAE_disp": fused_disparity[-1],
            "AverageAE_disp": AverageAE_disparity[-1],
            # "GradientAE_disp": GradientAE_dispariy[-1],
            # "NeuralAE_disp" : NeuralAE_disparity[-1],
            "GT_disp": disp
        }
        # for label, disp_map in situations.items():
        #     folder = os.path.join(base_output_folder, experiment_name, label)
        #     save_disp_as_svg(disp_map, folder, test_step, label, vmin, vmax)
        for label, disp_map in situations.items():
            folder = os.path.join(base_output_folder, experiment_name, label)
            save_disp_as_png(disp_map, folder, test_step, label, vmin, vmax)
                


        


        test_step +=1
        
        
        # #* Save Image
        # save_image((original_img_list[0][0]**(1/2.2)).cpu(), os.path.join(base_output_folder,'HDR_Left_tonemapped'), test_step, 'HDR_Left_tonemapped')
        # save_image((original_img_list[0][0]).cpu(), os.path.join(base_output_folder,'HDR_Left'), test_step, 'HDR_Left')
        # save_image(ldr_right_cap_fixed[0].cpu(), os.path.join(base_output_folder,'LDR_fixed_left'), test_step, 'LDR_fixed_left')
        # save_image(captured_adj_img_list[0][0].cpu(), os.path.join(base_output_folder,'LDR_exp_f_left'), test_step, 'LDR_exp_f_left')
        # save_image(captured_adj_img_list[1][0].cpu(), os.path.join(base_output_folder,'LDR_exp_s_left'), test_step, 'LDR_exp_s_left')
        
        
    # #* Create Video
    # # After the test loop, create videos from saved frames
    # for situation in situations.keys():
    #     situation_folder = os.path.join(base_output_folder, situation)
    #     video_output_path = os.path.join(base_output_folder, f"{situation}.mp4")
    #     create_video_from_frames(situation_folder, video_output_path, fps=10)
        
    # create_video_from_frames(os.path.join(base_output_folder,'HDR_Left_tonemapped'), os.path.join(base_output_folder,'HDR_Left_tonemapped','HDR_Left.mp4'), fps=10)
    # create_video_from_frames(os.path.join(base_output_folder,'HDR_Left'), os.path.join(base_output_folder,'HDR_Left','HDR_Left.mp4'), fps=10)
    # create_video_from_frames(os.path.join(base_output_folder,'LDR_fixed_left'), os.path.join(base_output_folder,'LDR_fixed_left', 'LDR_fixed_left.mp4'), fps=10)
    # create_video_from_frames(os.path.join(base_output_folder,'LDR_exp_f_left'), os.path.join(base_output_folder,'LDR_exp_f_left', 'LDR_exp_f_left.mp4'), fps=10)
    # create_video_from_frames(os.path.join(base_output_folder,'LDR_exp_s_left'), os.path.join(base_output_folder,'LDR_exp_s_left', 'LDR_exp_s_left.mp4'), fps=10)
    
    

# Example usage of test_sequence function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing script")
    parser.add_argument('--name', default='SAEC_test', help="name your experiment")
    parser.add_argument('--restore_ckpt', default='checkpoints/10_disp_fusion_mask_refinenet.pth', help="restore checkpoint")
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