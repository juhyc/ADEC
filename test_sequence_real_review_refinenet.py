import torch
# from core.combine_model3 import CombineModel_wo_net
from core.combine_model3_refinenet import CombineModel_wo_net
# from core.stereo_datasets3 import CARLASequenceDataset, fetch_dataloader
# from core.real_datasets import fetch_real_dataloader, RealDataset
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

# For comparison import reference method
from core.averaged_ae_reference import AverageBasedAutoExposure
from core.shim_et_al_reference import GradientExposureControl
from core.nae_reference import NeuralExposureControl

# For comparison import combined reference method (AEC + Stereo matching)
from core.combine_model_average import CombineModel_w_averageAE
from core.combine_model_gradient import CombineModel_w_gradientAE
from core.combine_model_nae import CombineModel_w_nae


DEVICE = 'cuda'

writer = SummaryWriter('runs/test_sequence_real_review_motion_refinenet')

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
    
def disparity_to_depth(disparity_map, baseline, focal_length):
    """Disparity map을 depth map으로 변환 (단위: mm)."""
    disparity_map = disparity_map.detach().cpu()
    baseline = baseline.detach().cpu()
    focal_length = focal_length.detach().cpu()

    
    depth_map = (baseline * focal_length) / (disparity_map + 1e-6)  # 1e-6은 분모가 0이 되는 것을 방지
    return depth_map
    
def save_depth_as_svg(depth_map, folder, frame_idx, label, vmin, vmax):
    """Depth map을 SVG 이미지로 저장."""
    os.makedirs(folder, exist_ok=True)
    
    # Generate depth map visualization with colorbar
    fig = visualize_disparity_with_colorbar_svg(depth_map, vmin, vmax, title=label)
    
    # Save figure as SVG
    svg_path = os.path.join(folder, f"{label}_depth_step{frame_idx}.svg")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)  # Close the figure to free memory


def create_video_from_frames(folder, output_path, fps=10, target_size=(800, 600)):
    frames = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")], key=sort_key_func)
    writer = imageio.get_writer(output_path, fps=fps)
    
    for frame in frames:
        img = imageio.imread(frame)

        # 이미지를 고정된 크기로 리사이즈
        img_resized = Image.fromarray(img).resize(target_size)
        writer.append_data(np.array(img_resized))

    writer.close()

def mae_rmse_loss_lidar(disp, points, focal_length, baseline, thres = 15000):

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
    mae_loss = torch.mean(torch.abs(sampled_depth - z))/1000 # m
    rmse_loss = torch.sqrt(torch.mean((sampled_depth - z) ** 2)) / 1000 #m    
    
    return mae_loss, rmse_loss

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

    
# For Raftstereo inference
args_raft = {
    # 'restore_ckpt': '/home/user/juhyung/SAEC/models/raftstereo-eth3d.pth',
    'restore_ckpt': '/home/user/juhyung/SAEC/models/raftstereo_carla_4000.pth',
    'save_numpy': False,
    'output_directory': "demo_output",
    'mixed_precision': False,
    'valid_iters': 7,
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


# For AverageAE+RAFT inference
args_averageAE_raft = {
    'restore_ckpt': 'checkpoints/combine_model_average_200_epoch.pth',
    # 'restore_ckpt': '/home/user/juhyung/SAEC/models/raftstereo-sceneflow.pth',
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
args_averageAE_raft = SimpleNamespace(**args_averageAE_raft)

# For graidentAE+RAFT inference
args_gradientAE_raft = {
    'restore_ckpt': 'checkpoints/combine_model_gradient_200_epoch.pth',
    # 'restore_ckpt': '/home/user/juhyung/SAEC/models/raftstereo-sceneflow.pth',
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
args_gradientAE_raft = SimpleNamespace(**args_gradientAE_raft)

# For graidentAE+RAFT inference
args_neuralAE_raft = {
    'restore_ckpt': 'checkpoints/combine_model_nae_200_epoch.pth',
    # 'restore_ckpt': '/home/user/juhyung/SAEC/models/raftstereo-sceneflow.pth',
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
args_neuralAE_raft = SimpleNamespace(**args_neuralAE_raft)

def normalize_exposure_shape(exp_tensor):
    if exp_tensor.dim() == 0:  # 스칼라 텐서
        exp_tensor = exp_tensor.unsqueeze(0).unsqueeze(1)  # torch.Size([1, 1])
    elif exp_tensor.dim() > 2:  # 3차원 이상 텐서
        exp_tensor = exp_tensor.squeeze(-1)  # 마지막 차원 제거
    return exp_tensor


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
        
    # # Load Overall-pipeline model
    # if args.restore_ckpt is not None:
    #     assert args.restore_ckpt.endswith(".pth")
    #     logging.info(f"Loading checkpoint... : {args.restore_ckpt}")
    #     checkpoint = torch.load(args.restore_ckpt)
    #     model.load_state_dict(checkpoint, strict=False)
    
    test_loader = fetch_real_dataloader(args)
    model.cuda()
    model.eval()
    
    
    #^ Load pretrained AverageAEC + RAFT
    averageAE_raft = nn.DataParallel(CombineModel_w_averageAE(args_averageAE_raft), device_ids=[0])
    averageAE_raft_ckpt = torch.load(args_averageAE_raft.restore_ckpt)
    averageAE_raft.load_state_dict(averageAE_raft_ckpt, strict=False)
    
    # # Load disp_recon finetuning model
    # if args_averageAE_raft is not None:
    #     logging.info(f"Loading checkpoint... : {args_averageAE_raft}")
    #     checkpoint = torch.load(args_averageAE_raft.restore_ckpt)
    #     # eliminate 'module.' prefix 
    #     new_checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    #     # Checkpoint load to disp_recon_net
    #     averageAE_raft.module.raft_stereo.load_state_dict(new_checkpoint, strict=False)  
    #     averageAE_raft.module.raft_stereo.load_state_dict(new_checkpoint, strict=False)  
    #     logging.info(f"Done loading checkpoint")

        
    averageAE_raft.cuda()
    averageAE_raft.eval()
    
    #^ Load pretrained gradientAEC + RAFT
    gradientAE_raft = nn.DataParallel(CombineModel_w_gradientAE(args_gradientAE_raft), device_ids=[0])
    gradientAE_raft_ckpt = torch.load(args_gradientAE_raft.restore_ckpt)
    gradientAE_raft.load_state_dict(gradientAE_raft_ckpt, strict=False)
    # # Load disp_recon finetuning model
    # if args_gradientAE_raft is not None:
    #     logging.info(f"Loading checkpoint... : {args_gradientAE_raft}")
    #     checkpoint = torch.load(args_gradientAE_raft.restore_ckpt)
    #     # eliminate 'module.' prefix 
    #     new_checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    #     # Checkpoint load to disp_recon_net
    #     gradientAE_raft.module.raft_stereo.load_state_dict(new_checkpoint, strict=False)  
    #     logging.info(f"Done loading checkpoint")
    
    gradientAE_raft.cuda()
    gradientAE_raft.eval()
    
    #^ Load pretrained neuralAEC + RAFT
    neuralAE_raft = nn.DataParallel(CombineModel_w_nae(args_neuralAE_raft), device_ids=[0])
    neuraltAE_raft_ckpt = torch.load(args_neuralAE_raft.restore_ckpt)
    neuralAE_raft.load_state_dict(neuraltAE_raft_ckpt, strict=False)
    # # Load disp_recon finetuning model
    # if args_neuralAE_raft is not None:
    #     logging.info(f"Loading checkpoint... : {args_neuralAE_raft}")
    #     checkpoint = torch.load(args_neuralAE_raft.restore_ckpt)
    #     # eliminate 'module.' prefix 
    #     new_checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    #     # Checkpoint load to disp_recon_net
    #     neuralAE_raft.module.raft_stereo.load_state_dict(new_checkpoint, strict=False)  
    #     logging.info(f"Done loading checkpoint")
    
    neuralAE_raft.cuda()
    neuralAE_raft.eval()
    
    
    # mae loss list
    mae_loss_dual_list = []
    mae_loss_average_list = []
    mae_loss_gradient_list = []
    mae_loss_neural_list = []

    # rmse loss list
    rmse_loss_dual_list = []
    rmse_loss_average_list = []
    rmse_loss_gradient_list = []
    rmse_loss_neural_list = []

    test_step = 0
    base_output_folder = 'test_results_real_review_motion'
    os.makedirs(base_output_folder, exist_ok=True)
    
    #* Initial exposure value
    initial_exp1 = torch.tensor([2.5], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    initial_exp2 = torch.tensor([2.5], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    # Initial exposure value for comparison
    exp_averageAE = torch.tensor([2.5], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    exp_gradientAE = torch.tensor([2.5], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    exp_neuralAE = torch.tensor([2.5], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    for i_batch, batch in enumerate(tqdm(test_loader)):
        
        if batch is None:
            continue
        
        img_list, *data_blob = batch
        
        # Folder name
        experiment_name = test_loader.dataset.experiment_name = test_loader.dataset.get_experiment_name(i_batch)
        experiment_folder = os.path.join(base_output_folder, experiment_name)
        os.makedirs(experiment_folder, exist_ok=True)
        
        print(f"===== Saving results in {experiment_folder} for step {test_step} =====")
        

        left_hdr, right_hdr, left_next_hdr, right_next_hdr, focal_length, base_line, points, hdr_denom1, hdr_denom2  = [x.cuda() for x in data_blob]
        
                
        #^ Our method inference
        # Dual exposure with 2 frame
        with torch.no_grad():
            # With stereo exposure control module
            fused_disparity, original_img_list, captured_rand_img_list, captured_adj_img_list, exp1, exp2, fmap_list, mask_list, flow_L = model(
                left_hdr, right_hdr, left_next_hdr, right_next_hdr, initial_exp1, initial_exp2)
        
        
        # Update exposure value
        initial_exp1 , initial_exp2 = exp1, exp2
        
        
        #^ AverageAE method inference
        with torch.no_grad():
            averageAE_disp, exp_averageAE_new, _, _, ldr_left_cap_averageAE = averageAE_raft(left_hdr, right_hdr, exp_averageAE)
        #update exposure 
        exp_averageAE = normalize_exposure_shape(exp_averageAE_new)
        
        #^ GradientAE method inference
        with torch.no_grad():
            gradientAE_disp, exp_gradientAE_new, _, _, ldr_left_cap_gradientAE = gradientAE_raft(left_hdr, right_hdr, exp_gradientAE)
        # update exposure
        exp_gradientAE = normalize_exposure_shape(exp_gradientAE_new)
        
        #^ NeuralAE method inference
        with torch.no_grad():
            neuralAE_disp, exp_neuralAE_new, _, _, ldr_left_cap_neuralAE = neuralAE_raft(left_hdr, right_hdr, exp_neuralAE)
        # updatae exposure
        exp_neuralAE = normalize_exposure_shape(exp_neuralAE_new)
        
        
        # # Exposure logging
        # writer.add_scalars("Exposures", {'Expsoure1' : initial_exp1.item(), 
        #                                  'Exposure2' : initial_exp2.item(), 
        #                                 #  'Exposure_average' : exp_averageAE.item(),
        #                                 #  'Exposure_gradient' : exp_gradientAE.item(),
        #                                 #  'Exposure_nae' : exp_neuralAE.item(),
        #                                  }, test_step)
        
        # # Exposure 텍스트 파일에 저장
        # exposure_log_path = os.path.join(experiment_folder, f"exposure_values_step{test_step}.txt")
        # with open(exposure_log_path, 'w') as f:
        #     f.write(f"Exposure1: {initial_exp1.item()}\n")
        #     f.write(f"Exposure2: {initial_exp2.item()}\n")
            # f.write(f"Exposure_average: {exp_averageAE.item()}\n")
            # f.write(f"Exposure_gradient: {exp_gradientAE.item()}\n")
            # f.write(f"Exposure_nae: {exp_neuralAE.item()}\n")


        
        #^ MAE loss
        mae_loss_dual_exp, rmse_loss_dual_exp = mae_rmse_loss_lidar(fused_disparity[-1], points, focal_length, base_line)
        mae_loss_single_averageAE, rmse_loss_single_averageAE  = mae_rmse_loss_lidar(averageAE_disp[-1], points, focal_length, base_line)
        mae_loss_single_gradientAE, rmse_loss_single_gradientAE  = mae_rmse_loss_lidar(gradientAE_disp[-1], points, focal_length, base_line)
        mae_loss_single_neuralAE, rmse_loss_single_neuralAE = mae_rmse_loss_lidar(neuralAE_disp[-1], points, focal_length, base_line)

        
        # MAE loss 텍스트 파일에 저장
        experiment_folder = os.path.join(base_output_folder, experiment_name)
        depth_mae_path = os.path.join(experiment_folder, f"depth_loss_step{test_step}.txt")
        with open(depth_mae_path, 'w') as f:
            f.write(f"DualAE_depth_mae: {mae_loss_dual_exp.item()}\n")
            f.write(f"AverageAE_depth_mae: {mae_loss_single_averageAE.item()}\n")
            f.write(f"GradientAE_depth_mae: {mae_loss_single_gradientAE.item()}\n")
            f.write(f"NeuralAE_depth_mae: {mae_loss_single_neuralAE.item()}\n")

        # # RMSE loss 텍스트 파일에 저장
        # experiment_folder = os.path.join(base_output_folder, experiment_name)
        # depth_rmse_path = os.path.join(experiment_folder, f"depth_loss_rmse_step{test_step}.txt")
        # with open(depth_rmse_path, 'w') as f:
        #     f.write(f"DualAE_depth_rmse: {rmse_loss_dual_exp.item()}\n")
        #     f.write(f"AverageAE_depth_rmse: {rmse_loss_single_averageAE.item()}\n")
        #     f.write(f"GradientAE_depth_rmse: {rmse_loss_single_gradientAE.item()}\n")
        #     f.write(f"NeuralAE_depth_rmse: {rmse_loss_single_neuralAE.item()}\n")

        
        # mae_loss_dual_list.append(mae_loss_dual_exp)
        # mae_loss_average_list.append(mae_loss_single_averageAE)
        # mae_loss_gradient_list.append(mae_loss_single_gradientAE)
        # mae_loss_neural_list.append(mae_loss_single_neuralAE)

        # rmse_loss_dual_list.append(rmse_loss_dual_exp)
        # rmse_loss_average_list.append(rmse_loss_single_averageAE)
        # rmse_loss_gradient_list.append(rmse_loss_single_gradientAE)
        # rmse_loss_neural_list.append(rmse_loss_single_neuralAE)
        
        # if mae_loss_dual_exp > mae_loss_single_exp1 + mae_loss_single_exp2:
        #     print(img_list)
        
        #* Logging
        vmin, vmax = 0, 30
        # writer.add_scalar("Fused disparity loss", disparity_loss.item(), test_step)
        # writer.add_scalar("Fixed disparity loss", fixed_disparity_loss.item(), test_step)
        # writer.add_scalars("LOSS comparison", {'Fused disparity loss' : disparity_loss.item(), 'Fixed_disparity_loss' : fixed_disparity_loss.item()}, test_step)
        
        # MAE loss logging with different method
        writer.add_scalars("MAE", {'ADEC' : mae_loss_dual_exp, 
        'Single_averageAE' : mae_loss_single_averageAE, 
        'Single_gradientAE' : mae_loss_single_gradientAE,
        'Single_neuralAE' : mae_loss_single_neuralAE,
        }, test_step)

    

        # writer.add_scalar("Expsoure gap", abs(exp1.item()-exp2.item()), test_step)
        # writer.add_scalar("Dynamic range", hdr_dynamic_range(hdr_img), test_step)
        # Dynamic range histogram
        # visualize_hdr_dr(writer, test_step, hdr_img)
        
        # Save images into different subfolders for each situation
        situations = {
            "DualAE_fused_disp": fused_disparity[-1],
            "AverageAE_disp": averageAE_disp[-1],
            "Gradient_disp": gradientAE_disp[-1],
            "NeuralAE_disp" : neuralAE_disp[-1],
        }
        experiment_folder = os.path.join(base_output_folder, experiment_name)
        # depth_rmse_path = os.path.join(experiment_folder, f"depth_loss_rmse_step{test_step}.txt")

        for label, disp_map in situations.items():
            folder = os.path.join(base_output_folder, experiment_name, label)
            save_disp_as_svg(disp_map, folder, test_step, label, vmin, vmax)
            # Depth map 변환 및 저장
            depth_map = disparity_to_depth(disp_map, base_line, focal_length)
            save_depth_as_svg(depth_map, folder, test_step, label, vmin, 50000)

        # LiDAR point Save
        lidar_path = os.path.join(base_output_folder, experiment_name, 'LiDAR')
        os.makedirs(lidar_path, exist_ok=True)  # 폴더가 없는 경우 생성

        # 파일명을 포함하여 저장 경로 지정
        fig_path = os.path.join(lidar_path, f"lidar_points_step{test_step}.svg")

        # LiDAR 포인트 플롯 생성 및 저장
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
            # "AverageAE_img" : ldr_left_cap_averageAE[0],
            # "GradientAE_img" : ldr_left_cap_gradientAE[0],
            # "NeuralAE_img" : ldr_left_cap_neuralAE[0] 
        }
        for label, cap_img in cap_situations.items():
            folder = os.path.join(base_output_folder, experiment_name, label)
            save_image(cap_img, folder, test_step, label)
            
            
        #* Logging
        writer.add_image('Disp/ADEC', visualize_disparity_with_colorbar(fused_disparity[-1], vmin, vmax), test_step)
        writer.add_image('Disp/AverageAE', visualize_disparity_with_colorbar(averageAE_disp[-1], vmin, vmax), test_step)
        writer.add_image('Disp/GradientAE', visualize_disparity_with_colorbar(gradientAE_disp[-1], vmin, vmax), test_step)
        writer.add_image('Disp/NeuralAE', visualize_disparity_with_colorbar(neuralAE_disp[-1], vmin, vmax), test_step)
        
        # Visualize captured image
        writer.add_image('Captured(T)/hdr_left_frame1', original_img_list[0][0], test_step)
        writer.add_image('Captured(T)/hdr_left_frame2', original_img_list[1][0], test_step)
        writer.add_image('Captured(T)/hdr_left_frame1_tonemapped', original_img_list[0][0]**(1/2.2), test_step)
        writer.add_image('Captured(T)/hdr_left_frame2_tonemapped', original_img_list[1][0]**(1/2.2), test_step)
        writer.add_image('Captured(T)/img1_adj_left', captured_adj_img_list[0][0], test_step)
        writer.add_image('Captured(T)/img2_adj_left', captured_adj_img_list[1][0], test_step)
        writer.add_image('Captured(T)/img_AverageAE_left', ldr_left_cap_averageAE[0][0], test_step)
        writer.add_image('Captured(T)/img_Gradient_left', ldr_left_cap_gradientAE[0][0], test_step)
        writer.add_image('Captured(T)/img_NeuralAE_left', ldr_left_cap_neuralAE[0][0], test_step)

        
        test_step += 1
        
        
    #     #* Save Image
    #     save_image((original_img_list[0][0]**(1/2.2)).cpu(), os.path.join(base_output_folder,'HDR_Left_tonemapped'), test_step, 'HDR_Left_tonemapped')
    #     save_image((original_img_list[0][0]).cpu(), os.path.join(base_output_folder,'HDR_Left'), test_step, 'HDR_Left')
    #     save_image(ldr_right_cap_fixed[0].cpu(), os.path.join(base_output_folder,'LDR_fixed_left'), test_step, 'LDR_fixed_left')
    #     save_image(captured_adj_img_list[0][0].cpu(), os.path.join(base_output_folder,'LDR_exp_f_left'), test_step, 'LDR_exp_f_left')
    #     save_image(captured_adj_img_list[1][0].cpu(), os.path.join(base_output_folder,'LDR_exp_s_left'), test_step, 'LDR_exp_s_left')
        
        
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
    parser.add_argument('--name', default='SAEC_test_real_comparison', help="name your experiment")
    parser.add_argument('--restore_ckpt', default='checkpoints/5000_disp_fusion_mask_finetuned_gru.pth', help="restore checkpoint")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--train_datasets', type=str, nargs='+', default=['test_real'])
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