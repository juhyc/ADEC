import torch
from core.combine_model3 import CombineModel_wo_net
# from core.stereo_datasets3 import CARLASequenceDataset, fetch_dataloader
# from core.real_datasets import fetch_real_dataloader, RealDataset
from core.real_datasets_wolidar import fetch_real_dataloader
from core.utils.display import *
from core.raft_stereo import *
from core.utils.simulate import *
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import logging
import argparse
import torchvision.models as models
import torchmetrics
from tqdm import tqdm
from types import SimpleNamespace

import imageio


DEVICE = 'cuda'

writer = SummaryWriter('runs/test_sequence')

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
        image = (image * 255).numpy().astype(np.uint8)

    # 이미지가 C x H x W 형식이면 H x W x C 형식으로 변환
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))  # Convert (C, H, W) to (H, W, C)

    image_path = os.path.join(folder, f"{label}_step{frame_idx}.png")
    imageio.imwrite(image_path, image)


def create_video_from_frames(folder, output_path, fps=10, target_size=(800, 600)):
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

    
# For Raftstereo inference
args_raft = {
    'restore_ckpt': '/home/user/juhyung/SAEC/models/raftstereo-middlebury.pth',
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
    
    # Load pretrained raft-stereo
    raft_stereo = nn.DataParallel(RAFTStereo(args_raft), device_ids=[0])
    raft_stereo_ckpt = torch.load(args_raft.restore_ckpt)
    logging.info(f"Loading RAFT_stereo checkpoint... {args_raft.restore_ckpt}")
    raft_stereo.load_state_dict(raft_stereo_ckpt, strict=False)
    logging.info(f"Done RAFT_stereo loading checkpoint...{args.restore_ckpt}")
    raft_stereo.cuda()
    raft_stereo.eval()
    
    
    test_step = 0
    base_output_folder = 'test_results_real2'
    
    # Initial exposure value
    initial_exp1 = torch.tensor([1.0], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    initial_exp2 = torch.tensor([1.0], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    # Fixed exposure value
    fixed_exp1 = torch.tensor([1.0], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    fixed_exp2 = torch.tensor([1.0], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    for i_batch, batch in enumerate(tqdm(test_loader)):
        
        if batch is None:
            continue
        
        img_list, *data_blob = batch

        # left_hdr, right_hdr, left_next_hdr, right_next_hdr, lidar_disp, u, v  = [x.cuda() for x in data_blob]
        left_hdr, right_hdr, left_next_hdr, right_next_hdr  = [x.cuda() for x in data_blob]
        print(f"@@@@@@@@@@@@@@@@@@@@@@@@HDR image min,max : {left_hdr.min()}  ,{left_hdr.max()}")
        
        # Image simulation with fixed exposure
        phi_l_ldr_fixed = ImageFormation(left_hdr, fixed_exp1, device=DEVICE)
        phi_r_ldr_fixed = ImageFormation(right_hdr, fixed_exp2, device=DEVICE)
        ldr_left_cap_fixed = QuantizeSTE.apply(phi_l_ldr_fixed.noise_modeling(), 8)
        ldr_right_cap_fixed = QuantizeSTE.apply(phi_r_ldr_fixed.noise_modeling(), 8)
        
        # Our method inference
        # Dual exposure with 2 frame

        with torch.no_grad():
            # With stereo exposure control module
            print(f"In test_sequence.py : {initial_exp1} {initial_exp2}")

            fused_disparity, original_img_list, captured_rand_img_list, captured_adj_img_list, exp1, exp2, fmap_list, mask_list, flow_L = model(
                left_hdr, right_hdr, left_next_hdr, right_next_hdr, initial_exp1, initial_exp2)
            # Fused disparity map with single exposure1
            fused_disparity_exp1, _, _, _, _, _, _ , _, _= model(
                left_hdr, right_hdr, left_next_hdr, right_next_hdr, initial_exp1, initial_exp1)
            # Fused disparity map with single exposure2
            fused_disparity_exp2, _, _, _, _, _, _ , _, _= model(
                left_hdr, right_hdr, left_next_hdr, right_next_hdr, initial_exp2, initial_exp2)
        
        # Update exposure value
        initial_exp1 , initial_exp2 = exp1, exp2
        
        # Image simulation with exposure 1
        phi_l_ldr_exp1 = ImageFormation(left_hdr, exp1, device=DEVICE)
        phi_r_ldr_exp1 = ImageFormation(right_hdr, exp1, device=DEVICE)
        ldr_left_cap_exp1 = QuantizeSTE.apply(phi_l_ldr_exp1.noise_modeling(), 8)
        ldr_right_cap_exp1 = QuantizeSTE.apply(phi_r_ldr_exp1.noise_modeling(), 8)
        
        # Image simulation with exposure 2
        phi_l_ldr_exp2 = ImageFormation(left_hdr, exp2, device=DEVICE)
        phi_r_ldr_exp2 = ImageFormation(right_hdr, exp2, device=DEVICE)
        ldr_left_cap_exp2 = QuantizeSTE.apply(phi_l_ldr_exp2.noise_modeling(), 8)
        ldr_right_cap_exp2 = QuantizeSTE.apply(phi_r_ldr_exp2.noise_modeling(), 8)
        
        # Inference with existed raft-stereo
        # Single exposure with 1 frame
        with torch.no_grad():
            fixed_disparity = raft_stereo(ldr_left_cap_fixed*255, ldr_right_cap_fixed*255)
            exp1_disparity = raft_stereo(ldr_left_cap_exp1*255, ldr_right_cap_exp1*255)
            exp2_disparity = raft_stereo(ldr_left_cap_exp2*255, ldr_right_cap_exp2*255)
            # Psuedo GT -> HDR original image, disparity map 
            disp = raft_stereo(left_hdr*255, right_hdr*255)
            disp = disp[-1]
            valid = disp[-1] < 512
        
        # disparity_loss, _ = sequence_loss(fused_disparity, disp, valid)
        # fixed_disparity_loss, _ = sequence_loss(fixed_disparity, disp, valid)
        
        # fused_disparity_l2loss = l2_loss(fused_disparity[-1], disp, valid)
        # fixed_disparity_l2loss = l2_loss(fixed_disparity[-1], disp, valid)
        # exp1_disparity_l2loss = l2_loss(exp1_disparity[-1], disp, valid)
        # exp2_disparity_l2loss = l2_loss(exp2_disparity[-1], disp, valid)
        
        # fused_disparity_exp1_loss = l2_loss(fused_disparity_exp1[-1], disp, valid)
        # fused_disparity_exp2_loss = l2_loss(fused_disparity_exp2[-1], disp, valid)
        
        #* Logging
        vmin, vmax = 0, 25
        # writer.add_scalar("Fused disparity loss", disparity_loss.item(), test_step)
        # writer.add_scalar("Fixed disparity loss", fixed_disparity_loss.item(), test_step)
        # writer.add_scalars("LOSS comparison", {'Fused disparity loss' : disparity_loss.item(), 'Fixed_disparity_loss' : fixed_disparity_loss.item()}, test_step)
        
        # Exposure logging
        writer.add_scalars("Exposures", {'Expsoure1' : exp1.item(), 'Exposure2' : exp2.item()}, test_step)
        writer.add_scalar("Expsoure gap", abs(exp1.item()-exp2.item()), test_step)
        # writer.add_scalar("Dynamic range", hdr_dynamic_range(hdr_img), test_step)
        # Dynamic range histogram
        # visualize_hdr_dr(writer, test_step, hdr_img)
        
        # Save images into different subfolders for each situation
        situations = {
            "Dual_adj_fused_disp": fused_disparity[-1],
            "Single_fixed_disp": fixed_disparity[-1],
            "Single_adj_f_disp": exp1_disparity[-1],
            "Single_adj_s_disp": exp2_disparity[-1],
            "Dual_adj_f_fused_disp": fused_disparity_exp1[-1],
            "Dual_adj_s_fused_disp": fused_disparity_exp2[-1],
            "GT_disp": disp
        }


        for label, disp_map in situations.items():
            folder = os.path.join(base_output_folder, label)
            save_disp_as_image(visualize_disparity_with_colorbar(disp_map, vmin, vmax, title=label), folder, test_step, label)
            
        
        # Loss logging
        # writer.add_scalar("Fused disparity l2loss", fused_disparity_l2loss.item(), test_step)
        # writer.add_scalar("Fixed disparity l2loss", fixed_disparity_l2loss.item(), test_step)
        # writer.add_scalar("Exp1 disparity l2loss", exp1_disparity_l2loss.item(), test_step)
        # writer.add_scalar("Exp2 disparity l2loss", exp2_disparity_l2loss.item(), test_step)
        # writer.add_scalars("l2LOSS comparison", {'Fused disparity l2loss' : fused_disparity_l2loss.item(), 'Fixed_disparity_l2loss' : fixed_disparity_l2loss.item()}, test_step)
        # writer.add_scalars("l2LOSS comparison2", {'Fused disparity l2loss' : fused_disparity_l2loss.item(), 'Fixed_disparity_l2loss' : fixed_disparity_l2loss.item(),
        #                                          'Exp1 disparity l2loss' : exp1_disparity_l2loss.item(), 'Exp2 disparity l2loss' : exp2_disparity_l2loss.item()}, test_step)
        # writer.add_scalars("L2Loss_comparison_fused_disp", {'Fused disparity' : fused_disparity_l2loss.item(), 'Fused disparity exp1' : fused_disparity_exp1_loss.item(), 'Fused disparity exp2' : fused_disparity_exp2_loss.item()}, test_step)
        # print(f"=====LOSS : fused{fused_disparity_l2loss.item():.4f}, EXP1:{fused_disparity_exp1_loss.item():.4f}, EXP2:{fused_disparity_exp2_loss.item():.4f}=====")
        
        # Disparity logging
        writer.add_image('Test/Fused_disparity', visualize_disparity_with_colorbar(fused_disparity[-1], vmin, vmax), test_step)
        writer.add_image('Test/Fixed_disparity', visualize_disparity_with_colorbar(fixed_disparity[-1], vmin, vmax), test_step)
        writer.add_image('Test/Exp1_disparity', visualize_disparity_with_colorbar(exp1_disparity[-1], vmin, vmax), test_step)
        writer.add_image('Test/Exp2_disparity', visualize_disparity_with_colorbar(exp2_disparity[-1], vmin, vmax), test_step)
        
        writer.add_image('Test/Fused_disparity_exp1', visualize_disparity_with_colorbar(fused_disparity_exp1[-1], vmin, vmax), test_step)
        writer.add_image('Test/Fused_disparity_exp2', visualize_disparity_with_colorbar(fused_disparity_exp2[-1], vmin, vmax), test_step)
        writer.add_image('Test/GT_disparity', visualize_disparity_with_colorbar(disp, vmin, vmax), test_step)
        
        
        # Visualize captured image
        writer.add_image('Captured(T)/hdr_left_frame1', original_img_list[0][0], test_step)
        writer.add_image('Captured(T)/hdr_left_frame2', original_img_list[1][0], test_step)
        writer.add_image('Captured(T)/hdr_left_frame1_tonemapped', original_img_list[0][0]**(1/2.2), test_step)
        writer.add_image('Captured(T)/hdr_left_frame2_tonemapped', original_img_list[1][0]**(1/2.2), test_step)
        # writer.add_image('Captured(T)/img1_rand_left', captured_rand_img_list[0][0], test_step)
        # writer.add_image('Captured(T)/img2_rand_left', captured_rand_img_list[1][0], test_step)
        writer.add_image('Captured(T)/img1_fixed_left', ldr_left_cap_fixed[0], test_step)
        writer.add_image('Captured(T)/img2_fixed_left', ldr_right_cap_fixed[0], test_step)
        writer.add_image('Captured(T)/img1_adj_left', captured_adj_img_list[0][0], test_step)
        writer.add_image('Captured(T)/img2_adj_left', captured_adj_img_list[1][0], test_step)

        # Feature map logging
        log_multiple_feature_map_with_colorbar(writer, fmap_list[0], "Train/Unwarped_Fmap1_L", test_step, num_channels=1)
        log_multiple_feature_map_with_colorbar(writer, fmap_list[1], "Train/Unwarped_Fmap2_L", test_step, num_channels=1)
        log_multiple_feature_map_with_colorbar(writer, fmap_list[3], "Train/Fused_Fmap", test_step, num_channels=1)
   
        visualize_tensor_with_DR(writer, test_step, original_img_list[0], original_img_list[2], original_img_list[3])
                
        
        # # img1_adj_left 히스토그램 logging

        # writer.add_image('Histogram/img1_adj_left', plot_histogram(captured_adj_img_list[0][0], title="Histogram First exp"), test_step)

        # # img2_adj_left 히스토그램 logging
        # writer.add_image('Histogram/img2_adj_left', plot_histogram(captured_adj_img_list[1][0], title="Histogram Second exp"), test_step)
        
        
        test_step += 1
        
        
        #* Save Image
        save_image((original_img_list[0][0]**(1/2.2)).cpu(), os.path.join(base_output_folder,'HDR_Left_tonemapped'), test_step, 'HDR_Left_tonemapped')
        save_image((original_img_list[0][0]).cpu(), os.path.join(base_output_folder,'HDR_Left'), test_step, 'HDR_Left')
        save_image(ldr_right_cap_fixed[0].cpu(), os.path.join(base_output_folder,'LDR_fixed_left'), test_step, 'LDR_fixed_left')
        save_image(captured_adj_img_list[0][0].cpu(), os.path.join(base_output_folder,'LDR_exp_f_left'), test_step, 'LDR_exp_f_left')
        save_image(captured_adj_img_list[1][0].cpu(), os.path.join(base_output_folder,'LDR_exp_s_left'), test_step, 'LDR_exp_s_left')
        
        
    #* Create Video
    # After the test loop, create videos from saved frames
    for situation in situations.keys():
        situation_folder = os.path.join(base_output_folder, situation)
        video_output_path = os.path.join(base_output_folder, f"{situation}.mp4")
        create_video_from_frames(situation_folder, video_output_path, fps=10)
        
    create_video_from_frames(os.path.join(base_output_folder,'HDR_Left_tonemapped'), os.path.join(base_output_folder,'HDR_Left_tonemapped','HDR_Left.mp4'), fps=10)
    create_video_from_frames(os.path.join(base_output_folder,'HDR_Left'), os.path.join(base_output_folder,'HDR_Left','HDR_Left.mp4'), fps=10)
    create_video_from_frames(os.path.join(base_output_folder,'LDR_fixed_left'), os.path.join(base_output_folder,'LDR_fixed_left', 'LDR_fixed_left.mp4'), fps=10)
    create_video_from_frames(os.path.join(base_output_folder,'LDR_exp_f_left'), os.path.join(base_output_folder,'LDR_exp_f_left', 'LDR_exp_f_left.mp4'), fps=10)
    create_video_from_frames(os.path.join(base_output_folder,'LDR_exp_s_left'), os.path.join(base_output_folder,'LDR_exp_s_left', 'LDR_exp_s_left.mp4'), fps=10)
    
    

# Example usage of test_sequence function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing script")
    parser.add_argument('--name', default='SAEC_test_real', help="name your experiment")
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