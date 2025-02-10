import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.transforms import ToTensor

from torch.utils.tensorboard import SummaryWriter
from core.saec import *
from core.utils.simulate import *
from core.raft_warp_stereo import RAFTStereoFusion
from core.utils.read_utils import prepare_inputs_custom
from core.utils.utils import InputPadder
from core.utils.mask import soft_binary_threshold_batch

from ptlflow import get_model
from ptlflow.utils import flow_utils
from ptlflow.utils.io_adapter import IOAdapter

from core.utils.display import *
from torch.cuda.amp import autocast

from core.update import BasicMultiUpdateBlock
from core.extractor import BasicEncoder, MultiBasicEncoder, ResidualBlock
from core.corr import CorrBlock1D, PytorchAlternateCorrBlock1D, CorrBlockFast1D, AlternateCorrBlock
from core.utils.utils import coords_grid, upflow8

import math

#* Feature warping pipeline

DEVICE = 'cuda'

writer = SummaryWriter('runs/disp_recon_model_feature_warp')

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass
        

# Quantize STE method
class QuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, n=8):
        # LDR image max value
        max_val = 2**n - 1
        
        # Quantize
        x_scaled = input * max_val
        x_clamped = torch.clamp(x_scaled, 0, max_val)   
        x_quantized = torch.round(x_clamped).to(torch.uint8)
        
        # Normalized to 0~1
        x_dequantized = x_quantized / max_val
        return x_dequantized

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


# RAFTStereo with attention feature fusion
class RAFTStereoFusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        context_dims = args.hidden_dims
        
        self.flow_model = get_model('rapidflow_it6', pretrained_ckpt='kitti').to(DEVICE)
        self.flow_model.train()
        
        # Unfreeze pretrained model parameter 
        # for param in self.flow_model.parameters():
        #     param.requires_grad = True
        
        # # Unfreeze the last few layers for fine-tuning
        # for param in list(self.flow_model.parameters())[-10:]:
        #     param.requires_grad = True

        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn=args.context_norm, downsample=args.n_downsample)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])

        # if args.shared_backbone:
        #     self.conv2 = nn.Sequential(
        #         ResidualBlock(128, 128, 'instance', stride=1),
        #         nn.Conv2d(128, 256, 3, padding=1))
        # else:
        #     self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', downsample=args.n_downsample)
            
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', downsample=args.n_downsample)
    
    # Compute optical flow on consecutive frame
    def compute_optical_flow_batch(self, model, image_pairs):
        device = next(model.parameters()).device if next(model.parameters(), None) is not None else 'cuda'
        
        all_flows = []
        for left_image, right_image in image_pairs:

            left_image = left_image.to(device)
            right_image = right_image.to(device)
            inputs = prepare_inputs_custom([left_image, right_image])
            
            inputs = {k: v.to(device) for k, v in inputs.items()} 

            with torch.no_grad():
                predictions = model(inputs)

            # Extract the flows
            flows = predictions['flows']
            all_flows.append(flows)
        
        return torch.cat(all_flows, dim=0).squeeze(1)
    
    # Resize flow to the feature map size
    def resize_flow(self, flow, target_size):
        resized_flow = F.interpolate(flow, size=target_size, mode='bilinear', align_corners=False)
        
        scale_factor_h = target_size[0] / flow.size(2)
        scale_factor_w = target_size[1] / flow.size(3)
        
        resized_flow[:, 0, :, :] *= scale_factor_w 
        resized_flow[:, 1, :, :] *= scale_factor_h  
        
        return resized_flow
    
    def resize_soft_binary_mask(self, mask, feature_map):
    # mask: torch.Size([B, 1, H_mask, W_mask])
    # feature_map: torch.Size([B, C, H_feature, W_feature])

        target_size = feature_map.shape[-2:]  # [H_feature, W_feature]
        
        resized_mask = F.interpolate(mask, size=target_size, mode='bilinear', align_corners=False)
        
        
        return resized_mask
    
    # Warp feature map using resized flow
    def warp_feature_map(self, feature_map, flow):
        n, c, h, w = feature_map.size()
        resized_flow = self.resize_flow(flow, target_size=(h, w))
        
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w), indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=0).float()  # [2, h, w]
        grid = grid.unsqueeze(0).repeat(n, 1, 1, 1).to(feature_map.device)  # [n, 2, h, w]
        
        grid = grid + resized_flow
        
        grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :].clone() / (w - 1) - 1.0
        grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :].clone() / (h - 1) - 1.0
        
        grid = grid.permute(0, 2, 3, 1)  # [n, h, w, 2]
        
        warped_feature_map = F.grid_sample(feature_map, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    
        return warped_feature_map


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape

        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)

        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, D, H, W = flow.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor*H, factor*W)
    
    def normalize_feature_map(self, fmap):
        mean = fmap.mean(dim=(2, 3), keepdim=True)
        std = fmap.std(dim=(2, 3), keepdim=True) + 1e-6  
        fmap_normalized = (fmap - mean) / std
        return fmap_normalized
    
    def compute_disparity(self, fmap1_input, fmap2_input, net_list_input, inp_list_input, iters, flow_init):
        if self.args.corr_implementation == "reg":  # 기본값
            corr_block = CorrBlock1D
            fmap1_input, fmap2_input = fmap1_input.float(), fmap2_input.float()
        elif self.args.corr_implementation == "alt":  # 메모리 효율적인 버전
            corr_block = PytorchAlternateCorrBlock1D
            fmap1_input, fmap2_input = fmap1_input.float(), fmap2_input.float()
        elif self.args.corr_implementation == "reg_cuda":  # 빠른 버전
            corr_block = CorrBlockFast1D
        elif self.args.corr_implementation == "alt_cuda":  # 빠른 대안 버전
            corr_block = AlternateCorrBlock

        corr_fn = corr_block(fmap1_input, fmap2_input, radius=self.args.corr_radius, num_levels=self.args.corr_levels)

        coords0, coords1 = self.initialize_flow(net_list_input[0])

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        net_list = net_list_input  # 제공된 net_list 사용
        inp_list = inp_list_input  # 제공된 inp_list 사용
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # Correlation Volume 인덱싱
            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                if self.args.n_gru_layers == 3 and self.args.slow_fast_gru:  # 저해상도 GRU 업데이트
                    net_list = self.update_block(net_list, inp_list, iter32=True, iter16=False, iter08=False, update=False)
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru:  # 저해상도 및 중해상도 GRU 업데이트
                    net_list = self.update_block(net_list, inp_list, iter32=self.args.n_gru_layers == 3, iter16=True, iter08=False, update=False)
                net_list, up_mask, delta_flow = self.update_block(net_list, inp_list, corr, flow,
                                                                iter32=self.args.n_gru_layers == 3, iter16=self.args.n_gru_layers >= 2)

            # 스테레오 모드에서 flow를 에피폴라 라인에 투영
            delta_flow[:, 1] = 0.0

            # Flow 업데이트
            coords1 = coords1 + delta_flow

            # 예측값 업샘플링
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_up = flow_up[:, :1]

            flow_predictions.append(flow_up)

        return flow_predictions


    def forward(self, image1, image2, image1_next, image2_next, exp_h, exp_l, iters=32, flow_init=None, test_mode=False):
        
        #* Noise modeling
        phi_l_exph = ImageFormation(image1, exp_h, device='cuda')
        phi_r_exph = ImageFormation(image2, exp_h, device='cuda')
        phi_l_expl = ImageFormation(image1_next, exp_l, device='cuda')
        phi_r_expl = ImageFormation(image2_next, exp_l, device='cuda')
        
        image1 = QuantizeSTE.apply(phi_l_exph.noise_modeling(), 8)
        image2 = QuantizeSTE.apply(phi_r_exph.noise_modeling(), 8)
        image1_next = QuantizeSTE.apply(phi_l_expl.noise_modeling(), 8)
        image2_next = QuantizeSTE.apply(phi_r_expl.noise_modeling(), 8)
        
        # For logging simulated LDR images
        cap1, cap1_next = image1, image1_next
        cap2, cap2_next = image2, image2_next
        cap_img_list = [cap1, cap2, cap1_next, cap2_next]
        
        #* Calculate softbinary mask
        print(f"In disp_recon_model_ablation.py : {image1.min()}, {image1.max()}")
        image1_mask = soft_binary_threshold_batch(image1)
        image2_mask = soft_binary_threshold_batch(image2)
        image1_next_mask = soft_binary_threshold_batch(image1_next)
        image2_next_mask = soft_binary_threshold_batch(image2_next)
              
        # Normalzie input image
        image1 = (2 * image1 - 1.0).contiguous()
        image2 = (2 * image2 - 1.0).contiguous()
        image1_next = (2 * image1_next - 1.0).contiguous()
        image2_next = (2 * image2_next - 1.0).contiguous()
        
        #* Extract features
        fmap1, fmap2 = self.fnet([image1, image2])
        fmap1_next, fmap2_next = self.fnet([image1_next, image2_next])
        
        # Feature map normalize
        # fmap1 = self.normalize_feature_map(fmap1)
        # fmap2 = self.normalize_feature_map(fmap2)
        # fmap1_next = self.normalize_feature_map(fmap1_next)
        # fmap2_next = self.normalize_feature_map(fmap2_next)
        # print(f"In diap_recon_model.py : image1_size : {image1.shape}, image2_size : {image2.shape}, fmap1_size : {fmap1.shape}, fmap1_next{fmap1_next.shape}")       
         
        with autocast(enabled=self.args.mixed_precision):
            # Compute flow between Frame1 and Frame2
            img_pair_left = [(image1[i], image1_next[i]) for i in range(image1.shape[0])]
            img_pair_right = [(image2[i], image2_next[i]) for i in range(image2.shape[0])]
            flow_left = self.compute_optical_flow_batch(self.flow_model, img_pair_left)
            flow_right = self.compute_optical_flow_batch(self.flow_model, img_pair_right)
            
        warped_fmap_left = self.warp_feature_map(fmap1_next, flow_left)
        warped_fmap_right = self.warp_feature_map(fmap2_next, flow_right)
        
        #* Resizing mask
        resized_image1_mask = self.resize_soft_binary_mask(image1_mask, fmap1) 
        resized_image2_mask = self.resize_soft_binary_mask(image2_mask, fmap2) 
        image1_next_mask = self.resize_soft_binary_mask(image1_next_mask, fmap1)
        image2_next_mask = self.resize_soft_binary_mask(image2_next_mask, fmap2)
        # Warp mask
        resized_image1_next_mask = self.warp_feature_map(image1_next_mask, flow_left) 
        resized_image2_next_mask = self.warp_feature_map(image2_next_mask, flow_right) 
        
        # print(f"Mask value : {resized_image1_mask.min()} {resized_image1_mask.max()}")
        # print(f"Mask shape : {resized_image1_mask.shape}")
        # print(f"Mask2 value : {resized_image1_next_mask.min()} {resized_image1_mask.max()}")
        # print(f"Mask2 shape : {resized_image1_next_mask.shape}")
                
        # *Feature Fusion with attention model
        # fused_fmap1 = self.attention_fusion(fmap1, warped_fmap_left)
        # fused_fmap2 = self.attention_fusion(fmap2, warped_fmap_right)
        
        # *Feature Fusion with confidence mask
        epsilon = 1e-6
        mask_sum1 = resized_image1_mask + resized_image1_next_mask
        mask_sum2 = resized_image2_mask + resized_image2_next_mask
        mask_sum_safe1 = torch.where(mask_sum1 == 0, epsilon * torch.ones_like(mask_sum1), mask_sum1)
        mask_sum_safe2 = torch.where(mask_sum2 == 0, epsilon * torch.ones_like(mask_sum2), mask_sum2)
        fused_fmap1 = (resized_image1_mask * fmap1 + resized_image1_next_mask * warped_fmap_left)/mask_sum_safe1
        fused_fmap2 = (resized_image2_mask * fmap2 + resized_image2_next_mask * warped_fmap_right)/mask_sum_safe2
        
        # print("fused_fmap1 shape:", fused_fmap1.shape)
        # print("fused_fmap2 shape:", fused_fmap2.shape)
        
        # For ablation study, each disparity map list
        flow_predictions_fused = []
        flow_predictions_fmap1 = []
        flow_predictions_fmap1_next = []
                
        # run the context network
        # 첫 번째 이미지의 Context Network 실행
        with autocast(enabled=self.args.mixed_precision):
            cnet_list_1 = self.cnet(image1, num_layers=self.args.n_gru_layers)
            net_list_1 = [torch.tanh(x[0]) for x in cnet_list_1]
            inp_list_1 = [torch.relu(x[1]) for x in cnet_list_1]
            inp_list_1 = [list(conv(i).split(split_size=conv.out_channels // 3, dim=1)) 
                        for i, conv in zip(inp_list_1, self.context_zqr_convs)]
        
        # 두 번째 이미지의 Context Network 실행 (필요한 경우)
        with autocast(enabled=self.args.mixed_precision):
            cnet_list_2 = self.cnet(image1_next, num_layers=self.args.n_gru_layers)
            net_list_2 = [torch.tanh(x[0]) for x in cnet_list_2]
            inp_list_2 = [torch.relu(x[1]) for x in cnet_list_2]
            inp_list_2 = [list(conv(i).split(split_size=conv.out_channels // 3, dim=1)) 
                        for i, conv in zip(inp_list_2, self.context_zqr_convs)]

        flow_predictions_fused = self.compute_disparity(fused_fmap1, fused_fmap2, net_list_1, inp_list_1, iters, flow_init)
        
        flow_predictions_fmap1 = self.compute_disparity(fmap1, fmap2, net_list_1, inp_list_1, iters, flow_init)
        
        flow_predictions_fmap1_next = self.compute_disparity(fmap1_next, fmap2_next, net_list_2, inp_list_2, iters, flow_init)
        
        mask_list = [resized_image1_mask, image1_next_mask, resized_image1_next_mask]

        
        return (flow_predictions_fused, flow_predictions_fmap1, flow_predictions_fmap1_next), fmap1, fmap1_next, warped_fmap_left, flow_left, fused_fmap1, cap_img_list, mask_list
        