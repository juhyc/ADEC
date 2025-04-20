import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.transforms import ToTensor

from torch.utils.tensorboard import SummaryWriter
from core.saec import *
from core.utils.simulate import *
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

writer = SummaryWriter('runs/disp_recon_dual')

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
class RAFTStereoFusion_refine(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device(args.device)
        context_dims = args.hidden_dims
        self.valid_iters = args.valid_iters
        
        # Load optical flow model
        self.flow_model = get_model('gmflow+', pretrained_ckpt='mix').to(self.device)

        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn=args.context_norm, downsample=args.n_downsample).to(self.device)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims).to(self.device)

        self.context_zqr_convs = nn.ModuleList(
            [nn.Conv2d(context_dims[i], args.hidden_dims[i] * 3, 3, padding=3 // 2).to(self.device)
             for i in range(self.args.n_gru_layers)]
        )
        
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', downsample=args.n_downsample).to(self.device)
    
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
        resized_flow = F.interpolate(flow, size=target_size, mode='bilinear', align_corners=True)
        
        scale_factor_h = target_size[0] / flow.size(2)
        scale_factor_w = target_size[1] / flow.size(3)
        
        resized_flow[:, 0, :, :] *= scale_factor_w 
        resized_flow[:, 1, :, :] *= scale_factor_h  
        
        return resized_flow
    
    def resize_soft_binary_mask(self, mask, feature_map):
        # mask: torch.Size([B, 1, H_mask, W_mask])
        # feature_map: torch.Size([B, C, H_feature, W_feature])

        target_size = feature_map.shape[-2:]  # [H_feature, W_feature]
        resized_mask = F.interpolate(mask, size=target_size, mode='bilinear', align_corners=True)
        
        return resized_mask
    
    # Feature map warping
    def warp_feature_map(self, feature_map, flow):
        n, c, h, w = feature_map.size()
        resized_flow = self.resize_flow(flow, target_size=(h, w))
        
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h, dtype=torch.float16, device=feature_map.device), 
            torch.arange(0, w, dtype=torch.float16, device=feature_map.device), 
            indexing='ij'
        )
        grid = torch.stack((grid_x, grid_y), dim=0)  # [2, h, w]
        grid = grid.unsqueeze(0).repeat(n, 1, 1, 1)  # [n, 2, h, w]
        
        grid = grid.to(torch.float32) + resized_flow

        grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :].clone() / (w - 0.5) - 1.0
        grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :].clone() / (h - 0.5) - 1.0
        
        grid = grid.permute(0, 2, 3, 1)  # [n, h, w, 2]
        
        warped_feature_map = F.grid_sample(feature_map, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        return warped_feature_map
    
    def mask_based_feature_fusion(self, fmap1, fmap2, mask1, mask2, fmap2_original, epsilon=1e-2):
        mask_sum = mask1 + mask2
        mask_sum_safe = torch.where(mask_sum == 0, epsilon * torch.ones_like(mask_sum), mask_sum)
        fused_fmap = (mask1 * fmap1 + mask2 * fmap2) / mask_sum_safe

        # Case 3 handling: Avoid overlapping addition
        warping_failure_mask = (mask1 == 0) & (mask2 > 0)
        
        # Update only where fused_fmap has near-zero values to avoid double addition
        near_zero_mask = (fused_fmap.abs() < epsilon).float()
        update_mask = (~warping_failure_mask) * near_zero_mask
        
        fused_fmap += update_mask * fmap2_original

        return fused_fmap, near_zero_mask, warping_failure_mask, update_mask

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.InstanceNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape

        coords0 = coords_grid(N, H, W).to(img.device)
        # coords1 = coords_grid(N, H, W).to(img.device)
        coords1 = coords0 + torch.normal(0, 0.1, coords0.shape).to(coords0.device)

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

    def forward(self, image1, image2, image1_next, image2_next, flow_init=None, test_mode=False):
        
        iters = self.valid_iters
        
        image1, image2 = image1.to(self.device), image2.to(self.device)
        image1_next, image2_next = image2_next.to(self.device), image2_next.to(self.device)
        
        # # # #*##### Noise modeling for finetuning#####
        # phi_l_exph = ImageFormation(image1, exp_h, device='cuda')   
        # phi_r_exph = ImageFormation(image2, exp_h, device='cuda')
        # phi_l_expl = ImageFormation(image1_next, exp_l, device='cuda')
        # phi_r_expl = ImageFormation(image2_next, exp_l, device='cuda')
        
        # image1 = QuantizeSTE.apply(phi_l_exph.noise_modeling(), 8)
        # image2 = QuantizeSTE.apply(phi_r_exph.noise_modeling(), 8)
        # image1_next = QuantizeSTE.apply(phi_l_expl.noise_modeling(), 8)
        # image2_next = QuantizeSTE.apply(phi_r_expl.noise_modeling(), 8)
        #*########################################
        
        # For logging simulated LDR images
        cap1, cap1_next = image1, image1_next
        cap2, cap2_next = image2, image2_next
        cap_img_list = [cap1, cap2, cap1_next, cap2_next]
        
        #* Calculate softbinary mask 
        image1_mask = soft_binary_threshold_batch(image1)
        image2_mask = soft_binary_threshold_batch(image2)
        image1_next_mask = soft_binary_threshold_batch(image1_next)
        image2_next_mask = soft_binary_threshold_batch(image2_next)
        
        # For logging
        temp_mask1 = image1_next_mask

        # Normalize input image
        image1 = (2 * image1 - 1.0).contiguous()
        image2 = (2 * image2 - 1.0).contiguous()
        image1_next = (2 * image1_next - 1.0).contiguous()
        image2_next = (2 * image2_next - 1.0).contiguous()
        
        #* Extract features
        fmap1, fmap2 = self.fnet([image1, image2])
        fmap1_next, fmap2_next = self.fnet([image1_next, image2_next])
        
        # * Compute flow between Frame1 and Frame2
        with autocast(enabled=self.args.mixed_precision):
            img_pair_left = [(image1[i], image1_next[i]) for i in range(image1.shape[0])]
            img_pair_right = [(image2[i], image2_next[i]) for i in range(image2.shape[0])]
            flow_left = self.compute_optical_flow_batch(self.flow_model, img_pair_left)
            flow_right = self.compute_optical_flow_batch(self.flow_model, img_pair_right)

        #* Warp feature maps using flow
        warped_fmap_left = self.warp_feature_map(fmap1_next, flow_left)
        warped_fmap_right = self.warp_feature_map(fmap2_next, flow_right)
            
        #* Resizing mask
        # For logging
        warped_image1_mask = self.warp_feature_map(image1_next_mask, flow_left)
        #####
        resized_image1_mask = self.resize_soft_binary_mask(image1_mask, fmap1) 
        resized_image2_mask = self.resize_soft_binary_mask(image2_mask, fmap2) 
        image1_next_mask = self.resize_soft_binary_mask(image1_next_mask, fmap1)
        image2_next_mask = self.resize_soft_binary_mask(image2_next_mask, fmap2)
        # Warp mask
        resized_image1_next_mask = self.warp_feature_map(image1_next_mask, flow_left) 
        resized_image2_next_mask = self.warp_feature_map(image2_next_mask, flow_right) 
                
        # *Feature Fusion with weighted mask
        epsilon = 1e-6
        mask_sum1 = resized_image1_mask + resized_image1_next_mask
        mask_sum2 = resized_image2_mask + resized_image2_next_mask
        mask_sum_safe1 = torch.where(mask_sum1 == 0, epsilon * torch.ones_like(mask_sum1), mask_sum1)
        mask_sum_safe2 = torch.where(mask_sum2 == 0, epsilon * torch.ones_like(mask_sum2), mask_sum2)
        temp_fused_fmap1 = (resized_image1_mask * fmap1 + resized_image1_next_mask * warped_fmap_left)/mask_sum_safe1
        temp_fused_fmap2 = (resized_image2_mask * fmap2 + resized_image2_next_mask * warped_fmap_right)/mask_sum_safe2
        
        fused_fmap1, near_zero_mask1, warping_failure_mask1, update_mask1 = self.mask_based_feature_fusion(fmap1, warped_fmap_left, resized_image1_mask, resized_image1_next_mask, fmap1_next)
        fused_fmap2, _, _, _ = self.mask_based_feature_fusion(fmap2, warped_fmap_right, resized_image2_mask, resized_image2_next_mask, fmap2_next)
        
        # For logging
        before_refined_fmap1 = temp_fused_fmap1

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            # cnet feature map from first frame left image
            cnet_list1 = self.cnet(image1, num_layers=self.args.n_gru_layers)
            
            # warp seconde frame left image cnet feature map
            cnet_list1_next = self.cnet(image1_next, num_layers=self.args.n_gru_layers)
            
            warped_outputs08 = self.warp_feature_map(cnet_list1_next[0][0], flow_left)
            warped_outputs16 = self.warp_feature_map(cnet_list1_next[1][0], flow_left)
            warped_outputs32 = self.warp_feature_map(cnet_list1_next[2][0], flow_left)
            
            # Resizing mask by resolution
            resized_image1_mask_08 = self.resize_soft_binary_mask(image1_mask, cnet_list1[0][0])
            resized_image1_mask_16 = self.resize_soft_binary_mask(image1_mask, cnet_list1[1][0])
            resized_image1_mask_32 = self.resize_soft_binary_mask(image1_mask, cnet_list1[2][0])
            
            resized_image1_next_mask_08 = self.resize_soft_binary_mask(image1_next_mask, cnet_list1_next[0][0])
            resized_image1_next_mask_16 = self.resize_soft_binary_mask(image1_next_mask, cnet_list1_next[1][0])
            resized_image1_next_mask_32 = self.resize_soft_binary_mask(image1_next_mask, cnet_list1_next[2][0])
            
            epsilon = 1e-6
            
            # Saturation mask_based fusion
            # resolution 8
            mask_sum_08 = resized_image1_mask_08 + resized_image1_next_mask_08
            mask_sum_safe_08 = torch.where(mask_sum_08 == 0, epsilon * torch.ones_like(mask_sum_08), mask_sum_08)
            fused_fmap_08 = (resized_image1_mask_08 * cnet_list1[0][0] + resized_image1_next_mask_08 * warped_outputs08) / mask_sum_safe_08
            
            # resolution 16
            mask_sum_16 = resized_image1_mask_16 + resized_image1_next_mask_16
            mask_sum_safe_16 = torch.where(mask_sum_16 == 0, epsilon * torch.ones_like(mask_sum_16), mask_sum_16)
            fused_fmap_16 = (resized_image1_mask_16 * cnet_list1[1][0] + resized_image1_next_mask_16 * warped_outputs16) / mask_sum_safe_16

            # resolution 32
            mask_sum_32 = resized_image1_mask_32 + resized_image1_next_mask_32
            mask_sum_safe_32 = torch.where(mask_sum_32 == 0, epsilon * torch.ones_like(mask_sum_32), mask_sum_32)
            fused_fmap_32 = (resized_image1_mask_32 * cnet_list1[2][0] + resized_image1_next_mask_32 * warped_outputs32) / mask_sum_safe_32
            
            net_list = [torch.tanh(fused_fmap_08), torch.tanh(fused_fmap_16), torch.tanh(fused_fmap_32)]
            inp_list = [torch.relu(cnet_list1[0][1]), torch.relu(cnet_list1[1][1]), torch.relu(cnet_list1[2][1])]
            
            inp_list = [list(conv(i).split(split_size=conv.out_channels // 3, dim=1)) for i, conv in zip(inp_list, self.context_zqr_convs)]            
        
        if self.args.corr_implementation == "reg": # Default
            corr_block = CorrBlock1D
            fused_fmap1, fused_fmap2 = fused_fmap1.float(), fused_fmap2.float()
        elif self.args.corr_implementation == "alt": # More memory efficient than reg
            corr_block = PytorchAlternateCorrBlock1D
            fused_fmap1, fused_fmap2 = fused_fmap1.float(), fused_fmap2.float()
        elif self.args.corr_implementation == "reg_cuda": # Faster version of reg
            corr_block = CorrBlockFast1D
        elif self.args.corr_implementation == "alt_cuda": # Faster version of alt
            corr_block = AlternateCorrBlock
            
        corr_fn = corr_block(fused_fmap1, fused_fmap2, radius=self.args.corr_radius, num_levels=self.args.corr_levels)

        coords0, coords1 = self.initialize_flow(net_list[0])

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume
            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                if self.args.n_gru_layers == 3 and self.args.slow_fast_gru: # Update low-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=True, iter16=False, iter08=False, update=False)
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru:# Update low-res GRU and mid-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=self.args.n_gru_layers==3, iter16=True, iter08=False, update=False)
                net_list, up_mask, delta_flow = self.update_block(net_list, inp_list, corr, flow, iter32=self.args.n_gru_layers==3, iter16=self.args.n_gru_layers>=2)

            # in stereo mode, project flow onto epipolar
            delta_flow[:,1] = 0.0

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # # We do not need to upsample or output intermediate results in test_mode
            # if test_mode and itr < iters-1:
            #     continue

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_up = flow_up[:,:1]
            
            flow_predictions.append(flow_up)

        # if test_mode:
        #     return coords1 - coords0, flow_up
        
        # For Viusalize
        mask_list = [image1_mask, temp_mask1, warped_image1_mask, resized_image1_mask * fmap1, resized_image1_next_mask * warped_fmap_left, near_zero_mask1, warping_failure_mask1, update_mask1]
        fmap_list = [fmap1, fmap1_next, warped_fmap_left, fused_fmap1, before_refined_fmap1]
        
        return flow_predictions, fmap1, fmap1_next, fused_fmap1, flow_left, fmap_list, cap_img_list, mask_list
        # return flow_predictions