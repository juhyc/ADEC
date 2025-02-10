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



#* Image warping pipeline

# DEVICE = 'cuda'

# writer = SummaryWriter('runs/disp_recon_model')sssssssssssssssssssssssssss

# class Disp_recon_model(nn.Module):
#     def __init__(self, args):
#         super(Disp_recon_model, self).__init__()
#         self.args = args
#         # Optical flow model
#         self.flow_model = get_model('rapidflow_it6', pretrained_ckpt='kitti').to(DEVICE)
#         self.flow_model.train()
#         # Raft warp stereo model
#         self.raft_warp_stereo = RAFTStereoFusion(args)

#         # Pretrained model parameter freeze
#         for param in self.flow_model.parameters():
#             param.requires_grad = False
        
#         # Unfreeze the last few layers for fine-tuning
#         for param in list(self.flow_model.parameters())[-10:]:
#             param.requires_grad = True
            
#     def compute_optical_flow_batch(self, model, image_pairs):
#         device = next(model.parameters()).device if next(model.parameters(), None) is not None else 'cuda'
        
#         all_flows = []
#         for left_image, right_image in image_pairs:

#             left_image = left_image.to(device)
#             right_image = right_image.to(device)
#             inputs = prepare_inputs_custom([left_image, right_image])
            
#             inputs = {k: v.to(device) for k, v in inputs.items()} 

#             with torch.no_grad():
#                 predictions = model(inputs)

#             # Extract the flows
#             flows = predictions['flows']
#             all_flows.append(flows)
        
#         return torch.cat(all_flows, dim=0).squeeze(1)
        
#     @staticmethod
#     def warp_image(image, flow):
#         image = image.float()
#         B, C, H, W = image.size()
#         flow = flow.permute(0, 2, 3, 1)  # [B, H, W, 2]

#         grid_y, grid_x = torch.meshgrid(torch.arange(H, device=flow.device), torch.arange(W, device=flow.device))
#         grid = torch.stack((grid_x, grid_y), 2).float()  # [H, W, 2]
#         grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 2]

#         flow = flow + grid  # [B, H, W, 2]

#         flow = 2 * flow / torch.tensor([W-1, H-1], device=flow.device).float() - 1

#         warped_image = F.grid_sample(image, flow, mode='bilinear', padding_mode='zeros', align_corners=False)

#         return warped_image
    
#     def forward(self, img1_left, img1_right, img2_left, img2_right, iters=32, test_mode=False):
#         with autocast():
#             # Compute flow between Frame1 and Frame2
#             img_pair_left = [(img1_left[i], img2_left[i]) for i in range(img1_left.shape[0])]
#             img_pair_right = [(img1_right[i], img2_right[i]) for i in range(img2_left.shape[0])]
            
#             left_flows_tensor = self.compute_optical_flow_batch(self.flow_model, img_pair_left)
#             right_flows_tensor = self.compute_optical_flow_batch(self.flow_model, img_pair_right)
            
#             # Warp stereo image
#             img2_left_warped = self.warp_image(img2_left, left_flows_tensor)
#             img2_right_warped = self.warp_image(img2_right, right_flows_tensor)
            
#             disparity = self.raft_warp_stereo(img1_left, img1_right, img2_left_warped, img2_right_warped, iters=iters, flow_init=None, test_mode=False)
            
#         return disparity, img2_left_warped


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
        
# * Test1
# # Channel Attention Module
# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
        
#         self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)

# # Spatial Attention Module
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)

# # Attention Fusion module
# class AttentionFusion(nn.Module):
#     def __init__(self, in_planes, ratio=16, out_planes = 256):
#         super(AttentionFusion, self).__init__()
#         self.channel_attention = ChannelAttention(in_planes, ratio)
#         self.spatial_attention = SpatialAttention(kernel_size=7)
#         self.conv1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

#     def forward(self, fmap1, fmap2):
#         # feature map fusion
#         combined_fmap = torch.cat([fmap1, fmap2], dim=1)
#         channel_att = self.channel_attention(combined_fmap)
#         spatial_att = self.spatial_attention(combined_fmap)
#         fused_map = channel_att * spatial_att * combined_fmap # [B, 512, H, W]
#         return self.conv1x1(fused_map) # [B, 256, H, W]
    
# class CrossAttention(nn.Module):
#     def __init__(self, embed_dim):
#         super(CrossAttention, self).__init__()
#         self.query_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)  # embed_dim = 256
#         self.key_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
#         self.value_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
#         self.softmax = nn.Softmax(dim=-1)
#         self.scale = math.sqrt(embed_dim)
    
#     def forward(self, fmap1, fmap2):
#         # Extract query, key, value
#         query = self.query_conv(fmap1)  # [B, 256, H, W]
#         key = self.key_conv(fmap2)      # [B, 256, H, W]
#         value = self.value_conv(fmap2)  # [B, 256, H, W]

#         # Reshape for matrix multiplication
#         B, C, H, W = query.shape
#         query = query.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
#         key = key.view(B, C, -1)                       # [B, C, HW]
#         value = value.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]

#         # Compute attention scores
#         attn_scores = torch.bmm(query, key) / self.scale  # [B, HW, HW]
#         attn_scores = self.softmax(attn_scores)  # [B, HW, HW]

#         # Compute weighted sum of values
#         attn_output = torch.bmm(attn_scores, value)  # [B, HW, C]
#         attn_output = attn_output.permute(0, 2, 1).view(B, C, H, W)
        
#         # Reshape attention scores for visualization
#         attn_scores = attn_scores.view(B, H, W, H, W).mean(dim=1).mean(dim=1)  # [B, H, W]
#         attn_scores = attn_scores.unsqueeze(1) # [B, 1, H, W]
#         attn_scores = (attn_scores - attn_scores.min()) / (attn_scores.max() - attn_scores.min())
        
#         return attn_output, attn_scores


# class CrossAttentionFusion(nn.Module):
#     def __init__(self, in_planes=256, out_planes=64):
#         super(CrossAttentionFusion, self).__init__()
#         self.cross_attention = CrossAttention(embed_dim=in_planes)

#     def forward(self, fmap1, fmap2):
#         # Apply cross attention directly on full feature maps
#         attn_output, attn_scores = self.cross_attention(fmap1, fmap2)
        
#         # Apply final 1x1 convolution to mix the information
#         return attn_output, attn_scores

# * Test2

# class MultiScaleChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super(MultiScaleChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return x * out

# class IterativeAttentionFusion(nn.Module):
#     def __init__(self, in_channels, num_iterations=3):
#         super(IterativeAttentionFusion, self).__init__()
#         self.num_iterations = num_iterations
#         self.attention = MultiScaleChannelAttention(in_channels)

#     def forward(self, x1, x2):
#         for _ in range(self.num_iterations):
#             x1_attn = self.attention(x1)
#             x2_attn = self.attention(x2)
            
#             x1_temp = x1_attn + x2_attn
#             x2_temp = x1_attn + x2_attn  
            
#             x1 = x1_temp
#             x2 = x2_temp
            
#         return x1
    
# class ImprovedAttention(nn.Module):
#     def __init__(self, in_channels, reduction=16, num_iterations=3):
#         super(ImprovedAttention, self).__init__()
#         self.iterative_attention_fusion = IterativeAttentionFusion(in_channels, num_iterations)
#         self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

#     def forward(self, fmap1, fmap2):
#         # Iterative attention fusion
#         fused_features = self.iterative_attention_fusion(fmap1, fmap2)
#         # Apply a final 1x1 convolution to fuse the features
#         fused_features = self.final_conv(fused_features)
#         return fused_features
    
# class AttentionFusion(nn.Module):
#     def __init__(self, in_planes=256):
#         super(AttentionFusion, self).__init__()
#         self.attention = ImprovedAttention(in_planes)

#     def forward(self, fmap1, fmap2):
#         fused_fmap = self.attention(fmap1, fmap2)
#         return fused_fmap

# class AttentionalFeatureFusion(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super(AttentionalFeatureFusion, self).__init__()
#         # Attention calculation for each input feature map
#         self.attention_u = MultiScaleChannelAttention(in_channels, reduction)
        
#         # Attention for the combined feature map
#         self.attention_m = MultiScaleChannelAttention(in_channels, reduction)

#     def forward(self, fmap1, fmap2):
#         # Step 1: Calculate attention maps for each feature map (a_u)
#         attn_score_fmap1 = self.attention_u(fmap1)  # Attention for feature map 1
#         attn_score_fmap2 = self.attention_u(fmap2)  # Attention for feature map 2

#         # Step 2: Generate upper-level feature maps F_u^g and F_u^c
#         attn_fmap1 = fmap1 * attn_score_fmap2  # Apply attention to feature map 1
#         attn_fmap2 = fmap2 * attn_score_fmap1  # Apply attention to feature map 2

#         # # Step 3: Compute the combined feature map F̅
#         # F_bar = (attn_fmap1 + attn_fmap2) / (attn_score_fmap1 + attn_score_fmap2 + 1e-8)  # Combine using attention weights

#         # # Step 4: Calculate the attention map for the combined feature map F̅ (a_m)
#         # a_m = self.attention_m(F_bar)

#         # # Step 5: Compute the final fused feature map F̂
#         # fused_fmap = (attn_fmap1 * a_m) + (attn_fmap2 * (1 - a_m))
        
#         fused_fmap = attn_fmap1 + attn_fmap2

#         return fused_fmap

    
# * Test3
class MultiScaleChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(MultiScaleChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return out  # This returns the attention map

class AttentionalFeatureFusion(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(AttentionalFeatureFusion, self).__init__()
        # Attention calculation for each input feature map
        self.attention_fmap1 = MultiScaleChannelAttention(in_channels, reduction)
        self.attention_fmap2 = MultiScaleChannelAttention(in_channels, reduction)
        
        self.scene_weight1 = nn.Linear(1, 1) 
        self.scene_weight2 = nn.Linear(1, 1)
        
    def compute_information(self, fmap):
        return fmap.var(dim=[2, 3], keepdim=True)

    def forward(self, fmap1, fmap2):
        attn_score_fmap1 = self.attention_fmap1(fmap1)  # Attention for feature map 1
        attn_score_fmap2 = self.attention_fmap2(fmap2)  # Attention for feature map 2
        
        info_fmap1 = self.compute_information(fmap1) 
        info_fmap2 = self.compute_information(fmap2)  
        
        # scene_weight1 = torch.sigmoid(self.scene_weight1(info_fmap1))
        # scene_weight2 = torch.sigmoid(self.scene_weight2(info_fmap2))
        # Apply softmax : Normalize so that the sum of the two scene weights equals 1
        scene_weight1 = torch.sigmoid(self.scene_weight1(info_fmap1))
        scene_weight2 = torch.sigmoid(self.scene_weight2(info_fmap2))
        total_weight = torch.cat([scene_weight1, scene_weight2], dim=1)
        norm_weights = torch.softmax(total_weight, dim=1)
        scene_weight1 = norm_weights[:, 0:1, :, :]
        scene_weight2 = norm_weights[:, 1:2, :, :]
        
        mask_fmap2 = (fmap2 != 0).float()

        attn_fmap1 = scene_weight1 * (fmap1 * attn_score_fmap2)  # Apply attention to feature map 1
        attn_fmap2 = scene_weight2 * (fmap2 * attn_score_fmap1) * mask_fmap2  # Apply attention to feature map 2

        fused_fmap = attn_fmap1 + attn_fmap2

        return fused_fmap

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

class FeatureRefiner(nn.Module):
    def __init__(self, in_channels):
        super(FeatureRefiner, self).__init__()
        
        # 기본적인 conv layer로 feature 추출
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        
        
        # self.bn1 = nn.BatchNorm2d(128)
        # self.bn2 = nn.BatchNorm2d(64)
        self.bn1 = nn.InstanceNorm2d(128)
        self.bn2 = nn.InstanceNorm2d(64)
        
        # Attention 메커니즘 추가 (필요시)
        self.attention = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        
        
        # Residual block로 refinement 적용
        self.refine_conv = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)
        
    def forward(self, fused_feature_map):
        # Feature 추출
        x = self.conv1(fused_feature_map)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        # Attention 적용
        attention_map = self.sigmoid(self.attention(x))
        
        # Refinement: 작은 correction map 생성
        refinement_map = self.refine_conv(x)
        
        # Attention 가중치를 사용하여 feature map 수정
        # alpha = 0.1
        refined_feature_map = fused_feature_map + refinement_map * attention_map
        
        return refined_feature_map

class SimpleRefineNet(nn.Module):
    """
    warp된 feature와 원본 feature를 concat or sum 한 뒤,
    작은 CNN으로 보정.
    """
    # def __init__(self, in_channels=256, out_channels=256):
    #     super().__init__()
        
    #     self.conv1 = nn.Sequential(
    #         nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
    #         nn.ReLU(inplace=True),
    #     )
    #     self.conv2 = nn.Sequential(
    #         nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
    #         nn.ReLU(inplace=True),
    #     )
    
    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.conv2(x)
    #     return x
    def __init__(self, in_channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + identity)

# RAFTStereo with attention feature fusion
class RAFTStereoFusion_refine(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        context_dims = args.hidden_dims
        
        self.flow_model = get_model('rapidflow_it6', pretrained_ckpt='kitti').to(DEVICE)
        # (1) 작은 보정 네트워크
        # self.refine_net = SimpleRefineNet(in_channels=256)
        
        # self.refine_net = FeatureRefiner(in_channels=256)
        
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
    
    
    # # Warp feature map using resized flow
    # def warp_feature_map(self, feature_map, flow):
    #     n, c, h, w = feature_map.size()
    #     resized_flow = self.resize_flow(flow, target_size=(h, w))
        
    #     grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w), indexing='ij')
    #     grid = torch.stack((grid_x, grid_y), dim=0).float()  # [2, h, w]
    #     grid = grid.unsqueeze(0).repeat(n, 1, 1, 1).to(feature_map.device)  # [n, 2, h, w]
        
    #     grid = grid + resized_flow
        
    #     grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :].clone() / (w - 1) - 1.0
    #     grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :].clone() / (h - 1) - 1.0
        
    #     grid = grid.permute(0, 2, 3, 1)  # [n, h, w, 2]
        
    #     warped_feature_map = F.grid_sample(feature_map, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    #     return warped_feature_map
    
    def warp_feature_map(self, feature_map, flow):
        n, c, h, w = feature_map.size()
        resized_flow = self.resize_flow(flow, target_size=(h, w))
        
        # 기존 grid 생성 방식에서 float16을 사용하여 메모리 사용을 줄임
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h, dtype=torch.float16, device=feature_map.device), 
            torch.arange(0, w, dtype=torch.float16, device=feature_map.device), 
            indexing='ij'
        )
        grid = torch.stack((grid_x, grid_y), dim=0)  # [2, h, w]
        grid = grid.unsqueeze(0).repeat(n, 1, 1, 1)  # [n, 2, h, w]
        
        # float32로 변환하여 flow와 연산 (float32로 변환해도 메모리 사용이 크게 늘지 않음)
        grid = grid.to(torch.float32) + resized_flow

        # 그리드 값을 -1에서 1 범위로 정규화 (경계 조건 개선을 위해 0.5 추가)
        grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :].clone() / (w - 0.5) - 1.0
        grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :].clone() / (h - 0.5) - 1.0
        
        # grid의 차원을 재배열하여 F.grid_sample에 전달
        grid = grid.permute(0, 2, 3, 1)  # [n, h, w, 2]
        
        # padding_mode를 'border'로 변경하여 경계 조건 개선, align_corners는 True로 설정
        warped_feature_map = F.grid_sample(feature_map, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        return warped_feature_map



    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
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
    
    def normalize_feature_map(self, fmap):
        mean = fmap.mean(dim=(2, 3), keepdim=True)
        std = fmap.std(dim=(2, 3), keepdim=True) + 1e-6  
        fmap_normalized = (fmap - mean) / std
        return fmap_normalized
    
    def global_brightness_align(self, imgA, imgB, eps=1e-6):
        """
        imgA, imgB: [B, C, H, W], 실수 범위가 [0,1]이라 가정
        (0~1 normalize된 이미지라고 가정. 만약 -1~1 범위라면 계산부 약간 수정)
        
        두 이미지의 평균/표준편차를 이용하여,
        imgB를 imgA에 대략 맞추는 간단한 방법 (혹은 반대로).
        """
        # imgA의 mean, std
        meanA = imgA.mean(dim=[1,2,3], keepdim=True)
        stdA = imgA.std(dim=[1,2,3], keepdim=True) + eps

        # imgB의 mean, std
        meanB = imgB.mean(dim=[1,2,3], keepdim=True)
        stdB = imgB.std(dim=[1,2,3], keepdim=True) + eps

        # imgB -> imgB_aligned: (imgB - meanB)/stdB * stdA + meanA
        imgB_aligned = (imgB - meanB)/stdB * stdA + meanA

        return imgA, imgB_aligned.clamp(0.0, 1.0)
        
    # Generate confidence mask for mask-based fusion
    def generate_confidence_mask(self, mask1, mask2, diff_threshold=0.5):
        """
        Generate a confidence mask based on softbinary masks of two frames.
        
        Args:
            mask1 (Tensor): Softbinary mask for the first frame (shape: [B, 1, H, W]).
            mask2 (Tensor): Softbinary mask for the second frame (shape: [B, 1, H, W]).
            diff_threshold (float): Threshold for intensity difference between the masks to determine exclusion.
            
        Returns:
            Tensor: A confidence mask where regions with extreme exposure differences are excluded (shape: [B, 1, H, W]).
        """
        mask_diff = torch.abs(mask1 - mask2)
        confidence_mask = (mask_diff < diff_threshold).float()
        return confidence_mask
    
    def mask_based_feature_fusion(self, fmap1, fmap2, mask1, mask2, epsilon=1e-6):
        """
        Perform mask-based feature fusion using two feature maps and their corresponding masks.
        
        Args:
            fmap1 (Tensor): Feature map from the first frame.
            fmap2 (Tensor): Warped feature map from the second frame.
            mask1 (Tensor): Soft-binary mask for the first frame.
            mask2 (Tensor): Soft-binary mask for the warped second frame.
            epsilon (float): Small value to prevent division by zero.
            
        Returns:
            Tensor: Fused feature map.
        """
        # Softbinary mask sum
        mask_sum = mask1 + mask2
        mask_sum_safe = torch.where(mask_sum == 0, epsilon * torch.ones_like(mask_sum), mask_sum)
        
        # Fuse feature maps based on the masks
        fused_fmap = (mask1 * fmap1 + mask2 * fmap2) / mask_sum_safe
        
        return fused_fmap
    
    def mask_based_fusion_with_confidence(self, fmap1, fmap2, mask1, mask2, confidence_mask, epsilon=1e-6):
        """
        Perform mask-based fusion using two feature maps and their corresponding masks, applying a confidence mask.

        Args:
            fmap1 (Tensor): Feature map from the first frame.
            fmap2 (Tensor): Warped feature map from the second frame.
            mask1 (Tensor): Soft-binary mask for the first frame.
            mask2 (Tensor): Soft-binary mask for the warped second frame.
            confidence_mask (Tensor): Confidence mask indicating areas with valid exposure differences.
            epsilon (float): Small value to prevent division by zero.

        Returns:
            Tensor: Fused feature map.
        """
        # Adjust the masks using the confidence mask
        adjusted_mask1 = mask1 * confidence_mask
        adjusted_mask2 = mask2 * confidence_mask

        # Calculate the safe mask sum
        mask_sum = adjusted_mask1 + adjusted_mask2
        mask_sum_safe = torch.where(mask_sum == 0, epsilon * torch.ones_like(mask_sum), mask_sum)
        
        # Fuse the feature maps based on the adjusted masks
        fused_fmap = (adjusted_mask1 * fmap1 + adjusted_mask2 * fmap2) / mask_sum_safe
        
        return fused_fmap

    def forward(self, image1, image2, image1_next, image2_next, iters=32, flow_init=None, test_mode=False):
        
        # #*##### Noise modeling for finetuning#####
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
        
        # # !Brightness align
        # image1, image1_next = self.global_brightness_align(image1, image1_next)
        # image2, image2_next = self.global_brightness_align(image2, image2_next)
        
        #* Calculate softbinary mask 
        image1_mask = soft_binary_threshold_batch(image1)
        image2_mask = soft_binary_threshold_batch(image2)
        image1_next_mask = soft_binary_threshold_batch(image1_next)
        image2_next_mask = soft_binary_threshold_batch(image2_next)
        temp_mask1 = image1_next_mask

        
        # Normalize input image
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

        #! Apply confidence mask to fusion process
        # adjusted_image1_mask = resized_image1_mask * confidence_mask1
        # adjusted_image1_next_mask = resized_image1_next_mask * confidence_mask1
        # adjusted_image2_mask = resized_image2_mask * confidence_mask2
        # adjusted_image2_next_mask = resized_image2_next_mask * confidence_mask2
                
        # # *Feature Fusion with confidence mask
        epsilon = 1e-6
        mask_sum1 = resized_image1_mask + resized_image1_next_mask
        mask_sum2 = resized_image2_mask + resized_image2_next_mask
        mask_sum_safe1 = torch.where(mask_sum1 == 0, epsilon * torch.ones_like(mask_sum1), mask_sum1)
        mask_sum_safe2 = torch.where(mask_sum2 == 0, epsilon * torch.ones_like(mask_sum2), mask_sum2)
        fused_fmap1 = (resized_image1_mask * fmap1 + resized_image1_next_mask * warped_fmap_left)/mask_sum_safe1
        fused_fmap2 = (resized_image2_mask * fmap2 + resized_image2_next_mask * warped_fmap_right)/mask_sum_safe2
        
        
        
        #! Perform mask-based fusion
        # fused_fmap1 = self.mask_based_feature_fusion(fmap1, warped_fmap_left, adjusted_image1_mask, adjusted_image1_next_mask)
        # fused_fmap2 = self.mask_based_feature_fusion(fmap2, warped_fmap_right, adjusted_image2_mask, adjusted_image2_next_mask)

        before_refined_fmap1 = fused_fmap1
        
        # # # ! *Refine feature map
        # fused_fmap1 = self.refine_net(fused_fmap1)
        # fused_fmap2 = self.refine_net(fused_fmap2)
                        
        # run the context network
        with autocast(enabled=self.args.mixed_precision):
        # 첫 번째 이미지에서 나온 cnet feature map
            cnet_list1 = self.cnet(image1, num_layers=self.args.n_gru_layers)
            
            # 두 번째 이미지에서 나온 cnet feature map을 flow로 warp
            cnet_list1_next = self.cnet(image1_next, num_layers=self.args.n_gru_layers)
            
            warped_outputs08 = self.warp_feature_map(cnet_list1_next[0][0], flow_left)
            warped_outputs16 = self.warp_feature_map(cnet_list1_next[1][0], flow_left)
            warped_outputs32 = self.warp_feature_map(cnet_list1_next[2][0], flow_left)
            
            # 마스크를 해상도별로 리사이즈
            resized_image1_mask_08 = self.resize_soft_binary_mask(image1_mask, cnet_list1[0][0])
            resized_image1_mask_16 = self.resize_soft_binary_mask(image1_mask, cnet_list1[1][0])
            resized_image1_mask_32 = self.resize_soft_binary_mask(image1_mask, cnet_list1[2][0])
            
            resized_image1_next_mask_08 = self.resize_soft_binary_mask(image1_next_mask, cnet_list1_next[0][0])
            resized_image1_next_mask_16 = self.resize_soft_binary_mask(image1_next_mask, cnet_list1_next[1][0])
            resized_image1_next_mask_32 = self.resize_soft_binary_mask(image1_next_mask, cnet_list1_next[2][0])
            
            epsilon = 1e-6
            
            # 해상도 08에서 마스크 기반 fusion
            mask_sum_08 = resized_image1_mask_08 + resized_image1_next_mask_08
            mask_sum_safe_08 = torch.where(mask_sum_08 == 0, epsilon * torch.ones_like(mask_sum_08), mask_sum_08)
            fused_fmap_08 = (resized_image1_mask_08 * cnet_list1[0][0] + resized_image1_next_mask_08 * warped_outputs08) / mask_sum_safe_08
            
            # 해상도 16에서 마스크 기반 fusion
            mask_sum_16 = resized_image1_mask_16 + resized_image1_next_mask_16
            mask_sum_safe_16 = torch.where(mask_sum_16 == 0, epsilon * torch.ones_like(mask_sum_16), mask_sum_16)
            fused_fmap_16 = (resized_image1_mask_16 * cnet_list1[1][0] + resized_image1_next_mask_16 * warped_outputs16) / mask_sum_safe_16

            # 해상도 32에서 마스크 기반 fusion
            mask_sum_32 = resized_image1_mask_32 + resized_image1_next_mask_32
            mask_sum_safe_32 = torch.where(mask_sum_32 == 0, epsilon * torch.ones_like(mask_sum_32), mask_sum_32)
            fused_fmap_32 = (resized_image1_mask_32 * cnet_list1[2][0] + resized_image1_next_mask_32 * warped_outputs32) / mask_sum_safe_32
            
            # 기존 방식과 동일하게 fused feature map을 net과 inp로 나누어 사용
            net_list = [torch.tanh(fused_fmap_08), torch.tanh(fused_fmap_16), torch.tanh(fused_fmap_32)]
            inp_list = [torch.relu(cnet_list1[0][1]), torch.relu(cnet_list1[1][1]), torch.relu(cnet_list1[2][1])]
            # inp_list = [torch.relu(fused_fmap_08), torch.relu(fused_fmap_16), torch.relu(fused_fmap_32)]

            # inp_list를 context_zqr_convs를 통해 변환
            inp_list = [list(conv(i).split(split_size=conv.out_channels // 3, dim=1)) for i, conv in zip(inp_list, self.context_zqr_convs)]            
        
            
        #     ######################################
            
            ############Existed Raft-stereo model################
            # net_list = [torch.tanh(x[0]) for x in cnet_list]
            # inp_list = [torch.relu(x[1]) for x in cnet_list]
            # inp_list = [list(conv(i).split(split_size=conv.out_channels // 3, dim=1)) for i, conv in zip(inp_list, self.context_zqr_convs)]
            ######################################
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
        
        # print(f"image1 shape, fmap1 shape : {image1.shape} {fmap1.shape}")
        # print(f"fused fmap shape : {fused_fmap1.shape}")
        # print(f"atten_score shape : {attn_scores1.shape}")
        
        # For Viusalize
        mask_list = [image1_mask, temp_mask1, warped_image1_mask, resized_image1_mask * fmap1, resized_image1_next_mask * warped_fmap_left]
        fmap_list = [fmap1, fmap1_next, warped_fmap_left, fused_fmap1]
        
        return flow_predictions, fmap1, fmap1_next, fused_fmap1, flow_left, fmap_list, cap_img_list, mask_list
        # return flow_predictions