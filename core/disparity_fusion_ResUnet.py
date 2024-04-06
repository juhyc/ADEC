import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################
# * Disparity fusion network
###############################################

def pad_to_match(x, target_shape):
    height, width = target_shape
    diffY = height - x.size(2)
    diffX = width - x.size(3)
    padding = [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
    x = F.pad(x, padding, "constant", value=0)  # 'constant'는 패딩 값으로 0을 사용
    return x

class ConvBlock(nn.Module):
    """Basic Convolution Block with residual connection."""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batchnorm2= nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        # For residual connection
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.residual_conv(x)
        
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x += residual  # Residual connection
        return self.relu2(x)


class UpConvBlock(nn.Module):
    """Up-convolution Block used in the decoder path."""
    def __init__(self, in_channels, out_channels):
        super(UpConvBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        skip = F.interpolate(skip, size=x.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)

class ChannelAttention(nn.Module):
    """Channel Attention Module."""
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class ResUNet(nn.Module):

    """Light-weight Residual U-Net architecture."""
    def __init__(self, n_channels, n_classes):
        super(ResUNet, self).__init__()
        self.down_conv1 = ConvBlock(n_channels, 64)
        self.down_conv2 = ConvBlock(64, 128)
        self.down_conv3 = ConvBlock(128, 256)
        self.down_conv4 = ConvBlock(256, 512)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.up_conv3 = UpConvBlock(512, 256)
        self.up_conv2 = UpConvBlock(256, 128)
        self.up_conv1 = UpConvBlock(128, 64)
        
        # * Add attention module before final convolution
        self.channel_attention = ChannelAttention(64)
        
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path
        shape = (x.shape[2], x.shape[3])
        x1 = self.down_conv1(x)
        x2 = self.maxpool(x1)
        x2 = self.down_conv2(x2)
        x3 = self.maxpool(x2)
        x3 = self.down_conv3(x3)
        x4 = self.maxpool(x3)
        x4 = self.down_conv4(x4)

        # Decoder path
        x = self.up_conv3(x4, x3)
        x = self.up_conv2(x, x2)
        x = self.up_conv1(x, x1)
        
        # * Apply channel attention before the final convolution
        attention_weights = self.channel_attention(x)
        x = x * attention_weights

        # Final convolution
        x = self.final_conv(x)
        
        # * Add pad
        x = pad_to_match(x, shape)

        return x


class DisparityFusion_ResUnet(nn.Module):
    def __init__(self):
        super(DisparityFusion_ResUnet, self).__init__()
        self.unet_fusion= ResUNet(n_channels=4, n_classes=1)
        
    
    def forward(self, stereo_depth1, stereo_depth2, mask1, mask2):
        left_combined = torch.cat([stereo_depth1, stereo_depth2, mask1, mask2], dim = 1)
        
        left_output = self.unet_fusion(left_combined)
        
        return left_output
    



