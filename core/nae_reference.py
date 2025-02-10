import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralExposureControl(nn.Module):
    def __init__(self, net_width=[128, 256, 512, 1024, 16], histo=True, use_group_norm=False, learn_delta=False,
                 proportional_delta=False, max_exposure=4, sigmoid_scale=3.0, **kwargs):
        super(NeuralExposureControl, self).__init__()

        self.net_width = net_width
        self.histo = histo
        self.learn_delta = learn_delta
        self.proportional_delta = proportional_delta
        self.max_exposure = max_exposure
        self.sigmoid_scale = sigmoid_scale

        # Define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=net_width[0], kernel_size=(4, 1), stride=4)
        self.conv2 = nn.Conv2d(in_channels=net_width[0], out_channels=net_width[1], kernel_size=(4, 1), stride=4)
        self.conv3 = nn.Conv2d(in_channels=net_width[1], out_channels=net_width[2], kernel_size=(4, 1), stride=4)
        self.conv4 = nn.Conv2d(in_channels=net_width[2], out_channels=net_width[3], kernel_size=(4, 1), stride=4)
        self.dense1 = nn.Conv2d(in_channels=net_width[3], out_channels=net_width[4], kernel_size=(1, 1))

        if use_group_norm:
            self.group_norm = nn.GroupNorm(num_groups=4, num_channels=net_width[3])

        self.final_layer = nn.Conv2d(in_channels=net_width[4], out_channels=1, kernel_size=(1, 1))

    def forward(self, img, **kwargs):
        B, C, H, W = img.shape  # 배치 차원 포함한 입력 형태
        if self.histo:
            img = merge_capture(img)  # Custom function
            mask = make_histo_sampling(height=H, width=W, crop_bottom=H, crop_right=W)[3]
            
            # Mask 크기를 img와 일치하도록 리사이즈
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(H, W), mode='nearest').bool().squeeze(0).squeeze(0)
            
            img_1d = img[mask.unsqueeze(0).unsqueeze(0)].view(B, -1)

            k = max(1, img_1d.size(1) // 2)  # k 동적 설정
            min_clip = 1 / ((1 << 20) - 1)
            median = torch.kthvalue(img_1d, k=k, dim=1).values
            median = torch.maximum(median, torch.tensor(min_clip, device=img.device))
            scale = 1 / (1024 * median).view(B, 1, 1, 1)
            img *= scale
            img = torch.minimum(img, torch.tensor(1.0, device=img.device))
            log_min_clip = torch.log(torch.tensor(min_clip, device=img.device))
            img = torch.log(torch.maximum(img, torch.tensor(min_clip, device=img.device)))
            img = (log_min_clip - img) / log_min_clip

            net = multi_scale_histogram(img, **kwargs)  # 배치 차원 지원
            net = F.relu(self.conv1(net))
            net = F.relu(self.conv2(net))
            net = F.relu(self.conv3(net))
            net = F.relu(self.conv4(net))
            net = F.relu(self.dense1(net))
        else:
            net = torch.zeros((B, 1, 1, 16), dtype=torch.float32, device=img.device)

        output = self.final_layer(net)
        sig_input = self.sigmoid_scale * output

        if self.learn_delta:
            output = (torch.sigmoid(sig_input) - torch.tensor([0.5, 0.0, 0.0], device=img.device)) * torch.tensor([2.0, 1.0, 1.0], device=img.device)
            output = output * torch.tensor([torch.log(self.max_exposure), torch.log(16), torch.log(16)], device=img.device) + torch.tensor([0.0, torch.log(8), torch.log(8)], device=img.device)
        elif self.proportional_delta:
            output = (torch.sigmoid(sig_input[:, 0]) - 0.5) * 2.0
            log_delta = torch.abs(output) * torch.log(16) + torch.log(8)
            output = torch.stack([output, log_delta, log_delta], dim=1)
        else:
            output = 2 * (torch.sigmoid(sig_input) - 0.5).squeeze(1)
            output *= torch.log(torch.tensor(self.max_exposure, device=img.device))

        new_exp = torch.exp(output)
        return new_exp


def merge_capture(img_stack, scale_factor=4095, high_thresh=255, mid_low_thresh=256, mid_high_thresh=4094,
                  high_scale=256, mid_scale=16):
    B, _, H, W = img_stack.shape

    img_stack *= scale_factor

    def torch_merge(img_stack):
        img_stack = torch.round(img_stack)
        img_low, img_mid, img_high = img_stack[:, 0], img_stack[:, 1], img_stack[:, 2]

        mask_low = ~(img_mid == scale_factor)
        mask_mid = ~((img_mid >= mid_low_thresh) & (img_mid <= mid_high_thresh))
        mask_high = ~(img_mid <= high_thresh)

        img_low[mask_low] = 0
        img_mid[mask_mid] = 0
        img_high[mask_high] = 0

        img_merged = img_high / high_scale + img_mid / mid_scale + img_low
        return img_merged.view(B, 1, H, W)

    return torch_merge(img_stack)


def make_histo_sampling(
        height, width,
        offset_x=5, offset_y=4,
        block_size=42, th_x_max=40, th_y_max=25,
        crop_top=4, crop_left=4, crop_bottom=None, crop_right=None,
        bayer_x=0, bayer_y=0
):
    if crop_bottom is None:
        crop_bottom = height - 4
    if crop_right is None:
        crop_right = width - 4

    w_block = width / block_size
    h_block = height / block_size
    buffer = torch.zeros((height, width, 3), dtype=torch.uint8)

    for block_y in range(block_size // 2):
        for block_x in range(block_size // 2):
            for th_y in range(th_y_max):
                for th_x in range(th_x_max):
                    x = ((th_x + offset_x + int(block_x * w_block)) << 1) + bayer_x
                    y = ((th_y + offset_y + int(block_y * h_block)) << 1) + bayer_y
                    if x < width and y < height:
                        buffer[y, x] = torch.tensor([0, 255, 0], dtype=torch.uint8)

    mask = buffer[:, :, 1].bool()
    buffer_cropped = buffer[crop_top:crop_bottom, crop_left:crop_right]
    mask_cropped = mask[crop_top:crop_bottom, crop_left:crop_right]

    return buffer, mask, buffer_cropped, mask_cropped


def multi_scale_histogram(im_cap, ds_step=2, scales=[1, 3, 7], image_shape=[1200, 1920], bayer_sampling=[1, 0], log_histogram=False):
    num_expo = im_cap.shape[0]
    image_height, image_width = image_shape
    bayer_i, bayer_j = bayer_sampling
    slice_list = []

    for i_expo in range(num_expo):
        for scale in scales:
            for i in range(scale):
                for j in range(scale):
                    x1 = int(j / scale * image_width)
                    x2 = int((j + 1) / scale * image_width)
                    y1 = int(i / scale * image_height)
                    y2 = int((i + 1) / scale * image_height)

                    x1 = x1 + (x1 % 2) + bayer_j * (1 - 2 * (x1 % 2))
                    y1 = y1 + (y1 % 2) + bayer_i * (1 - 2 * (y1 % 2))

                    im_crop = im_cap[:, :, y1:y2:ds_step, x1:x2:ds_step]
                    slice_list.append(make_histogram(im_crop[i_expo]))

    net = torch.cat(slice_list, dim=3)

    if log_histogram:
        net = torch.log(1 + net)
        net *= 256 / torch.sum(net, dim=1, keepdim=True)

    return net


def make_histogram(x, nbins=256):
    net = torch.histc(x, bins=nbins, min=0.0, max=1.0)
    net = net.float()
    net /= torch.clamp(net.sum(), min=1e-4)
    net *= nbins
    net = net.view(1, 1, nbins, 1)
    return net


# if __name__ == '__main__':
#     img_stack = torch.rand((1, 3, 384, 1280))
#     model = NeuralExposureControl()
#     alpha = model(img_stack)
#     print("Output:", alpha)
