import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import os
import glob
import re
import io
from torchvision.transforms.functional import to_tensor
from core.saec import calculate_histogram_global
from io import BytesIO
import torchvision.utils as vutils
from ptlflow.utils import flow_utils
import torch.nn.functional as F

###############################################
# * Visualize code for tensorboard logging
###############################################

def show_images(images):
    fig, axs = plt.subplots(1, len(images), figsize=(20, 10))
    for i, img in enumerate(images):
        axs[i].imshow(img[0].permute(1, 2, 0), cmap='magma')
        axs[i].axis('off')
    plt.show()

def show_dataloader_image(dataloader):
    batch = next(iter(dataloader))
    left_mono, right_mono, left_stereo, right_stereo, disparity = batch
    show_images([left_mono, right_mono, left_stereo, right_stereo, disparity])
    
   
def visualize_mask(batch_image):
    
    image_tensor = batch_image[0].clone().detach()
    image = image_tensor.cpu().numpy().squeeze()
    
    colored_image = plt.cm.gray(image)
    colored_image = (colored_image[..., : 3] * 255).astype(np.uint8)
    colored_image = np.transpose(colored_image, (2,0,1))
    
    return colored_image
    
    return image

# * Visualization flow_prediction during Training
def visualize_flow_cmap(batch_image):
    
    # For visualization multiply -1 to batch_image
    
    image_tensor = -batch_image[0].clone().detach()
    
    image = image_tensor.cpu().numpy().squeeze()
    image = (image - image.min())/(image.max() - image.min())
    
    colored_image = plt.cm.magma(image)
    colored_image = (colored_image[..., : 3] * 255).astype(np.uint8)
    colored_image = np.transpose(colored_image, (2,0,1))
    
    return colored_image

def save_image(image_tensor, filename):
    image = np.transpose(image_tensor, (1,2,0))
    im = Image.fromarray(image)
    im.save(filename)

def save_image_255(image_tensor, filename):
    image = image_tensor.cpu().permute(1,2,0)
    image = torch.clamp(image * 255, 0, 255)
    image = image.numpy().astype(np.uint8)
    image = Image.fromarray(image)
    image.save(filename)


# * Visualization prediciton depth map with color map
def visualize_flow_cmap_with_colorbar(batch_image,  figsize=(16, 10), dpi=100, colorbar_length=0.75, colorbar_aspect=20):
    
    image_tensor = batch_image[0].clone().detach()
    image = image_tensor.cpu().numpy().squeeze()
    # normalized_image = (image - image.min()) / (image.max() - image.min())
    
    
    fig, ax = plt.subplots(figsize = figsize, dpi = dpi)
    cax = ax.imshow(image, cmap='magma')
    
    fig.colorbar(cax, aspect = colorbar_aspect, shrink = colorbar_length)
    
    # TensorBoard에 로깅하기 위해 이미지를 PIL 형식으로 변환
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches = 'tight')
    buf.seek(0)
    image = Image.open(buf)
    image = np.array(image)
    image = np.transpose(image, (2, 0, 1))  # HWC to CHW
    
    return torch.tensor(image/255.0).float()

def visualize_mask(mask):
    
    mask_tensor = mask[0].clone().detach()
    
    mask = mask_tensor.cpu().numpy().squeeze()
    
    mask_image = plt.cm.gray(mask)
    mask_image = (mask_image[..., : 3] * 255).astype(np.uint8)
    mask_image = np.transpose(mask_image, (2,0,1))
    
    return mask_image

def check_hdr_image(image):
    print(type(image))
    print(image.shape)
    if isinstance(image, torch.Tensor):
        temp = image.clone().detach()
        temp = temp[0].cpu().permute(1,2,0)
        temp = temp.numpy().astype(np.uint8)
    plt.imshow(temp)
    plt.show()

def check_ldr_image(image):
    if isinstance(image, torch.Tensor):
        temp = image.clone().detach()
        temp = temp[0].cpu().permute(1,2,0)
        temp = torch.clamp(temp*255, 0, 255)
        temp = temp.numpy().astype(np.uint8)
    plt.imshow(temp)
    plt.show()
    
def check_ldr_image2(image):
    if isinstance(image, torch.Tensor):
        temp = image.clone().detach()
        temp = temp[1].cpu().permute(1,2,0)
        temp = torch.clamp(temp*255, 0, 255)
        temp = temp.numpy().astype(np.uint8)
    plt.imshow(temp)
    plt.show()
    

def visualize_dynamic_range(batch_image, HDR = True):
    image_tensor = batch_image[0].clone().detach()
    img= image_tensor.cpu().permute(1,2,0).numpy()
    
    if HDR:
        img = img*(2**16 - 1)
    else:
        img = img*(2**8 - 1)
    
    # Compute luminance for RGB image
    luminance = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    
    # Plot histogram
    
    luminance_flat = luminance.flatten()
    percentile_99_5 = np.percentile(luminance_flat, 95)
    hist, bins = np.histogram(luminance_flat, bins=256)
    
    # Identify the indices of bins representing data less than or equal to the 99.5th percentile
    mask = bins[:-1] <= percentile_99_5
    
    # Modified dynamic range based on 99.5th percentile
    modified_dr = 20 * np.log(bins[:-1][mask][-1] / np.min(luminance[luminance > 0]))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.bar(bins[:-1][mask], hist[mask], color='blue', alpha=0.7, width=np.diff(bins)[0])
    ax.hist(luminance_flat, bins, color='gray', alpha=0.7)
    ax.set_xlabel('Luminance')
    ax.set_ylabel('Frequency')
    ax.text(0.5, 0.9, f'Modified Dynamic Range: {modified_dr:.2f} dB', fontsize=11, ha='center', transform=ax.transAxes)
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf)
    image_tensor = to_tensor(image)
    
    return image_tensor

def visualize_dynamic_range_log(batch_image):
    image_tensor = batch_image[0].clone().detach()
    img= image_tensor.cpu().permute(1,2,0).numpy()
    
    # Compute luminance for RGB image
    luminance = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    
    # Plot histogram
    luminance_flat = luminance.flatten()
    
    p1 = np.percentile(luminance_flat, 1)
    p80 = np.percentile(luminance_flat, 80)
    
    modified_dr = 30 * np.log10((p80 + 1e-6) / (p1+ 1e-6))
    log_luminance= np.log10(luminance_flat + 1)
    
    hist, bins = np.histogram(log_luminance, bins= 256)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(bins[:-1], hist, width=np.diff(bins), color='blue', alpha=0.7)
    ax.set_xlabel('Log10 Luminance')
    ax.set_ylabel('Frequency')
    ax.text(0.5, 0.9, f'Modified Dynamic Range: {modified_dr:.2f} dB', fontsize=12, ha='center', transform=ax.transAxes)
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf)
    image_tensor = to_tensor(image)
    
    return image_tensor


def visualize_dynamic_range_entropy(batch_image, HDR = True):
    image_tensor = batch_image[0].clone().detach()
    img = image_tensor.cpu().permute(1, 2, 0).numpy()
    
    if HDR:
        img = img * (2**16 - 1)
    else:
        img = img * (2**8 - 1)
    
    # Compute luminance for RGB image
    luminance = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    
    # Plot histogram
    luminance_flat = luminance.flatten()
    hist, bins = np.histogram(luminance_flat, bins=256, density=True)
    
    # Calculate entropy
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.bar(bins[:-1], hist, color='blue', alpha=0.7, width=np.diff(bins)[0])
    ax.set_xlabel('Luminance')
    ax.set_ylabel('Probability')
    ax.text(0.5, 0.9, f'Entropy: {entropy:.2f} bits', fontsize=11, ha='center', transform=ax.transAxes)
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf)
    image_tensor = to_tensor(image)
    
    return image_tensor

# * Histogram visualize for tensorboard logging as image file
def visualize_histogram(img):
    histo = calculate_histogram_global(img)
    values = histo[0][0].cpu().numpy()
    frequency = np.arange(len(values))
    fig = plt.figure(figsize=(10,8))
    plt.bar(frequency, values, width=2.0)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of green channel intensity')

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = Image.open(buf)
    image = np.array(image)
    
    return image

def plot_to_image(figure, dpi=200):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and returns it as a numpy array."""
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=dpi)
    buf.seek(0)
    image = Image.open(buf)
    return np.array(image)

def visualize_tensor_with_DR(writer, step, hdr, ldr1, ldr2):
    image_np_hdr = hdr.squeeze(0).permute(1,2,0).cpu().numpy()
    image_np_ldr1 = ldr1.squeeze(0).permute(1,2,0).cpu().numpy()
    image_np_ldr2 = ldr2.squeeze(0).permute(1,2,0).cpu().numpy()
    
    hdr_dr = image_np_hdr.max()
    ldr_dr1 = image_np_ldr1.max()/image_np_ldr1.min()
    ldr_dr2 = image_np_ldr2.max()/image_np_ldr2.min()

    plt.figure(figsize=(18, 6))

    # # HDR 이미지 시각화
    # plt.subplot(1, 3, 1)
    # plt.title('HDR Image')
    # plt.imshow(image_np_hdr_norm)
    # plt.text(10, 10, f'DR: {hdr_dr:.2f} dB', color='white', backgroundcolor='black', fontsize=12, ha='left')
    # plt.axis('off')

    # # LDR 이미지 시각화
    # plt.subplot(1, 3, 2)
    # plt.title('LDR Image')
    # plt.imshow(image_np_ldr_norm)
    # plt.text(10, 10, f'DR: {ldr_dr:.2f} dB', color='white', backgroundcolor='black', fontsize=12, ha='left')
    # plt.axis('off')

    # 히스토그램 시각화
    # plt.subplot(1, 3, 3)
    plt.title('Dynamic Range Visualization')
    plt.hist(image_np_hdr.ravel(), bins=256, color='orange', alpha=0.75, label='HDR')
    plt.hist(image_np_ldr1.ravel(), bins=256, color='green', alpha=0.75, label='LDR1')
    plt.hist(image_np_ldr2.ravel(), bins=256, color='blue', alpha=0.75, label='LDR2')
    plt.yscale('log')
    plt.xlabel('Pixel Intensity (log scale)')
    plt.ylabel('Frequency')
    plt.legend()

    # plt.subplots_adjust(wspace=0.2, hspace=0.2)

    # 이미지로 변환하여 TensorBoard에 기록
    image_buf = plot_to_image(plt.gcf())
    writer.add_image('HDR_LDR_Comparison', image_buf, step, dataformats='HWC')
    plt.close()

def visualize_tensor_with_DR_save(writer, step, hdr, ldr1, ldr2, num_cnt):
    image_np_hdr = hdr.squeeze(0).permute(1,2,0).cpu().numpy()
    image_np_ldr1 = ldr1.squeeze(0).permute(1,2,0).cpu().numpy()
    image_np_ldr2 = ldr2.squeeze(0).permute(1,2,0).cpu().numpy()
    
    hdr_dr = image_np_hdr.max()
    ldr_dr1 = image_np_ldr1.max()/image_np_ldr1.min()
    ldr_dr2 = image_np_ldr2.max()/image_np_ldr2.min()

    plt.figure(figsize=(18, 6))

    plt.title('Dynamic Range Visualization')
    plt.hist(image_np_hdr.ravel(), bins=256, color='orange', alpha=0.75, label='HDR')
    plt.hist(image_np_ldr1.ravel(), bins=256, color='green', alpha=0.75, label='LDR1')
    plt.hist(image_np_ldr2.ravel(), bins=256, color='blue', alpha=0.75, label='LDR2')
    plt.yscale('log')
    plt.xlabel('Pixel Intensity (log scale)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f'test/sample/dynamic_range/{num_cnt}.png')

    # plt.subplots_adjust(wspace=0.2, hspace=0.2)

    # 이미지로 변환하여 TensorBoard에 기록
    image_buf = plot_to_image(plt.gcf())
    writer.add_image('HDR_LDR_Comparison', image_buf, step, dataformats='HWC')
    plt.close()

# For feature map logging
def log_feature_map(writer, feature_map, tag, step):
    
    first_sample = feature_map[0]
    # Normalize feature map to [0, 1] for visualization
    normalized_feature_map = (first_sample - first_sample.min()) / (first_sample.max() - first_sample.min())
    # Select first n feature maps (e.g., first 1 feature maps)
    n = min(1, normalized_feature_map.size(0))  # Select how many feature maps you want to log
    grid = vutils.make_grid(normalized_feature_map[:n], normalize=True, scale_each=True)
    # Log the feature map grid to TensorBoard
    writer.add_image(tag, grid, global_step=step)
    

def log_multiple_feature_map(writer, feature_map, tag, step, num_channels=4):
    # 첫 번째 배치에서 num_channels개의 채널 선택
    first_sample = feature_map[0, :num_channels]

    first_sample = first_sample.unsqueeze(1)  # [C, 1, H, W]

    # Normalize feature map to [0, 1] for visualization
    normalized_feature_map = (first_sample - first_sample.min()) / (first_sample.max() - first_sample.min())
    
    # Make grid of selected channels as separate grayscale images
    grid = vutils.make_grid(normalized_feature_map, nrow=num_channels, normalize=True, scale_each=True)
    
    writer.add_image(tag, grid, global_step=step)

def visualize_flow(writer, flow, tag, step):
    flow_rgb = flow_utils.flow_to_rgb(flow)
    
    writer.add_image(tag, flow_rgb[0], global_step = step)
    
def consine_similarity(feature_map1, feature_map2):
    feature_map1 = feature_map1.view(feature_map1.size(0), -1)  # Flattening
    feature_map2 = feature_map2.view(feature_map2.size(0), -1)  # Flattening
    similarity = F.cosine_similarity(feature_map1, feature_map2)
    return similarity.item()

def normalized_cosine_similarity(fmap1, fmap2):
    # Flatten and normalize each feature map
    fmap1_flat = fmap1.view(fmap1.size(0), -1)  # Flatten [B, C, H, W] -> [B, C*H*W]
    fmap2_flat = fmap2.view(fmap2.size(0), -1)  # Flatten [B, C, H, W] -> [B, C*H*W]
    
    fmap1_normalized = F.normalize(fmap1_flat, p=2, dim=1)
    fmap2_normalized = F.normalize(fmap2_flat, p=2, dim=1)
    
    cos_sim = F.cosine_similarity(fmap1_normalized, fmap2_normalized, dim=1)
    return cos_sim.mean().item()  # Average cosine similarity across the batch

# Loggig feature map differences
def log_difference_map(writer, fmap1, warped_fmap, fused_fmap, step, prefix="Train"):
    # Calculate the difference
    diff_fmap1 = torch.abs(fmap1 - fused_fmap)
    diff_warped_fmap = torch.abs(warped_fmap - fused_fmap)

    # Normalize the difference for visualization
    diff_fmap1 = (diff_fmap1 - diff_fmap1.min()) / (diff_fmap1.max() - diff_fmap1.min())
    diff_warped_fmap = (diff_warped_fmap - diff_warped_fmap.min()) / (diff_warped_fmap.max() - diff_warped_fmap.min())

    # Select one channel for visualization (or take the mean across channels)
    diff_fmap1 = diff_fmap1[:, 0:1, :, :]
    diff_warped_fmap = diff_warped_fmap[:, 0:1, :, :]

    # Convert the tensor to numpy for matplotlib
    diff_fmap1_np = diff_fmap1[0].detach().cpu().numpy().squeeze()
    diff_warped_fmap_np = diff_warped_fmap[0].detach().cpu().numpy().squeeze()

    # Create figure and axes
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot fmap1 vs fused map difference
    im1 = axs[0].imshow(diff_fmap1_np, cmap='viridis')
    axs[0].set_title('Difference: Fmap1 vs Fused')
    fig.colorbar(im1, ax=axs[0])

    # Plot warped fmap vs fused map difference
    im2 = axs[1].imshow(diff_warped_fmap_np, cmap='viridis')
    axs[1].set_title('Difference: Warped Fmap vs Fused')
    fig.colorbar(im2, ax=axs[1])

    # Save the plot as a numpy array
    fig.canvas.draw()
    plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Close the plot to free memory
    plt.close(fig)

    # Log the plot image to TensorBoard
    writer.add_image(f'{prefix}/Difference_Maps', plot_image, step, dataformats='HWC')
