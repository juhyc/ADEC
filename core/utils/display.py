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
