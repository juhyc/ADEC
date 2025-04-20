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
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import hsv_to_rgb
from matplotlib.colors import Normalize
from matplotlib import scale

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


def visualize_disparity_with_colorbar(batch_image, vmin=None, vmax=None, norm=False, title='Disparity Map'):
    image_tensor = -batch_image[0].clone().detach()
    image = image_tensor.cpu().numpy().squeeze()
    
    if norm:
        image = (image - image.min())/(image.max() - image.min())

    if vmin is None:
        vmin = image.min()
    if vmax is None:
        vmax = image.max()        
        
    dpi = 200 
    fig, ax = plt.subplots(figsize=(6, 3), dpi=dpi)
    
    cmap = plt.cm.magma
    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    ax.set_title(title)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Disparity Value')

    plt.tight_layout(pad=0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    
    image_pil = Image.open(buf)
    image_array = np.array(image_pil)
    image_array = np.transpose(image_array, (2, 0, 1))

    return image_array

def visualize_disparity_with_colorbar_svg(disp_map, vmin=None, vmax=None, norm=False, title='Disparity Map'):
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
        
    # Create a figure for the disparity map with colorbar
    fig, ax = plt.subplots(figsize=(6, 3), dpi=200)
    cmap = plt.cm.magma
    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    ax.set_title(title)

    # Add a colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Disparity Value')

    # Adjust layout to remove padding and extra whitespace
    plt.tight_layout(pad=0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    return fig

def visualize_disparity_only(disp_map, vmin=None, vmax=None, norm=False):
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
        
    # Create a figure for the disparity map only
    fig, ax = plt.subplots(figsize=(6, 3), dpi=200)
    cmap = plt.cm.magma
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')  # Hide axes

    return fig


def visualize_lidar_with_colorbar(batch_image, vmin=None, vmax=None, norm=False, title='Lidar'):
    image_tensor = batch_image[0].clone().detach()
    image = image_tensor.cpu().numpy().squeeze()
    
    if norm:
        image = (image - image.min())/(image.max() - image.min())

    if vmin is None:
        vmin = image.min()
    if vmax is None:
        vmax = image.max()        
        
    dpi = 200 
    fig, ax = plt.subplots(figsize=(6, 3), dpi=dpi)
    
    cmap = plt.cm.magma
    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    ax.set_title(title)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Disparity Value')

    plt.tight_layout(pad=0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    
    image_pil = Image.open(buf)
    image_array = np.array(image_pil)
    image_array = np.transpose(image_array, (2, 0, 1))

    return image_array

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
    
    # For tensorboard logging
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches = 'tight')
    buf.seek(0)
    image = Image.open(buf)
    image = np.array(image)
    image = np.transpose(image, (2, 0, 1))  # HWC to CHW
    
    return torch.tensor(image/255.0).float()


def visualize_mask(writer, batch_image, tag, step):
    
    image_tensor = batch_image[0].clone().detach()
    image = image_tensor.cpu().numpy().squeeze()
    
    colored_image = plt.cm.gray(image)
    colored_image = (colored_image[..., : 3] * 255).astype(np.uint8)
    colored_image = np.transpose(colored_image, (2,0,1))
    
    writer.add_image(tag, colored_image, step)

def log_mask_to_tensorboard(writer, mask_batch, tag, step):

    mask_image = mask_batch[0].clone().detach().cpu().numpy().squeeze()

    mask_image_normalized = (mask_image - mask_image.min()) / (mask_image.max() - mask_image.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(6, 3))

    im = ax.imshow(mask_image_normalized, cmap='gray', vmin=0, vmax=1)
    ax.axis('off')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Mask Value')

    plt.tight_layout(pad=0)

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    image = Image.open(buf).convert('RGB')
    image = transforms.ToTensor()(image)

    writer.add_image(tag, image, global_step=step)
    
def log_mask_to_tensorboard2(writer, mask_batch, tag, step):

    mask_image = mask_batch[0][0].clone().detach().cpu().numpy()

    mask_image_normalized = (mask_image - mask_image.min()) / (mask_image.max() - mask_image.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(6, 3))

    im = ax.imshow(mask_image_normalized, cmap='gray', vmin=0, vmax=1)
    ax.axis('off')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Mask Value')

    plt.tight_layout(pad=0)

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    image = Image.open(buf).convert('RGB')
    image = transforms.ToTensor()(image)

    writer.add_image(tag, image, global_step=step)


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

def visualize_tensor_with_DR(writer, step, tag, hdr, ldr1, ldr2):
    image_np_hdr = hdr.squeeze(0).permute(1,2,0).cpu().numpy()
    image_np_ldr1 = ldr1.squeeze(0).permute(1,2,0).cpu().numpy()
    image_np_ldr2 = ldr2.squeeze(0).permute(1,2,0).cpu().numpy()
    
    hdr_dr = image_np_hdr.max()
    ldr_dr1 = image_np_ldr1.max()/image_np_ldr1.min()
    ldr_dr2 = image_np_ldr2.max()/image_np_ldr2.min()

    plt.figure(figsize=(18, 6))

    plt.title('Dynamic Range Visualization')
    plt.hist(image_np_hdr.ravel(), bins=128, color='orange', alpha=0.75, label='Scene radiance')
    plt.hist(image_np_ldr1.ravel(), bins=128, color='blue', alpha=0.75, label='Cap1')
    plt.hist(image_np_ldr2.ravel(), bins=128, color='red', alpha=0.75, label='Cap2')
    plt.yscale('log')
    plt.xlabel('Scaled pixel intensity')
    plt.ylabel('Frequency')
    plt.legend(fontsize=20)

    # plt.subplots_adjust(wspace=0.2, hspace=0.2)

    image_buf = plot_to_image(plt.gcf())
    writer.add_image(f'{tag}/Dynamic Range', image_buf, step, dataformats='HWC')
    plt.close()

def visualize_DR_overay_3img(writer, step, tag, hdr, ldr1, ldr2):
    # Image to numpy array
    image_np_hdr = hdr.squeeze(0).permute(1,2,0).cpu().numpy()
    image_np_ldr1 = ldr1.squeeze(0).permute(1,2,0).cpu().numpy()
    image_np_ldr2 = ldr2.squeeze(0).permute(1,2,0).cpu().numpy()

    # Dynamic range
    hdr_dr = image_np_hdr.max()
    ldr_dr1 = image_np_ldr1.max() / image_np_ldr1.min()
    ldr_dr2 = image_np_ldr2.max() / image_np_ldr2.min()

    # Visualize histogream
    plt.figure(figsize=(18, 6))
    plt.title('Dynamic Range Visualization')

    plt.hist(image_np_hdr.ravel(), bins=512, color='orange', alpha=0.5, label=f'Scene radiance (DR: {hdr_dr:.2f})')

    # Add two LDR histogram on HDR histogram
    plt.hist(image_np_ldr1.ravel(), bins=128, color='blue', alpha=0.5, label=f'Cap1 (DR: {ldr_dr1:.2f})')
    plt.hist(image_np_ldr2.ravel(), bins=128, color='red', alpha=0.5, label=f'Cap2 (DR: {ldr_dr2:.2f})')

    # Set-axis
    plt.yscale('log')
    plt.xlabel('Scaled pixel intensity')
    plt.ylabel('Frequency')
    plt.legend(fontsize=15)

    # Tensorboard logging
    image_buf = plot_to_image(plt.gcf())
    writer.add_image(f'{tag}/Dynamic Range', image_buf, step, dataformats='HWC')
    plt.close()


def visualize_tensor_with_DR_single(writer, step, tag, hdr, ldr1):
    image_np_hdr = hdr.squeeze(0).permute(1,2,0).cpu().numpy()
    image_np_ldr1 = ldr1.squeeze(0).permute(1,2,0).cpu().numpy()
    
    hdr_dr = image_np_hdr.max()
    ldr_dr1 = image_np_ldr1.max()/image_np_ldr1.min()

    plt.figure(figsize=(18, 6))

    plt.title('Dynamic Range Visualization')
    plt.hist(image_np_hdr.ravel(), bins=128, color='orange', alpha=0.75, label='Scene radiance')
    plt.hist(image_np_ldr1.ravel(), bins=128, color='green', alpha=0.75, label='Cap1')
    plt.yscale('log')
    plt.xlabel('Scaled pixel intensity')
    plt.ylabel('Frequency')
    plt.legend(fontsize=20)

    # plt.subplots_adjust(wspace=0.2, hspace=0.2)

    image_buf = plot_to_image(plt.gcf())
    writer.add_image(f'{tag}/Dynamic Range', image_buf, step, dataformats='HWC')
    plt.close()

def visualize_tensor_with_DR_save(writer, step, tag, hdr, ldr1, ldr2, save_dir='test_results_real_lidar'):
    # HDR, LDR1, LDR2 이미지를 numpy 형식으로 변환
    image_np_hdr = hdr.squeeze(0).permute(1,2,0).cpu().numpy()
    image_np_ldr1 = ldr1.squeeze(0).permute(1,2,0).cpu().numpy()
    image_np_ldr2 = ldr2.squeeze(0).permute(1,2,0).cpu().numpy()
    
    # 히스토그램 그리기
    plt.figure(figsize=(18, 6))
    plt.title('Dynamic Range Visualization (Log Scale)')
    plt.hist(image_np_hdr.ravel() + 1e-6, bins=128, color='orange', alpha=0.75, label='Scene radiance')  # 작은 값 추가
    plt.hist(image_np_ldr1.ravel() + 1e-6, bins=128, color='green', alpha=0.75, label='Captured intensity1')  # 작은 값 추가
    plt.hist(image_np_ldr2.ravel() + 1e-6, bins=128, color='blue', alpha=0.75, label='Captured intensity2')  # 작은 값 추가
    plt.yscale('log')
    plt.xscale('log')  # x축을 로그 스케일로 설정
    plt.xlabel('Scaled pixel intensity (log)')
    plt.ylabel('Frequency (log scale)')
    plt.legend(fontsize=20)

    # TensorBoard에 이미지로 기록
    image_buf = plot_to_image(plt.gcf())
    writer.add_image(f'{tag}_Dynamic Range', image_buf, step, dataformats='HWC')

    # 히스토그램을 디렉토리에 저장
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f'{tag}_dynamic_range_step_{step}.svg'), format='svg')

    plt.close()


def visualize_tensor_with_DR_save(writer, step, tag, hdr, ldr1, ldr2, save_dir='test_results_real_lidar'):
    
    image_np_hdr = hdr.squeeze(0).permute(1,2,0).cpu().numpy()
    image_np_ldr1 = ldr1.squeeze(0).permute(1,2,0).cpu().numpy()
    image_np_ldr2 = ldr2.squeeze(0).permute(1,2,0).cpu().numpy()
    
    # Histogram
    plt.figure(figsize=(18, 6))
    plt.title('Dynamic Range Visualization (Log2 Scale)')
    plt.hist(image_np_hdr.ravel() + 1e-6, bins=128, color='orange', alpha=0.75, label='Scene radiance') 
    plt.hist(image_np_ldr1.ravel() + 1e-6, bins=128, color='green', alpha=0.75, label='Captured intensity1') 
    plt.hist(image_np_ldr2.ravel() + 1e-6, bins=128, color='blue', alpha=0.75, label='Captured intensity2') 
    plt.yscale('log')
    plt.xscale(scale.LogScale(plt.gca(), base=2))  # x축을 log2 스케일로 설정
    plt.xlabel('Scaled pixel intensity (log2 scale)')
    plt.ylabel('Frequency (log scale)')
    plt.legend(fontsize=20)

    image_buf = plot_to_image(plt.gcf())
    writer.add_image(f'{tag}_Dynamic Range', image_buf, step, dataformats='HWC')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f'{tag}_dynamic_range_step_{step}.svg'), format='svg')

    plt.close()
    
def visualize_tensor_with_DR_single_save(writer, step, tag, hdr, ldr1, save_dir='test_results_real_lidar'):

    image_np_hdr = hdr.squeeze(0).permute(1,2,0).cpu().numpy()
    image_np_ldr1 = ldr1.squeeze(0).permute(1,2,0).cpu().numpy()

    plt.figure(figsize=(18, 6))
    plt.title('Dynamic Range Visualization (Log2 Scale)')
    plt.hist(image_np_hdr.ravel() + 1e-6, bins=128, color='orange', alpha=0.75, label='Scene radiance')  
    plt.hist(image_np_ldr1.ravel() + 1e-6, bins=128, color='green', alpha=0.75, label='Captured intensity')  
    plt.yscale('log')
    plt.xscale(scale.LogScale(plt.gca(), base=2))  
    plt.xlabel('Scaled pixel intensity (log2 scale)')
    plt.ylabel('Frequency (log scale)')
    plt.legend(fontsize=20)

    image_buf = plot_to_image(plt.gcf())
    writer.add_image(f'{tag}_Dynamic Range', image_buf, step, dataformats='HWC')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f'{tag}_dynamic_range_step_{step}.svg'), format='svg')

    plt.close()
    
    
def visualize_hdr_dr(writer, step, hdr):
    image_np_hdr = hdr.squeeze(0).permute(1,2,0).cpu().numpy()
    
    hdr_dr = image_np_hdr.max()

    plt.figure(figsize=(18, 6))

    plt.title('Dynamic Range Visualization')
    plt.hist(image_np_hdr.ravel(), bins=256, color='orange', alpha=0.75, label='HDR')
    plt.yscale('log')
    plt.xlabel('Pixel Intensity (log scale)')
    plt.ylabel('Frequency')
    plt.legend()

    image_buf = plot_to_image(plt.gcf())
    writer.add_image('Dynamic range histogram', image_buf, step, dataformats='HWC')
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
    # Select feature map channel
    first_sample = feature_map[0, :num_channels]

    first_sample = first_sample.unsqueeze(1)  # [C, 1, H, W]

    # Normalize feature map to [0, 1] for visualization
    normalized_feature_map = (first_sample - first_sample.min()) / (first_sample.max() - first_sample.min())
    
    # Make grid of selected channels as separate grayscale images
    grid = vutils.make_grid(normalized_feature_map, nrow=num_channels, normalize=True, scale_each=True)
    
    writer.add_image(tag, grid, global_step=step)

def visualize_flow(writer, flow, tag, step):
    flow_rgb = flow_utils.flow_to_rgb(flow)
    # print(f"Shape of flow : {flow_rgb.shape}")
    
    writer.add_image(tag, flow_rgb[0], global_step = step)
    
def visualize_flow_colorbar(writer, flow, tag, step):
    # Flow magnitude
    flow_magnitude = torch.sqrt(flow[0, 0]**2 + flow[0, 1]**2).cpu().numpy()

    # flow to RGB mapping
    flow_rgb = flow_utils.flow_to_rgb(flow) 
    flow_rgb_np = flow_rgb[0].cpu().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)


    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(flow_rgb_np)
    ax.axis('off') 

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    norm = Normalize(vmin=flow_magnitude.min(), vmax=flow_magnitude.max())
    
    colorbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='hsv'), cax=cax, orientation="vertical")
    colorbar.set_label("Flow Magnitude", rotation=270, labelpad=15)
    
    writer.add_figure(tag, fig, global_step=step)
    plt.close(fig)
    
def visualize_flow_with_sphere(writer, flow, tag, step):

    flow_rgb = flow_utils.flow_to_rgb(flow)  
    flow_rgb_np = flow_rgb[0].cpu().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)

    # Set magnitude and angle
    flow_magnitude = torch.sqrt(flow[0, 0]**2 + flow[0, 1]**2).cpu().numpy()
    flow_angle = torch.atan2(flow[0, 1], flow[0, 0]).cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Visualize optical flow
    im = ax1.imshow(flow_rgb_np)
    ax1.axis('off')
    ax1.set_title("Optical Flow Map")

    sphere_size = 200  
    y, x = np.meshgrid(np.linspace(-1, 1, sphere_size), np.linspace(-1, 1, sphere_size))
    mask = x**2 + y**2 <= 1 
    angle = np.arctan2(y, x)  
    magnitude = np.sqrt(x**2 + y**2)  
    

    hsv_sphere = np.ones((sphere_size, sphere_size, 3))  
    hsv_sphere[..., 0] = (angle + np.pi) / (2 * np.pi)  
    hsv_sphere[..., 1] = 1.0 
    hsv_sphere[..., 2] = np.clip(magnitude, 0, 1)  

    # HSV -> RGB
    rgb_sphere = hsv_to_rgb(hsv_sphere)
    rgb_sphere[~mask] = 1  

    ax2.imshow(rgb_sphere)
    ax2.axis('off')
    # ax2.set_title("Optical Flow Color Sphere")


    ax2.text(sphere_size // 2, -10, 'Right', ha='center', va='center')
    ax2.text(sphere_size // 2, sphere_size + 10, 'Left', ha='center', va='center')
    ax2.text(-10, sphere_size // 2, 'Up', ha='center', va='center', rotation=90)
    ax2.text(sphere_size + 10, sphere_size // 2, 'Down', ha='center', va='center', rotation=270)
    # ax2.text(sphere_size // 2, sphere_size // 2, 'Low Magnitude', ha='center', va='center', color='black')
    # ax2.text(sphere_size // 2, sphere_size // 2 + 70, 'High Magnitude', ha='center', va='center', color='black')

    writer.add_figure(tag, fig, global_step=step)
    plt.close(fig)


    
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
    
def log_feature_map_with_colorbar(writer, feature_map, tag, step, num_channels=4):

    first_sample = feature_map[0, :num_channels]

    first_sample = first_sample.unsqueeze(1)  # [C, 1, H, W]

    # # Normalize feature map to [0, 1] for visualization
    # normalized_feature_map = (first_sample - first_sample.min()) / (first_sample.max() - first_sample.min())
    
    # Make grid of selected channels as separate grayscale images
    grid = vutils.make_grid(feature_map, nrow=num_channels, normalize=True, scale_each=True)
    
    # Add image to TensorBoard without color bar
    writer.add_image(tag, grid, global_step=step)

    # Convert grid to numpy for matplotlib visualization
    grid_np = grid.permute(1, 2, 0).cpu().numpy()

    # Plotting the grid with color bar using matplotlib
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_np, cmap='gray', interpolation='nearest')
    plt.colorbar()  # Add color bar to the side
    plt.title(f'{tag} at step {step}')
    plt.show()

def log_multiple_feature_map(writer, feature_map, tag, step, num_channels=4):
    first_sample = feature_map[0, :num_channels]

    first_sample = first_sample.unsqueeze(1)  # [C, 1, H, W]

    # Normalize feature map to [0, 1] for visualization
    normalized_feature_map = (first_sample - first_sample.min()) / (first_sample.max() - first_sample.min())
    
    # Make grid of selected channels as separate grayscale images
    grid = vutils.make_grid(normalized_feature_map, nrow=num_channels, normalize=True, scale_each=True)
    
    writer.add_image(tag, grid, global_step=step)
    

def log_multiple_feature_map_with_colorbar(writer, feature_map, tag, step, num_channels=4):

    first_sample = feature_map[0, :num_channels].detach().cpu().numpy()  # [C, H, W]

    for idx in range(num_channels):
        channel_map = first_sample[idx]  # [H, W]

        fig, ax = plt.subplots(figsize=(6, 3))

        im = ax.imshow(channel_map, cmap='RdGy', vmin = -3.0, vmax = 3.0)
        ax.axis('off')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        # cbar.ax.tick_params(labelsize=8)  
        # cbar.set_label('Feature Value')  

        plt.tight_layout(pad=0)

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)

        image = Image.open(buf)
        image = transforms.ToTensor()(image)

        writer.add_image(f'{tag}/channel_{idx}', image, global_step=step)

    
def log_average_feature_map(writer, feature_map1, feature_map2, tag, step, num_channels=4):
    first_sample1 = feature_map1[0, :num_channels]
    first_sample2 = feature_map2[0, :num_channels]

    averaged_feature_map = (first_sample1 + first_sample2) / 2

    averaged_feature_map = averaged_feature_map.unsqueeze(1)  # [C, 1, H, W]

    # Normalize feature map to [0, 1] for visualization
    normalized_feature_map = (averaged_feature_map - averaged_feature_map.min()) / (averaged_feature_map.max() - averaged_feature_map.min())
    
    # Make grid of selected channels as separate grayscale images
    grid = vutils.make_grid(normalized_feature_map, nrow=num_channels, normalize=True, scale_each=True)
    
    writer.add_image(tag, grid, global_step=step)

def log_edge_weight_to_tensorboard(writer, edge_weight, step):
    edge_weight_vis = edge_weight[0, 0].detach().cpu().numpy() 
    edge_weight_vis = (edge_weight_vis - edge_weight_vis.min()) / (edge_weight_vis.max() - edge_weight_vis.min())  # Normalize
    writer.add_image('Edge_Weight', torch.tensor(edge_weight_vis).unsqueeze(0), step, dataformats='CHW')

def plot_histogram(image_tensor, title="Histogram"):
    image_np = image_tensor.cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)
    
    # Histogram
    plt.figure(figsize=(10, 6), dpi=150)  
    plt.hist(image_np.ravel(), bins=256, range=(0, 255), fc='green', ec='green')
    plt.yscale('log', base=10)
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency (log scale, base 10)")

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    img = np.array(img)
    plt.close()

    return torch.tensor(img).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]


def plot_lidar_points(points, image_left, vmax=20000):
    """
    Project LiDAR points onto the image and return the plotted figure.

    Args:
        points (torch.Tensor): LiDAR 3D points to be projected
        image_left (torch.Tensor): Left camera image for overlay
        vmax (int): Maximum depth for color scaling

    Returns:
        fig (matplotlib.figure.Figure): The figure containing the plot
    """
    # Tensor를 numpy로 변환
    points = points.cpu().numpy()
    image_left = image_left.cpu().numpy().transpose(1, 2, 0)  # (height, width, channels) 형식으로 변환

    
    # points 좌표를 이미지 크기로 클리핑
    points[:, 0] = np.clip(points[:, 0], 0, image_left.shape[1] - 1)
    points[:, 1] = np.clip(points[:, 1], 0, image_left.shape[0] - 1)
    
    # 시각화
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].imshow(image_left)
    ax[0].set_title("Left Rectified Image")
    
    ax[1].imshow(np.zeros((image_left.shape[0], image_left.shape[1])), cmap='gray')
    scatter = ax[1].scatter(points[:, 0], points[:, 1], s=0.5, c=points[:, 2], cmap='magma', vmax=vmax)
    fig.colorbar(scatter, ax=ax[1], label="Depth (mm)")
    ax[1].set_title("Projected LiDAR Points")
    
    return fig

def plot_lidar_points_disp(points, image_left, focal_length, baseline, vmax=30):
    """
    Project LiDAR points onto the image and visualize using disparity values.

    Args:
        points (torch.Tensor): LiDAR 3D points (N, 3)
        image_left (torch.Tensor): Left image (C, H, W)
        focal_length (float or torch.Tensor): Camera focal length
        baseline (float or torch.Tensor): Stereo baseline
        vmax (float): Maximum disparity value for color scaling

    Returns:
        fig (matplotlib.figure.Figure): Visualization figure
    """
    # Tensor to numpy
    points = points.cpu().numpy()
    image_left = image_left.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)

    # Focal & baseline to float
    if isinstance(focal_length, torch.Tensor):
        focal_length = focal_length.item()
    if isinstance(baseline, torch.Tensor):
        baseline = baseline.item()

    # Compute disparity: disparity = (f * B) / Z
    disparities = (focal_length * baseline) / (points[:, 2] + 1e-6)  # Avoid divide by zero

    # Clip x/y for visualization
    points[:, 0] = np.clip(points[:, 0], 0, image_left.shape[1] - 1)
    points[:, 1] = np.clip(points[:, 1], 0, image_left.shape[0] - 1)

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].imshow(image_left)
    ax[0].set_title("Left Rectified Image")

    ax[1].imshow(np.zeros((image_left.shape[0], image_left.shape[1])), cmap='gray')
    scatter = ax[1].scatter(points[:, 0], points[:, 1], s=0.5, c=disparities, cmap='magma', vmin=0, vmax=vmax)
    fig.colorbar(scatter, ax=ax[1], label="Disparity (pixels)")
    ax[1].set_title("Projected LiDAR Points (Disparity View)")

    return fig


def plot_lidar_points_disp_only(points, image_shape, focal_length, baseline, vmax=30):
    """
    Visualize LiDAR points using disparity color map only (no axis, title, colorbar).

    Args:
        points (torch.Tensor): LiDAR 3D points (N, 3)
        image_shape (tuple): Image shape as (C, H, W) or (H, W)
        focal_length (float or torch.Tensor): Focal length
        baseline (float or torch.Tensor): Stereo baseline
        vmax (float): Max disparity value for color scale

    Returns:
        fig (matplotlib.figure.Figure): The figure with the visualization
    """
    # Convert to numpy
    points = points.cpu().numpy()
    if isinstance(focal_length, torch.Tensor):
        focal_length = focal_length.item()
    if isinstance(baseline, torch.Tensor):
        baseline = baseline.item()

    # Image shape 
    if len(image_shape) == 3:
        H, W = image_shape[1], image_shape[2]
    else:
        H, W = image_shape

    # disparity 
    disparities = (focal_length * baseline) / (points[:, 2] + 1e-6)

    # Clipping
    points[:, 0] = np.clip(points[:, 0], 0, W - 1)
    points[:, 1] = np.clip(points[:, 1], 0, H - 1)

    # Visualize
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(np.zeros((H, W)), cmap='gray') 
    ax.scatter(points[:, 0], points[:, 1], s=0.5, c=disparities, cmap='magma', vmin=0, vmax=vmax)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    return fig

def visualize_tensor_3mg_with_DR(writer, step, tag, save_dir, image_tensor_hdr, image_tensor_ldr1, image_tensor_ldr2):
    # Convert tensors to numpy arrays
    image_np_hdr = image_tensor_hdr.squeeze(0).permute(1,2,0).cpu().numpy()
    image_np_ldr1 = image_tensor_ldr1.squeeze(0).permute(1,2,0).cpu().numpy()
    image_np_ldr2 = image_tensor_ldr2.squeeze(0).permute(1,2,0).cpu().numpy()
    
    # Normalize images for display
    image_np_hdr_norm = (image_np_hdr - image_np_hdr.min()) / (image_np_hdr.max() - image_np_hdr.min())
    image_np_ldr_norm1 = (image_np_ldr1 - image_np_ldr1.min()) / (image_np_ldr1.max() - image_np_ldr1.min())
    image_np_ldr_norm2 = (image_np_ldr2 - image_np_ldr2.min()) / (image_np_ldr2.max() - image_np_ldr2.min())

    # Calculate dynamic ranges in dB
    hdr_max = image_np_hdr.max()
    hdr_min = image_np_hdr[image_np_hdr > 0].min()
    ldr_max1 = image_np_ldr1.max()
    ldr_min1 = image_np_ldr1[image_np_ldr1 > 0].min()
    ldr_max2 = image_np_ldr2.max()
    ldr_min2 = image_np_ldr2[image_np_ldr2 > 0].min()
    
    hdr_dr = 20 * np.log10(hdr_max / hdr_min)
    ldr_dr1 = 20 * np.log10(ldr_max1 / ldr_min1)
    ldr_dr2 = 20 * np.log10(ldr_max2 / ldr_min2)
    
    # Map LDR bounds to the same range as HDR for consistent visual scaling
    hdr_range = hdr_max - hdr_min
    ldr1_mapped_min = (ldr_min1 - hdr_min) / hdr_range
    ldr1_mapped_max = (ldr_max1 - hdr_min) / hdr_range
    ldr2_mapped_min = (ldr_min2 - hdr_min) / hdr_range
    ldr2_mapped_max = (ldr_max2 - hdr_min) / hdr_range

    # Calculate combined coverage using the min of both LDR lower bounds and max of upper bounds
    combined_min = min(ldr_min1, ldr_min2)
    combined_max = max(ldr_max1, ldr_max2)
    combined_dr = 20 * np.log10(combined_max / combined_min)

    # Print coverage information
    print(f"LDR1 covers {ldr_dr1:.2f} dB of the dynamic range.")
    print(f"LDR2 covers {ldr_dr2:.2f} dB of the dynamic range.")
    print(f"Combined LDR1 and LDR2 cover {combined_dr:.2f} dB of the dynamic range.")

    # Plotting the images and histograms
    plt.figure(figsize=(18, 6))
    
    # LDR1 Image
    plt.subplot(2, 2, 1)
    plt.title('Captured intensity1')
    plt.imshow(image_np_ldr_norm1 ** (1 / 2.2))
    plt.text(10, 10, f'DR: {ldr_dr1:.2f} dB', color='white', backgroundcolor='black', fontsize=12, ha='left')
    plt.axis('off')

    # HDR Image
    plt.subplot(2, 2, 2)
    plt.title('Scene radiance')
    plt.imshow(image_np_hdr_norm ** (1 / 2.2))
    plt.text(10, 10, f'DR: {hdr_dr:.2f} dB', color='white', backgroundcolor='black', fontsize=12, ha='left')
    plt.axis('off')
    
    # LDR2 Image
    plt.subplot(2, 2, 3)
    plt.title('Captured intensity2')
    plt.imshow(image_np_ldr_norm2 ** (1 / 2.2))
    plt.text(10, 10, f'DR: {ldr_dr2:.2f} dB', color='white', backgroundcolor='black', fontsize=12, ha='left')
    plt.axis('off')

    # Dynamic Range Visualization with normalized histograms
    plt.subplot(2, 2, 4)
    plt.title('Dynamic Range Visualization')
    
    # HDR Histogram with log scale on the x-axis
    plt.hist(image_np_hdr.ravel(), bins=2048, color='orange', alpha=0.75, label='Scene radiance')
    
    # Draw vertical lines to indicate the bounds for each LDR image on a normalized scale
    plt.axvline(x=ldr1_mapped_min * hdr_max, color='green', linestyle='--', label='Cap1 Bounds')
    plt.axvline(x=ldr1_mapped_max * hdr_max, color='green', linestyle='--')
    plt.axvline(x=ldr2_mapped_min * hdr_max, color='blue', linestyle='--', label='Cap2 Bounds')
    plt.axvline(x=ldr2_mapped_max * hdr_max, color='blue', linestyle='--')
    
    # Normalize LDR images to the HDR range for consistent display in log scale
    ldr1_mapped = (image_np_ldr1 - hdr_min) / hdr_range * hdr_max
    ldr2_mapped = (image_np_ldr2 - hdr_min) / hdr_range * hdr_max
    
    plt.hist(ldr1_mapped.ravel(), bins=256, color='green', alpha=0.5, label='Captured intensity1')
    plt.hist(ldr2_mapped.ravel(), bins=256, color='blue', alpha=0.5, label='Captured intensity2')
    
    plt.xscale('log')  # Set x-axis to log scale
    plt.yscale('log')
    plt.xlabel('Pixel Intensity (Log Scale)')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    
    image_buf = plot_to_image(plt.gcf())
    writer.add_image(f'{tag}/Dynamic Range', image_buf, step, dataformats='HWC')
    
    # Save to histgoram_directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f'{tag}_dynamic_range_step_{step}.svg'), format='svg')
    
    plt.close()
    
    return combined_dr

def visualize_tensor_2mg_with_DR(writer, step, tag, save_dir, image_tensor_hdr, image_tensor_ldr1):
    # Convert tensors to numpy arrays
    image_np_hdr = image_tensor_hdr.squeeze(0).permute(1,2,0).cpu().numpy()
    image_np_ldr1 = image_tensor_ldr1.squeeze(0).permute(1,2,0).cpu().numpy()
    
    # Normalize images for display
    image_np_hdr_norm = (image_np_hdr - image_np_hdr.min()) / (image_np_hdr.max() - image_np_hdr.min())
    image_np_ldr_norm1 = (image_np_ldr1 - image_np_ldr1.min()) / (image_np_ldr1.max() - image_np_ldr1.min())

    # Calculate dynamic ranges in dB
    hdr_max = image_np_hdr.max()
    hdr_min = image_np_hdr[image_np_hdr > 0].min()
    ldr_max1 = image_np_ldr1.max()
    ldr_min1 = image_np_ldr1[image_np_ldr1 > 0].min()

    
    hdr_dr = 20 * np.log10(hdr_max / hdr_min)
    ldr_dr1 = 20 * np.log10(ldr_max1 / ldr_min1)

    
    # Map LDR bounds to the same range as HDR for consistent visual scaling
    hdr_range = hdr_max - hdr_min
    ldr1_mapped_min = (ldr_min1 - hdr_min) / hdr_range
    ldr1_mapped_max = (ldr_max1 - hdr_min) / hdr_range


    # Calculate combined coverage using the min of both LDR lower bounds and max of upper bounds
    combined_dr = 20 * np.log10(ldr_max1 / ldr_min1)

    # Print coverage information
    # print(f"LDR1 covers {ldr_dr1:.2f} dB of the dynamic range.")
    # print(f"LDR2 covers {ldr_dr2:.2f} dB of the dynamic range.")
    print(f"Combined LDR1 and LDR2 cover {combined_dr:.2f} dB of the dynamic range.")

    # Plotting the images and histograms
    plt.figure(figsize=(18, 6))
    
    # LDR1 Image
    plt.subplot(2, 2, 1)
    plt.title('Captured intensity')
    plt.imshow(image_np_ldr_norm1 ** (1 / 2.2))
    plt.text(10, 10, f'DR: {ldr_dr1:.2f} dB', color='white', backgroundcolor='black', fontsize=12, ha='left')
    plt.axis('off')

    # Dynamic Range Visualization with normalized histograms
    plt.subplot(2, 2, 2)
    plt.title('Dynamic Range Visualization')
    
    # HDR Histogram with log scale on the x-axis
    plt.hist(image_np_hdr.ravel(), bins=2048, color='orange', alpha=0.75, label='Scene radiance')
    
    # Draw vertical lines to indicate the bounds for each LDR image on a normalized scale
    plt.axvline(x=ldr1_mapped_min * hdr_max, color='green', linestyle='--', label='Cap Bounds')
    plt.axvline(x=ldr1_mapped_max * hdr_max, color='green', linestyle='--')


    # Normalize LDR images to the HDR range for consistent display in log scale
    ldr1_mapped = (image_np_ldr1 - hdr_min) / hdr_range * hdr_max

    
    plt.hist(ldr1_mapped.ravel(), bins=256, color='green', alpha=0.5, label='Captured intensity')

    
    plt.xscale('log')  # Set x-axis to log scale
    plt.yscale('log')
    plt.xlabel('Pixel Intensity (Log Scale)')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    
    image_buf = plot_to_image(plt.gcf())
    writer.add_image(f'{tag}/Dynamic Range', image_buf, step, dataformats='HWC')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, f'{tag}_dynamic_range_step_{step}.svg'), format='svg')
    
    plt.close()
    
    return combined_dr


def visualize_error_map(error_map, vmin=0, vmax=5, title='Error Map'):
    error_map_np = error_map[0].cpu().numpy().squeeze()
    
    # vmin, vmax 
    if vmax is None or vmax <= vmin:
        vmax = error_map_np.max()
        vmin = error_map_np.min()


    if vmax - vmin < 0.1:  
        vmin, vmax = 0, 1

    dpi = 200
    fig, ax = plt.subplots(figsize=(6, 3), dpi=dpi)
    cmap = plt.cm.plasma  
    im = ax.imshow(error_map_np, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    ax.set_title(title)


    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Error Value')

    plt.tight_layout(pad=0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)


    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close(fig)
    buf.seek(0)


    image_pil = Image.open(buf)
    image_array = np.array(image_pil)
    image_array = np.transpose(image_array, (2, 0, 1))

    return image_array


def visualize_error_map_svg(error_map, vmin=0, vmax=5, title='Error Map'):
    # Convert error_map to numpy array
    error_map_np = error_map[0].cpu().numpy().squeeze()
    
    # Automatically set vmin and vmax if needed
    if vmax is None or vmax <= vmin:
        vmax = error_map_np.max()
        vmin = error_map_np.min()

    # Enhance contrast for very small error ranges
    if vmax - vmin < 0.1:
        vmin, vmax = 0, 1

    # Create figure with specific colormap and colorbar
    fig, ax = plt.subplots(figsize=(6, 3), dpi=200)
    cmap = plt.cm.plasma  # Use plasma colormap for error map
    im = ax.imshow(error_map_np, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    ax.set_title(title)

    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Error Value')

    # Adjust layout to remove padding and whitespace
    plt.tight_layout(pad=0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    return fig


#############################################
# Visualize Histogram and output dB coverage
def visualize_1img_with_DR(image_tensor_hdr, image_tensor_ldr, target_dynamic_range=48, save_path='histogram_single_ldr.svg', y_max=None):
    # Convert tensors to numpy arrays
    image_np_hdr = image_tensor_hdr.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    image_np_ldr = image_tensor_ldr.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

    # HDR Dynamic Range
    hdr_max = image_np_hdr.max()
    hdr_min = image_np_hdr[image_np_hdr > 0].min()
    hdr_dr = 20 * np.log10(hdr_max / hdr_min)
    
    # LDR dynamic range and scaling for DR calculation
    ldr_max, ldr_min = image_np_ldr.max(), image_np_ldr[image_np_ldr > 0].min()
    ldr_dr = 20 * np.log10(ldr_max / ldr_min)
    scale_factor = target_dynamic_range / ldr_dr
    adjusted_ldr_dr = ldr_dr * scale_factor
    increase_percent_ldr = ((adjusted_ldr_dr - target_dynamic_range) / target_dynamic_range) * 100

    # Visualization for dynamic range coverage
    bins = np.logspace(np.log10(hdr_min), np.log10(hdr_max), 300)
    plt.figure(figsize=(10, 6))

    plt.hist(image_np_hdr.ravel(), bins=bins, color='orange', alpha=1.0, histtype='stepfilled')
    plt.hist(image_np_ldr.ravel(), bins=bins, color='red', alpha=0.6, histtype='stepfilled')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Pixel Intensity (Log Scale)', fontsize=18)
    plt.ylabel('Frequency (Log Scale)', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    # plt.legend(fontsize=12)

    if y_max is not None:
        plt.ylim(top=y_max)

    plt.savefig(save_path, format='svg')
    plt.close()

    coverage_info = {
        'LDR_coverage': adjusted_ldr_dr,
        'LDR_increase_percent': increase_percent_ldr
    }
    return coverage_info


def visualize_2img_with_DR(image_tensor_hdr, image_tensor_ldr1, image_tensor_ldr2, target_dynamic_range=48, save_path='histogram_dual_ldr.svg', y_max=None):
    # Convert tensors to numpy arrays
    image_np_hdr = image_tensor_hdr.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    image_np_ldr1 = image_tensor_ldr1.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    image_np_ldr2 = image_tensor_ldr2.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

    # HDR Dynamic Range
    hdr_max = image_np_hdr.max()
    hdr_min = image_np_hdr[image_np_hdr > 0].min()
    hdr_dr = 20 * np.log10(hdr_max / hdr_min)
    
    # # LDR dynamic ranges and scaling for DR calculation
    # ldr_max1, ldr_min1 = image_np_ldr1.max(), image_np_ldr1[image_np_ldr1 > 0].min()
    # ldr_max2, ldr_min2 = image_np_ldr2.max(), image_np_ldr2[image_np_ldr2 > 0].min()
    
    image_np_ldr1 = np.clip(image_np_ldr1, 1e-5, None)
    image_np_ldr2 = np.clip(image_np_ldr2, 1e-5, None)

    # Recalculate dynamic range after clipping
    ldr_min1 = image_np_ldr1[image_np_ldr1 > 0].min()
    ldr_min2 = image_np_ldr2[image_np_ldr2 > 0].min()
    ldr_max1, ldr_max2 = image_np_ldr1.max(), image_np_ldr2.max()
    
    # Determine which LDR image has a lower min value
    # if ldr_min1 < ldr_min2:
    #     lower_bound_img, higher_bound_img = image_np_ldr1, image_np_ldr2
    #     lower_bound_color, higher_bound_color = 'red', 'blue'
    # else:
    #     lower_bound_img, higher_bound_img = image_np_ldr2, image_np_ldr1
    #     lower_bound_color, higher_bound_color = 'red', 'blue'
    
    # LDR min 값에 따라 자동 정렬 (밝은 이미지가 red, 어두운 이미지가 blue)
    if ldr_min1 > ldr_min2:
        lower_bound_img, higher_bound_img = image_np_ldr2, image_np_ldr1
        lower_bound_color, higher_bound_color = 'red', 'blue'  # 색상 교환
        ldr_min1, ldr_min2 = ldr_min2, ldr_min1
        ldr_max1, ldr_max2 = ldr_max2, ldr_max1
    else:
        lower_bound_img, higher_bound_img = image_np_ldr1, image_np_ldr2
        lower_bound_color, higher_bound_color = 'red', 'blue'  # 색상 교환


    # # Calculate dynamic ranges
    # lower_dr = 20 * np.log10(lower_bound_img.max() / lower_bound_img[lower_bound_img > 0].min())
    # scale_factor_lower = target_dynamic_range / lower_dr
    # adjusted_lower_dr = lower_dr * scale_factor_lower
    
    # higher_dr = 20 * np.log10(higher_bound_img.max() / higher_bound_img[higher_bound_img > 0].min())
    # scale_factor_higher = target_dynamic_range / higher_dr
    # adjusted_higher_dr = higher_dr * scale_factor_higher

    # # Combined DR
    # combined_min = min(ldr_min1, ldr_min2)
    # combined_max = max(ldr_max1, ldr_max2)
    # combined_dr = 20 * np.log10(combined_max / combined_min)
    # adjusted_combined_dr = combined_dr * min(scale_factor_lower, scale_factor_higher)
    
    # Adjusted Dynamic Range Scaling
    lower_dr = 20 * np.log10(ldr_max1 / max(ldr_min1, 1e-5))  # Prevent division by zero
    higher_dr = 20 * np.log10(ldr_max2 / max(ldr_min2, 1e-5))

    # Use target_dynamic_range to calculate scaling
    scale_factor_lower = target_dynamic_range / lower_dr
    scale_factor_higher = target_dynamic_range / higher_dr

    # Apply scaling symmetrically to avoid uneven gaps
    adjusted_lower_dr = lower_dr * min(scale_factor_lower, scale_factor_higher)
    adjusted_higher_dr = higher_dr * min(scale_factor_lower, scale_factor_higher)

    combined_min = min(ldr_min1, ldr_min2)
    combined_max = max(ldr_max1, ldr_max2)
    combined_dr = 20 * np.log10(combined_max / combined_min)
    
    print(f"Scale Factor Lower: {scale_factor_lower}, Scale Factor Higher: {scale_factor_higher}")
    print(f"Adjusted Lower DR: {adjusted_lower_dr}")
    print(f"Adjusted Higher DR: {adjusted_higher_dr}")
    print(f"LDR1 Min: {ldr_min1}, LDR1 Max: {ldr_max1}")
    print(f"LDR2 Min: {ldr_min2}, LDR2 Max: {ldr_max2}")

    # Visualization for dynamic range coverage
    # bins = np.logspace(np.log10(hdr_min), np.log10(hdr_max), 300)
    bins = np.logspace(np.log10(max(hdr_min, 1e-5)), np.log10(hdr_max), 300)


    plt.figure(figsize=(10, 6))

    plt.hist(image_np_hdr.ravel(), bins=bins, color='orange', alpha=1.0, histtype='stepfilled', label='HDR')
    plt.hist(lower_bound_img.ravel(), bins=bins, color=lower_bound_color, alpha=0.6, histtype='stepfilled', label='Lower Bound LDR')
    plt.hist(higher_bound_img.ravel(), bins=bins, color=higher_bound_color, alpha=0.6, histtype='stepfilled', label='Higher Bound LDR')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Pixel Intensity (Log Scale)', fontsize=18)
    plt.ylabel('Frequency (Log Scale)', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    # plt.legend(fontsize=12)

    if y_max is not None:
        plt.ylim(top=y_max)

    plt.savefig(save_path, format='svg')
    plt.close()

    # Return combined coverage info
    coverage_info = {
        # 'Combined_coverage': adjusted_combined_dr,
        'Lower_bound_coverage': adjusted_lower_dr,
        'Higher_bound_coverage': adjusted_higher_dr
    }
    return coverage_info
