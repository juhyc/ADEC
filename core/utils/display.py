import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import os
import glob
import re
import io

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