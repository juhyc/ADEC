from __future__ import print_function, division

import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
import glob

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from core.raft_stereo import RAFTStereo
from core.combine_model import CombineModel
from core.utils.simulate import *

from evaluate_stereo import *
import core.stereo_datasets as datasets

from core.saec import *
from PIL import Image
from torchvision.transforms import ToTensor, transforms

import matplotlib.pyplot as plt


# image = Image.open('/home/juhyung/SAEC/datasets/left.png')

# if isinstance(image, Image.Image):
#     to_tensor = ToTensor()
#     image = to_tensor(image)
    
# _, height, width = image.shape

# print(image.shape)
# print(height)

DEVICE = 'cuda'

def load_image(imfile):
    # img = np.array(Image.open(imfile)).astype(np.uint8)
    # img = img / 255.0
    # img = torch.from_numpy(img).permute(2, 0, 1).float()
    to_tensor = ToTensor()
    img = to_tensor(Image.open(imfile))

    return img[None].to(DEVICE)

def calculate_histograms(left_ldr_image, right_ldr_image):
    
    #^ Calculate histogram with multi scale
    histogram_coarest_l = histogram_subimage(left_ldr_image, 1)
    histogram_intermediate_l = histogram_subimage(left_ldr_image, 3)
    histogram_finest_l = histogram_subimage(left_ldr_image,7)
    
    histogram_coarest_r = histogram_subimage(right_ldr_image, 1)
    histogram_intermediate_r = histogram_subimage(right_ldr_image, 3)
    histogram_finest_r = histogram_subimage(right_ldr_image, 7)
    
    #^ Stack histogram [256,59]
    list_of_histograms_l = [histogram_coarest_l, histogram_intermediate_l, histogram_finest_l]
    stacked_histo_tensor_l = stack_histogram(list_of_histograms_l)

    list_of_histograms_r = [histogram_coarest_r, histogram_intermediate_r, histogram_finest_r]
    stacked_histo_tensor_r = stack_histogram(list_of_histograms_r)
    
    return stacked_histo_tensor_l, stacked_histo_tensor_r

def batch_to_image(image):
    image = image/255.0
    image = image.cpu()
    image = np.asarray(image.permute(0,2,3,1).squeeze(0))
    
    return image

def histogram_subimage2(image, grid_size):
    _, height, width = image.shape
    
    grid_height, grid_width = height // grid_size, width // grid_size
    
    for i in range(grid_size):
        for j in range(grid_size):
            pass
            

#^ Split Checkpoint

checkpoint = torch.load('/home/juhyung/SAEC/checkpoints/raft-stereo.pth')

combined_model_state_dict = checkpoint

# model의 가중치 추출
global_feature_net_state_dict = {k.replace('GlobalFeatureNet.', ''): v for k, v in combined_model_state_dict.items() if 'GlobalFeatureNet' in k}
raft_state_dict = {k.replace('RAFTStereo.',''): v for k, v in combined_model_state_dict.items() if 'RAFTStereo' in k}

# print(len(global_feature_net_state_dict.keys()))
# print(len(raft_state_dict.keys()))
# checkpoint2 = torch.load('/home/juhyung/SAEC/models/raftstereo-eth3d.pth')
# print(len(checkpoint.keys()))
#^

# print(type(global_feature_net_state_dict))
# print(type(checkpoint2))


model = torch.nn.DataParallel(GlobalFeatureNet())

#^ load state_dict check
# before_state_dict = {name: param.clone() for name, param in model.named_parameters()}
# model.load_state_dict(global_feature_net_state_dict)
# after_state_dict = {name: param.clone() for name, param in model.named_parameters()}


# for (name_before, param_before), (name_after, param_after) in zip(before_state_dict.items(), after_state_dict.items()):
#     assert name_before == name_after  # 이름이 동일한지 확인
#     if not torch.equal(param_before, param_after):
#         print(f"Parameter {name_before} has changed.")
#     else:
#         print(f"Parameter {name_before} remains the same.")

#^

model = model.module
model.to(DEVICE)
model.eval()

left_imgs = '/home/juhyung/SAEC/datasets/left.png'
right_imgs = '/home/juhyung/SAEC/datasets/right.png'

left_images = sorted(glob.glob(left_imgs, recursive=True))
right_images = sorted(glob.glob(right_imgs, recursive=True))

print(f"Found {len(left_images)} images")

for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
    image1 = load_image(imfile1)
    image2 = load_image(imfile2)
    
    display_img_list = []
    display_img_list.append(batch_to_image(image1))
    display_img_list.append(batch_to_image(image2))
    
    
    # * random expousre
    exp_rand_l, exp_rand_r = generate_random_exposure()
    
    # print(f"loaded image shape : {image1.shape}")
    
    
    
    image1_b = image1 * exp_rand_l
    image2_b = image2 * exp_rand_r
    
    image1_b = torch.clamp(image1_b, 0, 1)
    
    
    display_img_list.append(batch_to_image(image1_b))
    display_img_list.append(batch_to_image(image2_b))
    
    print("================================================================")
    print(f"Before model, image exposure. L : {exp_rand_l}, R : {exp_rand_r}")
    print("================================================================")
    
    #* Check histogram between two different exposure image
    # image1_b 자체는 문제 없음 배치도
    # 지금 image1_b 값이 0~1사이의 값이 아님, histogram_subimage를 계산할때 입력값은 0~1사이의 tensor값이 들어가야됌
    
    
    print(image1_b.shape)
    print(image1_b[0][0].min(), image1_b[0][0].max())
    
    
    test_hist = [torch.rand(256) for _ in range(3**2)]
    show_histogram(test_hist, 3)

    
    # histogram_intermediate_l = histogram_subimage(image1, 7)
    # # show_histogram(histogram_intermediate_l[0][0].cpu(),1)
    
    # print(histogram_intermediate_l[0][0][10].cpu().numpy())
    # show_histogram(histogram_intermediate_l[0][0].cpu(),7)
 

    
    # print(histogram_intermediate_l[0][0].min())
    # print(histogram_intermediate_l[0][0].max())
    # histogram_intermediate_r = histogram_subimage(image2_b, 3)
    
    # h_i_l = histogram_intermediate_l[0][0].cpu()
    # h_i_r = histogram_intermediate_r[0][0].cpu()
    # show_histogram(h_i_l, 3)
    # show_histogram(h_i_r, 3)
    
    # show_histogram(histogram_intermediate_l, 3)
    # show_histogram(histogram_intermediate_r, 3)
     
#     stacked_histo_tensor_l, stacked_histo_tensor_r = calculate_histograms(image1_b, image2_b)
    
#     output_l = model(stacked_histo_tensor_l.T)
#     output_r = model(stacked_histo_tensor_r.T)
    
#     print("================================================================")
#     print(f"Model output L : {output_l}, R : {output_r}")
#     print("================================================================")
    
#     shifted_exp_l = exposure_shift(exp_rand_l, output_l.mean().item())
#     shifted_exp_r = exposure_shift(exp_rand_r, output_r.mean().item())
#     print("================================================================")
#     print(f"Shifted exposure L : {shifted_exp_l}, R : {shifted_exp_r}")
#     print("================================================================")
    
#     image1_a = image1_b * shifted_exp_l
#     image2_a = image2_b * shifted_exp_r
    
#     display_img_list.append(batch_to_image(image1_a))
#     display_img_list.append(batch_to_image(image2_a))
    

# # Display plot
# fig, axes = plt.subplots(2,3,figsize = (20,5))
# plot_order = [0,2,4,1,3,5]

# for ax, idx in zip(axes.ravel(), plot_order):
#     if idx==0:
#         ax.set_title("HDR Original")
#     elif idx == 2:
#         ax.set_title("HDR with random exposure")
#     elif idx == 4:
#         ax.set_title("HDR with adjusted exposure")
#     ax.imshow(display_img_list[idx])
#     ax.axis('off')

    
# plt.tight_layout()
# plt.show()
    
    
    