import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import logging
import os
import glob 

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import re

def extract_number(filename):
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    return -1

# ^ Dataset for integrated pipeline (moudle 1,2,3)
class DepthDataset(Dataset):
    def __init__(self, left_mono_dir, right_mono_dir, left_stereo_dir, right_stereo_dir, left_mask_dir, right_mask_dir, disparity_dir, image_set='training', transform=None):
        self.left_mono_dir = left_mono_dir
        self.left_stereo_dir = left_stereo_dir
        self.left_mask_dir = left_mask_dir
        self.right_mono_dir = right_mono_dir
        self.right_stereo_dir = right_stereo_dir
        self.right_mask_dir = right_mask_dir
        self.disparity_dir = disparity_dir
        self.transform = transform
        self.image_set = image_set

        if image_set == 'training':
            self.images = sorted(os.listdir(left_mono_dir), key=extract_number)
        else:
            self.images = ['73.npy']        

    def __len__(self):
        return len(self.images)
    
    def scale2stereo(self, mono, stereo):
    
        mono_min, mono_max = torch.min(mono), torch.max(mono)
        stereo_min, stereo_max = torch.min(stereo), torch.max(stereo)
        
        scale_factor = (stereo_max - stereo_min) / (mono_max - mono_min)
        
        scaled_depth = (mono - mono_min) * scale_factor + stereo_min
        
        scaled_depth = torch.clamp(scaled_depth, stereo_min, stereo_max)
        
        return scaled_depth
        
    def __getitem__(self, idx):
        
        if self.image_set =='training':
            image_name = self.images[idx]
        else:
            image_name = self.images[0]
        
        # file path
        left_mono_path = os.path.join(self.left_mono_dir, image_name)
        right_mono_path = os.path.join(self.right_mono_dir, image_name)
        left_stereo_path = os.path.join(self.left_stereo_dir, image_name)
        right_stereo_path = os.path.join(self.right_stereo_dir, image_name)
        left_mask_path = os.path.join(self.left_mask_dir, image_name)
        right_mask_path = os.path.join(self.right_mask_dir, image_name)
        disparity_path = os.path.join(self.disparity_dir, image_name.split('.')[0]+'.png')
        
        # load file
        # ^ 수정 부분 cv2.imread() -> np.load
        left_mono_image =  np.load(left_mono_path).astype(np.float32)
        left_stereo_image = np.load(left_stereo_path).astype(np.float32)
        left_mask_image = np.load(left_mask_path).astype(np.float32)
        right_mono_image =  np.load(right_mono_path).astype(np.float32)
        right_stereo_image = np.load(right_stereo_path).astype(np.float32)
        right_mask_image = np.load(right_mask_path).astype(np.float32)
        disparity_image = cv2.imread(disparity_path, cv2.IMREAD_ANYDEPTH) / 256.0
        # ^
        
        # preprocessing
        # 1. 이미지를 disparity 사이즈로 resize
        # Todo 이미지 resize 크기 조절... Crop or Padding
        
        target_size = (1200, 370)
        left_mono_image = cv2.resize(left_mono_image, target_size)
        left_stereo_image = cv2.resize(left_stereo_image, target_size)
        left_mask_image = cv2.resize(left_mask_image, target_size)
        right_mono_image = cv2.resize(right_mono_image, target_size)
        right_stereo_image = cv2.resize(right_stereo_image, target_size)
        right_mask_image = cv2.resize(right_mask_image, target_size)
        disparity_image = cv2.resize(disparity_image, target_size)
        
        left_mono_image = torch.from_numpy(left_mono_image)
        left_stereo_image = torch.from_numpy(left_stereo_image)
        left_mask_image = torch.from_numpy(left_mask_image)
        right_mono_image = torch.from_numpy(right_mono_image)
        right_stereo_image = torch.from_numpy(right_stereo_image)
        right_mask_image = torch.from_numpy(right_mask_image)


        valid = disparity_image > 0.0
        disparity_image = torch.from_numpy(disparity_image.astype(np.float32)).unsqueeze(0)
        valid = torch.from_numpy(valid.astype(np.float32)).unsqueeze(0)
        
        # 2. 추가 전처리 scaling to disparity
        # * 수정 stereo의 depth의 scale이 적절하다는 기준하에 mono의 scale을 stereo scale에 맞춤.
        scaled_left_mono = self.scale2stereo(left_mono_image, left_stereo_image).unsqueeze(0)
        scaled_left_stereo = left_stereo_image.unsqueeze(0)
        left_mask = left_mask_image.unsqueeze(0)
        
        scaled_right_mono = self.scale2stereo(right_mono_image, right_stereo_image).unsqueeze(0)
        scaled_right_stereo = right_stereo_image.unsqueeze(0)
        right_mask = right_mask_image.unsqueeze(0)

        if self.transform:
            scaled_left_mono = self.transform(scaled_left_mono)
            scaled_right_mono = self.transform(scaled_right_mono)
            scaled_left_stereo = self.transform(scaled_left_stereo)
            scaled_right_stereo = self.transform(scaled_right_stereo)
            disparity_image = self.transform(disparity_image)
            valid = self.transform(valid)

        return scaled_left_mono, scaled_right_mono, scaled_left_stereo, scaled_right_stereo, left_mask, right_mask,  disparity_image, valid

# dataset for module 1
class DepthDataset_1(Dataset):
    def __init__(self, left_mono_dir, left_stereo_dir, left_mask_dir ,disparity_dir, transform=None, image_set='training'):
        self.left_mono_dir = left_mono_dir
        self.left_stereo_dir = left_stereo_dir
        self.left_mask_dir = left_mask_dir
        self.disparity_dir = disparity_dir
        self.transform = transform
        self.image_set = image_set
        
        if image_set == 'training':
            self.images = sorted(os.listdir(left_mono_dir), key=extract_number)
        else:
            self.images = ['73.npy']
        

    def __len__(self):
        return len(self.images)
    
    def scale2stereo(self, mono, stereo):
        
        mono_min, mono_max = torch.min(mono), torch.max(mono)
        stereo_min, stereo_max = torch.min(stereo), torch.max(stereo)
        
        scale_factor = (stereo_max - stereo_min) / (mono_max - mono_min)
        
        scaled_depth = (mono - mono_min) * scale_factor + stereo_min
        
        scaled_depth = torch.clamp(scaled_depth, stereo_min, stereo_max)
        
        return scaled_depth
        

    def __getitem__(self, idx):
        
        if self.image_set == 'training':
            image_name = self.images[idx]
        else:
            image_name = self.images[0]
            
        left_mono_path = os.path.join(self.left_mono_dir, image_name)
        left_stereo_path = os.path.join(self.left_stereo_dir, image_name)
        left_mask_path = os.path.join(self.left_mask_dir, image_name)
        disparity_path = os.path.join(self.disparity_dir, image_name.split('.')[0]+'.png')

        # ^ 수정 부분 cv2.imread() -> np.load
        left_mono_image =  np.load(left_mono_path).astype(np.float32)
        left_stereo_image = np.load(left_stereo_path).astype(np.float32)
        left_mask_image = np.load(left_mask_path).astype(np.float32)
        disparity_image = cv2.imread(disparity_path, cv2.IMREAD_ANYDEPTH) / 256.0
        # ^
        
        # ^ 이미지를 disparity 사이즈로 resize
        # Todo 이미지 resize 크기 조절. Crop or Padding
        target_size = (1200, 370)
        left_mono_image = cv2.resize(left_mono_image, target_size)
        left_stereo_image = cv2.resize(left_stereo_image, target_size)
        left_mask_image = cv2.resize(left_mask_image, target_size)
        disparity_image = cv2.resize(disparity_image, target_size)
        
        left_mono_image = torch.from_numpy(left_mono_image)
        left_stereo_image = torch.from_numpy(left_stereo_image)
        left_mask_image = torch.from_numpy(left_mask_image)

        valid = disparity_image > 0.0
        disparity_image = torch.from_numpy(disparity_image.astype(np.float32)).unsqueeze(0)
        valid = torch.from_numpy(valid.astype(np.float32)).unsqueeze(0)
        
        
        # * 추가 전처리 scaling to disparity
        # * 수정 stereo의 depth의 scale이 적절하다는 기준하에 mono의 scale을 stereo scale에 맞춤.
        scaled_left_mono = self.scale2stereo(left_mono_image, left_stereo_image).unsqueeze(0)
        scaled_left_stereo = left_stereo_image.unsqueeze(0)
        left_mask = left_mask_image.unsqueeze(0)
        

        if self.transform:
            scaled_left_mono = self.transform(scaled_left_mono)
            scaled_left_stereo = self.transform(scaled_left_stereo)
            left_mask = self.transform(left_mask)
            disparity_image = self.transform(disparity_image)
            valid = self.transform(valid)

        return scaled_left_mono, scaled_left_stereo, left_mask,  disparity_image, valid

#^ dataset for fusion module (stereo)
class DepthDataset_stereo(Dataset):
    def __init__(self, stereo_dir_1, stereo_dir_2, mask_dir_1, mask_dir_2, disparity_dir, transform=None, image_set='training'):
        self.stereo_dir_1 = stereo_dir_1
        self.stereo_dir_2 = stereo_dir_2
        self.mask_dir_1 = mask_dir_1
        self.mask_dir_2 = mask_dir_2
        self.disparity_dir = disparity_dir
        self.transform = transform
        self.image_set = image_set
        
        if image_set == 'training':
            self.images = sorted(os.listdir(stereo_dir_1), key=extract_number)
        else:
            self.images = ['000003_10.npy']
        
    def __len__(self):
        return len(self.images)
            
    def __getitem__(self, idx):
        
        if self.image_set == 'training':
            image_name = self.images[idx]
        else:
            image_name = self.images[0]

        # Set file path
        stereo_1_path = os.path.join(self.stereo_dir_1, image_name)
        stereo_2_path = os.path.join(self.stereo_dir_2, image_name)
        mask_1_path = os.path.join(self.mask_dir_1, image_name)
        mask_2_path = os.path.join(self.mask_dir_2, image_name)
        disparity_path = os.path.join(self.disparity_dir, image_name.split('.')[0]+'.png')

        # Load data
        stereo_image_1 =  np.load(stereo_1_path).astype(np.float32)
        stereo_image_2 = np.load(stereo_2_path).astype(np.float32)
        mask_image_1 = np.load(mask_1_path).astype(np.float32)
        mask_image_2 = np.load(mask_2_path).astype(np.float32)
        disparity_image = cv2.imread(disparity_path, cv2.IMREAD_ANYDEPTH) / 256.0
        
        # Resize image size to disparity size
        # Todo 이미지 resize 크기 조절. Crop or Padding
        target_size = (1200, 370)
        stereo_image_1 = cv2.resize(stereo_image_1, target_size)
        stereo_image_2 = cv2.resize(stereo_image_2, target_size)
        mask_image_1 = cv2.resize(mask_image_1, target_size)
        mask_image_2 = cv2.resize(mask_image_2, target_size)
        disparity_image = cv2.resize(disparity_image, target_size)
        
        stereo_image_1 = torch.from_numpy(stereo_image_1).unsqueeze(0)
        stereo_image_2 = torch.from_numpy(stereo_image_2).unsqueeze(0)
        mask_image_1 = torch.from_numpy(mask_image_1).unsqueeze(0)
        mask_image_2 = torch.from_numpy(mask_image_2).unsqueeze(0)

        valid = disparity_image > 0.0
        disparity_image = torch.from_numpy(disparity_image.astype(np.float32)).unsqueeze(0)
        valid = torch.from_numpy(valid.astype(np.float32)).unsqueeze(0) 

        if self.transform:
            stereo_image_1 = self.transform(stereo_image_1)
            stereo_image_2 = self.transform(stereo_image_2)
            mask_image_1 = self.transform(mask_image_1)
            mask_image_2 = self.transform(mask_image_2)
            disparity_image = self.transform(disparity_image)
            valid = self.transform(valid)

        return stereo_image_1, stereo_image_2, mask_image_1, mask_image_2, disparity_image, valid

    


