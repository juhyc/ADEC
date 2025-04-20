import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

from core.utils.read_utils import read_gen
from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor
from skimage.transform import resize

import os
import cv2
import logging
import numpy as np
from glob import glob
import random
import re

###############################################
# * Dataset class for synthetic CARLA dataset considering sequence
###############################################

def sort_key_func(file):
    numbers = re.findall(r'\d+', os.path.basename(file))
    return int(numbers[0]) if numbers else 0

def get_blur_parameters(epoch, total_epochs, initial_mean=5, final_mean=25, initial_std=2, final_std=6):
    
    progress = epoch / total_epochs  
    
    # Increase blur intensity linearly
    mean_degree = initial_mean + (final_mean - initial_mean) * progress
    std_degree = initial_std + (final_std - initial_std) * progress
    
    return mean_degree, std_degree

def apply_motion_blur(image, degree, angle):
    
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    kernel = np.diag(np.ones(degree))
    kernel = cv2.warpAffine(kernel, M, (degree, degree))
    kernel = kernel / degree
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

def apply_random_motion_blur(image, blur_prob=0.5, mean_degree=25, std_degree=6):
    
    if random.random() < blur_prob:
        degree = int(np.clip(np.random.normal(mean_degree, std_degree), 5, 50))
        angle = random.randint(0, 360)
        return apply_motion_blur(image, degree, angle)
    else:
        return image

class StereoDataset(data.Dataset):
    def __init__(self, reader=None, apply_blur=True, valid_mode=False, valid_case=0, mean_degree=5, std_degree=2):
        if reader is None:
            self.disparity_reader = read_gen
        else:
            self.disparity_reader = reader
            
        self.is_test = False
        self.apply_blur = apply_blur
        self.valid_mode = valid_mode
        self.valid_case = valid_case
        self.mean_degree = mean_degree
        self.std_degree = std_degree
        
        self.flow_list = []
        self.disparity_list = []
        self.image_list = []
        self.extra_info = []
    
    def update_blur_parameters(self, mean_degree, std_degree):
        
        self.mean_degree = mean_degree
        self.std_degree = std_degree

    
    def apply_random_blur_to_images(self, images, is_validation=False, case=None):
        """
        Apply blur to image list
        - Training : Choose one case randomly among 4 cases
        - Validation : Apply blur with input case
        """
        if is_validation and case is None:
            raise ValueError("Set validation case!")

        if not is_validation:
            case = random.choice([0, 1, 2, 3]) 

        blurred_images = []
        for idx, img in enumerate(images):
            img_np = img.permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)

            if case == 1 and idx < 2:  # first stereo pair blur
                img_blurred = apply_random_motion_blur(img_np, blur_prob=1.0, mean_degree=self.mean_degree, std_degree=self.std_degree)
            elif case == 2 and idx >= 2:  # second stereo pair만 blur
                img_blurred = apply_random_motion_blur(img_np, blur_prob=1.0, mean_degree=self.mean_degree, std_degree=self.std_degree)
            elif case == 3 and idx == random.randint(0, 3):  # One image is blurry among 4 images
                img_blurred = apply_random_motion_blur(img_np, blur_prob=1.0, mean_degree=self.mean_degree, std_degree=self.std_degree)
            else: # All image are clean
                img_blurred = img_np

            img_blurred = torch.from_numpy(img_blurred).permute(2, 0, 1).float() / 255.0
            blurred_images.append(img_blurred)

        return blurred_images

        
    def __getitem__(self, index):
        if self.is_test:
            img1 = read_gen(self.image_list)
            img2 = read_gen(self.image_list)
            img1 = np.array(img1)[...,:3]
            img2 = np.array(img2)[...,:3]
            img1 = torch.from_numpy(img1).permute(2,0,1).float()
            img2 = torch.from_numpy(img2).permute(2,0,1).float()
            
            return img1, img2, self.extra_info[index]
        
        index = index * 2 % len(self.image_list)
        disp_index = index // 2
        disp = self.disparity_reader(self.disparity_list[disp_index])
        
        if isinstance(disp, tuple):
            disp, valid = disp
        else:
            valid = disp < 512
            
        img1_left, hdr_img1 = read_gen(self.image_list[index][0])
        img1_right, _ = read_gen(self.image_list[index][1])
        img2_left, hdr_img2 = read_gen(self.image_list[index + 1][0])
        img2_right, _ = read_gen(self.image_list[index + 1][1])
        # img2_left = read_gen(self.image_list[(index + 1) % len(self.image_list)][0])
        # img2_right = read_gen(self.image_list[(index + 1) % len(self.image_list)][1])
        
        img1_left = np.array(img1_left)
        img1_right = np.array(img1_right)
        img2_left = np.array(img2_left)
        img2_right = np.array(img2_right)
        
        hdr_img1 = np.array(hdr_img1)
        hdr_img2 = np.array(hdr_img2)
        
        disp = np.array(disp).astype(np.float32)
        
        ##### Resizing for gpu memory############
        # height, width = img1_left.shape[1], img1_left.shape[0]
        # img1_left = cv2.resize(img1_left, (height // 2, width // 2))
        # img1_right = cv2.resize(img1_right, (height // 2, width // 2))
        # img2_left = cv2.resize(img2_left, (height // 2, width // 2))
        # img2_right = cv2.resize(img2_right, (height // 2, width // 2))
        # disp = cv2.resize(disp, (height // 2, width // 2))
        # disp = disp / 2
        # #########################################
        
        flow = np.stack([-disp, np.zeros_like(disp)], axis=-1)
        
        img1_left = torch.from_numpy(img1_left).permute(2, 0, 1).float()
        img1_right = torch.from_numpy(img1_right).permute(2, 0, 1).float()
        img2_left = torch.from_numpy(img2_left).permute(2, 0, 1).float()
        img2_right = torch.from_numpy(img2_right).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        
        # Check blur
        if self.apply_blur:
            if not self.valid_mode: # Training set -> apply blur randomly
                img1_left, img1_right, img2_left, img2_right = self.apply_random_blur_to_images(
                    [img1_left, img1_right, img2_left, img2_right]
                )
            else: # Validation set -> apply blur by case
                img1_left, img1_right, img2_left, img2_right = self.apply_random_blur_to_images(
                    [img1_left, img1_right, img2_left, img2_right], is_validation=True, case=self.valid_case
                )
            
        
        hdr_img1 = torch.from_numpy(hdr_img1).permute(2,0,1).float()
        hdr_img2 = torch.from_numpy(hdr_img2).permute(2,0,1).float()
        
        valid = (flow[0].abs() < 512) & (flow[1].abs() < 512)
        
        flow = flow[:1]
        
        return self.image_list[index] + self.image_list[index + 1] + [self.disparity_list[disp_index]], img1_left, img1_right, img2_left, img2_right, flow, valid.float()
    
    def __len__(self):
        return len(self.image_list) // 2


class CARLASequenceDataset(StereoDataset):
    def __init__(self, root='datasets/CARLA', image_set='training', apply_blur=True, valid_mode=False, valid_case=0):
        super(CARLASequenceDataset, self).__init__(reader=read_gen, apply_blur=apply_blur, valid_mode=valid_mode, valid_case=valid_case)
        assert os.path.exists(root), "Dataset root path does not exist."
        
        
        self.image_list = []
        self.disparity_list = []
        self.experiment_names = []
        
        if image_set == 'training':
            experiment_folders = sorted(glob(os.path.join(root, image_set, 'Experiment[1-9]*')))
        else:
            # Validation folder name
            experiment_folders = sorted(glob(os.path.join(root, image_set, 'Experiment483')))
    
        for experiment_folder in experiment_folders:
            
            experiment_name = os.path.basename(experiment_folder)
            self.experiment_names.append(experiment_name)
            
            # image load from each folders
            image1_list = sorted(glob(os.path.join(experiment_folder, 'hdr_left/*.npy')), key=sort_key_func)
            image2_list = sorted(glob(os.path.join(experiment_folder, 'hdr_right/*.npy')), key=sort_key_func)
            disp_list = sorted(glob(os.path.join(experiment_folder, 'ground_truth_disparity_left/*.npy')), key=sort_key_func)

            # Check the number of files
            if not (len(image1_list) == len(image2_list) == len(disp_list)):
                logging.warning(f"File count mismatch in {experiment_folder}: "
                                f"image1_list={len(image1_list)}, image2_list={len(image2_list)}, disp_list={len(disp_list)}")
            
            # Add to image_list and disparity_list in even index pairs
            for idx in range(0, len(image1_list) - 1, 2):
                if idx + 1 < len(image1_list):
                    self.image_list.append([image1_list[idx], image2_list[idx]])
                    self.image_list.append([image1_list[idx + 1], image2_list[idx + 1]])
                    self.disparity_list.append(disp_list[idx])
        
        if len(self.image_list) // 2 != len(self.disparity_list):
            logging.warning(f"Data count mismatch: image_list size={len(self.image_list)}, disparity_list size={len(self.disparity_list)}")

        print(f"image_list size: {len(self.image_list)}, disp_list size: {len(self.disparity_list)}")
        
    def get_experiment_name(self, index):
        """Return the experiment name based on the index."""
        images_per_experiment = len(self.image_list) // len(self.experiment_names)
        experiment_index = index // images_per_experiment
        return self.experiment_names[experiment_index]


def fetch_dataloader(args):
    train_dataset = None
    
    for dataset_name in args.train_datasets:
        if dataset_name.startswith('carla'):
            new_dataset = CARLASequenceDataset(apply_blur=True)
            print(f"Samples : {len(new_dataset)}")
            logging.info(f"Adding {len(new_dataset)} training samples from CARLA")
            
            train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset  
    
            train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                   pin_memory=True, shuffle=True, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6)) - 2, drop_last=True)
            
        if dataset_name.startswith('test_carla'):
            new_dataset = CARLASequenceDataset(image_set='test', apply_blur=False)
            print(f"Samples : {len(new_dataset)}")
            logging.info(f"Adding {len(new_dataset)} test samples from CARLA")
            
            train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset  
    
            train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                   pin_memory=True, shuffle=False, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6)) - 2, drop_last=True)
    
    logging.info('Training with %d image pairs' % len(train_dataset))
    return train_loader