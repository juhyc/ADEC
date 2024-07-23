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
import re

###############################################
# * Dataset class for synthetic CARLA dataset considering sequence
###############################################

def sort_key_func(file):
    numbers = re.findall(r'\d+', os.path.basename(file))
    return int(numbers[0]) if numbers else 0

class StereoDataset(data.Dataset):
    def __init__(self, reader=None):
        if reader is None:
            self.disparity_reader = read_gen
        else:
            self.disparity_reader = reader
            
        self.is_test = False
        self.flow_list = []
        self.disparity_list = []
        self.image_list = []
        self.extra_info = []
        
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
            
        img1_left = read_gen(self.image_list[index][0])
        img1_right = read_gen(self.image_list[index][1])
        img2_left = read_gen(self.image_list[index + 1][0])
        img2_right = read_gen(self.image_list[index + 1][1])
        # img2_left = read_gen(self.image_list[(index + 1) % len(self.image_list)][0])
        # img2_right = read_gen(self.image_list[(index + 1) % len(self.image_list)][1])
        
        img1_left = np.array(img1_left)
        img1_right = np.array(img1_right)
        img2_left = np.array(img2_left)
        img2_right = np.array(img2_right)
        
        disp = np.array(disp).astype(np.float32)
        
        height, width = img1_left.shape[1], img1_left.shape[0]
        img1_left = cv2.resize(img1_left, (height // 2, width // 2))
        img1_right = cv2.resize(img1_right, (height // 2, width // 2))
        img2_left = cv2.resize(img2_left, (height // 2, width // 2))
        img2_right = cv2.resize(img2_right, (height // 2, width // 2))
        disp = cv2.resize(disp, (height // 2, width // 2))
        disp = disp / 2
        
        flow = np.stack([-disp, np.zeros_like(disp)], axis=-1)
        
        img1_left = torch.from_numpy(img1_left).permute(2, 0, 1).float()
        img1_right = torch.from_numpy(img1_right).permute(2, 0, 1).float()
        img2_left = torch.from_numpy(img2_left).permute(2, 0, 1).float()
        img2_right = torch.from_numpy(img2_right).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        
        valid = (flow[0].abs() < 512) & (flow[1].abs() < 512)
        
        flow = flow[:1]
        
        return self.image_list[index] + [self.disparity_list[disp_index]], img1_left, img1_right, img2_left, img2_right, flow, valid.float()
    
    def __len__(self):
        return len(self.image_list) // 2

class CARLASequenceDataset(StereoDataset):
    def __init__(self, root='datasets/CARLA', image_set='training'):
        super(CARLASequenceDataset, self).__init__(reader=read_gen)
        assert os.path.exists(root), "Dataset root path does not exist."
        
        image1_list = []
        image2_list = []
        disp_list = []
        
        if image_set == 'training':
            # experiment_folders = sorted(glob(os.path.join(root, image_set, 'Experiment[1-9]*')))
            # ! To test overfitting on specific dataset
            experiment_folders = sorted(glob(os.path.join(root, image_set, 'Experiment177')))
            for experiment_folder in experiment_folders:
                image1_list += sorted(glob(os.path.join(experiment_folder, 'hdr_left/*.npy')))
                image2_list += sorted(glob(os.path.join(experiment_folder, 'hdr_right/*.npy')))
                disp_list += sorted(glob(os.path.join(experiment_folder, 'ground_truth_disparity_left/*.npy')))
                
        else:
            image_set = 'test'
            experiment_folders = sorted(glob(os.path.join(root, image_set, 'Experiment33')))
            for experiment_folder in experiment_folders:
                image1_list += sorted(glob(os.path.join(experiment_folder, 'hdr_left/*.npy')), key=sort_key_func)
                image2_list += sorted(glob(os.path.join(experiment_folder, 'hdr_right/*.npy')), key=sort_key_func)
                disp_list += sorted(glob(os.path.join(experiment_folder, 'ground_truth_disparity_left/*.npy')), key=sort_key_func)
        
        for idx in range(0, len(image1_list) - 1, 2):  # Only take even index pairs
            if idx + 1 < len(image1_list):  # Ensure idx + 1 is within range
                self.image_list.append([image1_list[idx], image2_list[idx]])
                self.image_list.append([image1_list[idx + 1], image2_list[idx + 1]])
                self.disparity_list.append(disp_list[idx])

def fetch_dataloader(args):
    train_dataset = None
    
    for dataset_name in args.train_datasets:
        if dataset_name.startswith('carla'):
            new_dataset = CARLASequenceDataset()
            print(f"Samples : {len(new_dataset)}")
            logging.info(f"Adding {len(new_dataset)} samples from CARLA")
    
    train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset  
    
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                   pin_memory=True, shuffle=True, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6)) - 2, drop_last=True)
    
    logging.info('Training with %d image pairs' % len(train_dataset))
    return train_loader