import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

from core.utils.read_utils import read_gen
from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor

import os
import cv2
import logging
import numpy as np
from glob import glob

class StereoDataset(data.Dataset):
    def __init__(self,  reader = None):
  
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
            return img1,img2, self.extra_info[index]
        
        index = index % len(self.image_list)
        disp = self.disparity_reader(self.disparity_list[index])
        
        if isinstance(disp, tuple):
            disp, valid = disp
        else:
            valid = disp < 512
            
        img1 = read_gen(self.image_list[index][0])
        img2 = read_gen(self.image_list[index][1])
        
        img1 = np.array(img1)
        img2 = np.array(img2)
        
        disp = np.array(disp).astype(np.float32)
        flow = np.stack([-disp, np.zeros_like(disp)], axis = -1)
        
        img1 = torch.from_numpy(img1).permute(2,0,1).float()
        img2 = torch.from_numpy(img2).permute(2,0,1).float()
        flow = torch.from_numpy(flow).permute(2,0,1).float()
        
        valid = (flow[0].abs() < 512) & (flow[1].abs() < 512)
            
        flow = flow[:1]
        
        return self.image_list[index] + [self.disparity_list[index]], img1, img2, flow, valid.float()
    
    def __len__(self):
        return len(self.image_list)
    
class CARLA(StereoDataset):
    def __init__(self, root = '/home/juhyung/SAEC/test/dataset/Experiment18', image_set='training'):
        super(CARLA, self).__init__(reader=read_gen)
        assert os.path.exists(root)
        
        if image_set =='training':
            image1_list = sorted(glob(os.path.join(root, image_set, 'hdr_left/*.hdr')))
            image2_list = sorted(glob(os.path.join(root, image_set, 'hdr_right/*.hdr')))
            disp_list = sorted(glob(os.path.join(root, 'training', 'ground_truth_disparity_left/*.npy')))
        else:
            image_set = 'training'
            image1_list = sorted(glob(os.path.join(root, image_set, 'hdr_left/0.hdr')))
            image2_list = sorted(glob(os.path.join(root, image_set, 'hdr_right/0.hdr')))
            disp_list = sorted(glob(os.path.join(root, 'training', 'ground_truth_disparity_left/0.npy')))
        
        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]

def fetch_dataloader(args):
    
    train_dataset = None
    
    for dataset_name in args.train_datasets:
        if dataset_name.startswith('carla'):
            new_dataset = CARLA()
            logging.info(f"Adding {len(new_dataset)} samples from CARLA")
    
    train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset  
            
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                   pin_memory=True, shuffle=True, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6))-2, drop_last=True)
    
    logging.info('Training with %d image pairs' % len(train_dataset))
    return train_loader