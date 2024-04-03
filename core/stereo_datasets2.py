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

###############################################
# * Dataset class for synthetic CARLA dataset
###############################################

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
        
        # # ! Add resizing
        height, width = img1.shape[1], img1.shape[0]
        img1 = cv2.resize(img1, (height//2, width//2))
        img2 = cv2.resize(img2, (height//2, width//2))
        disp = cv2.resize(disp, (height//2, width//2))
        disp = disp/2
        
        flow = np.stack([-disp, np.zeros_like(disp)], axis = -1)
        
        img1 = torch.from_numpy(img1).permute(2,0,1).float()
        img2 = torch.from_numpy(img2).permute(2,0,1).float()
        flow = torch.from_numpy(flow).permute(2,0,1).float()
        
        valid = (flow[0].abs() < 512) & (flow[1].abs() < 512)
            
        flow = flow[:1]
        
        return self.image_list[index] + [self.disparity_list[index]], img1, img2, flow, valid.float()
    
    def __len__(self):
        return len(self.image_list)

# * For CARLA synthetic dataset class

class CARLA(StereoDataset):
    def __init__(self, root = 'datasets/CARLA', image_set='training'):
        super(CARLA, self).__init__(reader=read_gen)
        assert os.path.exists(root), "Dataset root path does not exist."
        
        image1_list = []
        image2_list = []
        disp_list = []
        
        if image_set == 'training':
            experiment_folders = sorted(glob(os.path.join(root, image_set, 'Experiment[1-9]*')))                
            for experiment_folder in experiment_folders:
                image1_list += sorted(glob(os.path.join(experiment_folder, 'hdr_left/*.npy')))
                image2_list += sorted(glob(os.path.join(experiment_folder, 'hdr_right/*.npy')))
                disp_list += sorted(glob(os.path.join(experiment_folder, 'ground_truth_disparity_left/*.npy')))
        else:
            # Validation set
            image_set = 'test'
            
            # * For Single image         
            # experiment_folders = os.path.join(root, image_set, 'Valid')
            # image1_list = [os.path.join(experiment_folder, 'hdr_left/*.hdr')]
            # image2_list = [os.path.join(experiment_folder, 'hdr_right/*.hdr')]
            # disp_list = [os.path.join(experiment_folder, 'ground_truth_disparity_left/*.npy')]
            
            # * For Full sequence image
            # experiment_folders = sorted(glob(os.path.join(root, image_set, 'Experiment[1-9]*')))        
            # for experiment_folder in experiment_folders:
            #     image1_list += sorted(glob(os.path.join(experiment_folder, 'hdr_left/*.npy')))
            #     image2_list += sorted(glob(os.path.join(experiment_folder, 'hdr_right/*.npy')))
            #     disp_list += sorted(glob(os.path.join(experiment_folder, 'ground_truth_disparity_left/*.npy')))
                
            # * For first sequence image
            experiment_folders = sorted(glob(os.path.join(root, image_set, 'Experiment[1-9]*')))
            for experiment_folder in experiment_folders:
                hdr_left_files = sorted(glob(os.path.join(experiment_folder, 'hdr_left/*.npy')))
                if hdr_left_files:  
                    image1_list.append(hdr_left_files[0])  
                hdr_right_files = sorted(glob(os.path.join(experiment_folder, 'hdr_right/*.npy')))
                if hdr_right_files:
                    image2_list.append(hdr_right_files[0])
                disp_files = sorted(glob(os.path.join(experiment_folder, 'ground_truth_disparity_left/*.npy')))
                if disp_files:
                    disp_list.append(disp_files[0])

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]

def fetch_dataloader(args):
    
    train_dataset = None
    
    for dataset_name in args.train_datasets:
        if dataset_name.startswith('carla'):
            new_dataset = CARLA()
            print(f"Samples : {len(new_dataset)}")
            logging.info(f"Adding {len(new_dataset)} samples from CARLA")
    
    train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset  
            
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                   pin_memory=True, shuffle=True, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6))-2, drop_last=True)
    
    logging.info('Training with %d image pairs' % len(train_dataset))
    return train_loader