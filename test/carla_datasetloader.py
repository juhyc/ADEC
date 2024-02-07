import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import os
import cv2
from glob import glob

# CUDA
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# file의 확장자명을 파악하고 읽는 형태
def read_gen(file_name):
    ext = os.path.splitext(file_name)[-1]
    
    if ext == '.hdr':
        return cv2.imread(file_name, cv2.IMREAD_ANYDEPTH)
    elif ext == '.npy':
        return np.load(file_name)
    
    return []

class StereoDataset(data.Dataset):
    def __init__(self, reader = None):
        
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

def train(args):
    train_loader = fetch_dataloader(args)
    print("Load train loader")
    
    for i_batch, (_, *data_blob) in enumerate(tqdm(train_loader)):
        print(f"i_batch size : {i_batch}")
        print(f"data blob size : {len(data_blob)}")
        left_hdr, right_hdr, disparity, valid = [x.cuda() for x in data_blob]
        
        print(left_hdr.shape)
        print(right_hdr.shape)
        print(disparity.shape)
        print(valid.shape)
        


if __name__  == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='SAEC', help="name your experiment")
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=6, help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['carla'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=100000, help="length of training schedule.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[320, 720], help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=16, help="number of updates to the disparity field in each forward pass.")

    args = parser.parse_args()
    
    train(args)