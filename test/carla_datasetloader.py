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
import re

import torch.nn.functional as F
import sys
core_path = os.path.abspath('/home/user/juhyung/SAEC/core')
sys.path.append(core_path)
from core.utils.read_utils import read_gen
from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor
from skimage.transform import resize

import sys
core_path = os.path.abspath('/home/user/juhyung/SAEC/core')
sys.path.append(core_path)
###############################################
# * Dataset class for synthetic CARLA dataset considering sequence
###############################################

def sort_key_func(file):
    numbers = re.findall(r'\d+', os.path.basename(file))
    return int(numbers[0]) if numbers else 0

class StereoDataset(data.Dataset):
    def __init__(self, reader=None):
        self.disparity_reader = reader if reader is not None else read_gen
        self.is_test = False
        self.flow_list = []
        self.disparity_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):
        if self.is_test:
            img1 = read_gen(self.image_list[index][0])
            img2 = read_gen(self.image_list[index][1])
            img1 = np.array(img1)[..., :3]
            img2 = np.array(img2)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        # Adjust index not to exceed last frame
        index = index * 2 # To ensure the batch has a step of 2
        if index >= len(self.image_list) - 1:
            index = len(self.image_list) - 2

        current_frame = self.image_list[index]
        next_frame = self.image_list[index + 1]
        current_disp = self.disparity_reader(self.disparity_list[index])

        if isinstance(current_disp, tuple):
            current_disp, valid = current_disp
        else:
            valid = current_disp < 512

        # Load image and disparity map as numpy array
        img1 = np.array(read_gen(current_frame[0]))
        img2 = np.array(read_gen(current_frame[1]))
        img1_next = np.array(read_gen(next_frame[0])) if index + 1 < len(self.image_list) else img1
        img2_next = np.array(read_gen(next_frame[1])) if index + 1 < len(self.image_list) else img2
        disp = np.array(current_disp).astype(np.float32)

        # Resize images and disparity map
        height, width = img1.shape[1], img1.shape[0]
        img1 = cv2.resize(img1, (height // 2, width // 2))
        img2 = cv2.resize(img2, (height // 2, width // 2))
        img1_next = cv2.resize(img1_next, (height // 2, width // 2))
        img2_next = cv2.resize(img2_next, (height // 2, width // 2))
        disp = cv2.resize(disp, (height // 2, width // 2))
        disp = disp / 2

        flow = np.stack([-disp, np.zeros_like(disp)], axis=-1)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        img1_next = torch.from_numpy(img1_next).permute(2, 0, 1).float()
        img2_next = torch.from_numpy(img2_next).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        valid = (flow[0].abs() < 512) & (flow[1].abs() < 512)

        flow = flow[:1]

        return self.image_list[index] + [self.disparity_list[index]], img1, img2, img1_next, img2_next, flow, valid.float()

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
            experiment_folders = sorted(glob(os.path.join(root, image_set, 'Experiment[1-9]*')))
            for experiment_folder in experiment_folders:
                image1_files = sorted(glob(os.path.join(experiment_folder, 'hdr_left/*.npy')), key=sort_key_func)
                image2_files = sorted(glob(os.path.join(experiment_folder, 'hdr_right/*.npy')), key=sort_key_func)
                disp_files = sorted(glob(os.path.join(experiment_folder, 'ground_truth_disparity_left/*.npy')), key=sort_key_func)

                image1_list.extend(image1_files)
                image2_list.extend(image2_files)
                disp_list.extend(disp_files)
        else:
            image_set = 'test'
            experiment_folders = sorted(glob(os.path.join(root, image_set, 'Experiment34')))
            for experiment_folder in experiment_folders:
                image1_files = sorted(glob(os.path.join(experiment_folder, 'hdr_left/*.npy')), key=sort_key_func)
                image2_files = sorted(glob(os.path.join(experiment_folder, 'hdr_right/*.npy')), key=sort_key_func)
                disp_files = sorted(glob(os.path.join(experiment_folder, 'ground_truth_disparity_left/*.npy')), key=sort_key_func)

                image1_list.extend(image1_files)
                image2_list.extend(image2_files)
                disp_list.extend(disp_files)

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]

def fetch_dataloader(args):
    train_dataset = None

    for dataset_name in args.train_datasets:
        if (dataset_name.startswith('carla')):
            new_dataset = CARLASequenceDataset()
            print(f"Samples : {len(new_dataset)}")
            logging.info(f"Adding {len(new_dataset)} samples from CARLA")

            if train_dataset is None:
                train_dataset = new_dataset
            else:
                train_dataset.image_list += new_dataset.image_list
                train_dataset.disparity_list += new_dataset.disparity_list

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   pin_memory=True, shuffle=False, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6))-2, drop_last=False)

    logging.info('Training with %d image pairs' % len(train_dataset))
    return train_loader


def train(args):
    train_loader = fetch_dataloader(args)
    print("Load train loader")
    
    for i_batch, (temp, *data_blob) in enumerate(tqdm(train_loader)):
        print(f"i_batch size : {i_batch}")
        print(f"data blob size : {len(data_blob)}")
        left_hdr, right_hdr, left_next_hdr, right_next_hdr, disparity, valid = [x.cuda() for x in data_blob]
        
        print(temp)
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