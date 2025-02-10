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
    """
    에폭에 따라 Blur 강도를 조절하는 함수.
    
    Args:
        epoch (int): 현재 에폭
        total_epochs (int): 총 학습 에폭 수
        initial_mean (int): 초기 Blur 강도
        final_mean (int): 최종 Blur 강도
        initial_std (int): 초기 Blur 표준편차
        final_std (int): 최종 Blur 표준편차
    
    Returns:
        tuple: (mean_degree, std_degree)
    """
    progress = epoch / total_epochs  # 학습 진행 비율 (0.0 ~ 1.0)
    
    # 선형적으로 Blur 강도 증가
    mean_degree = initial_mean + (final_mean - initial_mean) * progress
    std_degree = initial_std + (final_std - initial_std) * progress
    
    return mean_degree, std_degree

def apply_motion_blur(image, degree, angle):
    """
    이미지에 Motion Blur 적용
    Args:
        image (np.ndarray): 입력 이미지 (H, W, C)
        degree (int): Blur 강도 (kernel size)
        angle (int): Blur 방향 (0~360도)
    Returns:
        np.ndarray: Blur가 적용된 이미지
    """
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    kernel = np.diag(np.ones(degree))
    kernel = cv2.warpAffine(kernel, M, (degree, degree))
    kernel = kernel / degree
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

def apply_random_motion_blur(image, blur_prob=0.5, mean_degree=25, std_degree=6):
    """
    랜덤하게 Motion Blur를 적용
    Args:
        image (np.ndarray): 입력 이미지
        blur_prob (float): blur를 적용할 확률
        mean_degree (int): 평균 blur 강도
        std_degree (int): blur 강도의 표준편차
    Returns:
        np.ndarray: Blur가 적용된 이미지 (또는 원본 이미지)
    """
    if random.random() < blur_prob:
        degree = int(np.clip(np.random.normal(mean_degree, std_degree), 5, 50))
        angle = random.randint(0, 360)
        return apply_motion_blur(image, degree, angle)
    else:
        return image

class StereoDataset(data.Dataset):
    def __init__(self, reader=None, apply_blur=True, mean_degree=5, std_degree=2):
        if reader is None:
            self.disparity_reader = read_gen
        else:
            self.disparity_reader = reader
            
        self.is_test = False
        self.apply_blur = apply_blur
        self.mean_degree = mean_degree
        self.std_degree = std_degree
        
        self.flow_list = []
        self.disparity_list = []
        self.image_list = []
        self.extra_info = []
    
    def update_blur_parameters(self, mean_degree, std_degree):
        """
        학습 진행에 따라 Blur 강도 업데이트
        Args:
            mean_degree (float): 새로운 Blur 평균 강도
            std_degree (float): 새로운 Blur 표준편차
        """
        self.mean_degree = mean_degree
        self.std_degree = std_degree

    def apply_random_blur_to_images(self, images):
        """
        입력된 이미지 리스트 중 랜덤으로 blur 적용
        """
        num_images_to_blur = random.randint(0, 4)
        
        # 이미지 ID로 Blur 적용 대상을 결정
        images_to_blur_ids = set(id(img) for img in random.sample(images, num_images_to_blur))
        
        blurred_images = []
        for img in images:
            img_np = img.permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)
            
            if id(img) in images_to_blur_ids:  # 이미지의 ID를 기준으로 비교
                img_blurred = apply_random_motion_blur(img_np, blur_prob=1.0, mean_degree=self.mean_degree, std_degree=self.std_degree)
            else:
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
        
        # Blur 적용 여부 확인
        if self.apply_blur:
            img1_left, img1_right, img2_left, img2_right = self.apply_random_blur_to_images(
                [img1_left, img1_right, img2_left, img2_right]
            )
        
        hdr_img1 = torch.from_numpy(hdr_img1).permute(2,0,1).float()
        hdr_img2 = torch.from_numpy(hdr_img2).permute(2,0,1).float()
        
        valid = (flow[0].abs() < 512) & (flow[1].abs() < 512)
        
        flow = flow[:1]
        
        return self.image_list[index] + self.image_list[index + 1] + [self.disparity_list[disp_index]], img1_left, img1_right, img2_left, img2_right, flow, valid.float()
    
    def __len__(self):
        return len(self.image_list) // 2


class CARLASequenceDataset(StereoDataset):
    def __init__(self, root='datasets/CARLA2', image_set='training', apply_blur=True):
        super(CARLASequenceDataset, self).__init__(reader=read_gen, apply_blur=apply_blur)
        assert os.path.exists(root), "Dataset root path does not exist."
        
        
        self.image_list = []
        self.disparity_list = []
        self.experiment_names = []
        
        if image_set == 'training':
            experiment_folders = sorted(glob(os.path.join(root, image_set, 'Experiment[1-9]*')))
        else:
            experiment_folders = sorted(glob(os.path.join(root, image_set, 'Experiment14')))
    
        for experiment_folder in experiment_folders:
            # Experiment 폴더 이름을 추출하고 저장
            experiment_name = os.path.basename(experiment_folder)
            self.experiment_names.append(experiment_name)
            
            # 각 실험 폴더에서 이미지를 로드
            image1_list = sorted(glob(os.path.join(experiment_folder, 'hdr_left/*.npy')), key=sort_key_func)
            image2_list = sorted(glob(os.path.join(experiment_folder, 'hdr_right/*.npy')), key=sort_key_func)
            disp_list = sorted(glob(os.path.join(experiment_folder, 'ground_truth_disparity_left/*.npy')), key=sort_key_func)

            # 파일 수 불일치 체크
            if not (len(image1_list) == len(image2_list) == len(disp_list)):
                logging.warning(f"File count mismatch in {experiment_folder}: "
                                f"image1_list={len(image1_list)}, image2_list={len(image2_list)}, disp_list={len(disp_list)}")

            # 짝수 인덱스 쌍으로 image_list와 disparity_list에 추가
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
        # 한 실험당 이미지 쌍 개수를 구해서 나눈 값을 사용해 experiment 이름을 가져옵니다.
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