import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import logging

from core.utils.read_utils import *
from torch.utils.data import DataLoader, default_collate
import re

def sort_key_func(file):
    numbers = re.findall(r'\d+', os.path.basename(file))
    return int(numbers[0]) if numbers else 0

class RealDataset(Dataset):
    def __init__(self, root='datasets/Real', transform=None, image_set = 'training', test_date_folder=None):
        self.image_list = []
        self.transform = transform
        
        date_folders = sorted(os.listdir(root)) # 날짜별 폴더
        
        if image_set == 'test':
            # test인 경우 test_date_folder에 해당하는 폴더만 사용
            if test_date_folder in date_folders:
                date_folders = [test_date_folder]
            else:
                raise ValueError(f"Test date folder '{test_date_folder}' not found in dataset.")
        
        for date_folder in date_folders:
            date_path = os.path.join(root, date_folder)
            time_folders = sorted(os.listdir(date_path))  # 시간별 폴더
            
            for time_folder in time_folders:
                folder_path = os.path.join(date_path, time_folder)
                raw_path = os.path.join(folder_path, 'raw.npz')
                post_path = os.path.join(folder_path, 'post.npz')

                if os.path.exists(raw_path) and os.path.exists(post_path):
                    self.image_list.append((raw_path, post_path))
                else:
                    print(f"Missing data in {folder_path}, skipping this folder.")

        if image_set == 'training':
            np.random.shuffle(self.image_list)  # training일 경우 전체 데이터를 섞
        
    def __len__(self):
        return len(self.image_list) - 1  # Stereo 쌍을 위해 홀수 개는 제거
    
    def __getitem__(self, index):
        # 첫번째 이미지쌍 불러오기
        raw_path1, post_path1 = self.image_list[index]
        raw_data1 = np.load(raw_path1)
        post_data1 = np.load(post_path1)
        left1 = raw_data1['left']
        right1 = raw_data1['right']
        
        try:
        # Lidar GT point (depth 값을 disparity로 변환)
            lidar_gt = post_data1['projected_depth']  # 여기서 KeyError가 발생할 수 있음
        except KeyError:
            print(f"Missing 'projected_depth' in {post_path1}. Skipping this entry.")
            return None 
        
        # 두번째 이미지쌍 불러오기
        raw_path2, post_path2 = self.image_list[index + 1]
        raw_data2 = np.load(raw_path2)
        left2 = raw_data2['left']
        right2 = raw_data2['right']
        
        # focal_length, baseline
        focal_length = post_data1['k_left'][0,0]
        baseline = np.abs(post_data1['T'][0])
        
        
        left_32bit_1 = convert_to_32bit_bayer_rg24_2(left1, left1.shape[1], left1.shape[0])
        right_32bit_1 = convert_to_32bit_bayer_rg24_2(right1, right1.shape[1], right1.shape[0])
        left_32bit_2 = convert_to_32bit_bayer_rg24_2(left2, left2.shape[1], left2.shape[0])
        right_32bit_2 = convert_to_32bit_bayer_rg24_2(right2, right2.shape[1], right2.shape[0])
        
        # Debayering
        left1_rgb = cv2.cvtColor(bayerToBgr(left_32bit_1), cv2.COLOR_BGR2RGB)
        right1_rgb = cv2.cvtColor(bayerToBgr(right_32bit_1), cv2.COLOR_BGR2RGB)
        left2_rgb = cv2.cvtColor(bayerToBgr(left_32bit_2), cv2.COLOR_BGR2RGB)
        right2_rgb = cv2.cvtColor(bayerToBgr(right_32bit_2), cv2.COLOR_BGR2RGB)
        
        # Bilateral filtering
        left1_rgb = bilateralFilter(left1_rgb)
        right1_rgb = bilateralFilter(right1_rgb)
        left2_rgb = bilateralFilter(left2_rgb)
        right2_rgb = bilateralFilter(right2_rgb)
        
        # Rectification 수행
        left1_rect, right1_rect = self.calibrate_frame(left1_rgb, right1_rgb, post_path1)
        left2_rect, right2_rect = self.calibrate_frame(left2_rgb, right2_rgb, post_path2)
        
        # Tensor로 변환
        left1_rect = torch.from_numpy(left1_rect).permute(2,0,1).float()
        right1_rect = torch.from_numpy(right1_rect).permute(2,0,1).float()
        left2_rect = torch.from_numpy(left2_rect).permute(2,0,1).float()
        right2_rect = torch.from_numpy(right2_rect).permute(2,0,1).float()
        
        # Lidar GT를 depth에서 disparity로 변환
        depth_lidar = lidar_gt.copy()
        depth_lidar[:, 2] = focal_length * baseline / depth_lidar[:, 2]  # disparity 값으로 변환

        # 이미지 u, v 좌표
        u = depth_lidar[:, 0].astype(np.int32)
        v = depth_lidar[:, 1].astype(np.int32)

        # 유효한 u, v 좌표 필터링 (이미지 크기 내에 있는지 확인)
        valid_indices = (u >= 0) & (u < left1_rect.shape[2]) & (v >= 0) & (v < left1_rect.shape[1])
        u = u[valid_indices]
        v = v[valid_indices]
        disparity_lidar = depth_lidar[valid_indices, 2]  # 필터링된 depth_lidar
        
        return [self.image_list[index], self.image_list[index + 1]], left1_rect, right1_rect, left2_rect, right2_rect, torch.from_numpy(disparity_lidar).float(), u, v
    
    def calibrate_frame(self, left, right, post_path):
        post_np = np.load(post_path)
        
        # Calibration 값 불러오기
        k_left = post_np["k_left"]
        dist_left = post_np["d_left"]
        k_right = post_np["k_right"]
        dist_right = post_np["d_right"]
        R = post_np["R"]
        T = post_np["T"]
        
        # Calibration 수행
        calib_image_size = (1440, 928)
        input_image_size = (left.shape[1], left.shape[0])  # (width, height)
        
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            k_left, dist_left, k_right, dist_right, calib_image_size, R, T, alpha=0
        )
        
        map1x, map1y = cv2.initUndistortRectifyMap(
            k_left, dist_left, R1, P1, input_image_size, cv2.CV_32FC1
        )
        map2x, map2y = cv2.initUndistortRectifyMap(
            k_right, dist_right, R2, P2, input_image_size, cv2.CV_32FC1
        )
        
        left_rectified = cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR)
        
        return left_rectified, right_rectified

def collate_fn(batch):
    # None 값을 제거하고 유효한 데이터만 반환
    batch = [data for data in batch if data is not None]
    
    if len(batch) == 0:
        return None
    
    return default_collate(batch)

def fetch_real_dataloader(args):
    
    if 'real' in args.train_datasets:
        train_dataset = RealDataset(root='datasets/Real', image_set='training')
        print(f"Samples : {len(train_dataset)}")
        logging.info(f"Adding {len(train_dataset)} training samples from Real")
    if 'test_real' in args.train_datasets:
        test_dataset = RealDataset(root='datasets/Real', image_set='test', test_date_folder='08_22_15_4')
        print(f"Samples : {len(test_dataset)}")
        logging.info(f"Adding {len(test_dataset)} training samples from test_Real")

    real_loader = DataLoader(
        train_dataset if 'real' in args.train_datasets else test_dataset, 
        batch_size=args.batch_size, 
        shuffle=('real' in args.train_datasets),  # training일 경우만 shuffle
        num_workers=4, 
        pin_memory=True, 
        drop_last=True, 
        collate_fn=collate_fn
    )
    
    return real_loader
