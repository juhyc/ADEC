import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import logging

from core.utils.read_utils import *
from torch.utils.data import DataLoader, default_collate
# Dataset 클래스 정의

class RealDataset(Dataset):
    def __init__(self, root='datasets/Real', transform=None, crop_size=(300, 800)):
        self.image_list = []
        self.transform = transform
        self.crop_size = crop_size  # 크롭 크기 설정
        
        # 날짜별 폴더 읽기
        date_folders = sorted(os.listdir(root)) 
        
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
        
    def __len__(self):
        return len(self.image_list) - 1  # Stereo 쌍을 위해 홀수 개는 제거
    
    # 이미지와 Lidar 데이터를 중심 기준으로 크롭하는 함수
    def crop_image_vertical(self, image, lidar_u, lidar_v, lidar_disparity):
        H, W = image.shape[:2]
        crop_h = self.crop_size[0]  # 높이만 설정

        # 이미지 중심 y좌표 (높이 기준으로)
        center_y = H // 2
        
        # 크롭 시작점 (위아래를 자름)
        y_start = center_y - crop_h // 2
        
        # 이미지 크롭 (좌우는 그대로 두고 위아래만 크롭)
        cropped_image = image[y_start:y_start + crop_h, :]
        
        # Lidar u, v 좌표 조정 (y_start만큼 뺌)
        lidar_v_crop = lidar_v - y_start

        # 유효한 u, v 좌표 필터링 (위아래 자른 이미지 내에 있는지 확인)
        valid_indices = (lidar_v_crop >= 0) & (lidar_v_crop < crop_h)
        
        # 필터링된 Lidar 데이터 반환
        lidar_u_crop = lidar_u[valid_indices]  # u 좌표는 변화 없음
        lidar_v_crop = lidar_v_crop[valid_indices]
        lidar_disparity_crop = lidar_disparity[valid_indices]

        return cropped_image, lidar_u_crop, lidar_v_crop, lidar_disparity_crop
    
    def __getitem__(self, index):
        # 첫번째 이미지쌍 불러오기
        raw_path1, post_path1 = self.image_list[index]
        raw_data1 = np.load(raw_path1)
        post_data1 = np.load(post_path1)
        left1 = raw_data1['left']
        right1 = raw_data1['right']
        
        try:
            # Lidar GT point (depth 값을 disparity로 변환)
            lidar_gt = post_data1['projected_depth']  
        except KeyError:
            print(f"Missing 'projected_depth' in {post_path1}. Skipping this entry.")
            return None 
        
        # 두번째 이미지쌍 불러오기
        raw_path2, post_path2 = self.image_list[index + 1]
        raw_data2 = np.load(raw_path2)
        left2 = raw_data2['left']
        right2 = raw_data2['right']
        
        # focal_length, baseline
        focal_length = post_data1['k_left'][0, 0]
        baseline = np.abs(post_data1['T'][0])
        
        # 이미지 Bayer 포맷을 32비트로 변환
        left_32bit_1 = convert_to_32bit_bayer_rg24_2(left1, left1.shape[1], left1.shape[0])
        right_32bit_1 = convert_to_32bit_bayer_rg24_2(right1, right1.shape[1], right1.shape[0])
        left_32bit_2 = convert_to_32bit_bayer_rg24_2(left2, left2.shape[1], left2.shape[0])
        right_32bit_2 = convert_to_32bit_bayer_rg24_2(right2, right2.shape[1], right2.shape[0])
        
        # Debayering
        left1_rgb = cv2.cvtColor(bayerToBgr(left_32bit_1), cv2.COLOR_BGR2RGB)
        right1_rgb = cv2.cvtColor(bayerToBgr(right_32bit_1), cv2.COLOR_BGR2RGB)
        left2_rgb = cv2.cvtColor(bayerToBgr(left_32bit_2), cv2.COLOR_BGR2RGB)
        right2_rgb = cv2.cvtColor(bayerToBgr(right_32bit_2), cv2.COLOR_BGR2RGB)
        
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
        disparity_lidar = depth_lidar[:, 2]

        # 크롭된 이미지와 Lidar 데이터를 얻기 위해 crop_image_center 함수 호출
        left1_crop, u_crop, v_crop, disparity_lidar_crop = self.crop_image_vertical(left1_rect.numpy(), u, v, disparity_lidar)
        right1_crop, _, _, _ = self.crop_image_vertical(right1_rect.numpy(), u, v, disparity_lidar)
        left2_crop, _, _, _ = self.crop_image_vertical(left2_rect.numpy(), u, v, disparity_lidar)
        right2_crop, _, _, _ = self.crop_image_vertical(right2_rect.numpy(), u, v, disparity_lidar)
        
        # 크롭된 이미지와 Lidar GT 반환
        return [self.image_list[index], self.image_list[index + 1]], torch.from_numpy(left1_crop), torch.from_numpy(right1_crop), torch.from_numpy(left2_crop), torch.from_numpy(right2_crop), torch.from_numpy(disparity_lidar_crop).float(), u_crop, v_crop
    
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

# DataLoader의 collate_fn
def collate_fn(batch):
    # None 값을 제거하고 유효한 데이터만 반환
    batch = [data for data in batch if data is not None]
    
    if len(batch) == 0:
        return None
    
    return default_collate(batch)

# Dataloader 함수 정의
def fetch_real_dataloader(args):
    real_dataset = RealDataset(root='datasets/Real')
    print(f"Samples : {len(real_dataset)}")
    logging.info(f"Adding {len(real_dataset)} samples from Real")
    
    real_loader = DataLoader(
        real_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True, collate_fn=collate_fn
    )
    
    logging.info('Training with %d image pairs' % len(real_dataset))
    
    return real_loader
