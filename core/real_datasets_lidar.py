import os
import glob as glob_module  
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

def transfrom_points(points: np.ndarray, transform_mtx: np.ndarray):

    """
    Transform points using a 4x4 transformation matrix
    Args:
        points (np.ndarray): 3D points to transform
        transform_mtx (np.ndarray): 4x4 transformation matrix
    Returns:
        np.ndarray: Transformed points
    """
    points = points.reshape(-1, 3)
    points = points[(points[:, 0] != 0) | (points[:, 1] != 0)]
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points = transform_mtx @ points.T

    return points[:3].T

def transform_point_inverse(points: np.ndarray, transform_mtx: np.ndarray):

    """
    Transform points using a 4x4 transformation matrix
    Args:
        points (np.ndarray): 3D points to transform
        transform_mtx (np.ndarray): 4x4 transformation matrix
    Returns
        np.ndarray: Transformed points
    """
    transform_mtx = np.linalg.pinv(transform_mtx)
    return transfrom_points(points, transform_mtx)

def project_points_on_camera(

    points: np.ndarray,
    focal_length: float,
    cx: float,
    cy: float,
    image_width: float = 0,
    image_height: float = 0,
):

    """
    Project 3D points to 2D image plane

    Args:
        points (np.ndarray): 3D points to project
        focal_length (float): Focal length of the camera
        cx (float): Principal point x-coordinate
        cy (float): Principal point y-coordinate
        image_width (float): Image width, Optional
        image_height (float): Image height, Optional
    Returns:
        np.ndarray: Projected points

    """
    points[:, 0] = points[:, 0] * focal_length / points[:, 2] + cx
    points[:, 1] = points[:, 1] * focal_length / points[:, 2] + cy

    if image_width > 0 and image_height > 0:
        points = points[
            (points[:, 0] >= 0)
            & (points[:, 0] <= image_width - 1)
            & (points[:, 1] <= image_height - 1)
            & (points[:, 2] > 0)
        ]

    return points

class RealDataset(Dataset):
    def __init__(self, root='datasets/Real', transform=None, image_set = 'training', test_date_folders=None):
        self.image_list = []
        self.transform = transform
        self.camera_param = np.load('datasets/camera_params/post.npz')
        self.camera_param_path  ='datasets/camera_params/post.npz'

        self.experiment_names = []
        
        self.transform_mtx = np.array([
        [9.74168269e-01, -2.16619390e-02, -2.24781992e-01, 3.68182351e+01],
        [2.23991311e-01, -3.38457838e-02, 9.74003263e-01, 2.71851960e+02],
        [-2.87067220e-02, -9.99192285e-01, -2.81194025e-02, -2.35719906e+02],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
         ])

        date_folders = sorted(os.listdir(root)) 
        
        # Training
        if image_set == 'training':
            date_folders = sorted(os.listdir(root))
        # Test
        elif image_set == 'test' and test_date_folders:
            date_folders = [os.path.basename(folder) for folder in test_date_folders]
        else:
            raise ValueError("Invalid image set or no test folders specified.")
        
        
        for date_folder in date_folders:
            date_path = os.path.join(root, date_folder)
            time_folders = sorted(os.listdir(date_path)) 
            
            for time_folder in time_folders:
                folder_path = os.path.join(date_path, time_folder)
                left_path = os.path.join(folder_path, 'left.npy')
                right_path = os.path.join(folder_path, 'right.npy')
                point_path = os.path.join(folder_path, 'points.npy')
                

                if os.path.exists(left_path) and os.path.exists(right_path) and os.path.exists(point_path):
                    self.image_list.append((left_path, right_path, point_path))
                    self.experiment_names.append(date_folder)
                else:
                    print(f"Missing data in {folder_path}, skipping this folder.")

        if image_set == 'training':
            np.random.shuffle(self.image_list)
        
    def __len__(self):
        return len(self.image_list) - 1  
    
    def __getitem__(self, index):
        # First frame stereo
        left_path1, right_path1, point_path1 = self.image_list[index]
        left_data1 = np.load(left_path1)
        right_data1 = np.load(right_path1)
        point_data1 = np.load(point_path1).reshape(-1, 3)*1000 # mm
        left1 = left_data1
        right1 = right_data1
        
        # Second frame stereo
        left_path2, right_path2, point_path2 = self.image_list[index + 1]
        left_data2 = np.load(left_path2)
        right_data2 = np.load(right_path2)
        # point_data2 = np.load(point_path2)
        left2 = left_data2
        right2 = right_data2
        
        # focal_length, baseline
        focal_length = self.camera_param['k_left'][0,0]
        baseline = np.abs(self.camera_param['T'][0,0])
        
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
        
        # Rectification
        left1_rect, right1_rect = self.calibrate_frame(left1_rgb, right1_rgb, self.camera_param_path)
        left2_rect, right2_rect = self.calibrate_frame(left2_rgb, right2_rgb, self.camera_param_path)
        
        left1_rect = torch.from_numpy(left1_rect).permute(2,0,1).float()
        right1_rect = torch.from_numpy(right1_rect).permute(2,0,1).float()
        left2_rect = torch.from_numpy(left2_rect).permute(2,0,1).float()
        right2_rect = torch.from_numpy(right2_rect).permute(2,0,1).float()
        
        hdr_left1 = left1_rect*(2**24)
        hdr_left2 = left2_rect*(2**24)
        
        # left1_rect = (left1_rect-left1_rect.min())/(left1_rect.max()-left1_rect.min())
        # right1_rect = (right1_rect-right1_rect.min())/(right1_rect.max()-right1_rect.min())
        # left2_rect = (left2_rect-left2_rect.min())/(left2_rect.max()-left2_rect.min())
        # right2_rect = (right2_rect-right2_rect.min())/(right2_rect.max()-right2_rect.min())
        
        # Lidar point sampling
        points = transform_point_inverse(point_data1, self.transform_mtx)
        points = project_points_on_camera(points, 1323.50, 684, 557,1440,928)
        
        return [self.image_list[index], self.image_list[index + 1]], left1_rect, right1_rect, left2_rect, right2_rect, focal_length, baseline, points, hdr_left1, hdr_left2
    
    def calibrate_frame(self, left, right, post_path):
        post_np = np.load(post_path)
        
        # Load calibration configurations
        k_left = post_np["k_left"]
        dist_left = post_np["d_left"]
        k_right = post_np["k_right"]
        dist_right = post_np["d_right"]
        R = post_np["R"]
        T = post_np["T"]
        
        # Calibration
        calib_image_size = (1440, 928)
        # input_image_size = (left.shape[1], left.shape[0])  # (width, height)
        
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            k_left, dist_left, k_right, dist_right, calib_image_size, R, T, alpha=0
        )
        
        map1x, map1y = cv2.initUndistortRectifyMap(
            k_left, dist_left, R1, P1, (1440, 928), cv2.CV_32FC1
        )
        map2x, map2y = cv2.initUndistortRectifyMap(
            k_right, dist_right, R2, P2, (1440, 928), cv2.CV_32FC1
        )
        
        left_rectified = cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR)
        
        return left_rectified, right_rectified
    
    def get_experiment_name(self, index):
        """Return the experiment name based on the index."""
        images_per_experiment = len(self.image_list) // len(self.experiment_names)
        experiment_index = index // images_per_experiment
        return self.experiment_names[experiment_index]

def collate_fn(batch):
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
        test_date_folders = sorted(glob_module.glob('datasets/Real/Test7'))
        # test_date_folders = sorted(glob_module.glob('datasets/Real_test/Test15'))
        if not test_date_folders:
            raise ValueError("No test folders found with pattern 'Test[1-9]*' in Real_test directory.")
        
        test_dataset = RealDataset(root='datasets/Real', image_set='test', test_date_folders=test_date_folders)
        print(f"Samples : {len(test_dataset)}")
        logging.info(f"Adding {len(test_dataset)} testing samples from Real_test")

    real_loader = DataLoader(
        train_dataset if 'real' in args.train_datasets else test_dataset, 
        batch_size=args.batch_size, 
        shuffle=('real' in args.train_datasets), 
        num_workers=4, 
        pin_memory=True, 
        drop_last=True, 
        collate_fn=collate_fn
    )
    
    return real_loader
