import os
import cv2
import numpy as np
import torch
import logging
from glob import glob
from ptlflow.data.flow_transforms import ToTensor
from ptlflow.models.base_model.base_model import BaseModel
from ptlflow.utils.utils import InputScaler

from typing import Any, Dict, List, Optional, Tuple, Union
###############################################
#* read file by file extensions
###############################################

def read_gen(file_name):
    ext = os.path.splitext(file_name)[-1]
    
    if ext == '.hdr':
        img = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        min_val = np.min(img)
        max_val = np.max(img)
        normalized_img = (img - min_val) / (max_val - min_val)
        return normalized_img

    elif ext == '.npy':
        img = np.load(file_name)
        # read disparity file
        if len(img.shape)==2:
            return img
        # read image file
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            min_val = np.min(img)
            max_val = np.max(img)
            normalized_img = (img - min_val) / (max_val - min_val)
            return normalized_img, img
    
    return []

def prepare_inputs_custom(
    images: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    flows: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    inputs: Optional[Dict[str, Any]] = None,
    image_only: bool = False,
    **kwargs: Union[torch.Tensor, List[torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Transform torch inputs into the input format of the optical flow models.

    This version handles torch tensors directly, ensuring proper handling of tensors on GPU.

    Parameters
    ----------
    images : Union[torch.Tensor, List[torch.Tensor]]
        One or more images to use to estimate the optical flow. Typically, it will be at least with two images in the
        CHW format.
    flows : Optional[Union[torch.Tensor, List[torch.Tensor]]], optional
        One or more groundtruth optical flow, which can be used for validation. Typically it will be an array CHW.
    inputs : Optional[Dict[str, Any]]
        Dict containing input tensors or other metadata. Only the tensors will be transformed.
    image_only : Optional[bool]
        If True, only applies scaling and padding to the images.
    kwargs : Union[torch.Tensor, List[torch.Tensor]]
        Any other tensor inputs can be provided as keyworded arguments. This function will create an entry in the input dict
        for each keyworded tensor given.

    Returns
    -------
    Dict[str, Any]
        The inputs converted and transformed to the input format of the optical flow models.
    """
    if inputs is None:
        inputs = {"images": images, "flows": flows}
        inputs.update(kwargs)
        keys_to_remove = []
        for k, v in inputs.items():
            if v is None or len(v) == 0:
                keys_to_remove.append(k)
        for k in keys_to_remove:
            del inputs[k]

    for k, v in inputs.items():
        if image_only and k != "images":
            continue

        if isinstance(v, list):
            v = torch.stack(v)  # list to tensor
        if isinstance(v, torch.Tensor):
            while len(v.shape) < 5:
                v = v.unsqueeze(0)
            inputs[k] = v

    return inputs

# Convert Raw image to 32bit format
def convert_to_32bit_bayer_rg24_2(raw_data, width, height):
    # raw_data = np.fromfile(raw_data, dtype=np.uint8)
    raw_data = raw_data.reshape(-1, 3)
    raw_int32 = (raw_data[:, 0].astype(np.uint32) +
                (raw_data[:, 1].astype(np.uint32) << 8) +
                (raw_data[:, 2].astype(np.uint32) << 16))
    return raw_int32.reshape(height, width)

# # Debayering 32bit image to BGR image
# def bayerToBgr(bayer_img: np.ndarray) -> np.ndarray:
#     height, width = bayer_img.shape

#     # Initialize the BGR channels
#     red_channel = np.zeros((height, width), dtype=np.float32)
#     green_channel = np.zeros((height, width), dtype=np.float32)
#     blue_channel = np.zeros((height, width), dtype=np.float32)

#     # Interpolate R, G, B channels
#     # Red channel
#     red_channel[0:height:2, 0:width:2] = bayer_img[0:height:2, 0:width:2]
#     red_channel[1 : height - 1 : 2, 0:width:2] = (
#         bayer_img[0 : height - 2 : 2, 0:width:2] + bayer_img[2:height:2, 0:width:2]
#     ) / 2.0
#     red_channel[0:height:2, 1 : width - 1 : 2] = (
#         bayer_img[0:height:2, 0 : width - 2 : 2] + bayer_img[0:height:2, 2:width:2]
#     ) / 2.0
#     red_channel[1 : height - 1 : 2, 1 : width - 1 : 2] = (
#         bayer_img[0 : height - 2 : 2, 0 : width - 2 : 2]
#         + bayer_img[0 : height - 2 : 2, 2:width:2]
#         + bayer_img[2:height:2, 0 : width - 2 : 2]
#         + bayer_img[2:height:2, 2:width:2]
#     ) / 4.0

#     # Green channel
#     green_channel[0:height:2, 1:width:2] = bayer_img[0:height:2, 1:width:2]
#     green_channel[1:height:2, 0:width:2] = bayer_img[1:height:2, 0:width:2]
#     green_channel[0 : height - 2 : 2, 0 : width - 2 : 2] = (
#         bayer_img[0 : height - 2 : 2, 1 : width - 1 : 2]
#         + bayer_img[1 : height - 1 : 2, 0 : width - 2 : 2]
#     ) / 2.0
#     green_channel[1 : height - 1 : 2, 1 : width - 1 : 2] = (
#         bayer_img[1 : height - 2 : 2, 0 : width - 2 : 2]
#         + bayer_img[0 : height - 3 : 2, 1 : width - 1 : 2]
#         + bayer_img[1 : height - 1 : 2, 2:width:2]
#         + bayer_img[2:height:2, 1 : width - 1 : 2]
#     ) / 4.0

#     # Blue channel
#     blue_channel[1:height:2, 1:width:2] = bayer_img[1:height:2, 1:width:2]
#     blue_channel[0 : height - 2 : 2, 1:width:2] = (
#         bayer_img[0 : height - 2 : 2, 1:width:2] + bayer_img[2:height:2, 1:width:2]
#     ) / 2.0
#     blue_channel[1:height:2, 0 : width - 2 : 2] = (
#         bayer_img[1:height:2, 0 : width - 2 : 2] + bayer_img[1:height:2, 2:width:2]
#     ) / 2.0
#     blue_channel[0 : height - 2 : 2, 0 : width - 2 : 2] = (
#         bayer_img[0 : height - 2 : 2, 0 : width - 2 : 2]
#         + bayer_img[0 : height - 2 : 2, 2:width:2]
#         + bayer_img[2:height:2, 0 : width - 2 : 2]
#         + bayer_img[2:height:2, 2:width:2]
#     ) / 4.0

#     # Merge the channels into a BGR image
#     bgr_image = np.stack((blue_channel, green_channel, red_channel), axis=-1)
#     bgr_image = (bgr_image / (2 << 24)).astype(np.float32)
#     return bgr_image

# Debayering 32bit image to BGR image using bilinear interpolation (based on Lucid Vision Labs approach)
import numpy as np

# Debayering 32bit image to BGR image using bilinear interpolation (RGGB pattern)
def bayerToBgr(bayer_img: np.ndarray) -> np.ndarray:
    height, width = bayer_img.shape

    # Initialize the BGR channels
    red_channel = np.zeros((height, width), dtype=np.float32)
    green_channel = np.zeros((height, width), dtype=np.float32)
    blue_channel = np.zeros((height, width), dtype=np.float32)

    # Interpolate Red channel (located at even rows and even columns)
    red_channel[0:height:2, 0:width:2] = bayer_img[0:height:2, 0:width:2]  # Copy red pixels
    red_channel[1:height-1:2, 0:width:2] = (bayer_img[0:height-2:2, 0:width:2] + bayer_img[2:height:2, 0:width:2]) / 2.0
    red_channel[0:height:2, 1:width-1:2] = (bayer_img[0:height:2, 0:width-2:2] + bayer_img[0:height:2, 2:width:2]) / 2.0
    red_channel[1:height-1:2, 1:width-1:2] = (
        bayer_img[0:height-2:2, 0:width-2:2] + bayer_img[0:height-2:2, 2:width:2] +
        bayer_img[2:height:2, 0:width-2:2] + bayer_img[2:height:2, 2:width:2]
    ) / 4.0

    # Interpolate Green channel (located at even rows, odd columns and odd rows, even columns)
    green_channel[0:height:2, 1:width:2] = bayer_img[0:height:2, 1:width:2]  # Copy green pixels on even rows
    green_channel[1:height:2, 0:width:2] = bayer_img[1:height:2, 0:width:2]  # Copy green pixels on odd rows
    green_channel[0:height-2:2, 0:width-2:2] = (bayer_img[0:height-2:2, 1:width-1:2] + bayer_img[1:height-1:2, 0:width-2:2]) / 2.0
    green_channel[1:height-1:2, 1:width-1:2] = (
        bayer_img[1:height-2:2, 0:width-2:2] + bayer_img[0:height-3:2, 1:width-1:2] +
        bayer_img[1:height-1:2, 2:width:2] + bayer_img[2:height:2, 1:width-1:2]
    ) / 4.0

    # Interpolate Blue channel (located at odd rows and odd columns)
    blue_channel[1:height:2, 1:width:2] = bayer_img[1:height:2, 1:width:2]  # Copy blue pixels
    blue_channel[0:height-2:2, 1:width:2] = (bayer_img[0:height-2:2, 1:width:2] + bayer_img[2:height:2, 1:width:2]) / 2.0
    blue_channel[1:height:2, 0:width-2:2] = (bayer_img[1:height:2, 0:width-2:2] + bayer_img[1:height:2, 2:width:2]) / 2.0
    blue_channel[0:height-2:2, 0:width-2:2] = (
        bayer_img[0:height-2:2, 0:width-2:2] + bayer_img[0:height-2:2, 2:width:2] +
        bayer_img[2:height:2, 0:width-2:2] + bayer_img[2:height:2, 2:width:2]
    ) / 4.0

    # Merge the channels into a BGR image
    bgr_image = np.stack((blue_channel, green_channel, red_channel), axis=-1)
    bgr_image = (bgr_image/2**24).astype(np.float32)
    
    return bgr_image

# Bilateral filter 
def bilateralFilter(bgr_image: np.ndarray) -> np.ndarray:
    # Bilateral filter to reduce grid-like artifacts while preserving edges
    filtered_image = cv2.bilateralFilter(bgr_image, d=1, sigmaColor=20, sigmaSpace=20)

    return filtered_image


# Rectifictation 
def calibrate_frame(left, right, folder: str):
    print(folder)
    
    # Load raw and post calibration data
    raw_np = np.load(folder + "/raw.npz")
    post_np = np.load(folder + "/post.npz")  # Load calibration data from post.npz
    ranges = raw_np["ranges"]
    
    # Extract calibration data from post.npz
    k_left = post_np["k_left"]
    dist_left = post_np["d_left"]
    k_right = post_np["k_right"]
    dist_right = post_np["d_right"]
    R = post_np["R"]
    T = post_np["T"]

    # Calibration resolution
    calib_image_size = (1440, 928)

    # Input image shape (800x600)
    input_image_size = (left.shape[1], left.shape[0])  # (width, height) -> (800, 600)

    # Stereo rectification  (useing calibration resolution)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        k_left,
        dist_left,
        k_right,
        dist_right,
        calib_image_size,
        R,
        T,
        alpha=0,
    )

    map1x, map1y = cv2.initUndistortRectifyMap(
        k_left, dist_left, R1, P1, input_image_size, cv2.CV_32FC1  # 800x600
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        k_right, dist_right, R2, P2, input_image_size, cv2.CV_32FC1  # 800x600
    )

    left_rectified = cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR)

    return left_rectified, right_rectified

# Image cropping
def image_crop(image, crop_width=700, crop_height=500):
    if len(image.shape)==3:
        h, w = image.shape[:2]
    
    if len(image.shape)==2:
        h, w = image.shape
        
    x_start = (w - crop_width) // 2
    y_start = (h - crop_height) // 2
    
    cropped_img = image[y_start:y_start + crop_height, x_start:x_start + crop_width]
    
    return cropped_img
