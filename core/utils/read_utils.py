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
            log_max_val = np.log1p(img.max())
            min_val = np.min(img)
            max_val = np.max(img)
            normalized_img = (img - min_val) / (max_val - min_val)
            return normalized_img, log_max_val
    
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