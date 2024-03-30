import os
import cv2
import numpy as np
from glob import glob

# file의 확장자명을 파악하고 읽는 형태
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
            return normalized_img
    
    return []