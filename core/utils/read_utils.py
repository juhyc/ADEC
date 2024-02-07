import os
import cv2
import numpy as np
from glob import glob

# file의 확장자명을 파악하고 읽는 형태
def read_gen(file_name):
    ext = os.path.splitext(file_name)[-1]
    
    if ext == '.hdr':
        return cv2.imread(file_name, cv2.IMREAD_ANYDEPTH)
    elif ext == '.npy':
        return np.load(file_name)
    
    return []