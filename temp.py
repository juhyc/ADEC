from __future__ import print_function, division

import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from core.raft_stereo import RAFTStereo
from core.combine_model import CombineModel

from evaluate_stereo import *
import core.stereo_datasets as datasets

from core.saec import histogram_subimage
from PIL import Image
from torchvision.transforms import ToTensor

image = Image.open('/home/juhyung/SAEC/datasets/left.png')

if isinstance(image, Image.Image):
    to_tensor = ToTensor()
    image = to_tensor(image)
    
_, height, width = image.shape

print(image.shape)
print(height)