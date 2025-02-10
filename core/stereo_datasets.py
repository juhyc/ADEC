# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import logging
import os
import re
import copy
import math
import random
from pathlib import Path
from glob import glob
import os.path as osp
import cv2

from core.utils import frame_utils
from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor

###############################################
# ! Original RAFT-stereo dataset class
###############################################

def sort_key_func(file):
    numbers = re.findall(r'\d+', os.path.basename(file))
    return int(numbers[0]) if numbers else 0

class StereoDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, reader=None):
        self.augmentor = None
        self.sparse = sparse
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        if aug_params is not None and "crop_size" in aug_params:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        if reader is None:
            self.disparity_reader = frame_utils.read_gen
        else:
            self.disparity_reader = reader        

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.disparity_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):
        index = index * 2 % len(self.image_list)
        disp_index = index // 2
        disp = self.disparity_reader(self.disparity_list[disp_index])

        if isinstance(disp, tuple):
            disp, valid = disp
        else:
            valid = disp < 512

        img1_left = frame_utils.read_gen(self.image_list[index][0])
        img1_right = frame_utils.read_gen(self.image_list[index][1])
        img2_left = frame_utils.read_gen(self.image_list[index + 1][0])
        img2_right = frame_utils.read_gen(self.image_list[index + 1][1])

        img1_left = np.array(img1_left).astype(np.uint8)
        img1_right = np.array(img1_right).astype(np.uint8)
        img2_left = np.array(img2_left).astype(np.uint8)
        img2_right = np.array(img2_right).astype(np.uint8)
        disp = np.array(disp).astype(np.float32)

        # 원본 이미지와 시차 맵의 높이와 너비를 가져옵니다.
        h, w = img1_left.shape[:2]

        # 모델의 입력 크기에 맞추기 위해 필요한 패딩 크기를 계산합니다.
        desired_h = 376  # 모델이 예상하는 높이
        desired_w = 1244  # 모델이 예상하는 너비

        pad_h = desired_h - h
        pad_w = desired_w - w

        # 패딩이 필요한 경우에만 패딩을 적용합니다.
        if pad_h > 0 or pad_w > 0:
            # 이미지에 패딩 추가
            img1_left = np.pad(img1_left, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
            img1_right = np.pad(img1_right, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
            img2_left = np.pad(img2_left, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
            img2_right = np.pad(img2_right, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
            # 시차 맵에 패딩 추가
            disp = np.pad(disp, ((0, pad_h), (0, pad_w)), mode='edge')
            valid = np.pad(valid, ((0, pad_h), (0, pad_w)), mode='edge')

        # 만약 이미지가 예상 크기보다 크다면 크롭합니다.
        if pad_h < 0 or pad_w < 0:
            crop_h = desired_h
            crop_w = desired_w
            img1_left = img1_left[:crop_h, :crop_w]
            img1_right = img1_right[:crop_h, :crop_w]
            img2_left = img2_left[:crop_h, :crop_w]
            img2_right = img2_right[:crop_h, :crop_w]
            disp = disp[:crop_h, :crop_w]
            valid = valid[:crop_h, :crop_w]

        flow = np.stack([-disp, np.zeros_like(disp)], axis=-1)

        img1_left = torch.from_numpy(img1_left).permute(2, 0, 1).float()
        img1_right = torch.from_numpy(img1_right).permute(2, 0, 1).float()
        img2_left = torch.from_numpy(img2_left).permute(2, 0, 1).float()
        img2_right = torch.from_numpy(img2_right).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        valid = (flow[0].abs() < 512) & (flow[1].abs() < 512)
        flow = flow[:1]

        return self.image_list[index] + self.image_list[index + 1] + [self.disparity_list[disp_index]], img1_left, img1_right, img2_left, img2_right, flow, valid.float()

    
    
    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.flow_list = v * copy_of_self.flow_list
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self
        
    def __len__(self):
        return len(self.image_list)

class SceneFlowDatasets(StereoDataset):
    def __init__(self, aug_params=None, root='datasets', dstype='frames_cleanpass', things_test=False):
        super(SceneFlowDatasets, self).__init__(aug_params)
        self.root = root
        self.dstype = dstype

        if things_test:
            self._add_things("TEST")
        else:
            self._add_things("TRAIN")
            self._add_monkaa()
            self._add_driving()

    def _add_things(self, split='TRAIN'):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'FlyingThings3D')
        left_images = sorted( glob(osp.join(root, self.dstype, split, '*/*/left/*.png')) )
        right_images = [ im.replace('left', 'right') for im in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        # Choose a random subset of 400 images for validation
        state = np.random.get_state()
        np.random.seed(1000)
        val_idxs = set(np.random.permutation(len(left_images))[:400])
        np.random.set_state(state)

        for idx, (img1, img2, disp) in enumerate(zip(left_images, right_images, disparity_images)):
            if (split == 'TEST' and idx in val_idxs) or split == 'TRAIN':
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from FlyingThings {self.dstype}")

    def _add_monkaa(self):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'Monkaa')
        left_images = sorted( glob(osp.join(root, self.dstype, '*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Monkaa {self.dstype}")


    def _add_driving(self):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'Driving')
        left_images = sorted( glob(osp.join(root, self.dstype, '*/*/*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Driving {self.dstype}")


class ETH3D(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/ETH3D', split='training'):
        super(ETH3D, self).__init__(aug_params, sparse=True)

        image1_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im0.png')) )
        image2_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im1.png')) )
        disp_list = sorted( glob(osp.join(root, 'two_view_training_gt/*/disp0GT.pfm')) ) if split == 'training' else [osp.join(root, 'two_view_training_gt/playground_1l/disp0GT.pfm')]*len(image1_list)

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class SintelStereo(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/SintelStereo'):
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispSintelStereo)

        image1_list = sorted( glob(osp.join(root, 'training/*_left/*/frame_*.png')) )
        image2_list = sorted( glob(osp.join(root, 'training/*_right/*/frame_*.png')) )
        disp_list = sorted( glob(osp.join(root, 'training/disparities/*/frame_*.png')) ) * 2

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            assert img1.split('/')[-2:] == disp.split('/')[-2:]
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class FallingThings(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/FallingThings'):
        super().__init__(aug_params, reader=frame_utils.readDispFallingThings)
        assert os.path.exists(root)

        with open(os.path.join(root, 'filenames.txt'), 'r') as f:
            filenames = sorted(f.read().splitlines())

        image1_list = [osp.join(root, e) for e in filenames]
        image2_list = [osp.join(root, e.replace('left.jpg', 'right.jpg')) for e in filenames]
        disp_list = [osp.join(root, e.replace('left.jpg', 'left.depth.png')) for e in filenames]

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class TartanAir(StereoDataset):
    def __init__(self, aug_params=None, root='datasets', keywords=[]):
        super().__init__(aug_params, reader=frame_utils.readDispTartanAir)
        assert os.path.exists(root)

        with open(os.path.join(root, 'tartanair_filenames.txt'), 'r') as f:
            filenames = sorted(list(filter(lambda s: 'seasonsforest_winter/Easy' not in s, f.read().splitlines())))
            for kw in keywords:
                filenames = sorted(list(filter(lambda s: kw in s.lower(), filenames)))

        image1_list = [osp.join(root, e) for e in filenames]
        image2_list = [osp.join(root, e.replace('_left', '_right')) for e in filenames]
        disp_list = [osp.join(root, e.replace('image_left', 'depth_left').replace('left.png', 'left_depth.npy')) for e in filenames]

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class KITTI(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/KITTI', image_set='training'):
        super(KITTI, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI)
        assert os.path.exists(root)

        # Todo) image_set : training/validate 에 따라 loading 되는 부분 수정하기
        if image_set == 'training':
            image1_list = sorted(glob(os.path.join(root, image_set, 'image_2/*_10.png')))
            image2_list = sorted(glob(os.path.join(root, image_set, 'image_3/*_10.png')))
            disp_list = sorted(glob(os.path.join(root, 'training', 'disp_occ_0/*_10.png'))) if image_set == 'training' else [osp.join(root, 'training/disp_occ_0/000085_10.png')]*len(image1_list)
        else:
            image_set = 'training'
            image1_list = sorted(glob(os.path.join(root, image_set, 'image_2/000016_10.png')))
            image2_list = sorted(glob(os.path.join(root, image_set, 'image_3/000016_10.png')))
            disp_list = sorted(glob(os.path.join(root, 'training', 'disp_occ_0/000016_10.png')))
            
        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
            
class KITTI_Sequence(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/KITTI', image_set='training'):
        super(KITTI_Sequence, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI)
        assert os.path.exists(root)
        
        self.image_list = []
        self.disparity_list = []

        if image_set == 'training':
            image1_list = sorted(glob(os.path.join(root, image_set, 'image_2/*_10.png')))
            image2_list = sorted(glob(os.path.join(root, image_set, 'image_3/*_10.png')))
            image1_next_list = sorted(glob(os.path.join(root, image_set, 'image_2/*_11.png')))
            image2_next_list = sorted(glob(os.path.join(root, image_set, 'image_3/*_11.png')))
            disp_list = sorted(glob(os.path.join(root, 'training', 'disp_occ_0/*_10.png'))) if image_set == 'training' else [osp.join(root, 'training/disp_occ_0/000085_10.png')]*len(image1_list)
        else:
            image_set = 'training'
            image1_list = sorted(glob(os.path.join(root, image_set, 'image_2/000016_10.png')))
            image2_list = sorted(glob(os.path.join(root, image_set, 'image_3/000016_10.png')))
            image1_next_list = sorted(glob(os.path.join(root, image_set, 'image_2/000016_11.png')))
            image2_next_list = sorted(glob(os.path.join(root, image_set, 'image_3/000016_11.png')))
            disp_list = sorted(glob(os.path.join(root, 'training', 'disp_occ_0/000016_10.png')))
            
        for idx in range(len(image1_list)):
            self.image_list.append([image1_list[idx], image2_list[idx]])
            self.image_list.append([image1_next_list[idx], image2_next_list[idx]])
            self.disparity_list.append(disp_list[idx])

        # 데이터 개수 일관성 확인
        if not (len(self.image_list) / 2 == len(self.disparity_list)):
            logging.warning(f"Data count mismatch: image_list {len(self.image_list)}, disparity_list {len(self.disparity_list)}")

        print(f"image_list size : {len(self.image_list)}, disp_list size: {len(self.disparity_list)}")
        
    def __len__(self):
        return len(self.image_list) //2 
    

class Middlebury(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/Middlebury', split='F'):
        super(Middlebury, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispMiddlebury)
        assert os.path.exists(root)
        assert split in ["F", "H", "Q", "2014"]
        if split == "2014": # datasets/Middlebury/2014/Pipes-perfect/im0.png
            scenes = list((Path(root) / "2014").glob("*"))
            for scene in scenes:
                for s in ["E","L",""]:
                    self.image_list += [ [str(scene / "im0.png"), str(scene / f"im1{s}.png")] ]
                    self.disparity_list += [ str(scene / "disp0.pfm") ]
        else:
            lines = list(map(osp.basename, glob(os.path.join(root, "MiddEval3/trainingF/*"))))
            lines = list(filter(lambda p: any(s in p.split('/') for s in Path(os.path.join(root, "MiddEval3/official_train.txt")).read_text().splitlines()), lines))
            image1_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/im0.png') for name in lines])
            image2_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/im1.png') for name in lines])
            disp_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/disp0GT.pfm') for name in lines])
            assert len(image1_list) == len(image2_list) == len(disp_list) > 0, [image1_list, split]
            for img1, img2, disp in zip(image1_list, image2_list, disp_list):
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]

  
def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """

    # aug_params = {'crop_size': args.image_size, 'min_scale': args.spatial_scale[0], 'max_scale': args.spatial_scale[1], 'do_flip': False, 'yjitter': not args.noyjitter}
    # if hasattr(args, "saturation_range") and args.saturation_range is not None:
    #     aug_params["saturation_range"] = args.saturation_range
    # if hasattr(args, "img_gamma") and args.img_gamma is not None:
    #     aug_params["gamma"] = args.img_gamma
    # if hasattr(args, "do_flip") and args.do_flip is not None:
    #     aug_params["do_flip"] = args.do_flip

    train_dataset = None
    for dataset_name in args.train_datasets:
        if dataset_name.startswith("middlebury_"):
            new_dataset = Middlebury(split=dataset_name.replace('middlebury_',''))
        elif dataset_name == 'sceneflow':
            clean_dataset = SceneFlowDatasets(dstype='frames_cleanpass')
            final_dataset = SceneFlowDatasets(dstype='frames_finalpass')
            new_dataset = (clean_dataset*4) + (final_dataset*4)
            logging.info(f"Adding {len(new_dataset)} samples from SceneFlow")
        elif 'kitti' in dataset_name:
            new_dataset = KITTI_Sequence()
            logging.info(f"Adding {len(new_dataset)} samples from KITTI")
        elif dataset_name == 'sintel_stereo':
            new_dataset = SintelStereo()*140
            logging.info(f"Adding {len(new_dataset)} samples from Sintel Stereo")
        elif dataset_name == 'falling_things':
            new_dataset = FallingThings()*5
            logging.info(f"Adding {len(new_dataset)} samples from FallingThings")
        elif dataset_name.startswith('tartan_air'):
            new_dataset = TartanAir(keywords=dataset_name.split('_')[2:])
            logging.info(f"Adding {len(new_dataset)} samples from Tartain Air")
        train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=True, shuffle=True, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6))-2, drop_last=True)

    logging.info('Training with %d image pairs' % len(train_dataset))
    return train_loader

