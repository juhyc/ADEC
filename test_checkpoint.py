import sys
sys.path.append('core')

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from core.utils.simulate import *
from core.raft_stereo import RAFTStereo
from core.utils.utils import InputPadder
from PIL import Image
from core.saec import *
from core.combine_model import CombineModel

DEVICE = 'cuda'


def load_image(imfile):
    img = np.array(imfile).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)



# * Depth reconstruction RAFT

def demo(args):
    
    raft_model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    combined_model = torch.nn.DataParallel(CombineModel(args), device_ids=[0])
    
    raft_checkpoint = torch.load(args.restore_ckpt) # 기존 체크 포인트 module.layer
    
    new_raft_state_dict = {} # module 이름 빠진거 layer.
    for k, v in raft_checkpoint.items():
        if k.startswith('module.'):
            new_k = k[7:]  # 'module.' 접두사 제거
        else:
            new_k = k
        new_raft_state_dict[new_k] = v
    
    combined_state_dict = combined_model.state_dict() # module.GFN, module.RAFT.layer.
    count= 0
    
    for k in new_raft_state_dict.keys():
        combined_keys = "module.RAFTStereo." + k
        if combined_keys in combined_state_dict:
            combined_state_dict[combined_keys] = new_raft_state_dict[k]
            count += 1
    
    print(f"original raft parameter : {len(raft_checkpoint.keys())}")
    print(f"Combine model parameter keys len : {len(combined_model.state_dict().keys())}")    
    print(f"count : {count}")
    
    
    # check keys
    current_keys = set(raft_model.state_dict().keys())
    checkpoint_keys = set(raft_checkpoint.keys())
    combined_keys = set(combined_model.state_dict().keys())

    combined_state_dict_keys = set(combined_state_dict.keys())
    # print(combined_state_dict_keys)
    
    missing_keys = combined_model.state_dict().keys() - combined_state_dict_keys
    unexpected_keys = combined_state_dict_keys - combined_model.state_dict().keys()
    
    
    print("Missing keys:", missing_keys)
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("Unexpected keys:", unexpected_keys)
    print("===============================================")

    
    model = model.module
    model.to(DEVICE)
    model.eval()
    
    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():

        left_image_path = 'datasets/left.png'
        left_imgs = Image.open(left_image_path)
        right_image_path = 'datasets/right.png'
        right_imgs = Image.open(right_image_path)

        image1 = load_image(left_imgs)
        image2 = load_image(right_imgs)
        print("===========Load Simulated imgs with shifted exposure factor============")

        # padder = InputPadder(image1.shape, divis_by=32)
        # image1, image2 = padder.pad(image1, image2)

        _, flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
        flow_up = flow_up.squeeze()


        file_stem = '/home/juhyung/SAEC/datasets'.split('/')[-1]
        
        if args.save_numpy:
            np.save(output_directory / f"{file_stem}.npy", flow_up.cpu().numpy().squeeze())
        plt.imsave(output_directory / f"{file_stem}.png", -flow_up.cpu().numpy().squeeze(), cmap='jet')
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="datasets/Middlebury/MiddEval3/testH/*/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="datasets/Middlebury/MiddEval3/testH/*/im1.png")
    parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    
    args = parser.parse_args()

    demo(args)