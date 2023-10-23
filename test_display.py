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
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def image_tensor_batch(image):
    to_tensor = ToTensor()
    image = to_tensor(Image.open(image))
    return image[None].to(DEVICE)

def batch_to_image(image):
    image = image/255.0
    image = image.cpu()
    image = np.asarray(image.permute(0,2,3,1).squeeze(0))
    return image

def exposure_change(image, exposure_factor):
    image = image * exposure_factor
    image = torch.clamp(image, min = 0 , max = 1)
    return image

def plot_img(display_imglist):
    fig, axes = plt.subplots(2,3,figsize = (20,5))
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    plot_order = [0,2,4,1,3,5]

    for ax, idx in zip(axes.ravel(), plot_order):
        if idx==0:
            ax.set_title("HDR Original")
        elif idx == 2:
            ax.set_title("HDR with random exposure")
        elif idx == 4:
            ax.set_title("HDR with adjusted exposure")
        ax.imshow(display_imglist[idx]*255)
        ax.axis('off')
        
def plot_img2(display_imglist):
    fig, axes = plt.subplots(3,3,figsize = (20,5))
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    plot_order = [0,3,6,1,4,7,2,5,8]

    for ax, idx in zip(axes.ravel(), plot_order):
        if idx==0:
            ax.set_title("HDR Original")
        elif idx == 3:
            ax.set_title("HDR with random exposure")
        elif idx == 6:
            ax.set_title("HDR with adjusted exposure")
        ax.imshow(display_imglist[idx]*255)
        ax.axis('off')
        
def plot_histogram_list(stacked_histo_tensor_list):
    
    fig, axes = plt.subplots(2,3, figsize = (20,5))
    # fig.subplots_adjust(wspace=0.1, hspace=0.1)
    plot_order = [0,2,4,1,3,5]
    
    for ax, idx in zip(axes.ravel(), plot_order):
        
        if idx == 0:
            ax.set_title("HDR Original stacked histogram")
        elif idx == 2:
            ax.set_title("HDR with random exposure stacked histogram")
        elif idx == 4:
            ax.set_title("HDR with adjusted exposure stacked histogram")
        stacked_histo_tensor_temp = stacked_histo_tensor_list[idx].permute(2,1,0).squeeze(-1).to('cpu')
        n_channels, n_bins = stacked_histo_tensor_temp.shape
        
        for ch in range(n_channels):
            ax.plot(stacked_histo_tensor_temp[ch])
        

def test_display(args):
    
    # ^ checkpoint split 
    checkpoint = torch.load(args.restore_ckpt)
    combined_model_state_dict = checkpoint
    checkpoint_origin = torch.load('/home/juhyung/SAEC/models/raftstereo-middlebury.pth')

    
    global_feature_net_state_dict = {k.replace('GlobalFeatureNet.', ''): v for k, v in combined_model_state_dict.items() if 'GlobalFeatureNet' in k}
    raft_state_dict = {k.replace('RAFTStereo.',''): v for k, v in combined_model_state_dict.items() if 'RAFTStereo' in k}
    # ^
    
    # * model load (GlobalFeatureNet)
    # model_gfn = torch.nn.DataParallel(GlobalFeatureNet())
    # model_gfn.load_state_dict(global_feature_net_state_dict)
    # model_gfn = model_gfn.module
    # model_gfn.to(DEVICE)
    # model_gfn.eval()
    # *
    
    # * model load (RAFT)
    
    combine_model = torch.nn.DataParallel(CombineModel(args))
    del checkpoint['module.RAFTStereo.update_block.mask.2.weight'], checkpoint['module.RAFTStereo.update_block.mask.2.bias']
    combine_model.load_state_dict(checkpoint, strict=False)
    combine_model = combine_model.module
    
    model_gfn = combine_model.GlobalFeatureNet
    model_gfn.to(DEVICE)
    model_gfn.eval()
    
    model_raft = torch.nn.DataParallel(RAFTStereo(args))
    del raft_state_dict['module.update_block.mask.2.weight'], raft_state_dict['module.update_block.mask.2.bias']
    model_raft.load_state_dict(raft_state_dict, strict = False)
    model_raft.load_state_dict(checkpoint_origin, strict=False)
    model_raft = model_raft.module
    
    # model_raft = combine_model.RAFTStereo
    model_raft.to(DEVICE)
    model_raft.eval()
    
    
    # model_raft = torch.nn.DataParallel(CombineModel.RAFTStereo(args))
    # ! 기존 모델과 파라미터 형태 안맞아서 strict false 추후 변경
    # model_raft.load_state_dict(raft_state_dict)
    # model_raft = model_raft.module
    # model_raft.to(DEVICE)
    # model_raft.eval()
    #*
    
    # model = torch.nn.DataParallel(CombineModel(args))
    # model.load_state_dict(combined_model_state_dict)
    # print(model)
    
    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)
    
    
    image_path = '/home/juhyung/SAEC/datasets/left.png'
    image_path2 = '/home/juhyung/SAEC/datasets/right.png'
    
    left_images = sorted(glob.glob(args.left_imgs, recursive=True))
    right_images = sorted(glob.glob(args.right_imgs, recursive=True))
    print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1 = image_tensor_batch(imfile1)
            image2 = image_tensor_batch(imfile2)
            
            left_img_b = image1
            right_img_b = image2
            
            exp_rand_l, exp_rand_r = generate_random_exposure()
            display_imglist = []
            stacked_histo_list = []

            # Original Image
            display_imglist.append(batch_to_image(left_img_b))
            display_imglist.append(batch_to_image(right_img_b))
            stacked_histo_tensor_l, stacked_histo_tensor_r = calculate_histograms(left_img_b, right_img_b)
            stacked_histo_list.append(stacked_histo_tensor_l)
            stacked_histo_list.append(stacked_histo_tensor_r)
            
            # Origianl image depth map
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)
            _, flow_up = model_raft(image1, image2, iters=args.valid_iters, test_mode=True)
            flow_up = padder.unpad(flow_up).squeeze()
            display_imglist.append(-flow_up.cpu().numpy().squeeze())
            

            # Random exposure image
            left_img_b_exp = exposure_change(left_img_b, exp_rand_l)
            right_img_b_exp = exposure_change(right_img_b, exp_rand_r)

            display_imglist.append(batch_to_image(left_img_b_exp))
            display_imglist.append(batch_to_image(right_img_b_exp))
            
            # Random exposure image depth map
            padder = InputPadder(left_img_b_exp.shape, divis_by=32)
            image1, image2 = padder.pad(left_img_b_exp, right_img_b_exp)
            _, flow_up = model_raft(image1, image2, iters=args.valid_iters, test_mode=True)
            flow_up = padder.unpad(flow_up).squeeze()
            display_imglist.append(-flow_up.cpu().numpy().squeeze())

            # Stack histogram (random exposure value)
            stacked_histo_tensor_l_exp, stacked_histo_tensor_r_exp = calculate_histograms(left_img_b_exp, right_img_b_exp)
            stacked_histo_list.append(stacked_histo_tensor_l_exp)
            stacked_histo_list.append(stacked_histo_tensor_r_exp)

            print(f"Initial Random exposure output, L : {exp_rand_l}, R : {exp_rand_r}")

            # Model output
            output_l_exp = model_gfn(stacked_histo_tensor_l_exp.T.to(DEVICE))
            output_r_exp = model_gfn(stacked_histo_tensor_r_exp.T.to(DEVICE))
            print(f"Model exposure output, L : {output_l_exp.item()}, R : {output_r_exp.item()}")

            # Adjust exposure image
            adjusted_img_l = exposure_change(left_img_b_exp, output_l_exp.item())
            adjusted_img_r = exposure_change(right_img_b_exp, output_r_exp.item())

            stacked_histo_tensor_l_adjust, stacked_histo_tensor_r_adjust = calculate_histograms(adjusted_img_l, adjusted_img_r)
            stacked_histo_list.append(stacked_histo_tensor_l_adjust)
            stacked_histo_list.append(stacked_histo_tensor_r_adjust)

            display_imglist.append(batch_to_image(adjusted_img_l))
            display_imglist.append(batch_to_image(adjusted_img_r))
            
            # Adjust exposure image depth map
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(adjusted_img_l, adjusted_img_r)
            _, flow_up = model_raft(image1, image2, iters=args.valid_iters, test_mode=True)
            flow_up = padder.unpad(flow_up).squeeze()
            display_imglist.append(-flow_up.cpu().numpy().squeeze())

            print(f"len(display_imglist) : {len(display_imglist)}")
            print(f"len(stacked_histo_list) : {len(stacked_histo_list)}")
            


            # * Plot image & histgoram
            plot_img2(display_imglist)
            plot_histogram_list(stacked_histo_list)
            plt.show()
            

            # padder = InputPadder(image1.shape, divis_by=32)
            # image1, image2 = padder.pad(image1, image2)

            # _, flow_up = model_raft(image1, image2, iters=args.valid_iters, test_mode=True)

            # flow_up = padder.unpad(flow_up).squeeze()
            
            # # depth_map = flow_up.cpu().numpy().squeeze()

            # file_stem = imfile1.split('/')[-2]
            # if args.save_numpy:
            #     np.save(output_directory / f"{file_stem}.npy", flow_up.cpu().numpy().squeeze())
            # plt.imsave(output_directory / f"{file_stem}.png", -flow_up.cpu().numpy().squeeze(), cmap='jet')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # base checkpoint
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    # parser.add_argument('--base_ckpt', help = "base checkpoint", required=True)
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

    test_display(args)