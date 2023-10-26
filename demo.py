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

def convert_to_tensor(self, image):
    if isinstance(image, Image.Image):
        to_tensor = ToTensor()
        return to_tensor(image)
    return image

def simul_to_saec_to_simul():
    # * Load image (HDR scene)
    left_image_path = 'datasets/left.png'
    left_original_image = Image.open(left_image_path)
    right_image_path = 'datasets/right.png'
    right_original_image = Image.open(right_image_path)

    # * Simulate image (HDR -> LDR)

    left_image_np_n = convert_to_tensor(left_original_image)
    right_image_np_n = convert_to_tensor(right_original_image)

    exp_rand_l, exp_rand_r = generate_random_exposure()
    a_l, b_l = cal_dynamic_range(left_image_np_n, exp_rand_l)
    a_r, b_r = cal_dynamic_range(right_image_np_n, exp_rand_r)

    # ? test value
    print(f"left exposure value : {exp_rand_l} | left image dynamic value : {round(a_l,3)} ~ {round(b_l,3)}")
    print(f"right exposure value : {exp_rand_r} | right image dynamic value : {round(a_r,3)} ~ {round(b_r,3)}")
    # ?

    adjust_img_np_l = adjust_dr(left_original_image, exp_rand_l, (a_l,b_l))
    adjust_img_np_r = adjust_dr(right_original_image, exp_rand_r, (a_r, b_r))

    simulated_img_np_l = poisson_gauss_noise(adjust_img_np_l, iso=100)
    simulated_img_np_r = poisson_gauss_noise(adjust_img_np_r, iso = 100)

    simulated_img_np_disp_l = Image.fromarray((simulated_img_np_l*255).astype(np.uint8))
    simulated_img_np_disp_r = Image.fromarray((simulated_img_np_r*255).astype(np.uint8))

    # ? display test
    simulated_img_np_disp_l.show()
    simulated_img_np_disp_r.show()
    # ?

    # * SAEC module
    # ^ Caculate histogram by different scale
    histogram_coarest_l = histogram_subimage(simulated_img_np_l, 1)
    histogram_intermediate_l = histogram_subimage(simulated_img_np_l, 3)
    histogram_finest_l = histogram_subimage(simulated_img_np_disp_l,7)

    histogram_coarest_r = histogram_subimage(simulated_img_np_r, 1)
    histogram_intermediate_r = histogram_subimage(simulated_img_np_r, 3)
    histogram_finest_r = histogram_subimage(simulated_img_np_disp_r,7)

    #^ Stack histogram [256,59]
    list_of_histograms_l = [histogram_coarest_l, histogram_intermediate_l, histogram_finest_l]
    merged_list_l = [item for sublist in list_of_histograms_l for item in sublist]
    stacked_histo_tensor_l = torch.stack(merged_list_l, dim = 1)

    list_of_histograms_r = [histogram_coarest_r, histogram_intermediate_r, histogram_finest_r]
    merged_list_r = [item for sublist in list_of_histograms_r for item in sublist]
    stacked_histo_tensor_r = torch.stack(merged_list_r, dim = 1)

    # ^ Input network model
    saec_model_l = GlobalFeatureNet()
    output_l = saec_model_l(stacked_histo_tensor_l.T)

    saec_model_r = GlobalFeatureNet()
    output_r = saec_model_r(stacked_histo_tensor_r.T)

    # ^ Shifted exposure value
    # ToDo shifted exposure value값이 항상 커지기만 함. 이게 맞나?
    # Todo : Threshold 값 이용해서 높은건 낮추고, 낮춘건 높이기?
    shifted_exp_l = exposure_shift(exp_rand_l, output_l.item())
    shifted_exp_r = exposure_shift(exp_rand_r, output_r.item())

    # ? expected exposure values test
    # Todo output expected value가 Mexp 값을 넘는경우가 발생 체크
    print(f"Left expected value : {output_l.item()} | Right expected value : {output_r.item()}")
    print(f"Shifted left expected value : {shifted_exp_l} | Shifted right expected value : {shifted_exp_r}")
    # ?

    # * Simulate Image LDR with shifted exposure value
    a_l_s, b_l_s = cal_dynamic_range(left_image_np_n, shifted_exp_l)
    a_r_s, b_r_s = cal_dynamic_range(right_image_np_n, shifted_exp_r)

    adjust_img_np_l = adjust_dr(left_original_image, shifted_exp_l, (a_l_s, b_l_s))
    adjust_img_np_r = adjust_dr(right_original_image, shifted_exp_r, (a_r_s, b_r_s))

    simulated_img_np_l = poisson_gauss_noise(adjust_img_np_l, iso = 100)
    simulated_img_np_r = poisson_gauss_noise(adjust_img_np_r, iso = 100)

    simulated_img_np_disp_l = Image.fromarray((simulated_img_np_l*255).astype(np.uint8))
    simulated_img_np_disp_r = Image.fromarray((simulated_img_np_r*255).astype(np.uint8))

    # ? display test
    simulated_img_np_disp_l.show()
    simulated_img_np_disp_r.show()
    
    return simulated_img_np_disp_l, simulated_img_np_disp_r

# * Depth reconstruction RAFT

def demo(args):
    
    left_imgs, right_imgs = simul_to_saec_to_simul()
    
    combine_model = CombineModel(args)
    checkpoint_state_dict = torch.load(args.restore_ckpt)
    raft_stereo_state_dict = combine_model.RAFTStereo.state_dict()
    
    for name, param in checkpoint_state_dict.items():
        if name.startswith('RAFTStereo.'):  # RAFTStereo 모듈에 해당하는 키만 선택
            key = name[len('RAFTStereo.'):]  # 'RAFTStereo.' 접두사 제거
            if key in raft_stereo_state_dict:
                raft_stereo_state_dict[key] = param
    
    combine_model.RAFTStereo.load_state_dict(raft_stereo_state_dict)
    
    # model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model = torch.nn.DataParallel(CombineModel(args), device_ids=[0])
    
    # model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    # global_feature_net = model.module.GlobalFeatureNet
    # for param in global_feature_net.parameters():
    #     print(param.requires_grad)
        
    
    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        # left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        # right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        
        # print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        # for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
        #     image1 = load_image(imfile1)
        #     image2 = load_image(imfile2)

        #     padder = InputPadder(image1.shape, divis_by=32)
        #     image1, image2 = padder.pad(image1, image2)

        #     _, flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
        #     flow_up = padder.unpad(flow_up).squeeze()

        #     file_stem = imfile1.split('/')[-2]
        #     if args.save_numpy:
        #         np.save(output_directory / f"{file_stem}.npy", flow_up.cpu().numpy().squeeze())
        #     plt.imsave(output_directory / f"{file_stem}.png", -flow_up.cpu().numpy().squeeze(), cmap='jet')
        

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