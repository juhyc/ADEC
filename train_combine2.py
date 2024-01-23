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
from core.utils.utils import InputPadder

from evaluate_stereo import *
import core.stereo_datasets as datasets
import matplotlib.pyplot as plt
from PIL import Image

from core.loss import BerHuLoss
from core.depth_datasets import DepthDataset_stereo
from core.utils.display import *

# try:
#     from torch.cuda.amp import GradScaler
# except:
#     # dummy GradScaler for PyTorch < 1.6
#     class GradScaler:
#         def __init__(self):
#             pass
#         def scale(self, loss):
#             return loss
#         def unscale_(self, optimizer):
#             pass
#         def step(self, optimizer):
#             optimizer.step()
#         def update(self):
#             pass

# ^ CombineModle(SAEC + RAFT + Fusion) trainging code.

# Tensorboard를 위한 Writer 초기화
writer = SummaryWriter('runs/combine_pipeline_demo')

# CUDA
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ^ Train
def train(args):

    model = torch.nn.DataParallel(CombineModel(args), device_ids=[0])
    # model = nn.DataParallel(RAFTStereo(args))
    print("Parameter Count: %d" % count_parameters(model))
    
    # Todo) dataloader 수정
    train_loader = datasets.fetch_dataloader(args)
    criterion = BerHuLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr = 0.0002)
    total_steps = 0

    # ToDo) RAFT load_state_dict 수정
    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        raft_checkpoint = torch.load(args.restore_ckpt)
        
        # * downsampling = 3 인 경우
        if args.n_downsample == 3:
            del raft_checkpoint['module.update_block.mask.2.weight'], raft_checkpoint['module.update_block.mask.2.bias']
        
        new_raft_state_dict = {}
        for k, v in raft_checkpoint.items():
            if k.startswith('module.'):
                new_k = k[7:]
            else:
                new_k = k
            new_raft_state_dict[new_k] = v
        
        combined_state_dict = model.state_dict()
        count = 0
        
        for k in new_raft_state_dict.keys():
            combined_keys = "module.RAFTStereo." + k
            if combined_keys in combined_state_dict:
                combined_state_dict[combined_keys] = new_raft_state_dict[k] 
                count += 1
        
        model.load_state_dict(combined_state_dict)
                
        logging.info(f"Done loading checkpoint")
    
    model.cuda()
    model.train()
    
    # RAFTStereo 의 가중치 고정
    for param in model.module.RAFTStereo.parameters():
        param.requires_grad = False
    
    model.module.RAFTStereo.freeze_bn() # We keep BatchNorm frozen

    validation_frequency = 10 #10000

    should_keep_training = True
    global_batch_num = 0
    while should_keep_training:

        for i_batch, (_, *data_blob) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            left_hdr, right_hdr, disparity, valid = [x.cuda() for x in data_blob]

            assert model.training
            fused_disparity, disparity1, disparity2 = model(left_hdr, right_hdr, iters=args.train_iters ) 

            # ^ visualize during training

            writer.add_image('disparity1', visualize_flow_cmap(disparity1), global_batch_num)
            writer.add_image('disparity2', visualize_flow_cmap(disparity2), global_batch_num)
            writer.add_image('Disparity_prediction', visualize_flow_cmap(fused_disparity), global_batch_num)

            assert model.training

            valid_mask = (valid >= 0.5)
            valid_mask = valid_mask.unsqueeze(1)
            loss = criterion(fused_disparity[valid_mask], disparity[valid_mask])
            
            writer.add_scalar("live_loss", loss.item(), global_batch_num)
            writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], global_batch_num)
            global_batch_num += 1
            
            loss.backward()
            optimizer.step()
            
            # # Todo) Validation code 수정
            # if total_steps % validation_frequency == validation_frequency - 1:
            #     print("====Validation====")
            #     valid_num = (total_steps / validation_frequency) * 10 + 1
            #     save_path = Path('checkpoints/%d_%s.pth' % (total_steps + 1, args.name))
            #     logging.info(f"Saving file {save_path.absolute()}")
            #     torch.save(model.state_dict(), save_path)

            #     results, flow_valid, flow_valid_gt, left_ldr_cap_valid, right_ldr_cap_valid, left_ldr_adj_denom_valid, right_ldr_adj_denom_valid, exposure_dict_valid, left_ldr_adj_valid, right_ldr_adj_valid = validate_kitti(model.module, iters=args.valid_iters)
                
            #     # valid output_exp 확인용
            #     output_exp_valid = {"Valid " + key : value for key, value in exposure_dict_valid.items()}
                
            #     logger.write_dict(results)
            #     logger.write_dict(output_exp_valid)
            #     logger.writer.add_image('E_Disparity gt(Valid)', visualize_flow_cmap(flow_valid_gt, image1), valid_num)
            #     logger.writer.add_image('E_Disparity prediction_c(Valid)', visualize_flow_cmap(flow_valid, image1), valid_num)
            #     logger.writer.add_image('F_Captured left LDR image(Valid)', check_ldr_image(left_ldr_cap_valid), valid_num)
            #     logger.writer.add_image('F_Captured right LDR image(Valid)', check_ldr_image(right_ldr_cap_valid), valid_num)
            #     logger.writer.add_image("F'_Before normalized left LDR image(Valid)", check_ldr_image(left_ldr_adj_valid), valid_num)
            #     logger.writer.add_image("F'_Before normalized right LDR image(Valid)", check_ldr_image(right_ldr_adj_valid), valid_num)
            #     logger.writer.add_image('G_Adjusted left LDR image(Valid)', check_ldr_image(left_ldr_adj_denom_valid), valid_num)
            #     logger.writer.add_image('G_Adjusted right LDR image(Valid)', check_ldr_image(right_ldr_adj_denom_valid), valid_num)
                
            #     # print("Save valid disparity numpy file")
            #     # np.save(f'/home/juhyung/SAEC/demo_output/valid_{valid_num}.npy', flow_valid)
            #     print("===Save LDR image to PNG file===")
            #     print(left_ldr_adj_valid.shape)
            #     left_ldr_np = left_ldr_adj_valid.squeeze().permute(1,2,0).cpu().numpy()
            #     right_ldr_np = right_ldr_adj_valid.squeeze().permute(1,2,0).cpu().numpy()
            #     left_ldr = Image.fromarray((left_ldr_np*255).astype(np.uint8))
            #     right_ldr = Image.fromarray((right_ldr_np*255).astype(np.uint8))
            #     left_ldr.save(f"/home/juhyung/SAEC/demo_output/left_ldr_{int(valid_num)}.png")
            #     right_ldr.save(f"/home/juhyung/SAEC/demo_output/right_ldr_{int(valid_num)}.png")
            #     print("===Complete save LDR image pairs===")
            #     print("===Save Valid disparity to numpy file===")
            #     np.save(f'/home/juhyung/SAEC/demo_output/valid_{int(valid_num)}.npy', flow_valid)
            #     print("===Complete save Valid disparity ===")
                

            #     model.train()
            #     model.module.RAFTStereo.freeze_bn() # 수정

            # total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break
            
            #^ save intermediate checkpoint file to display
            if total_steps%100 == 0:
                save_path = Path('checkpoints/%s_%d_epoch.pth' % (args.name, total_steps))
                logging.info(f"Saving intermediate file {save_path}")
                torch.save(model.state_dict(), save_path)

        if len(train_loader) >= 10000:
            save_path = Path('checkpoints/%d_epoch_%s.pth.gz' % (total_steps + 1, args.name))
            logging.info(f"Saving file {save_path}")
            torch.save(model.state_dict(), save_path)

    print("FINISHED TRAINING")
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='SAEC', help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=6, help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['kitti'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=100000, help="length of training schedule.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[320, 720], help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=16, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

    # Validation parameters
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during validation forward pass')

    # Architecure choices
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")

    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=None, help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[0, 0], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    Path("checkpoints").mkdir(exist_ok=True, parents=True)

    train(args)