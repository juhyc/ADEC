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

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

# ^ CombineModle(SAEC + RAFT) trainging code.

# ^ Raft sequence loss
def sequence_loss(flow_preds, flow_gt, valid, loss_gamma=0.9, max_flow=700):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()

    # exclude extremly large displacements
    valid = ((valid >= 0.5) & (mag < max_flow)).unsqueeze(1)
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    for i in range(n_predictions):
        assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
        flow_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

# ^ Optimizer
def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

# * Visualization flow_prediction during Training
def visualize_flow_cmap(flow_prediction, image1):
    
    # flow_prediction list case
    if isinstance(flow_prediction, list):
        flow_up_image = flow_prediction[-1][0].clone().detach()
    # flow_gt
    elif isinstance(flow_prediction, torch.Tensor) and len(flow_prediction.shape) == 4:
        # padder = InputPadder(image1.shape, divis_by = 32)
        # flow_prediction = padder.unpad(flow_prediction)
        flow_up_image = flow_prediction[0].detach()

    # Valid
    elif isinstance(flow_prediction, torch.Tensor) and len(flow_prediction.shape) == 3:
        flow_up_image = -flow_prediction[0].unsqueeze(0).detach()
    
    
    flow_up_image = -flow_up_image.cpu().numpy().squeeze()
    
    flow_up_image = (flow_up_image - flow_up_image.min()) / (flow_up_image.max() - flow_up_image.min())
    
    colored_flow_image = plt.cm.magma(flow_up_image)
    colored_flow_image = (colored_flow_image[...,:3] * 255).astype(np.uint8)
    colored_flow_image = np.transpose(colored_flow_image, (2,0,1))
    
    return colored_flow_image

# * Visualization train stereo image during Training
def visualize_img(image):
    if len(image.shape) == 3:
        temp = image.unsqueeze(0)
    
    temp = image.clone().detach()
    temp = temp[0].cpu().permute(1,2,0)
    temp = torch.clamp(temp, 0, 255)
    temp = temp.numpy().astype(np.uint8)
    temp = np.transpose(temp, (2,0,1))
    return temp


def check_ldr_image(image):
    if isinstance(image, torch.Tensor):
        temp = image.clone().detach()
        temp = temp[0].cpu().permute(1,2,0)
        temp = torch.clamp(temp*255, 0, 255)
        temp = temp.numpy().astype(np.uint8)
    return np.transpose(temp, (2,0,1))
    # plt.imshow(temp)
    # plt.show()

class Logger:

    SUM_FREQ = 100

    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir='runs')

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir='runs')

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir='runs')

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


# ^ Train
def train(args):

    model = torch.nn.DataParallel(CombineModel(args), device_ids=[0])
    # model = nn.DataParallel(RAFTStereo(args))
    print("Parameter Count: %d" % count_parameters(model))
    
    # !dataloader 수정
    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0
    logger = Logger(model, scheduler)

    # ! RAFT load_state_dict 수정
    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        raft_checkpoint = torch.load(args.restore_ckpt)
        # * downsampling = 3 인 경우
        if args.n_downsample == 3:
            del raft_checkpoint['module.update_block.mask.2.weight'], raft_checkpoint['module.update_block.mask.2.bias']
            # model.module.RAFTStereo.load_state_dict(checkpoint, strict=False)
            # model.load_state_dict(checkpoint, strict=False)
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
        
        # * Check loading checkpoint keys
        # print(f"Combine model parameter keys : {len(model.state_dict().keys())}")
        # print(f"Number of changed weight : {count}")
        # print("======Check keys======")
        
        # missing_keys = model.state_dict().keys() - combined_state_dict.keys()
        # unexpected_keys = combined_state_dict.keys() - model.state_dict().keys()
        
        # print("Missing keys:", missing_keys)
        # print("Unexpected keys:", unexpected_keys)
        # print("===============================================")
                
        logging.info(f"Done loading checkpoint")
    
    model.cuda()
    model.train()
    # ? RAFT freeze_bn 수정
    model.module.RAFTStereo.freeze_bn() # We keep BatchNorm frozen
    # model.module.freeze_bn() # We keep BatchNorm frozen

    validation_frequency = 10 #10000

    scaler = GradScaler(enabled=args.mixed_precision)

    should_keep_training = True
    global_batch_num = 0
    while should_keep_training:

        for i_batch, (_, *data_blob) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data_blob]

            assert model.training
            flow_predictions, left_ldr_cap, right_ldr_cap, left_ldr_adj_denom, right_ldr_adj_denom, exposure_dict, left_ldr_adj, right_ldr_adj = model(image1, image2, iters=args.train_iters ) 
            
            # check_ldr_image(sim_left_ldr_image)
            # check_ldr_image(sim_right_ldr_image)

            # ^ visualize during training

            # logger.write_dict(output_exp)
            
            logger.writer.add_image('B_Training left',visualize_img(image1), global_batch_num)
            logger.writer.add_image('B_Training right',visualize_img(image2), global_batch_num)
            
            logger.writer.add_image('C_Captured left LDR image', check_ldr_image(left_ldr_cap), global_batch_num)
            logger.writer.add_image('C_Captured right LDR image', check_ldr_image(right_ldr_cap), global_batch_num)
            
            logger.writer.add_image("C'_Before normalized left LDR image", check_ldr_image(left_ldr_adj), global_batch_num)
            logger.writer.add_image("C'_Before normalized right LDR iamge", check_ldr_image(right_ldr_adj), global_batch_num)
            
            logger.writer.add_image('D_Adjusted left LDR image', check_ldr_image(left_ldr_adj_denom), global_batch_num)
            logger.writer.add_image('D_Adjusted right LDR image', check_ldr_image(right_ldr_adj_denom), global_batch_num)
            
            logger.writer.add_image('A_Disparity_gt', visualize_flow_cmap(flow, image1), global_batch_num)
            logger.writer.add_image('Valid correspondences mask', valid[0].unsqueeze(0), global_batch_num)
            logger.writer.add_image('A_Disparity_prediction_c', visualize_flow_cmap(flow_predictions, image1), global_batch_num)

            assert model.training

            loss, metrics = sequence_loss(flow_predictions, flow, valid)
            logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
            logger.writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], global_batch_num)
            global_batch_num += 1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)
            
            # Todo) Validation code 수정
            if total_steps % validation_frequency == validation_frequency - 1:
                print("====Validation====")
                valid_num = (total_steps / validation_frequency) * 10 + 1
                save_path = Path('checkpoints/%d_%s.pth' % (total_steps + 1, args.name))
                logging.info(f"Saving file {save_path.absolute()}")
                torch.save(model.state_dict(), save_path)

                results, flow_valid, flow_valid_gt, left_ldr_cap_valid, right_ldr_cap_valid, left_ldr_adj_denom_valid, right_ldr_adj_denom_valid, exposure_dict_valid, left_ldr_adj_valid, right_ldr_adj_valid = validate_kitti(model.module, iters=args.valid_iters)
                
                # valid output_exp 확인용
                output_exp_valid = {"Valid " + key : value for key, value in exposure_dict_valid.items()}
                
                logger.write_dict(results)
                logger.write_dict(output_exp_valid)
                logger.writer.add_image('E_Disparity gt(Valid)', visualize_flow_cmap(flow_valid_gt, image1), valid_num)
                logger.writer.add_image('E_Disparity prediction_c(Valid)', visualize_flow_cmap(flow_valid, image1), valid_num)
                logger.writer.add_image('F_Captured left LDR image(Valid)', check_ldr_image(left_ldr_cap_valid), valid_num)
                logger.writer.add_image('F_Captured right LDR image(Valid)', check_ldr_image(right_ldr_cap_valid), valid_num)
                logger.writer.add_image("F'_Before normalized left LDR image(Valid)", check_ldr_image(left_ldr_adj_valid), valid_num)
                logger.writer.add_image("F'_Before normalized right LDR image(Valid)", check_ldr_image(right_ldr_adj_valid), valid_num)
                logger.writer.add_image('G_Adjusted left LDR image(Valid)', check_ldr_image(left_ldr_adj_denom_valid), valid_num)
                logger.writer.add_image('G_Adjusted right LDR image(Valid)', check_ldr_image(right_ldr_adj_denom_valid), valid_num)
                
                # print("Save valid disparity numpy file")
                # np.save(f'/home/juhyung/SAEC/demo_output/valid_{valid_num}.npy', flow_valid)
                print("===Save LDR image to PNG file===")
                print(left_ldr_adj_valid.shape)
                left_ldr_np = left_ldr_adj_valid.squeeze().permute(1,2,0).cpu().numpy()
                right_ldr_np = right_ldr_adj_valid.squeeze().permute(1,2,0).cpu().numpy()
                left_ldr = Image.fromarray((left_ldr_np*255).astype(np.uint8))
                right_ldr = Image.fromarray((right_ldr_np*255).astype(np.uint8))
                left_ldr.save(f"/home/juhyung/SAEC/demo_output/left_ldr_{int(valid_num)}.png")
                right_ldr.save(f"/home/juhyung/SAEC/demo_output/right_ldr_{int(valid_num)}.png")
                print("===Complete save LDR image pairs===")
                print("===Save Valid disparity to numpy file===")
                np.save(f'/home/juhyung/SAEC/demo_output/valid_{int(valid_num)}.npy', flow_valid)
                print("===Complete save Valid disparity ===")
                

                model.train()
                model.module.RAFTStereo.freeze_bn() # 수정

            total_steps += 1

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
    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='saec', help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=6, help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['sceneflow'], help="training datasets.")
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