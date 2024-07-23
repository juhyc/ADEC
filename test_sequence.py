import torch
from core.combine_model3 import CombineModel_wo_net
from core.stereo_datasets3 import CARLASequenceDataset, fetch_dataloader

DEVICE = 'cuda'

def test_sequence(model, dataloader):
    model.eval()

    initial_exp_high = torch.tensor([1.3], dtype=torch.float32).to(DEVICE)
    initial_exp_low = torch.tensor([1.3], dtype=torch.float32).to(DEVICE)

    for batch in dataloader:
        left_hdr, right_hdr, left_next_hdr, right_next_hdr, flow, valid = batch
        left_hdr, right_hdr = left_hdr.to(DEVICE), right_hdr.to(DEVICE)
        left_next_hdr, right_next_hdr = left_next_hdr.to(DEVICE), right_next_hdr.to(DEVICE)

        with torch.no_grad():
            fused_disparity, disparity_exph, disparity_expl, original_img_list, captured_rand_img_list, captured_adj_img_list, mask_list, mask_mul_list, disparity_list, shifted_exp_f1, shifted_exp_f2 = model(
                left_hdr, right_hdr, left_next_hdr, right_next_hdr, initial_exp_high, initial_exp_low, test_mode=True)

        # Update the initial exposure values for the next frame pair
        initial_exp_high = shifted_exp_f1
        initial_exp_low = shifted_exp_f2

        # Add visualize step
        # Add save file step

# Example usage of test_sequence function
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Testing script")
    parser.add_argument('--test_datasets', type=str, nargs='+', default=['carla'])
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    model = CombineModel_wo_net(args).to(DEVICE)
    dataset = CARLASequenceDataset(root='datasets/CARLA', image_set='test')
    dataloader = fetch_dataloader(dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False, num_workers=4, drop_last=False)

    test_sequence(model, dataloader)
