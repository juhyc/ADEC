## Dataset directory
```
datasets/
└── CARLA/
    └── training/
        └── Experiment1/
            ├── ground_truth_depth_left
            ├── ground_truth_depth_right
            ├── ground_truth_disparity_left
            ├── ground_truth_disparity_right
            ├── hdr_left
            ├── hdr_right
            ├── calibration_file_extrinsic_left.npy
            ├── calibration_file_extrinsic_right.npy
            └── calibration_file_intrinsic.npy
```

## Model parameter directory
```
models/
└── 5000_disp_gru_eth3d.pth
└── 5000_disp_gru_sceneflow.pth
└── 10000_disp_gru_eth3d.pth
└── 10000_disp_gru_sceneflow.pth

```


## Training code

```
python train_disp_recon_dual.py --restore_ckpt models/raftstereo-eth3d.pth --num_steps 10000 --batch_size 4
```

## Test code
Inference code for real dataset.
```
python test_sequence_real.py --restore_ckpt models/5000_disp_gru_sceneflow_blur_gmflow.pth --batch_size 1
```

Inference code for synthetic dataset.
```
python test_sequence_carla.py --restore_ckpt models/5000_disp_gru_sceneflow_blur_gmflow.pth --batch_size 1
```

## Model Parameter link
https://drive.google.com/drive/folders/1T_WXItkuY6egEVAQgqO7DuJ2AtrPErWM?usp=sharing

