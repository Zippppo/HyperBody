
  | File                            | Purpose                                                |
  |---------------------------------|--------------------------------------------------------|
  | pasco/models/dense_unet3d.py    | Pure 3D UNet with 4 encoder/decoder levels             |
  | pasco/models/body_net.py        | PyTorch Lightning module with training/val logic       |
  | pasco/data/body/__init__.py     | Package init                                           |
  | pasco/data/body/params.py       | 71 classes, frequencies, weights utilities             |
  | pasco/data/body/body_dataset.py | Dataset: loads .npz, voxelizes PC, pads to 160×128×256 |
  | pasco/data/body/body_dm.py      | DataModule for train/val/test splits                   |
  | scripts/train_body.py           | Training script with CLI args                          |
  | scripts/eval_body.py            | Evaluation script with mIoU metrics                    |

   Architecture

  Input: [B, 1, 160, 128, 256] occupancy grid (skin surface)
      ↓
  Encoder: 4 levels (32→64→128→256→512 channels)
      ↓
  Bottleneck: 3D convolutions
      ↓
  Decoder: 4 levels with skip connections
      ↓
  Output: [B, 71, 160, 128, 256] class logits

  Usage

  Training:
  python scripts/train_body.py --dataset_root E:\CODE\PaSCo-main\PaSCo\voxel-output\merged_data --batch_size 2 --lr 1e-4 --max_epochs 100 --use_class_weights

  Evaluation:
  python scripts/eval_body.py \
      --checkpoint logs/body_unet/checkpoints/last.ckpt \
      --dataset_root /path/to/body_data \
      --split test

  Data Format Expected

  dataset_root/
  ├── train.txt          # Sample IDs for training
  ├── val.txt            # Sample IDs for validation
  ├── test.txt           # Sample IDs for testing
  └── data/
      ├── sample1.npz    # Each .npz contains:
      ├── sample2.npz    #   - sensor_pc: (N, 3) skin surface points
      └── ...            #   - voxel_labels: (H, W, D) target labels
                         #   - grid_voxel_size: [4, 4, 4]
                         #   - grid_world_min: (3,)