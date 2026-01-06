# PaSCo Migration Plan: Medical Body Dataset (Simplified)

## Task Goal

**Input**: Skin surface point cloud → **Output**: Full body voxel labels (semantic completion, 71 classes)

**Key insight**: This is a **semantic completion** task - predict internal organs from external skin surface.

## Design Decisions

- **Architecture**: Pure dense 3D CNN (no MinkowskiEngine)
- **Task**: Semantic segmentation only (no instance/panoptic)
- **Grid size**: Pad to 160×128×256 (covers max data: 129×110×249)
- **Classes**: 71 (0-70 found in data)

## Why Simplified Approach

| Original Plan         | Problem                                | Simplified Solution |
| --------------------- | -------------------------------------- | ------------------- |
| Use MinkowskiEngine   | 16 files integrated, hard to configure | Pure PyTorch 3D CNN |
| Panoptic segmentation | Complex (matcher, queries, etc.)       | Semantic only       |
| Point cloud input     | Data already voxelized                 | Direct voxel input  |
| 128×128×256 grid      | Data max is 129×110×249                | 160×128×256         |

## Files to Create

| File                              | Purpose                      |
| --------------------------------- | ---------------------------- |
| `pasco/models/dense_unet3d.py`    | Pure 3D UNet (no ME)         |
| `pasco/models/body_net.py`        | Simple semantic seg network  |
| `pasco/data/body/__init__.py`     | Package init                 |
| `pasco/data/body/params.py`       | Class names, frequencies     |
| `pasco/data/body/body_dataset.py` | Load .npz, pad to fixed size |
| `pasco/data/body/body_dm.py`      | PyTorch Lightning DataModule |
| `scripts/train_body.py`           | Training script              |
| `scripts/eval_body.py`            | Evaluation script            |

## Key Parameters

| Parameter       | Value           | Note                    |
| --------------- | --------------- | ----------------------- |
| `n_classes`     | 71              | 0-70 found in data      |
| `in_channels`   | 1               | Occupancy grid (binary) |
| `grid_size`     | (160, 128, 256) | Padded size             |
| `base_channels` | 32              | UNet base channels      |
| `lr`            | 1e-4            | Learning rate           |
| `batch_size`    | 2               | Memory dependent        |

## Architecture Overview

```
Input: Occupancy grid [B, 1, 160, 128, 256]
         |
         v
    Encoder (3D Conv + MaxPool) × 4
         |
         v
    Bottleneck
         |
         v
    Decoder (3D ConvTranspose + Skip) × 4
         |
         v
Output: Segmentation [B, 71, 160, 128, 256]
```

## Implementation Steps

1. Create `dense_unet3d.py` - Pure 3D UNet
2. Create `body_dataset.py` - Load npz, create occupancy, pad
3. Create `body_dm.py` - DataModule
4. Create `params.py` - Class definitions
5. Create `train_body.py` - Training loop
6. Create `eval_body.py` - Evaluation metrics
7. Test training

## Data Processing

Task: **Semantic Completion** - predict full internal organs from skin surface

```python
def load_sample(npz_path, target_size=(160, 128, 256)):
    data = np.load(npz_path)
    sensor_pc = data['sensor_pc']      # Skin surface point cloud (N, 3)
    voxel_labels = data['voxel_labels']  # Target organ labels (H, W, D)
    voxel_size = data['grid_voxel_size']  # [4, 4, 4]
    world_min = data['grid_world_min']

    # Voxelize point cloud to create input occupancy
    pc_voxel = ((sensor_pc - world_min) / voxel_size).astype(np.int32)
    pc_voxel = np.clip(pc_voxel, 0, np.array(voxel_labels.shape) - 1)

    # Create binary occupancy grid (skin surface)
    occupancy = np.zeros(voxel_labels.shape, dtype=np.float32)
    occupancy[pc_voxel[:, 0], pc_voxel[:, 1], pc_voxel[:, 2]] = 1.0

    # Pad to target size
    pad_h = target_size[0] - voxel_labels.shape[0]
    pad_w = target_size[1] - voxel_labels.shape[1]
    pad_d = target_size[2] - voxel_labels.shape[2]

    occupancy = np.pad(occupancy, ((0, pad_h), (0, pad_w), (0, pad_d)))
    labels = np.pad(voxel_labels, ((0, pad_h), (0, pad_w), (0, pad_d)))

    return occupancy[None], labels  # [1, H, W, D], [H, W, D]
```

**Data flow:**

```
sensor_pc (N, 3)          voxel_labels (H, W, D)
     |                           |
     v                           v
Voxelize                    Pad to 160x128x256
     |                           |
     v                           v
Input: skin surface      Target: full organ labels
[1, 160, 128, 256]       [160, 128, 256]
```

## Loss Function

Cross-entropy with class weights for imbalanced classes:

```python
criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
```

## Metrics

- mIoU (mean Intersection over Union)
- Per-class IoU
- Overall accuracy

## Memory Estimate

- Input: 160×128×256×1×4 bytes = 20 MB
- Feature maps (32ch): ~640 MB
- Total VRAM: ~2-3 GB per sample

## Dataset Statistics (verified)

- **Total samples**: 4,028
- **Shape range**: 75-129 × 50-110 × 37-249
- **Classes**: 71 (labels 0-70, all present)
- **Point cloud size**: ~100K points per sample
