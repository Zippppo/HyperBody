# 3D U-Net Baseline Design for Human Body Organ Segmentation

**Date:** 2026-02-01
**Goal:** Implement a baseline 3D U-Net to predict full human body voxels and organ classes from partial body surface point clouds

## Project Overview

**Task:** Input partial body surface point cloud → Output full 3D voxel grid with 70 anatomical structure classes

**Dataset:**
- 10,779 samples total
- Split: 9,779 train / 500 val / 500 test (pre-defined in `Dataset/dataset_split.json`)
- Input: Point clouds with 24k-105k points (x,y,z coordinates in mm)
- Output: 3D voxel grids with variable sizes (X: 78-117, Y: 56-92, Z: 54-241)
- Voxel resolution: 4mm × 4mm × 4mm
- Classes: 70 anatomical structures (organs, bones, muscles)

**Hardware:** 2× NVIDIA A100 40GB GPUs

## Design Decisions

### 1. Point Cloud to Volume Conversion

**Approach:** Voxelize point cloud to binary occupancy grid
- Convert variable-sized point clouds to fixed 3D binary volumes
- Resolution: 4mm (matching target voxel resolution)
- Value: 1 where points exist, 0 elsewhere
- Simple and matches output resolution

### 2. Fixed Volume Size

**Strategy:** Pad all samples to fixed size
- Target size: **128 × 96 × 256** voxels
- Covers maximum dimensions (117 × 92 × 241) with margin
- Padding: Zero-padding for both input and output
- Preserves all original data without cropping

### 3. Architecture

**3D U-Net with 4 encoder/decoder levels + Dense Bottleneck:**
```
Encoder path:
- Level 1: 1 → 32 channels (128×96×256)
- Level 2: 32 → 64 channels (64×48×128)
- Level 3: 64 → 128 channels (32×24×64)
- Level 4: 128 → 256 channels (16×12×32)

Dense Bottleneck: 256 → 384 channels (16×12×32)
  - 4-layer DenseBlock with growth_rate=32
  - Each layer: BN→ReLU→Conv1×1×1→BN→ReLU→Conv3×3×3
  - 1×1×1 compression: bn_size=4 (reduces to 128 channels before 3×3×3)
  - Output: 256 + 4×32 = 384 channels

Decoder path:
- Level 4: 384+128 → 128 channels (32×24×64)
- Level 3: 128+128 → 64 channels (64×48×128)
- Level 2: 64+64 → 32 channels (128×96×256)
- Level 1: 32+32 → 70 channels (128×96×256)
```

**Components:**
- Convolution: 3×3×3 kernels, padding=1
- Downsampling: MaxPool3d (2×2×2)
- Upsampling: Trilinear interpolation + Conv3d
- Skip connections: Concatenation
- Activation: ReLU
- Normalization: BatchNorm3d
- Dense Bottleneck: DenseBlock with 1×1×1 compression (DenseNet-BC style)
- Output: 70 channels (one per class)

**Dense Bottleneck Parameters (configurable):**
- `growth_rate`: 32 (channels added per layer)
- `dense_layers`: 4 (number of dense layers)
- `bn_size`: 4 (1×1×1 compression factor)

### 4. Loss Function

**Modular combined loss: Weighted CE + Dice**

**Cross-Entropy Loss:**
- Weighted by inverse class frequency
- Handles class imbalance (class 0 is 73.8% of voxels)

**Dice Loss:**
- Focuses on overlap rather than per-voxel accuracy
- Works well for small organs

**Combined Loss:**
```python
total_loss = alpha * weighted_ce_loss + beta * dice_loss
```
- Default: alpha=0.5, beta=0.5
- Configurable to use CE only, Dice only, or combined

### 5. Training Configuration

**Hyperparameters:**
- Optimizer: Adam
- Learning rate: 1e-3
- LR scheduler: ReduceLROnPlateau (factor=0.5, patience=10)
- Batch size: 8-12 (depending on memory)
- Epochs: 120
- Gradient clipping: 1.0 (prevent exploding gradients)

**Multi-GPU:**
- Use `torch.nn.DataParallel` for 2× A100s
- Effective batch size: 16-24 across both GPUs

**Data Augmentation:**
- None for baseline (human anatomy constraints make augmentation tricky)
- Can be added later if needed

### 6. Evaluation Metrics

**Primary Metrics:**
- Dice score per class (all 70 classes)
- Mean Dice score (average across classes)
- Mean Dice score (weighted by class frequency)

**Tracking:**
- Validation metrics computed every epoch
- Log to TensorBoard
- Save best model based on mean Dice score

### 7. Training Infrastructure

**Checkpointing:**
- Save best model (highest validation Dice)
- Save latest model (for resuming)
- Save every 10 epochs as backup
- Checkpoint includes: model state, optimizer state, epoch, metrics

**Logging:**
- TensorBoard for loss curves and metrics
- Console output with progress bars (tqdm)
- Log file with detailed training info

**Resume Capability:**
- Can resume from any checkpoint
- Restores model, optimizer, epoch, and scheduler state

## Project Structure

```
HyperBody/
├── models/
│   ├── unet3d.py          # 3D U-Net architecture
│   ├── dense_block.py     # Dense Bottleneck (DenseLayer, DenseBlock)
│   └── losses.py          # Modular loss functions (CE, Dice, Combined)
├── data/
│   ├── dataset.py         # PyTorch Dataset class
│   ├── voxelizer.py       # Point cloud to voxel conversion
│   └── transforms.py      # Future augmentations (placeholder)
├── utils/
│   ├── metrics.py         # Dice score calculation
│   └── checkpoint.py      # Model saving/loading utilities
├── train.py               # Main training script
├── config.py              # Hyperparameters and configuration
├── requirements.txt       # Python dependencies
└── docs/
    └── plans/             # Design and implementation plans
```

## Implementation Pipeline

1. **Data Loading:**
   - Load .npz files with sensor_pc and voxel_labels
   - Apply train/val/test split from dataset_split.json

2. **Preprocessing:**
   - Voxelize point cloud to binary occupancy grid (128×96×256)
   - Pad voxel_labels to same size (128×96×256)
   - Convert to PyTorch tensors

3. **Training Loop:**
   - Forward pass through 3D U-Net
   - Compute combined loss (CE + Dice)
   - Backward pass and optimizer step
   - Log metrics to TensorBoard

4. **Validation:**
   - Compute Dice scores per class
   - Save best model checkpoint
   - Update learning rate scheduler

5. **Checkpointing:**
   - Save model state every epoch
   - Keep best and latest checkpoints

## Success Criteria

**Baseline is successful if:**
- Training completes without OOM errors
- Training loss decreases consistently
- Validation Dice score > 0.3 (showing the model learns something meaningful)
- Major organs (liver, lungs, heart) have Dice > 0.5
- Pipeline is modular and easy to extend

## Future Improvements (Post-Baseline)

- Data augmentation (careful with anatomical constraints)
- Attention mechanisms (e.g., attention gates in U-Net)
- Multi-scale inputs
- Post-processing (CRF, morphological operations)
- Ensemble methods
- Different architectures (V-Net, nnU-Net, Transformer-based)

## Dependencies

```
torch>=2.0.0
numpy>=1.24.0
tensorboard>=2.12.0
tqdm>=4.65.0
scikit-learn>=1.2.0
```

## Notes

- Conda environment: `pasco`
- No augmentation for baseline (human body constraints)
- Modular loss design allows easy experimentation
- Focus on getting end-to-end pipeline working first
- Optimization and tuning come after baseline validation
