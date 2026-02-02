from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    # Data
    data_dir: str = "Dataset/voxel_data"
    split_file: str = "Dataset/dataset_split.json"
    tree_file: str = "Dataset/tree.json"
    dataset_info_file: str = "Dataset/dataset_info.json"
    num_classes: int = 70
    voxel_size: float = 4.0
    volume_size: Tuple[int, int, int] = (144, 128, 268)  # X, Y, Z

    # Model
    in_channels: int = 1
    base_channels: int = 32
    num_levels: int = 4

    # Dense Bottleneck
    growth_rate: int = 32      # channels added per layer
    dense_layers: int = 4      # number of dense layers
    bn_size: int = 4           # 1x1x1 compression factor

    # Hyperbolic
    hyp_embed_dim: int = 32
    hyp_curv: float = 1.0
    hyp_weight: float = 0.05      # Loss weight
    hyp_margin: float = 0.1       # Triplet margin
    hyp_samples_per_class: int = 64
    hyp_num_negatives: int = 8    # Negative classes per anchor
    hyp_min_radius: float = 0.1   # Shallow organ init norm
    hyp_max_radius: float = 2.0   # Deep organ init norm

    # Training
    batch_size: int = 4  # per GPU
    num_workers: int = 0
    epochs: int = 120
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # AMP
    use_amp: bool = True

    # Loss
    ce_weight: float = 0.5
    dice_weight: float = 0.5

    # LR scheduler
    lr_patience: int = 10
    lr_factor: float = 0.5

    # Checkpoint
    checkpoint_dir: str = "checkpoints"
    save_every: int = 10
    log_dir: str = "runs"

    # GPU
    gpu_ids: List[int] = field(default_factory=lambda: [0, 1])

    # Resume
    resume: str = ""
