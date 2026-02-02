from dataclasses import dataclass, field, fields
from typing import List, Tuple
import yaml


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
    batch_size: int = 1  # per GPU
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
    checkpoint_dir: str = "checkpoints/test_lorentz"
    save_every: int = 10
    log_dir: str = "runs/test_lorentz"

    # GPU
    gpu_ids: List[int] = field(default_factory=lambda: [0, 1])

    # Resume
    resume: str = ""

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load config from YAML file, overriding defaults."""
        cfg = cls()
        with open(yaml_path, "r") as f:
            yaml_cfg = yaml.safe_load(f)

        if yaml_cfg is None:
            return cfg

        # Get valid field names
        valid_fields = {f.name for f in fields(cls)}

        for key, value in yaml_cfg.items():
            if key in valid_fields:
                # Handle tuple conversion for volume_size
                if key == "volume_size" and isinstance(value, list):
                    value = tuple(value)
                setattr(cfg, key, value)
            else:
                print(f"Warning: Unknown config key '{key}' in YAML, ignored.")

        return cfg

    def to_yaml(self, yaml_path: str) -> None:
        """Save current config to YAML file."""
        cfg_dict = {}
        for f in fields(self):
            value = getattr(self, f.name)
            # Convert tuple to list for YAML
            if isinstance(value, tuple):
                value = list(value)
            cfg_dict[f.name] = value

        with open(yaml_path, "w") as f:
            yaml.dump(cfg_dict, f, default_flow_style=False, sort_keys=False)
