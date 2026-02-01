"""
Body dataset for semantic segmentation.

Loads .npz files containing:
  - sensor_pc: (N, 3) skin surface point cloud
  - voxel_labels: (H, W, D) target voxel labels (0-70)
  - grid_voxel_size: voxel size (e.g., [4, 4, 4])
  - grid_world_min: (3,) world coordinate minimum
  - grid_world_max: (3,) world coordinate maximum

Creates a binary occupancy grid from the point cloud and pads to a fixed target size.
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class BodyDataset(Dataset):
    """
    Body semantic segmentation dataset.

    Args:
        root: Dataset root directory
        split: "train", "val", or "test"
        target_size: Target grid size (H, W, D) for padding
    """

    def __init__(self, root, split="train", target_size=(160, 160, 256)):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.target_size = target_size

        # Read split file
        split_file = self.root / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, 'r') as f:
            self.sample_ids = [line.strip() for line in f if line.strip()]

        # Determine data directory
        self.data_dir = self.root / "data"
        if not self.data_dir.exists():
            self.data_dir = self.root

        print(f"[BodyDataset] split={split}, samples={len(self.sample_ids)}, "
              f"target_size={target_size}")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        npz_path = self.data_dir / f"{sample_id}.npz"

        data = np.load(npz_path)
        sensor_pc = data["sensor_pc"]          # (N, 3) point cloud
        voxel_labels = data["voxel_labels"]    # (H, W, D) labels
        voxel_size = data["grid_voxel_size"]   # voxel size
        world_min = data["grid_world_min"]     # world min coordinates

        # Voxelize point cloud to create input occupancy grid
        pc_voxel = ((sensor_pc - world_min) / voxel_size).astype(np.int32)
        pc_voxel = np.clip(pc_voxel, 0, np.array(voxel_labels.shape) - 1)

        # Create binary occupancy grid (skin surface)
        occupancy = np.zeros(voxel_labels.shape, dtype=np.float32)
        occupancy[pc_voxel[:, 0], pc_voxel[:, 1], pc_voxel[:, 2]] = 1.0

        # Pad to target size
        occupancy, voxel_labels = self._pad_to_target(occupancy, voxel_labels)

        # Convert to tensors
        occupancy = torch.from_numpy(occupancy).unsqueeze(0)     # [1, H, W, D]
        labels = torch.from_numpy(voxel_labels.astype(np.int64))  # [H, W, D]

        return {
            "occupancy": occupancy,
            "labels": labels,
            "sample_id": sample_id,
        }

    def _pad_to_target(self, occupancy, labels):
        """Pad arrays to target size with zeros (occupancy) and 255 (labels)."""
        H, W, D = occupancy.shape
        tH, tW, tD = self.target_size

        pad_h = max(0, tH - H)
        pad_w = max(0, tW - W)
        pad_d = max(0, tD - D)

        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            occupancy = np.pad(occupancy, ((0, pad_h), (0, pad_w), (0, pad_d)),
                               mode='constant', constant_values=0)
            labels = np.pad(labels, ((0, pad_h), (0, pad_w), (0, pad_d)),
                            mode='constant', constant_values=255)

        return occupancy, labels


class BodyDatasetFromList(Dataset):
    """
    Body dataset initialized from a list of sample IDs (for prediction).

    Args:
        root: Dataset root directory
        sample_ids: List of sample identifiers
        target_size: Target grid size (H, W, D)
    """

    def __init__(self, root, sample_ids, target_size=(160, 160, 256)):
        super().__init__()
        self.root = Path(root)
        self.sample_ids = sample_ids
        self.target_size = target_size

        # Determine data directory
        self.data_dir = self.root / "data"
        if not self.data_dir.exists():
            self.data_dir = self.root

        print(f"[BodyDatasetFromList] samples={len(self.sample_ids)}, "
              f"target_size={target_size}")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        npz_path = self.data_dir / f"{sample_id}.npz"

        data = np.load(npz_path)
        sensor_pc = data["sensor_pc"]
        voxel_labels = data["voxel_labels"]
        voxel_size = data["grid_voxel_size"]
        world_min = data["grid_world_min"]

        # Voxelize point cloud to create input occupancy grid
        pc_voxel = ((sensor_pc - world_min) / voxel_size).astype(np.int32)
        pc_voxel = np.clip(pc_voxel, 0, np.array(voxel_labels.shape) - 1)

        # Create binary occupancy grid
        occupancy = np.zeros(voxel_labels.shape, dtype=np.float32)
        occupancy[pc_voxel[:, 0], pc_voxel[:, 1], pc_voxel[:, 2]] = 1.0

        # Pad to target size
        H, W, D = occupancy.shape
        tH, tW, tD = self.target_size

        pad_h = max(0, tH - H)
        pad_w = max(0, tW - W)
        pad_d = max(0, tD - D)

        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            occupancy = np.pad(occupancy, ((0, pad_h), (0, pad_w), (0, pad_d)),
                               mode='constant', constant_values=0)
            voxel_labels = np.pad(voxel_labels, ((0, pad_h), (0, pad_w), (0, pad_d)),
                                  mode='constant', constant_values=255)

        # Convert to tensors
        occupancy = torch.from_numpy(occupancy).unsqueeze(0)
        labels = torch.from_numpy(voxel_labels.astype(np.int64))

        return {
            "occupancy": occupancy,
            "labels": labels,
            "sample_id": sample_id,
        }


def collate_fn(batch):
    """
    Custom collate function for body dataset.

    Stacks occupancy grids and labels into batches.
    """
    occupancy = torch.stack([item["occupancy"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    sample_ids = [item["sample_id"] for item in batch]

    return {
        "occupancy": occupancy,
        "labels": labels,
        "sample_id": sample_ids,
    }
