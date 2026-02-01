"""
Prediction script for body semantic segmentation.

Generates standardized prediction results that can be evaluated by evaluate.py.

Usage:
    python scripts/body/eval/CE.py --checkpoint logs/body_unet_bs2_lr0.0001_ch16/checkpoints/best_model.ckpt --dataset_root Dataset/voxel_data --split test --output_dir model_eval_res/baseline --gpuids 0

"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from pasco.data.body import BodyDataModule, N_CLASSES
from pasco.models.body_net import BodyNet


def parse_args():
    parser = argparse.ArgumentParser(description="Generate predictions for body semantic segmentation")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Path to dataset root directory")
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test"],
                        help="Dataset split to predict on")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save predictions")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--target_size", type=int, nargs=3, default=[128, 128, 256],
                        help="Target grid size (H W D)")
    parser.add_argument("--gpuids", type=int, default=None,
                        help="GPU device ID to use (default: auto-select)")

    return parser.parse_args()


def validate_args(args):
    """Validate command line arguments."""
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")
    if args.num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {args.num_workers}")
    if any(s <= 0 for s in args.target_size):
        raise ValueError(f"target_size must be positive, got {args.target_size}")


def get_data_dir(dataset_root):
    """Get the data directory (data/ subdirectory or root)."""
    data_dir = os.path.join(dataset_root, "data")
    if os.path.exists(data_dir):
        return data_dir
    return dataset_root


def load_raw_metadata(data_dir, sample_id):
    """Load metadata from raw data file."""
    npz_path = os.path.join(data_dir, f"{sample_id}.npz")
    data = np.load(npz_path)
    return {
        "grid_world_min": data["grid_world_min"].astype(np.float32),
        "grid_world_max": data.get("grid_world_max", compute_grid_world_max(data)).astype(np.float32),
        "grid_voxel_size": data["grid_voxel_size"].astype(np.float32),
        "grid_occ_size": np.array(data["voxel_labels"].shape, dtype=np.int32),
    }


def compute_grid_world_max(data):
    """Compute grid_world_max if not present in data."""
    world_min = data["grid_world_min"]
    voxel_size = data["grid_voxel_size"]
    occ_size = np.array(data["voxel_labels"].shape)
    return world_min + occ_size * voxel_size


def main():
    """Main prediction function."""
    args = parse_args()
    validate_args(args)

    # Set device based on --gpuids
    if args.gpuids is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot use --gpuids")
        if args.gpuids >= torch.cuda.device_count():
            raise ValueError(f"GPU {args.gpuids} not available. Available GPUs: {torch.cuda.device_count()}")
        device = torch.device(f"cuda:{args.gpuids}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    try:
        model = BodyNet.load_from_checkpoint(args.checkpoint)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {args.checkpoint}\nError: {e}")

    model = model.to(device)
    model.eval()

    # Setup data
    dm = BodyDataModule(
        root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=tuple(args.target_size),
    )

    if args.split == "val":
        dm.setup("fit")
        dataloader = dm.val_dataloader()
        dataset = dm.val_ds
    else:
        dm.setup("test")
        dataloader = dm.test_dataloader()
        dataset = dm.test_ds

    # Validate dataset is not empty
    if len(dataset) == 0:
        raise ValueError(f"No samples found for split '{args.split}' in {args.dataset_root}")

    print(f"Predicting on {len(dataset)} samples")

    # Create output directory
    pred_dir = os.path.join(args.output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    # Get data directory for loading raw metadata
    data_dir = get_data_dir(args.dataset_root)

    # Collect sample IDs
    all_sample_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            occupancy = batch["occupancy"].to(device)
            sample_ids = batch["sample_id"]

            # Forward pass
            logits = model(occupancy)
            pred = logits.argmax(dim=1).cpu().numpy()

            # Save each sample
            for i, sample_id in enumerate(sample_ids):
                all_sample_ids.append(sample_id)

                # Load metadata from raw data file
                metadata = load_raw_metadata(data_dir, sample_id)

                # Save prediction with metadata
                np.savez_compressed(
                    os.path.join(pred_dir, f"{sample_id}.npz"),
                    pred_labels=pred[i].astype(np.uint8),
                    grid_world_min=metadata["grid_world_min"],
                    grid_world_max=metadata["grid_world_max"],
                    grid_voxel_size=metadata["grid_voxel_size"],
                    grid_occ_size=metadata["grid_occ_size"],
                )

    # Save meta.json
    meta = {
        "model_name": os.path.basename(args.output_dir),
        "checkpoint": args.checkpoint,
        "split": args.split,
        "target_size": args.target_size,
        "n_classes": N_CLASSES,
        "timestamp": datetime.now().isoformat(),
        "sample_ids": all_sample_ids,
    }

    meta_path = os.path.join(args.output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nPredictions saved to: {pred_dir}")
    print(f"Metadata saved to: {meta_path}")
    print(f"Total samples: {len(all_sample_ids)}")


if __name__ == "__main__":
    main()
