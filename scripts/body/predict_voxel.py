"""
Prediction script for generating voxel labels following LOOC protocol.

This script loads a trained model and generates predictions in the standardized
npz format for evaluation.

Usage:
    python scripts/body/predict_voxel.py --checkpoint <model.ckpt> --split test --output predictions/

Example (using default split files):
    python scripts/body/predict_voxel.py \
        --checkpoint logs/body_unet_bs2_lr0.0001_ch16/checkpoints/epoch=098-val_mIoU=0.0000.ckpt \
        --dataset_root voxel-output/merged_data \
        --split test \
        --output predictions/pasco-20260102/

Example (using dataset_split.json):
    python scripts/body/predict_voxel.py \
        --checkpoint logs/body_unet_bs2_lr0.0001_ch16/checkpoints/epoch=098-val_mIoU=0.0000.ckpt \
        --dataset_root voxel-output/merged_data \
        --split test \
        --split_json dataset_split.json \
        --output predictions/pasco-20260102/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pasco.data.body import BodyDataModule, N_CLASSES
from pasco.data.body.body_dataset import BodyDatasetFromList, collate_fn
from pasco.models.body_net import BodyNet
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Generate voxel predictions for LOOC evaluation")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Path to dataset root directory")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Dataset split to predict on")
    parser.add_argument("--split_json", type=str, default=None,
                        help="Path to dataset_split.json file (if provided, overrides default split files)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for predictions")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for prediction")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--target_size", type=int, nargs=3, default=[160, 160, 256],
                        help="Target grid size (H W D)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to run predictions on")

    return parser.parse_args()


def load_split_samples(split_json_path, split):
    """
    Load sample IDs from dataset_split.json.

    Args:
        split_json_path: Path to dataset_split.json
        split: Split name ('train', 'val', or 'test')

    Returns:
        List of sample IDs
    """
    with open(split_json_path, 'r') as f:
        data = json.load(f)

    if 'splits' not in data or split not in data['splits']:
        raise ValueError(f"Split '{split}' not found in {split_json_path}")

    sample_ids = data['splits'][split]
    print(f"Loaded {len(sample_ids)} samples from {split_json_path} (split: {split})")

    return sample_ids


def load_original_metadata(dataset_root, sample_id):
    """
    Load original metadata from the ground truth file.

    Args:
        dataset_root: Path to dataset root
        sample_id: Sample identifier

    Returns:
        dict with grid metadata (grid_world_min, grid_world_max, etc.)
    """
    # Determine data directory
    data_dir = os.path.join(dataset_root, "data")
    if not os.path.exists(data_dir):
        data_dir = dataset_root

    npz_path = os.path.join(data_dir, f"{sample_id}.npz")
    data = np.load(npz_path)

    return {
        "grid_world_min": data["grid_world_min"],
        "grid_world_max": data["grid_world_max"],
        "grid_voxel_size": data["grid_voxel_size"],
        "grid_occ_size": data["grid_occ_size"] if "grid_occ_size" in data else data["voxel_labels"].shape,
    }


def unpad_prediction(pred, original_shape):
    """
    Remove padding from prediction to match original shape.

    Args:
        pred: [H, W, D] padded prediction
        original_shape: (H, W, D) original shape

    Returns:
        [H_orig, W_orig, D_orig] unpadded prediction
    """
    H, W, D = original_shape
    return pred[:H, :W, :D]


def save_prediction(pred, metadata, output_path):
    """
    Save prediction in LOOC evaluation format.

    Args:
        pred: [H, W, D] predicted voxel labels
        metadata: dict with grid metadata
        output_path: Path to save npz file
    """
    np.savez_compressed(
        output_path,
        voxel_labels=pred.astype(np.uint8),
        grid_world_min=metadata["grid_world_min"],
        grid_world_max=metadata["grid_world_max"],
        grid_voxel_size=metadata["grid_voxel_size"],
        grid_occ_size=metadata["grid_occ_size"],
    )


def main():
    args = parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model = BodyNet.load_from_checkpoint(args.checkpoint)
    model = model.to(device)
    model.eval()

    # Setup data
    print(f"\nSetting up data from: {args.dataset_root}")

    # Use custom split from JSON if provided
    if args.split_json:
        print(f"Using custom split from: {args.split_json}")
        sample_ids = load_split_samples(args.split_json, args.split)

        # Create dataset directly with sample IDs
        dataset = BodyDatasetFromList(
            root=args.dataset_root,
            sample_ids=sample_ids,
            target_size=tuple(args.target_size),
            data_aug=False,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    else:
        # Use default split files (train.txt, val.txt, test.txt)
        dm = BodyDataModule(
            root=args.dataset_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            target_size=tuple(args.target_size),
            data_aug=False,  # Never augment during prediction
        )

        # Get appropriate dataloader
        if args.split == "train":
            dm.setup("fit")
            dataloader = dm.train_dataloader()
            dataset = dm.train_ds
        elif args.split == "val":
            dm.setup("fit")
            dataloader = dm.val_dataloader()
            dataset = dm.val_ds
        else:  # test
            dm.setup("test")
            dataloader = dm.test_dataloader()
            dataset = dm.test_ds

    print(f"Predicting on {len(dataset)} samples from '{args.split}' split")

    # Generate predictions
    print("\n" + "=" * 60)
    print("GENERATING PREDICTIONS")
    print("=" * 60)

    num_saved = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            occupancy = batch["occupancy"].to(device)
            sample_ids = batch["sample_id"]

            # Forward pass
            logits = model(occupancy)  # [B, n_classes, H, W, D]
            pred = logits.argmax(dim=1).cpu().numpy()  # [B, H, W, D]

            # Save each prediction
            for i, sample_id in enumerate(sample_ids):
                # Load original metadata
                try:
                    metadata = load_original_metadata(args.dataset_root, sample_id)
                except Exception as e:
                    print(f"\nWarning: Could not load metadata for {sample_id}: {e}")
                    continue

                # Get original shape from metadata
                original_shape = tuple(metadata["grid_occ_size"])

                # Remove padding
                pred_unpadded = unpad_prediction(pred[i], original_shape)

                # Save prediction
                output_path = output_dir / f"{sample_id}.npz"
                save_prediction(pred_unpadded, metadata, output_path)

                num_saved += 1

    print("\n" + "=" * 60)
    print("PREDICTION COMPLETE")
    print("=" * 60)
    print(f"Saved {num_saved} predictions to: {output_dir}")
    print(f"\nNext step: Run evaluation")
    print(f"  python scripts/body/evaluate_predictions.py --pred_dir {output_dir} --split {args.split}")


if __name__ == "__main__":
    main()
