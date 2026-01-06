"""
Evaluation script for body semantic segmentation.

Usage:
    python scripts/eval_body.py --checkpoint /path/to/checkpoint.ckpt --dataset_root /path/to/data

Example:
    python scripts/eval_body.py \
        --checkpoint logs/body_unet/checkpoints/last.ckpt \
        --dataset_root ./data/body \
        --split test
"""

import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pasco.data.body import BodyDataModule, N_CLASSES, body_class_names
from pasco.models.body_net import BodyNet


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate body semantic segmentation")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Path to dataset root directory")
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test"],
                        help="Evaluation split")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--target_size", type=int, nargs=3, default=[160, 160, 256],
                        help="Target grid size (H W D)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save predictions (optional)")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Save prediction results to files")

    return parser.parse_args()


def compute_metrics(pred, target, n_classes, ignore_index=255):
    """
    Compute evaluation metrics.

    Args:
        pred: (H, W, D) predicted labels
        target: (H, W, D) ground truth labels
        n_classes: Number of classes
        ignore_index: Label to ignore

    Returns:
        dict with metrics
    """
    # Flatten
    pred = pred.flatten()
    target = target.flatten()

    # Remove ignored pixels
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    # Per-class IoU
    iou_per_class = np.zeros(n_classes)
    class_counts = np.zeros(n_classes)

    for cls in range(n_classes):
        pred_cls = pred == cls
        target_cls = target == cls

        if target_cls.sum() == 0:
            continue

        class_counts[cls] = target_cls.sum()
        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()

        if union > 0:
            iou_per_class[cls] = intersection / union

    # Overall accuracy
    accuracy = (pred == target).mean()

    return {
        "iou_per_class": iou_per_class,
        "class_counts": class_counts,
        "accuracy": accuracy,
    }


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = BodyNet.load_from_checkpoint(args.checkpoint)
    model = model.to(device)
    model.eval()

    # Setup data
    dm = BodyDataModule(
        root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=tuple(args.target_size),
        data_aug=False,
    )

    if args.split == "val":
        dm.setup("fit")
        dataloader = dm.val_dataloader()
        dataset = dm.val_ds
    else:
        dm.setup("test")
        dataloader = dm.test_dataloader()
        dataset = dm.test_ds

    print(f"Evaluating on {len(dataset)} samples")

    # Create output directory
    if args.save_predictions and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Evaluate
    all_iou = np.zeros(N_CLASSES)
    all_counts = np.zeros(N_CLASSES)
    all_accuracy = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            occupancy = batch["occupancy"].to(device)
            labels = batch["labels"].numpy()
            sample_ids = batch["sample_id"]

            # Forward pass
            logits = model(occupancy)
            pred = logits.argmax(dim=1).cpu().numpy()

            # Compute metrics per sample
            for i in range(len(sample_ids)):
                metrics = compute_metrics(pred[i], labels[i], N_CLASSES)
                all_iou += metrics["iou_per_class"] * (metrics["class_counts"] > 0)
                all_counts += (metrics["class_counts"] > 0)
                all_accuracy.append(metrics["accuracy"])

                # Save predictions
                if args.save_predictions and args.output_dir:
                    sample_id = sample_ids[i]
                    np.save(
                        os.path.join(args.output_dir, f"{sample_id}_pred.npy"),
                        pred[i].astype(np.uint8)
                    )

    # Compute final metrics
    valid_classes = all_counts > 0
    iou_per_class = np.zeros(N_CLASSES)
    iou_per_class[valid_classes] = all_iou[valid_classes] / all_counts[valid_classes]

    mean_iou = iou_per_class[valid_classes].mean()
    mean_accuracy = np.mean(all_accuracy)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Split: {args.split}")
    print(f"Samples: {len(dataset)}")
    print(f"Mean IoU: {mean_iou * 100:.2f}%")
    print(f"Mean Accuracy: {mean_accuracy * 100:.2f}%")
    print(f"Valid classes: {valid_classes.sum()} / {N_CLASSES}")

    # Per-class IoU
    print("\n" + "-" * 60)
    print("Per-class IoU:")
    print("-" * 60)

    # Sort by IoU
    sorted_idx = np.argsort(iou_per_class)[::-1]
    for idx in sorted_idx:
        if all_counts[idx] > 0:
            name = body_class_names[idx] if idx < len(body_class_names) else f"class_{idx}"
            print(f"  {idx:3d} {name:20s}: {iou_per_class[idx] * 100:6.2f}%")

    # Save results
    if args.output_dir:
        results = {
            "mean_iou": mean_iou,
            "mean_accuracy": mean_accuracy,
            "iou_per_class": iou_per_class,
            "class_counts": all_counts,
        }
        np.savez(os.path.join(args.output_dir, "results.npz"), **results)
        print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
