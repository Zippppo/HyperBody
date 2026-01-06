"""
Training script for body semantic segmentation.

Usage:
    python scripts/train_body.py --dataset_root /path/to/data

Example:
    python scripts/body/train_body.py --dataset_root Dataset/voxel_data --batch_size 2 --lr 1e-4 --max_epochs 100 --exp_name body_unet --n_gpus 2 --precision 16 --base_channels 16 --log
"""

import argparse
import os
import sys
import json
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger
import torch

# Add project root directory to path (scripts/body -> scripts -> root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pasco.data.body import BodyDataModule, N_CLASSES
from pasco.data.body.params import compute_class_frequencies, compute_class_weights
from pasco.models.body_net import BodyNet


class TrainingInfoLogger(Callback):
    """Callback to log training information to JSON file."""

    def __init__(self, log_path):
        super().__init__()
        self.log_path = Path(log_path)
        self.training_history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_mIoU": [],
            "learning_rate": [],
            "epoch": [],
        }
        self.best_metrics = {
            "best_val_mIoU": 0.0,
            "best_epoch": 0,
        }

    def on_train_epoch_end(self, trainer, pl_module):
        """Log training metrics at epoch end."""
        # Get current epoch
        epoch = trainer.current_epoch

        # Get logged metrics
        metrics = trainer.callback_metrics

        # Append to history
        self.training_history["epoch"].append(epoch)
        self.training_history["train_loss"].append(
            float(metrics.get("train/loss", 0.0))
        )
        self.training_history["train_accuracy"].append(
            float(metrics.get("train/accuracy", 0.0))
        )

        # Get learning rate
        if trainer.optimizers:
            lr = trainer.optimizers[0].param_groups[0]['lr']
            self.training_history["learning_rate"].append(lr)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log validation metrics at epoch end."""
        metrics = trainer.callback_metrics

        # Append validation metrics
        self.training_history["val_loss"].append(
            float(metrics.get("val/loss", 0.0))
        )
        val_miou = float(metrics.get("val/mIoU", 0.0))
        self.training_history["val_mIoU"].append(val_miou)

        # Update best metrics
        if val_miou > self.best_metrics["best_val_mIoU"]:
            self.best_metrics["best_val_mIoU"] = val_miou
            self.best_metrics["best_epoch"] = trainer.current_epoch

        # Save to file after each validation
        self._save_log()

    def on_train_end(self, trainer, pl_module):
        """Final save when training ends."""
        self._save_log()
        print(f"\nTraining log saved to: {self.log_path}")

    def _save_log(self):
        """Save training history to JSON file."""
        log_data = {
            "training_history": self.training_history,
            "best_metrics": self.best_metrics,
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(self.log_path, 'w') as f:
            json.dump(log_data, f, indent=2)


def save_training_config(args, exp_dir, class_weights=None, model_params=None):
    """
    Save training configuration to JSON file.

    Args:
        args: Training arguments
        exp_dir: Experiment directory
        class_weights: Optional class weights array
        model_params: Optional model parameter count
    """
    config = {
        "experiment": {
            "name": args.exp_name,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "log_dir": args.log_dir,
        },
        "data": {
            "dataset_root": args.dataset_root,
            "target_size": args.target_size,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "data_aug": args.data_aug,
        },
        "model": {
            "n_classes": N_CLASSES,
            "base_channels": args.base_channels,
            "use_light_model": args.use_light_model,
            "total_parameters": model_params,
        },
        "training": {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "max_epochs": args.max_epochs,
            "warmup_epochs": args.warmup_epochs,
            "n_gpus": args.n_gpus,
            "precision": args.precision,
            "seed": args.seed,
        },
        "loss": {
            "use_class_weights": args.use_class_weights,
            "weight_alpha": args.weight_alpha if args.use_class_weights else None,
        },
        "system": {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        }
    }

    # Add class weights statistics if provided
    if class_weights is not None:
        config["loss"]["class_weights_stats"] = {
            "min": float(class_weights.min()),
            "max": float(class_weights.max()),
            "mean": float(class_weights.mean()),
            "std": float(class_weights.std()),
        }

    # Save config
    config_path = Path(exp_dir) / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Training configuration saved to: {config_path}")
    return config_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train body semantic segmentation")

    # Data
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Path to dataset root directory")
    parser.add_argument("--target_size", type=int, nargs=3, default=[160, 160, 256],
                        help="Target grid size (H W D)")

    # Model
    parser.add_argument("--base_channels", type=int, default=32,
                        help="Base channel count for UNet")
    parser.add_argument("--use_light_model", action="store_true",
                        help="Use lighter 3-level UNet")

    # Training
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay")
    parser.add_argument("--max_epochs", type=int, default=100,
                        help="Maximum training epochs")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Warmup epochs")

    # Loss
    parser.add_argument("--use_class_weights", action="store_true",
                        help="Use class-weighted loss")
    parser.add_argument("--weight_alpha", type=float, default=0.5,
                        help="Class weight exponent")

    # Data augmentation
    parser.add_argument("--data_aug", action="store_true",
                        help="Enable data augmentation")

    # System
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--n_gpus", type=int, default=1,
                        help="Number of GPUs")
    parser.add_argument("--precision", type=str, default="32",
                        choices=["16", "32", "bf16"],
                        help="Training precision")

    # Logging
    parser.add_argument("--exp_name", type=str, default="body_unet",
                        help="Experiment name")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Log directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    pl.seed_everything(args.seed)

    # Create experiment name
    exp_name = (
        f"{args.exp_name}_"
        f"bs{args.batch_size}_"
        f"lr{args.lr}_"
        f"ch{args.base_channels}"
    )
    if args.use_class_weights:
        exp_name += f"_cw{args.weight_alpha}"
    if args.data_aug:
        exp_name += "_aug"
    if args.use_light_model:
        exp_name += "_light"

    # Create experiment directory
    exp_dir = Path(args.log_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Experiment: {exp_name}")
    print(f"Log directory: {exp_dir}")
    print("=" * 60)

    # Setup data module
    print("\nSetting up data...")
    dm = BodyDataModule(
        root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=tuple(args.target_size),
        data_aug=args.data_aug,
    )
    dm.setup("fit")

    print(f"Train samples: {len(dm.train_ds)}")
    print(f"Val samples: {len(dm.val_ds)}")

    # Compute class weights if requested
    class_weights = None
    if args.use_class_weights:
        print("\nComputing class frequencies...")
        frequencies = compute_class_frequencies(args.dataset_root, split="train")
        class_weights = compute_class_weights(frequencies, alpha=args.weight_alpha)
        print(f"Class weights - min: {class_weights.min():.3f}, max: {class_weights.max():.3f}, mean: {class_weights.mean():.3f}")

    # Create model
    print("\nCreating model...")
    model = BodyNet(
        n_classes=N_CLASSES,
        in_channels=1,
        base_channels=args.base_channels,
        lr=args.lr,
        weight_decay=args.weight_decay,
        class_weights=class_weights,
        use_light_model=args.use_light_model,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
    )

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Save training configuration
    print("\nSaving training configuration...")
    save_training_config(
        args=args,
        exp_dir=exp_dir,
        class_weights=class_weights,
        model_params=total_params,
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=exp_dir / "checkpoints",
        filename="best_model",  # Only save best model with fixed name
        save_top_k=1,  # Only keep the best model
        monitor="val/mIoU",
        mode="max",
        save_last=False,  # Don't save last checkpoint
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Training info logger
    training_logger = TrainingInfoLogger(log_path=exp_dir / "training_log.json")

    # Setup tensorboard logger
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=exp_name,
    )

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if args.n_gpus > 0 else "cpu",
        devices=args.n_gpus if args.n_gpus > 0 else 1,
        strategy="ddp" if args.n_gpus > 1 else "auto",
        precision=args.precision,
        callbacks=[checkpoint_callback, lr_monitor, training_logger],
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=1.0,
    )

    # Print training info
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    # Train
    trainer.fit(model, dm, ckpt_path=args.resume)

    # Print summary
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best val mIoU: {checkpoint_callback.best_model_score:.4f}")
    print(f"Training log: {exp_dir / 'training_log.json'}")
    print(f"Config file: {exp_dir / 'training_config.json'}")
    print("=" * 60)

    # Save final summary
    summary = {
        "status": "completed",
        "completion_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_checkpoint": str(checkpoint_callback.best_model_path),
        "best_val_mIoU": float(checkpoint_callback.best_model_score),
        "total_epochs": trainer.current_epoch + 1,
    }

    summary_path = exp_dir / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nTraining summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
