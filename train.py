import argparse
import os
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from data.dataset import HyperBodyDataset
from models.unet3d import UNet3D
from models.losses import CombinedLoss, compute_class_weights
from utils.metrics import DiceMetric
from utils.checkpoint import save_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train 3D U-Net for body segmentation")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint path to resume from")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--gpuids", type=str, default=None, help="GPU IDs (e.g., '0,1')")
    return parser.parse_args()


def setup_logging(log_dir: str):
    """Setup console and file logging."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip):
    """Train for one epoch.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits = model(inputs)
        loss = criterion(logits, targets)

        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, metric, device):
    """Validate and compute metrics.

    Returns:
        Tuple of (val_loss, dice_per_class, mean_dice).
    """
    model.eval()
    metric.reset()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="  Val  ", leave=False)
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = criterion(logits, targets)

        total_loss += loss.item()
        num_batches += 1

        metric.update(logits, targets)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    val_loss = total_loss / max(num_batches, 1)
    dice_per_class, mean_dice, _ = metric.compute()

    return val_loss, dice_per_class, mean_dice


def main():
    args = parse_args()
    cfg = Config()

    # Override config with CLI args
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.gpuids is not None:
        cfg.gpu_ids = [int(x) for x in args.gpuids.split(",")]
    if args.resume:
        cfg.resume = args.resume

    # Setup logging
    logger = setup_logging(cfg.log_dir)
    logger.info("=" * 60)
    logger.info("Training 3D U-Net with Dense Bottleneck")
    logger.info("=" * 60)
    logger.info(f"Config: {cfg}")

    # Device
    device = torch.device(f"cuda:{cfg.gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}, GPU IDs: {cfg.gpu_ids}")

    # Datasets
    logger.info("Loading datasets...")
    train_dataset = HyperBodyDataset(cfg.data_dir, cfg.split_file, "train", cfg.volume_size)
    val_dataset = HyperBodyDataset(cfg.data_dir, cfg.split_file, "val", cfg.volume_size)
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # Compute class weights
    logger.info("Computing class weights from 100 samples...")
    class_weights = compute_class_weights(train_dataset, cfg.num_classes, num_samples=100)
    class_weights = class_weights.to(device)
    logger.info(f"Class weights range: [{class_weights.min():.4f}, {class_weights.max():.4f}]")

    # Model
    logger.info("Creating model...")
    model = UNet3D(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        base_channels=cfg.base_channels,
        growth_rate=cfg.growth_rate,
        dense_layers=cfg.dense_layers,
        bn_size=cfg.bn_size,
    )

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,} ({num_params / 1e6:.1f}M)")

    model = model.to(device)

    # Multi-GPU
    if len(cfg.gpu_ids) > 1 and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=cfg.gpu_ids)
        logger.info(f"Using DataParallel on {len(cfg.gpu_ids)} GPUs")

    # Loss, optimizer, scheduler
    criterion = CombinedLoss(
        num_classes=cfg.num_classes,
        ce_weight=cfg.ce_weight,
        dice_weight=cfg.dice_weight,
        class_weights=class_weights,
    )
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg.lr_factor, patience=cfg.lr_patience
    )

    # Metrics
    metric = DiceMetric(num_classes=cfg.num_classes)

    # Resume from checkpoint
    start_epoch = 0
    best_dice = 0.0
    if cfg.resume:
        logger.info(f"Resuming from checkpoint: {cfg.resume}")
        start_epoch, best_dice = load_checkpoint(
            cfg.resume, model, optimizer, scheduler, device=device
        )
        logger.info(f"Resumed at epoch {start_epoch}, best_dice={best_dice:.4f}")

    # TensorBoard writer
    writer = SummaryWriter(log_dir=cfg.log_dir)

    # Training loop
    logger.info(f"Starting training from epoch {start_epoch} to {cfg.epochs}")
    logger.info("-" * 60)

    for epoch in range(start_epoch, cfg.epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch [{epoch + 1}/{cfg.epochs}]  LR: {current_lr:.6f}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, cfg.grad_clip
        )

        # Validate
        val_loss, dice_per_class, mean_dice = validate(
            model, val_loader, criterion, metric, device
        )

        # Update scheduler
        scheduler.step(mean_dice)

        # Log to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Dice/mean", mean_dice, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        # Log key organ Dice scores (if present)
        # These indices depend on the dataset label mapping
        key_organs = {
            "Dice/class_00_background": 0,
            "Dice/class_01": 1,
            "Dice/class_02": 2,
            "Dice/class_03": 3,
            "Dice/class_04": 4,
        }
        for name, idx in key_organs.items():
            writer.add_scalar(name, dice_per_class[idx].item(), epoch)

        # Check if best model
        is_best = mean_dice > best_dice
        if is_best:
            best_dice = mean_dice

        # Log epoch summary
        logger.info(
            f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Mean Dice: {mean_dice:.4f} | Best Dice: {best_dice:.4f}"
            f"{' *' if is_best else ''}"
        )

        # Save checkpoint
        # Get model state (unwrap DataParallel if needed)
        model_state = (
            model.module.state_dict()
            if isinstance(model, nn.DataParallel)
            else model.state_dict()
        )

        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_dice": best_dice,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mean_dice": mean_dice,
        }

        # Save latest (every epoch)
        save_checkpoint(checkpoint_state, cfg.checkpoint_dir, "latest.pth", is_best=is_best)

        # Save periodic checkpoint
        if (epoch + 1) % cfg.save_every == 0:
            save_checkpoint(checkpoint_state, cfg.checkpoint_dir, f"epoch_{epoch + 1}.pth")
            logger.info(f"  Saved periodic checkpoint: epoch_{epoch + 1}.pth")

        logger.info("-" * 60)

    writer.close()
    logger.info(f"Training complete. Best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
