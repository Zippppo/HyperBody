import argparse
import os
import logging
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from data.dataset import HyperBodyDataset
from models.unet3d import UNet3D
from models.losses import CombinedLoss, compute_class_weights
from utils.metrics import DiceMetric
from utils.checkpoint import save_checkpoint, load_checkpoint


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def is_distributed() -> bool:
    """Check if running in distributed mode."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get current process rank (0 if not distributed)."""
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    """Get total number of processes (1 if not distributed)."""
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def setup_distributed(gpu_ids=None):
    """Initialize distributed training if launched with torchrun.

    If gpu_ids is provided AND we are NOT already in a torchrun env,
    set CUDA_VISIBLE_DEVICES so only specified GPUs are visible.

    Returns:
        local_rank: Local rank of this process (0 if not distributed)
    """
    if gpu_ids is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

    if "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
    return 0


def cleanup_distributed():
    """Clean up distributed training."""
    if is_distributed():
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="Train 3D U-Net for body segmentation")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint path to resume from")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--gpuids", type=str, default=None, help="GPU IDs (e.g., '0,1')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def setup_logging(log_dir: str, is_main: bool = True):
    """Setup console and file logging.

    Args:
        log_dir: Directory to store log files
        is_main: If True, log to file and console. If False, use NullHandler.
    """
    os.makedirs(log_dir, exist_ok=True)

    handlers = []
    if is_main:
        log_file = os.path.join(log_dir, "training.log")
        handlers.append(logging.FileHandler(log_file))
        handlers.append(logging.StreamHandler())
    else:
        handlers.append(logging.NullHandler())

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )
    return logging.getLogger(__name__)


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip, epoch=0, scaler=None):
    """Train for one epoch.

    Args:
        model: The model to train
        loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        grad_clip: Gradient clipping value (0 to disable)
        epoch: Current epoch number (used for DistributedSampler shuffling)
        scaler: Optional GradScaler for AMP training (None to disable AMP)

    Returns:
        Average training loss for the epoch.
    """
    model.train()

    # Set epoch for DistributedSampler (ensures proper shuffling)
    if hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(epoch)

    total_loss = 0.0
    num_batches = 0

    # Only show progress bar on main process
    pbar = tqdm(loader, desc="  Train", leave=False, disable=not is_main_process())
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast(device_type='cuda'):
                logits = model(inputs)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
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

    # Only show progress bar on main process
    pbar = tqdm(loader, desc="  Val  ", leave=False, disable=not is_main_process())
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = criterion(logits, targets)

        total_loss += loss.item()
        num_batches += 1

        metric.update(logits, targets)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    # Sync metrics across processes before compute
    if is_distributed():
        metric.sync_across_processes()

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

    # Setup distributed training (passes gpu_ids to set CUDA_VISIBLE_DEVICES)
    local_rank = setup_distributed(gpu_ids=cfg.gpu_ids)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Setup logging (only main process logs)
    logger = setup_logging(cfg.log_dir, is_main=is_main_process())
    logger.info("=" * 60)
    logger.info("Training 3D U-Net with Dense Bottleneck")
    logger.info("=" * 60)
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Config: {cfg}")

    # Log distributed info
    if is_distributed():
        logger.info(f"Distributed training: rank {get_rank()}/{get_world_size()}, local_rank {local_rank}")
    logger.info(f"Device: {device}")

    # Datasets
    logger.info("Loading datasets...")
    train_dataset = HyperBodyDataset(cfg.data_dir, cfg.split_file, "train", cfg.volume_size)
    val_dataset = HyperBodyDataset(cfg.data_dir, cfg.split_file, "val", cfg.volume_size)
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create samplers for distributed training
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed() else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed() else None

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),  # Only shuffle if not using sampler
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # Compute class weights (with caching)
    class_weights_cache = os.path.join(cfg.checkpoint_dir, "class_weights.pt")
    if os.path.exists(class_weights_cache):
        logger.info(f"Loading cached class weights from {class_weights_cache}")
    else:
        logger.info("Computing class weights from 100 samples (will be cached)...")
    class_weights = compute_class_weights(
        train_dataset, cfg.num_classes, num_samples=100, cache_path=class_weights_cache
    )
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

    # Wrap model with DDP if distributed
    if is_distributed():
        model = DDP(model, device_ids=[local_rank])
        logger.info(f"Using DistributedDataParallel on {get_world_size()} GPUs")

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

    # AMP GradScaler (only if use_amp is enabled)
    scaler = GradScaler() if cfg.use_amp else None
    logger.info(f"AMP enabled: {cfg.use_amp}")

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

    # TensorBoard writer (only main process)
    writer = SummaryWriter(log_dir=cfg.log_dir) if is_main_process() else None

    # Training loop
    logger.info(f"Starting training from epoch {start_epoch} to {cfg.epochs}")
    logger.info("-" * 60)

    for epoch in range(start_epoch, cfg.epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch [{epoch + 1}/{cfg.epochs}]  LR: {current_lr:.6f}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, cfg.grad_clip, epoch=epoch, scaler=scaler
        )

        # Validate
        val_loss, dice_per_class, mean_dice = validate(
            model, val_loader, criterion, metric, device
        )

        # Update scheduler
        scheduler.step(mean_dice)

        # Log to TensorBoard (only main process)
        if writer:
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

        # Save checkpoint (only main process)
        if is_main_process():
            # Get model state (unwrap DDP if needed)
            model_state = (
                model.module.state_dict()
                if hasattr(model, 'module')  # Works for both DDP and DataParallel
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

    if writer:
        writer.close()
    cleanup_distributed()
    logger.info(f"Training complete. Best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
