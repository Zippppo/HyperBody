"""
PyTorch Lightning module for body semantic segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from pasco.models.dense_unet3d import DenseUNet3D, DenseUNet3DLight


class BodyNet(pl.LightningModule):
    """
    PyTorch Lightning module for body semantic segmentation.

    This network takes a binary occupancy grid (skin surface) and predicts
    full body voxel labels (71 classes for different organs).

    Args:
        n_classes: Number of output classes (71)
        in_channels: Number of input channels (1 for occupancy)
        base_channels: Base channel count for UNet
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        ignore_index: Label index to ignore (default 255)
        use_light_model: Use lighter 3-level UNet instead of 4-level
    """

    def __init__(
        self,
        n_classes=71,
        in_channels=1,
        base_channels=32,
        lr=1e-4,
        weight_decay=0.0,
        ignore_index=255,
        use_light_model=False,
        warmup_epochs=5,
        max_epochs=100,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.n_classes = n_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.ignore_index = ignore_index
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        # Build model
        if use_light_model:
            self.model = DenseUNet3DLight(
                in_channels=in_channels,
                n_classes=n_classes,
                base_channels=base_channels,
            )
        else:
            self.model = DenseUNet3D(
                in_channels=in_channels,
                n_classes=n_classes,
                base_channels=base_channels,
            )

        # Metrics storage
        self.val_iou_sum = None
        self.val_iou_count = None

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)

    def compute_loss(self, logits, labels):
        """Compute cross-entropy loss."""
        criterion = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index,
        )
        return criterion(logits, labels)

    def compute_iou(self, pred, target, n_classes):
        """
        Compute per-class IoU.

        Args:
            pred: [B, H, W, D] predicted class indices
            target: [B, H, W, D] ground truth class indices

        Returns:
            iou_per_class: [n_classes] IoU for each class
            valid_mask: [n_classes] which classes have GT samples
        """
        iou_per_class = torch.zeros(n_classes, device=pred.device)
        valid_mask = torch.zeros(n_classes, dtype=torch.bool, device=pred.device)

        for cls in range(n_classes):
            pred_cls = pred == cls
            target_cls = target == cls

            # Skip if no GT for this class
            if target_cls.sum() == 0:
                continue

            valid_mask[cls] = True
            intersection = (pred_cls & target_cls).sum().float()
            union = (pred_cls | target_cls).sum().float()

            if union > 0:
                iou_per_class[cls] = intersection / union

        return iou_per_class, valid_mask

    def training_step(self, batch, batch_idx):
        """Training step."""
        occupancy = batch["occupancy"]  # [B, 1, H, W, D]
        labels = batch["labels"]        # [B, H, W, D]

        # Forward pass
        logits = self(occupancy)  # [B, n_classes, H, W, D]

        # Compute loss
        loss = self.compute_loss(logits, labels)

        # Compute accuracy
        pred = logits.argmax(dim=1)
        valid_mask = labels != self.ignore_index
        accuracy = (pred[valid_mask] == labels[valid_mask]).float().mean()

        # Log metrics
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/accuracy", accuracy, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        occupancy = batch["occupancy"]
        labels = batch["labels"]

        # Forward pass
        logits = self(occupancy)

        # Compute loss
        loss = self.compute_loss(logits, labels)

        # Compute predictions
        pred = logits.argmax(dim=1)

        # Compute IoU
        iou_per_class, valid_mask = self.compute_iou(pred, labels, self.n_classes)

        # Accumulate IoU
        if self.val_iou_sum is None:
            self.val_iou_sum = torch.zeros(self.n_classes, device=self.device)
            self.val_iou_count = torch.zeros(self.n_classes, device=self.device)

        self.val_iou_sum += iou_per_class
        self.val_iou_count += valid_mask.float()

        # Log loss
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        return {"loss": loss, "iou": iou_per_class, "valid": valid_mask}

    def on_validation_epoch_end(self):
        """Compute and log validation metrics at epoch end."""
        if self.val_iou_sum is not None:
            # Compute mean IoU per class
            valid = self.val_iou_count > 0
            iou_per_class = torch.zeros_like(self.val_iou_sum)
            iou_per_class[valid] = self.val_iou_sum[valid] / self.val_iou_count[valid]

            # Mean IoU across valid classes
            miou = iou_per_class[valid].mean()

            self.log("val/mIoU", miou, prog_bar=True, sync_dist=True)

            # Reset accumulators
            self.val_iou_sum = None
            self.val_iou_count = None

    def test_step(self, batch, batch_idx):
        """Test step - same as validation."""
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        """Same as validation epoch end."""
        self.on_validation_epoch_end()

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Warmup + Cosine annealing scheduler
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return epoch / self.warmup_epochs
            else:
                progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


if __name__ == "__main__":
    # Test the module
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = BodyNet(n_classes=71, base_channels=32).to(device)

    # Print info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test forward pass
    batch = {
        "occupancy": torch.randn(2, 1, 64, 64, 64).to(device),
        "labels": torch.randint(0, 71, (2, 64, 64, 64)).to(device),
    }

    with torch.no_grad():
        loss = model.training_step(batch, 0)
    print(f"Loss: {loss.item():.4f}")
