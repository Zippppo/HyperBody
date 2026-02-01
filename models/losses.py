import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Multi-class Dice loss for 3D segmentation"""

    def __init__(self, smooth: float = 1.0):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, D, H, W) raw model output
            targets: (B, D, H, W) ground truth labels (int64)

        Returns:
            Scalar Dice loss (1 - mean_dice)
        """
        num_classes = logits.shape[1]

        # Force float32 for numerical stability in AMP
        logits = logits.float()

        # Softmax to get probabilities
        probs = F.softmax(logits, dim=1)  # (B, C, D, H, W)

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes)  # (B, D, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)

        # Flatten spatial dimensions
        probs_flat = probs.view(probs.shape[0], num_classes, -1)  # (B, C, N)
        targets_flat = targets_one_hot.view(targets_one_hot.shape[0], num_classes, -1)  # (B, C, N)

        # Compute Dice per class
        intersection = (probs_flat * targets_flat).sum(dim=2)  # (B, C)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)  # (B, C)

        dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)  # (B, C)

        # Average over classes and batch
        mean_dice = dice_per_class.mean()

        return 1.0 - mean_dice


class CombinedLoss(nn.Module):
    """Combined Cross-Entropy and Dice loss for 3D segmentation"""

    def __init__(
        self,
        num_classes: int = 70,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        class_weights: torch.Tensor = None,
        smooth: float = 1.0,
    ):
        """
        Args:
            num_classes: Number of segmentation classes
            ce_weight: Weight for cross-entropy loss
            dice_weight: Weight for Dice loss
            class_weights: (C,) tensor of per-class weights for CE loss
            smooth: Smoothing factor for Dice loss
        """
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, D, H, W) raw model output
            targets: (B, D, H, W) ground truth labels (int64)

        Returns:
            Combined loss scalar
        """
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        return self.ce_weight * ce + self.dice_weight * dice


def compute_class_weights(
    dataset,
    num_classes: int = 70,
    num_samples: int = 100,
    method: str = "inverse_sqrt",
    cache_path: str = None,
) -> torch.Tensor:
    """
    Compute per-class weights from dataset samples.

    Args:
        dataset: PyTorch Dataset with (input, label) items
        num_classes: Number of segmentation classes
        num_samples: Number of samples to use for weight computation
        method: Weight computation method ('inverse_sqrt' or 'inverse')
        cache_path: Path to cache the weights. If provided and file exists,
                    loads weights from cache instead of computing.

    Returns:
        (num_classes,) tensor of normalized class weights
    """
    import os
    import random

    # Try to load from cache
    if cache_path and os.path.exists(cache_path):
        cached = torch.load(cache_path, weights_only=True)
        # Validate cached weights match current config
        if (
            cached.get("num_classes") == num_classes
            and cached.get("num_samples") == num_samples
            and cached.get("method") == method
        ):
            return cached["weights"]

    # Sample indices
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    # Count class frequencies
    class_counts = torch.zeros(num_classes, dtype=torch.float64)

    for idx in indices:
        _, labels = dataset[idx]
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        for c in range(num_classes):
            class_counts[c] += (labels == c).sum()

    # Compute weights
    total_voxels = class_counts.sum()
    class_freq = class_counts / total_voxels

    # Avoid division by zero for absent classes
    class_freq = torch.clamp(class_freq, min=1e-8)

    if method == "inverse_sqrt":
        weights = 1.0 / torch.sqrt(class_freq)
    elif method == "inverse":
        weights = 1.0 / class_freq
    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize weights to sum to num_classes
    weights = weights / weights.sum() * num_classes
    weights = weights.float()

    # Save to cache
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        torch.save(
            {
                "weights": weights,
                "num_classes": num_classes,
                "num_samples": num_samples,
                "method": method,
            },
            cache_path,
        )

    return weights
