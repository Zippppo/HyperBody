import torch
import torch.nn.functional as F
from typing import Tuple, Optional


class DiceMetric:
    """Accumulate per-class Dice scores across batches"""

    def __init__(self, num_classes: int = 70, smooth: float = 1e-5):
        """
        Args:
            num_classes: Number of segmentation classes
            smooth: Smoothing factor to avoid division by zero
        """
        self.num_classes = num_classes
        self.smooth = smooth
        self.reset()

    def reset(self):
        """Reset accumulators for new epoch"""
        self.intersection = torch.zeros(self.num_classes, dtype=torch.float64)
        self.pred_sum = torch.zeros(self.num_classes, dtype=torch.float64)
        self.target_sum = torch.zeros(self.num_classes, dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Accumulate statistics from a batch.

        Args:
            logits: (B, C, D, H, W) raw model output
            targets: (B, D, H, W) ground truth labels (int64)
        """
        # Get predictions
        preds = logits.argmax(dim=1)  # (B, D, H, W)

        # Move to CPU for accumulation
        preds = preds.cpu()
        targets = targets.cpu()

        # Accumulate per-class statistics
        for c in range(self.num_classes):
            pred_c = (preds == c)
            target_c = (targets == c)

            self.intersection[c] += (pred_c & target_c).sum().item()
            self.pred_sum[c] += pred_c.sum().item()
            self.target_sum[c] += target_c.sum().item()

    def compute(self) -> Tuple[torch.Tensor, float, Optional[torch.Tensor]]:
        """
        Compute Dice scores from accumulated statistics.

        Returns:
            dice_per_class: (num_classes,) Dice score for each class
            mean_dice: Mean Dice across classes present in targets
            valid_mask: (num_classes,) boolean mask of classes present in targets
        """
        # Compute Dice per class
        dice_per_class = (2.0 * self.intersection + self.smooth) / (
            self.pred_sum + self.target_sum + self.smooth
        )

        # Mask for classes present in targets (avoid inflating mean with absent classes)
        valid_mask = self.target_sum > 0

        # Mean Dice only over present classes
        if valid_mask.sum() > 0:
            mean_dice = dice_per_class[valid_mask].mean().item()
        else:
            mean_dice = 0.0

        return dice_per_class.float(), mean_dice, valid_mask

    def compute_per_class_dict(self, class_names: list = None) -> dict:
        """
        Compute Dice scores as a dictionary.

        Args:
            class_names: Optional list of class names

        Returns:
            Dictionary mapping class index/name to Dice score
        """
        dice_per_class, mean_dice, valid_mask = self.compute()

        result = {"mean_dice": mean_dice}

        for c in range(self.num_classes):
            if valid_mask[c]:
                key = class_names[c] if class_names else f"class_{c}"
                result[key] = dice_per_class[c].item()

        return result
