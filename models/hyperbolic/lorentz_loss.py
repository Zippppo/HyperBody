"""
Lorentz space ranking loss for hyperbolic embeddings.

Uses triplet margin loss: pull voxel embeddings toward their class embeddings,
push away from other class embeddings.
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from models.hyperbolic.lorentz_ops import pointwise_dist


class LorentzRankingLoss(nn.Module):
    """
    Triplet ranking loss in Lorentz hyperbolic space.

    For each sampled voxel:
    - anchor = voxel embedding
    - positive = class embedding of voxel's true class
    - negatives = M embeddings of random other classes

    Loss = mean(max(0, margin + d(anchor, positive) - d(anchor, negative)))
    """

    def __init__(
        self,
        margin: float = 0.1,
        curv: float = 1.0,
        num_samples_per_class: int = 64,
        num_negatives: int = 8,
    ):
        """
        Args:
            margin: Triplet margin
            curv: Curvature (for distance computation)
            num_samples_per_class: Max voxels to sample per class
            num_negatives: Number of negative classes per anchor
        """
        super().__init__()
        self.margin = margin
        self.curv = curv
        self.num_samples_per_class = num_samples_per_class
        self.num_negatives = num_negatives

    def forward(
        self,
        voxel_emb: Tensor,
        labels: Tensor,
        label_emb: Tensor,
    ) -> Tensor:
        """
        Compute ranking loss.

        Args:
            voxel_emb: [B, D, H, W, Z] Lorentz voxel embeddings
            labels: [B, H, W, Z] ground truth labels (int64)
            label_emb: [num_classes, D] Lorentz class embeddings

        Returns:
            Scalar loss
        """
        # Force float32 for numerical stability (AMP compatibility)
        voxel_emb = voxel_emb.float()
        label_emb = label_emb.float()

        device = voxel_emb.device
        B, D, H, W, Z = voxel_emb.shape
        num_classes = label_emb.shape[0]

        # Reshape: [B, D, H, W, Z] -> [N, D] where N = B*H*W*Z
        voxel_flat = voxel_emb.permute(0, 2, 3, 4, 1).reshape(-1, D)  # [N, D]
        labels_flat = labels.reshape(-1)  # [N]

        # Find unique classes in this batch
        unique_classes = torch.unique(labels_flat)

        # Sample voxels per class
        sampled_indices = []
        sampled_classes = []

        for cls in unique_classes:
            cls_mask = labels_flat == cls
            cls_indices = torch.where(cls_mask)[0]

            # Sample up to num_samples_per_class
            n_samples = min(len(cls_indices), self.num_samples_per_class)
            if n_samples > 0:
                perm = torch.randperm(len(cls_indices), device=device)[:n_samples]
                sampled = cls_indices[perm]
                sampled_indices.append(sampled)
                sampled_classes.append(cls.expand(n_samples))

        if len(sampled_indices) == 0:
            # No valid samples
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Concatenate all samples
        sampled_indices = torch.cat(sampled_indices)  # [K]
        sampled_classes = torch.cat(sampled_classes)  # [K]
        K = len(sampled_indices)

        # Get anchor embeddings
        anchors = voxel_flat[sampled_indices]  # [K, D]

        # Get positive embeddings (class embedding for each anchor's true class)
        positives = label_emb[sampled_classes]  # [K, D]

        # Compute positive distances
        d_pos = pointwise_dist(anchors, positives, self.curv)  # [K]

        # Sample negative classes for each anchor
        # For each anchor, sample num_negatives classes different from its true class
        all_classes = torch.arange(num_classes, device=device)

        total_loss = torch.tensor(0.0, device=device)
        valid_count = 0

        for i in range(K):
            true_cls = sampled_classes[i]
            # Available negative classes (all except true class)
            neg_mask = all_classes != true_cls
            neg_classes = all_classes[neg_mask]

            if len(neg_classes) == 0:
                continue

            # Sample num_negatives from available
            n_neg = min(self.num_negatives, len(neg_classes))
            perm = torch.randperm(len(neg_classes), device=device)[:n_neg]
            neg_class_indices = neg_classes[perm]  # [n_neg]

            # Get negative embeddings
            negatives = label_emb[neg_class_indices]  # [n_neg, D]

            # Expand anchor for pointwise distance
            anchor_expanded = anchors[i:i+1].expand(n_neg, -1)  # [n_neg, D]

            # Compute negative distances
            d_neg = pointwise_dist(anchor_expanded, negatives, self.curv)  # [n_neg]

            # Triplet loss: max(0, margin + d_pos - d_neg)
            # d_pos[i] is scalar, d_neg is [n_neg]
            triplet_loss = torch.clamp(self.margin + d_pos[i] - d_neg, min=0)  # [n_neg]

            # Average over negatives
            total_loss = total_loss + triplet_loss.mean()
            valid_count += 1

        if valid_count == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return total_loss / valid_count
