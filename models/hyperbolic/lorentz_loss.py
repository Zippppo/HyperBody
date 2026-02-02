"""
Lorentz space ranking loss for hyperbolic embeddings.

Uses triplet margin loss: pull voxel embeddings toward their class embeddings,
push away from other class embeddings.
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from models.hyperbolic.lorentz_ops import pointwise_dist, pairwise_dist


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

        # Fully vectorized sampling: sample up to num_samples_per_class per class
        N = labels_flat.shape[0]

        # Create random priorities for sampling
        random_priorities = torch.rand(N, device=device)

        # Sort by (class, random_priority) to group by class with random order within
        # Use composite key: class * 2 + random (since random in [0,1))
        sort_key = labels_flat.float() * 2.0 + random_priorities
        sorted_indices = torch.argsort(sort_key)
        sorted_labels = labels_flat[sorted_indices]

        # Compute position within each class using cumsum trick
        # label_changes[i] = 1 if sorted_labels[i] != sorted_labels[i-1], else 0
        label_changes = torch.cat([
            torch.ones(1, device=device, dtype=torch.long),
            (sorted_labels[1:] != sorted_labels[:-1]).long()
        ])
        # cumsum gives group id, subtract to get position within group
        group_ids = torch.cumsum(label_changes, dim=0) - 1
        # Position within class: for each element, count how many before it have same label
        # Use scatter to count positions
        positions = torch.zeros(N, device=device, dtype=torch.long)
        # For each group, positions should be 0, 1, 2, ...
        # We can compute this by: position[i] = i - first_index_of_group[group_ids[i]]
        unique_groups, inverse_indices = torch.unique(group_ids, return_inverse=True)
        # Get first occurrence of each group
        first_occurrence = torch.zeros(len(unique_groups), device=device, dtype=torch.long)
        # scatter_reduce to get min index for each group
        first_occurrence.scatter_reduce_(
            0, inverse_indices,
            torch.arange(N, device=device, dtype=torch.long),
            reduce='amin', include_self=False
        )
        positions = torch.arange(N, device=device, dtype=torch.long) - first_occurrence[inverse_indices]

        # Select samples where position < num_samples_per_class
        sample_mask = positions < self.num_samples_per_class
        sampled_indices = sorted_indices[sample_mask]  # [K]
        sampled_classes = sorted_labels[sample_mask]   # [K]
        K = sampled_indices.shape[0]

        if K == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Get anchor embeddings
        anchors = voxel_flat[sampled_indices]  # [K, D]

        # Get positive embeddings (class embedding for each anchor's true class)
        positives = label_emb[sampled_classes]  # [K, D]

        # Compute positive distances
        d_pos = pointwise_dist(anchors, positives, self.curv)  # [K]

        # Vectorized negative sampling and loss computation
        # Compute all pairwise distances: anchors to all class embeddings
        all_dists = pairwise_dist(anchors, label_emb, self.curv)  # [K, num_classes]

        # Create mask for valid negatives (exclude true class for each anchor)
        # neg_mask[i, j] = True if class j is a valid negative for anchor i
        class_indices = torch.arange(num_classes, device=device)  # [num_classes]
        neg_mask = class_indices.unsqueeze(0) != sampled_classes.unsqueeze(1)  # [K, num_classes]

        # For each anchor, randomly select num_negatives from valid negatives
        # Generate random scores and mask out invalid negatives
        neg_scores = torch.rand(K, num_classes, device=device)
        neg_scores = torch.where(neg_mask, neg_scores, torch.tensor(-1.0, device=device))

        # Get top-k negative indices for each anchor
        n_neg = min(self.num_negatives, num_classes - 1)
        if n_neg <= 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        _, neg_indices = torch.topk(neg_scores, n_neg, dim=1)  # [K, n_neg]

        # Gather negative distances
        d_neg = torch.gather(all_dists, 1, neg_indices)  # [K, n_neg]

        # Compute triplet loss: max(0, margin + d_pos - d_neg)
        # d_pos: [K], d_neg: [K, n_neg]
        triplet_loss = torch.clamp(self.margin + d_pos.unsqueeze(1) - d_neg, min=0)  # [K, n_neg]

        # Average over negatives, then over anchors
        loss = triplet_loss.mean()

        return loss
