"""
LOOC (Learning Organ Occupancy from Calibrated skin surface) Evaluation Metrics.

This module provides utilities for computing evaluation metrics based on
organ-level Axis-Aligned Bounding Boxes (AABBs).

Metrics:
    - CD (Center Distance): Distance between AABB centers in mm
    - IoU (Intersection over Union): Overlap ratio of AABBs
    - ESF (Encompassment Scaling Factor): Scaling factor for predicted AABB to encompass GT
"""

import numpy as np
from typing import Dict, Tuple, Optional


class AABB:
    """
    Axis-Aligned Bounding Box.

    Attributes:
        min_coords: [3] minimum corner (x, y, z) in mm
        max_coords: [3] maximum corner (x, y, z) in mm
    """

    def __init__(self, min_coords: np.ndarray, max_coords: np.ndarray):
        """
        Initialize AABB.

        Args:
            min_coords: [3] minimum corner coordinates
            max_coords: [3] maximum corner coordinates
        """
        self.min_coords = np.array(min_coords, dtype=np.float32)
        self.max_coords = np.array(max_coords, dtype=np.float32)

    @property
    def center(self) -> np.ndarray:
        """Get AABB center."""
        return (self.min_coords + self.max_coords) / 2.0

    @property
    def size(self) -> np.ndarray:
        """Get AABB size (width, height, depth)."""
        return self.max_coords - self.min_coords

    @property
    def volume(self) -> float:
        """Get AABB volume."""
        size = self.size
        return float(size[0] * size[1] * size[2])

    def intersection(self, other: "AABB") -> Optional["AABB"]:
        """
        Compute intersection AABB with another AABB.

        Args:
            other: Another AABB

        Returns:
            Intersection AABB, or None if no intersection
        """
        min_coords = np.maximum(self.min_coords, other.min_coords)
        max_coords = np.minimum(self.max_coords, other.max_coords)

        # Check if valid intersection
        if np.any(min_coords >= max_coords):
            return None

        return AABB(min_coords, max_coords)

    def union_volume(self, other: "AABB") -> float:
        """
        Compute union volume with another AABB.

        Args:
            other: Another AABB

        Returns:
            Union volume
        """
        intersection = self.intersection(other)
        if intersection is None:
            return self.volume + other.volume
        return self.volume + other.volume - intersection.volume

    def iou(self, other: "AABB") -> float:
        """
        Compute IoU (Intersection over Union) with another AABB.

        Args:
            other: Another AABB

        Returns:
            IoU value in [0, 1]
        """
        intersection = self.intersection(other)
        if intersection is None:
            return 0.0

        intersection_vol = intersection.volume
        union_vol = self.union_volume(other)

        if union_vol == 0:
            return 0.0

        return intersection_vol / union_vol

    def __repr__(self):
        return f"AABB(min={self.min_coords}, max={self.max_coords}, center={self.center})"


def extract_organ_aabb(voxel_labels: np.ndarray, organ_class: int,
                       grid_world_min: np.ndarray, grid_voxel_size: np.ndarray) -> Optional[AABB]:
    """
    Extract AABB for a specific organ from voxel labels.

    Args:
        voxel_labels: [H, W, D] voxel-wise organ labels
        organ_class: Target organ class ID
        grid_world_min: [3] grid origin in world coordinates (mm)
        grid_voxel_size: [3] voxel size (mm)

    Returns:
        AABB in world coordinates (mm), or None if organ not present
    """
    # Find voxels belonging to this organ
    organ_mask = voxel_labels == organ_class

    if not organ_mask.any():
        return None

    # Get voxel indices
    voxel_indices = np.array(np.where(organ_mask))  # [3, N]

    # Compute AABB in voxel space
    min_voxel = voxel_indices.min(axis=1)  # [3]
    max_voxel = voxel_indices.max(axis=1)  # [3]

    # Convert to world coordinates (mm)
    # Note: Add 1 to max_voxel to include the full extent of the last voxel
    min_world = grid_world_min + min_voxel * grid_voxel_size
    max_world = grid_world_min + (max_voxel + 1) * grid_voxel_size

    return AABB(min_world, max_world)


def compute_center_distance(aabb1: AABB, aabb2: AABB) -> float:
    """
    Compute Center Distance (CD) between two AABBs.

    Args:
        aabb1: First AABB
        aabb2: Second AABB

    Returns:
        Euclidean distance between centers (mm)
    """
    return float(np.linalg.norm(aabb1.center - aabb2.center))


def compute_encompassment_scaling_factor(pred_aabb: AABB, gt_aabb: AABB) -> float:
    """
    Compute Encompassment Scaling Factor (ESF).

    ESF measures how much the predicted AABB needs to be scaled (from its center)
    to fully encompass the ground truth AABB.

    Args:
        pred_aabb: Predicted AABB
        gt_aabb: Ground truth AABB

    Returns:
        Scaling factor (>=1.0, where 1.0 means perfect encompassment)
    """
    # Compute relative positions of GT corners with respect to pred center
    pred_center = pred_aabb.center
    pred_half_size = pred_aabb.size / 2.0

    # GT corners relative to pred center
    gt_min_rel = gt_aabb.min_coords - pred_center
    gt_max_rel = gt_aabb.max_coords - pred_center

    # Find max scaling factor needed along each axis
    # We need to check all corners of GT AABB
    scales = []

    for dim in range(3):
        if pred_half_size[dim] == 0:
            # Degenerate predicted AABB (zero size in this dimension)
            if gt_aabb.size[dim] > 0:
                scales.append(float('inf'))
            else:
                scales.append(1.0)
        else:
            # Max distance from pred center to GT along this axis
            max_dist = max(abs(gt_min_rel[dim]), abs(gt_max_rel[dim]))
            scale = max_dist / pred_half_size[dim]
            scales.append(scale)

    # Overall scaling factor is the maximum across all dimensions
    esf = max(scales)

    # ESF should be at least 1.0
    return max(1.0, float(esf))


def evaluate_organ(pred_labels: np.ndarray, gt_labels: np.ndarray,
                   organ_class: int, grid_world_min: np.ndarray,
                   grid_voxel_size: np.ndarray) -> Optional[Dict[str, float]]:
    """
    Evaluate predictions for a single organ.

    Args:
        pred_labels: [H, W, D] predicted voxel labels
        gt_labels: [H, W, D] ground truth voxel labels
        organ_class: Organ class ID to evaluate
        grid_world_min: [3] grid origin (mm)
        grid_voxel_size: [3] voxel size (mm)

    Returns:
        dict with metrics {cd, iou, esf}, or None if organ not present in GT
    """
    # Extract AABBs
    gt_aabb = extract_organ_aabb(gt_labels, organ_class, grid_world_min, grid_voxel_size)

    if gt_aabb is None:
        # Organ not present in ground truth
        return None

    pred_aabb = extract_organ_aabb(pred_labels, organ_class, grid_world_min, grid_voxel_size)

    if pred_aabb is None:
        # Organ not predicted (treat as worst case)
        return {
            "cd": float('inf'),
            "iou": 0.0,
            "esf": float('inf'),
        }

    # Compute metrics
    cd = compute_center_distance(pred_aabb, gt_aabb)
    iou = pred_aabb.iou(gt_aabb)
    esf = compute_encompassment_scaling_factor(pred_aabb, gt_aabb)

    return {
        "cd": cd,
        "iou": iou,
        "esf": esf,
    }


def evaluate_sample(pred_labels: np.ndarray, gt_labels: np.ndarray,
                    grid_world_min: np.ndarray, grid_voxel_size: np.ndarray,
                    n_classes: int = 71) -> Dict[int, Dict[str, float]]:
    """
    Evaluate predictions for all organs in a sample.

    Args:
        pred_labels: [H, W, D] predicted voxel labels
        gt_labels: [H, W, D] ground truth voxel labels
        grid_world_min: [3] grid origin (mm)
        grid_voxel_size: [3] voxel size (mm)
        n_classes: Number of organ classes

    Returns:
        dict mapping organ_class -> metrics {cd, iou, esf}
        Only includes organs present in GT
    """
    results = {}

    for organ_class in range(1, n_classes):  # Skip background (class 0)
        metrics = evaluate_organ(
            pred_labels, gt_labels, organ_class,
            grid_world_min, grid_voxel_size
        )

        if metrics is not None:
            results[organ_class] = metrics

    return results


def aggregate_metrics(all_results: Dict[int, list]) -> Dict[int, Dict[str, float]]:
    """
    Aggregate metrics across multiple samples.

    Args:
        all_results: dict mapping organ_class -> list of metric dicts

    Returns:
        dict mapping organ_class -> mean metrics {cd, iou, esf}
    """
    aggregated = {}

    for organ_class, metrics_list in all_results.items():
        if not metrics_list:
            continue

        # Compute means
        cd_values = [m["cd"] for m in metrics_list if not np.isinf(m["cd"])]
        iou_values = [m["iou"] for m in metrics_list]
        esf_values = [m["esf"] for m in metrics_list if not np.isinf(m["esf"])]

        aggregated[organ_class] = {
            "cd_mean": np.mean(cd_values) if cd_values else float('inf'),
            "cd_median": np.median(cd_values) if cd_values else float('inf'),
            "iou_mean": np.mean(iou_values),
            "esf_mean": np.mean(esf_values) if esf_values else float('inf'),
            "esf_median": np.median(esf_values) if esf_values else float('inf'),
            "count": len(metrics_list),
        }

    return aggregated


if __name__ == "__main__":
    # Test AABB operations
    aabb1 = AABB([0, 0, 0], [10, 10, 10])
    aabb2 = AABB([5, 5, 5], [15, 15, 15])

    print(f"AABB1: {aabb1}")
    print(f"AABB2: {aabb2}")
    print(f"Intersection: {aabb1.intersection(aabb2)}")
    print(f"IoU: {aabb1.iou(aabb2):.4f}")
    print(f"Center Distance: {compute_center_distance(aabb1, aabb2):.2f} mm")
    print(f"ESF: {compute_encompassment_scaling_factor(aabb2, aabb1):.4f}")

    # Test with voxel data
    print("\n" + "=" * 60)
    print("Testing with synthetic voxel data")
    print("=" * 60)

    # Create synthetic voxel labels
    voxel_labels_gt = np.zeros((50, 50, 50), dtype=np.uint8)
    voxel_labels_gt[10:20, 10:20, 10:20] = 1  # Organ 1

    voxel_labels_pred = np.zeros((50, 50, 50), dtype=np.uint8)
    voxel_labels_pred[12:22, 12:22, 12:22] = 1  # Organ 1 (slightly shifted)

    grid_world_min = np.array([0.0, 0.0, 0.0])
    grid_voxel_size = np.array([4.0, 4.0, 4.0])

    # Evaluate
    metrics = evaluate_organ(
        voxel_labels_pred, voxel_labels_gt, 1,
        grid_world_min, grid_voxel_size
    )

    print(f"\nOrgan 1 metrics:")
    print(f"  CD: {metrics['cd']:.2f} mm")
    print(f"  IoU: {metrics['iou']:.4f}")
    print(f"  ESF: {metrics['esf']:.4f}")
