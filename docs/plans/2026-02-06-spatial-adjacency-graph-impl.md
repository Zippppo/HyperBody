# Spatial Adjacency Graph — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the anatomical tree distance matrix with spatial adjacency edges so that physically neighboring organs (even across body systems) are treated as hard negatives in curriculum learning.

**Architecture:** Precompute an asymmetric contact matrix from training GT labels using vectorized MaxPool3d dilation + einsum. Fuse it with the existing tree distance via per-pair min formula. The resulting `D_final` matrix replaces `tree_dist_matrix` in `LorentzTreeRankingLoss` — no changes to the Loss class itself.

**Tech Stack:** PyTorch (F.max_pool3d, torch.einsum), numpy, plotly (visualizations), pytest (TDD)

**Design Doc:** `docs/plans/2026-02-06-spatial-adjacency-graph-design.md`

---

## Codebase Context (READ THIS FIRST)

### Key Files

| File | Role |
|------|------|
| `data/organ_hierarchy.py` | `compute_tree_distance_matrix()` — current symmetric [70,70] tree distance |
| `models/hyperbolic/lorentz_loss.py:226-450` | `LorentzTreeRankingLoss` — uses `tree_dist_matrix` buffer for sampling weights |
| `train.py:381-406` | Creates loss: `if cfg.hyp_distance_mode == "tree"` → `LorentzTreeRankingLoss(tree_dist_matrix=...)` |
| `config.py:44` | `hyp_distance_mode: str = "hyperbolic"` — config field |
| `configs/LR-Curriculum-TreeDistance.yaml` | Reference config for tree mode |
| `Dataset/tree.json` | Anatomical hierarchy (nested JSON dict) |
| `Dataset/dataset_info.json` | `class_names` list (70 classes, indices 0-69) |
| `Dataset/dataset_split.json` | Train/val/test splits (train: 9779 samples) |
| `Dataset/voxel_data/*.npz` | GT labels in `data["voxel_labels"]`, int64 |
| `data/dataset.py` | `HyperBodyDataset.__getitem__` → returns `(inp: [1,144,128,268], lbl: [144,128,268])` |
| `tests/hyperbolic/test_tree_distance_matrix.py` | Existing tree distance tests (13 tests) — pattern to follow |
| `tests/hyperbolic/test_lorentz_loss.py` | Existing loss tests (30+ tests) — integration pattern |

### How LorentzTreeRankingLoss Uses the Distance Matrix

```python
# lorentz_loss.py:417 — indexes by anchor class (row = anchor u, col = negative v)
tree_dists = self.tree_dist_matrix[sampled_classes]  # [K, num_classes]

# lorentz_loss.py:432 — sampling weight: smaller distance = higher probability
neg_weights = torch.exp(-tree_dists / temperature)
```

The matrix is only read by row index. Asymmetric `D_final[u,v] != D_final[v,u]` is automatically handled: when u is anchor, row u is used; when v is anchor, row v is used.

### Volume Dimensions

`volume_size = (144, 128, 268)` → labels shape per sample: `[144, 128, 268]` int64, values 0-69.

---

## Task 1: `compute_contact_matrix()` — Core Function + Tests

**Files:**
- Create: `data/spatial_adjacency.py`
- Create: `tests/hyperbolic/test_spatial_adjacency.py`

### Step 1: Write the failing test for single-sample contact matrix

The core function processes one label volume and returns per-sample overlap and volume tensors.

```python
# tests/hyperbolic/test_spatial_adjacency.py
"""
Tests for spatial adjacency: contact matrix computation and distance fusion.
"""
import pytest
import torch


class TestComputeSingleSampleOverlap:
    """Test _compute_single_sample_overlap() on synthetic label volumes."""

    def test_two_adjacent_cubes_have_contact(self):
        """
        Two organs as adjacent cubes. After dilation, they overlap.

        Layout (1D cross-section, 20 voxels):
            [0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2, 0,0,0,0,0]
            organ 1: voxels 5-9, organ 2: voxels 10-14
            They touch at boundary (9,10). Dilation radius=2 should create overlap.
        """
        from data.spatial_adjacency import _compute_single_sample_overlap

        num_classes = 3  # 0=background, 1=organA, 2=organB
        labels = torch.zeros(20, 20, 20, dtype=torch.long)
        labels[5:10, 5:15, 5:15] = 1   # organ 1: 5x10x10 = 500 voxels
        labels[10:15, 5:15, 5:15] = 2  # organ 2: 5x10x10 = 500 voxels

        overlap, volume = _compute_single_sample_overlap(
            labels, num_classes=num_classes, dilation_radius=2
        )

        assert overlap.shape == (num_classes, num_classes)
        assert volume.shape == (num_classes,)

        # Organ 1 and 2 are adjacent → after dilation, overlap > 0
        assert overlap[1, 2].item() > 0, "organ1 dilated should overlap with organ2"
        assert overlap[2, 1].item() > 0, "organ2 dilated should overlap with organ1"

        # Diagonal should be >= volume (dilation covers own voxels)
        assert overlap[1, 1].item() >= volume[1].item()

    def test_distant_organs_no_contact(self):
        """Two organs far apart should have zero overlap after dilation."""
        from data.spatial_adjacency import _compute_single_sample_overlap

        num_classes = 3
        labels = torch.zeros(30, 30, 30, dtype=torch.long)
        labels[0:5, 0:5, 0:5] = 1      # organ 1: corner
        labels[25:30, 25:30, 25:30] = 2 # organ 2: far corner

        overlap, volume = _compute_single_sample_overlap(
            labels, num_classes=num_classes, dilation_radius=2
        )

        assert overlap[1, 2].item() == 0, "distant organs should have zero overlap"
        assert overlap[2, 1].item() == 0

    def test_small_organ_inside_large_organ_asymmetry(self):
        """
        Small organ surrounded by large organ.
        Contact(small->large) should be >> Contact(large->small).
        """
        from data.spatial_adjacency import _compute_single_sample_overlap

        num_classes = 3
        labels = torch.zeros(30, 30, 30, dtype=torch.long)
        labels[5:25, 5:25, 5:25] = 1                  # large organ: 20^3 = 8000
        labels[12:18, 12:18, 12:18] = 2               # small organ: 6^3 = 216 (carved out)

        overlap, volume = _compute_single_sample_overlap(
            labels, num_classes=num_classes, dilation_radius=2
        )

        vol_large = volume[1].item()
        vol_small = volume[2].item()
        assert vol_small < vol_large

        # small->large overlap should be much larger relative to small's volume
        contact_small_to_large = overlap[2, 1].item() / max(vol_small, 1)
        contact_large_to_small = overlap[1, 2].item() / max(vol_large, 1)
        assert contact_small_to_large > contact_large_to_small, \
            f"small->large ({contact_small_to_large:.3f}) should > large->small ({contact_large_to_small:.3f})"

    def test_output_shapes_and_dtypes(self):
        """Check output shapes and dtypes."""
        from data.spatial_adjacency import _compute_single_sample_overlap

        num_classes = 5
        labels = torch.zeros(10, 10, 10, dtype=torch.long)
        labels[2:5, 2:5, 2:5] = 1
        labels[6:9, 6:9, 6:9] = 3

        overlap, volume = _compute_single_sample_overlap(
            labels, num_classes=num_classes, dilation_radius=2
        )

        assert overlap.shape == (5, 5)
        assert volume.shape == (5,)
        assert overlap.dtype == torch.float32
        assert volume.dtype == torch.float32

    def test_empty_volume_returns_zeros(self):
        """All-zero labels (only background) should return zero overlap."""
        from data.spatial_adjacency import _compute_single_sample_overlap

        num_classes = 5
        labels = torch.zeros(10, 10, 10, dtype=torch.long)

        overlap, volume = _compute_single_sample_overlap(
            labels, num_classes=num_classes, dilation_radius=2
        )

        # Only class 0 (background) has volume, everything else is zero
        assert volume[1:].sum().item() == 0
        assert overlap[1:, 1:].sum().item() == 0
```

**Step 2: Run test to verify it fails**

Run: `cd /home/comp/25481568/code/HyperBody && python -m pytest tests/hyperbolic/test_spatial_adjacency.py::TestComputeSingleSampleOverlap -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data.spatial_adjacency'`

**Step 3: Implement `_compute_single_sample_overlap()`**

```python
# data/spatial_adjacency.py
"""
Compute spatial adjacency (contact) matrix between organs from GT labels.

Uses MaxPool3d dilation + einsum for fully vectorized computation.
No python for-loops over organ pairs.
"""
import torch
import torch.nn.functional as F
from typing import Tuple


def _compute_single_sample_overlap(
    labels: torch.Tensor,
    num_classes: int,
    dilation_radius: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute pairwise overlap between dilated organ masks for one sample.

    Args:
        labels: (D, H, W) int64 label volume, values in [0, num_classes)
        num_classes: total number of classes (C)
        dilation_radius: radius for cube dilation (kernel = 2*r + 1)

    Returns:
        overlap: (C, C) float32 — overlap[u, v] = number of voxels where
                 dilated mask of u intersects with original mask of v
        volume:  (C,) float32 — volume[u] = number of voxels belonging to class u
    """
    device = labels.device
    kernel = 2 * dilation_radius + 1

    # One-hot encode: (D,H,W) -> (C, D, H, W)
    one_hot = F.one_hot(labels, num_classes).permute(3, 0, 1, 2).float()  # (C, D, H, W)

    # Volume per class
    volume = one_hot.sum(dim=(1, 2, 3))  # (C,)

    # Dilate each class mask: (C, 1, D, H, W) -> max_pool3d -> (C, 1, D, H, W)
    dilated = F.max_pool3d(
        one_hot.unsqueeze(1),  # (C, 1, D, H, W)
        kernel_size=kernel,
        stride=1,
        padding=dilation_radius,
    ).squeeze(1)  # (C, D, H, W)

    # Pairwise overlap via einsum: overlap[u, v] = sum of (dilated_u AND original_v)
    overlap = torch.einsum('cdhw, kdhw -> ck', dilated, one_hot)  # (C, C)

    return overlap, volume
```

**Step 4: Run test to verify it passes**

Run: `cd /home/comp/25481568/code/HyperBody && python -m pytest tests/hyperbolic/test_spatial_adjacency.py::TestComputeSingleSampleOverlap -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add data/spatial_adjacency.py tests/hyperbolic/test_spatial_adjacency.py
git commit -m "feat: add _compute_single_sample_overlap() with vectorized MaxPool3d+einsum"
```

---

## Task 2: `compute_contact_matrix_from_dataset()` — Aggregate Across Samples

**Files:**
- Modify: `data/spatial_adjacency.py`
- Modify: `tests/hyperbolic/test_spatial_adjacency.py`

### Step 1: Write the failing test

```python
# Append to tests/hyperbolic/test_spatial_adjacency.py

class TestComputeContactMatrixFromDataset:
    """Test compute_contact_matrix_from_dataset() aggregation."""

    def _make_fake_dataset(self, samples):
        """Create a minimal list-like dataset returning (inp, lbl) tuples."""
        class FakeDataset:
            def __init__(self, label_list):
                self.label_list = label_list
            def __len__(self):
                return len(self.label_list)
            def __getitem__(self, idx):
                lbl = self.label_list[idx]
                inp = torch.zeros(1, *lbl.shape)  # dummy input
                return inp, lbl
        return FakeDataset(samples)

    def test_aggregation_two_samples(self):
        """
        Two samples with different organ layouts. Contact matrix should
        aggregate overlaps and volumes across both.
        """
        from data.spatial_adjacency import compute_contact_matrix_from_dataset

        num_classes = 3

        # Sample 1: organ 1 and 2 adjacent
        lbl1 = torch.zeros(20, 20, 20, dtype=torch.long)
        lbl1[5:10, 5:15, 5:15] = 1
        lbl1[10:15, 5:15, 5:15] = 2

        # Sample 2: only organ 1 exists (no organ 2)
        lbl2 = torch.zeros(20, 20, 20, dtype=torch.long)
        lbl2[5:15, 5:15, 5:15] = 1

        dataset = self._make_fake_dataset([lbl1, lbl2])

        contact = compute_contact_matrix_from_dataset(
            dataset, num_classes=num_classes, dilation_radius=2
        )

        assert contact.shape == (num_classes, num_classes)
        assert contact.dtype == torch.float32

        # Contact(1->2) should be > 0 (from sample 1)
        assert contact[1, 2].item() > 0

        # Diagonal should be 0
        assert contact[0, 0].item() == 0
        assert contact[1, 1].item() == 0
        assert contact[2, 2].item() == 0

        # Contact values should be in [0, 1]
        assert contact.min().item() >= 0
        assert contact.max().item() <= 1.0

    def test_contact_matrix_is_asymmetric(self):
        """
        Small organ surrounded by large organ should produce
        asymmetric contact: Contact(small->large) > Contact(large->small).
        """
        from data.spatial_adjacency import compute_contact_matrix_from_dataset

        num_classes = 3
        lbl = torch.zeros(30, 30, 30, dtype=torch.long)
        lbl[5:25, 5:25, 5:25] = 1           # large organ: ~7784 voxels
        lbl[12:18, 12:18, 12:18] = 2        # small organ: 216 voxels (carved)

        dataset = self._make_fake_dataset([lbl])
        contact = compute_contact_matrix_from_dataset(
            dataset, num_classes=num_classes, dilation_radius=2
        )

        assert contact[2, 1].item() > contact[1, 2].item(), \
            f"small->large ({contact[2,1]:.3f}) should > large->small ({contact[1,2]:.3f})"

    def test_save_and_load_roundtrip(self, tmp_path):
        """Contact matrix should survive save/load via torch.save."""
        from data.spatial_adjacency import compute_contact_matrix_from_dataset

        num_classes = 3
        lbl = torch.zeros(10, 10, 10, dtype=torch.long)
        lbl[2:5, 2:5, 2:5] = 1
        lbl[5:8, 5:8, 5:8] = 2

        dataset = self._make_fake_dataset([lbl])
        contact = compute_contact_matrix_from_dataset(
            dataset, num_classes=num_classes, dilation_radius=2
        )

        save_path = tmp_path / "contact_matrix.pt"
        torch.save(contact, save_path)
        loaded = torch.load(save_path, weights_only=True)

        assert torch.allclose(contact, loaded)
```

**Step 2: Run test to verify it fails**

Run: `cd /home/comp/25481568/code/HyperBody && python -m pytest tests/hyperbolic/test_spatial_adjacency.py::TestComputeContactMatrixFromDataset -v`
Expected: FAIL — `ImportError: cannot import name 'compute_contact_matrix_from_dataset'`

**Step 3: Implement `compute_contact_matrix_from_dataset()`**

Append to `data/spatial_adjacency.py`:

```python
from torch.utils.data import Dataset, DataLoader


def compute_contact_matrix_from_dataset(
    dataset: Dataset,
    num_classes: int,
    dilation_radius: int = 2,
    num_workers: int = 0,
) -> torch.Tensor:
    """
    Compute global asymmetric contact matrix from all samples in a dataset.

    Contact(u, v) = fraction of organ u's dilated boundary that overlaps organ v,
    aggregated across all training samples.

    Args:
        dataset: Dataset returning (input, labels) where labels is (D, H, W) int64
        num_classes: total number of classes
        dilation_radius: cube dilation radius in voxels
        num_workers: DataLoader workers (0 for main process)

    Returns:
        contact_matrix: (C, C) float32, asymmetric, diagonal=0, values in [0,1]
    """
    global_overlap = torch.zeros(num_classes, num_classes, dtype=torch.float64)
    global_volume = torch.zeros(num_classes, dtype=torch.float64)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    for inp, lbl in loader:
        # lbl: (B, D, H, W) with B=1 → squeeze to (D, H, W)
        labels = lbl.squeeze(0).long()

        overlap, volume = _compute_single_sample_overlap(
            labels, num_classes=num_classes, dilation_radius=dilation_radius
        )

        global_overlap += overlap.double()
        global_volume += volume.double()

    # Compute contact ratio: Contact(u, v) = overlap(u, v) / volume(u)
    # Use unsqueeze(1) so division broadcasts: [C,1] divides [C,C] row-wise
    contact_matrix = global_overlap / global_volume.unsqueeze(1).clamp(min=1)

    # Diagonal = 0 (self-contact is not meaningful)
    contact_matrix.fill_diagonal_(0)

    return contact_matrix.float()
```

**Step 4: Run test to verify it passes**

Run: `cd /home/comp/25481568/code/HyperBody && python -m pytest tests/hyperbolic/test_spatial_adjacency.py::TestComputeContactMatrixFromDataset -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add data/spatial_adjacency.py tests/hyperbolic/test_spatial_adjacency.py
git commit -m "feat: add compute_contact_matrix_from_dataset() with global accumulation"
```

---

## Task 3: `compute_graph_distance_matrix()` — Fuse Tree + Spatial

**Files:**
- Modify: `data/spatial_adjacency.py`
- Modify: `tests/hyperbolic/test_spatial_adjacency.py`

### Step 1: Write the failing test

```python
# Append to tests/hyperbolic/test_spatial_adjacency.py

class TestComputeGraphDistanceMatrix:
    """Test compute_graph_distance_matrix() — fusion of tree + spatial."""

    def test_basic_fusion(self):
        """
        With known D_tree and contact matrix, verify the per-pair min formula.

        D_final(u,v) = min(D_tree(u,v), lambda / (Contact(u,v) + epsilon))
        """
        from data.spatial_adjacency import compute_graph_distance_matrix

        # 3x3 toy matrices
        D_tree = torch.tensor([
            [0, 4, 10],
            [4, 0, 8],
            [10, 8, 0],
        ], dtype=torch.float32)

        contact = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],   # organ1->organ2: 50% contact
            [0.0, 0.02, 0.0],  # organ2->organ1: 2% contact
        ], dtype=torch.float32)

        D_final = compute_graph_distance_matrix(
            D_tree, contact, lambda_=1.0, epsilon=0.01
        )

        assert D_final.shape == (3, 3)

        # D_final[1,2] = min(8, 1.0/(0.5+0.01)) = min(8, 1.96) = 1.96
        expected_12 = min(8.0, 1.0 / (0.5 + 0.01))
        assert abs(D_final[1, 2].item() - expected_12) < 1e-4

        # D_final[2,1] = min(8, 1.0/(0.02+0.01)) = min(8, 33.33) = 8.0
        expected_21 = min(8.0, 1.0 / (0.02 + 0.01))
        assert abs(D_final[2, 1].item() - expected_21) < 1e-4

    def test_asymmetry(self):
        """D_final should be asymmetric when contact matrix is asymmetric."""
        from data.spatial_adjacency import compute_graph_distance_matrix

        D_tree = torch.tensor([
            [0, 10],
            [10, 0],
        ], dtype=torch.float32)

        contact = torch.tensor([
            [0.0, 0.5],   # organ0->organ1: 50%
            [0.0, 0.0],   # organ1->organ0: 0% (no contact from organ1's perspective)
        ], dtype=torch.float32)

        D_final = compute_graph_distance_matrix(D_tree, contact, lambda_=1.0, epsilon=0.01)

        # D_final[0,1] should be shortened (high contact)
        # D_final[1,0] should stay at D_tree (no contact)
        assert D_final[0, 1].item() < D_final[1, 0].item()

    def test_diagonal_is_zero(self):
        """Diagonal should always be 0."""
        from data.spatial_adjacency import compute_graph_distance_matrix

        D_tree = torch.tensor([[0, 5], [5, 0]], dtype=torch.float32)
        contact = torch.tensor([[0.0, 0.3], [0.1, 0.0]], dtype=torch.float32)

        D_final = compute_graph_distance_matrix(D_tree, contact, lambda_=1.0, epsilon=0.01)
        assert D_final[0, 0].item() == 0
        assert D_final[1, 1].item() == 0

    def test_no_contact_preserves_tree(self):
        """Zero contact matrix should return D_tree unchanged."""
        from data.spatial_adjacency import compute_graph_distance_matrix

        D_tree = torch.tensor([[0, 5, 8], [5, 0, 3], [8, 3, 0]], dtype=torch.float32)
        contact = torch.zeros(3, 3, dtype=torch.float32)

        D_final = compute_graph_distance_matrix(D_tree, contact, lambda_=1.0, epsilon=0.01)

        # lambda/(0+epsilon) = 100, which is >> any tree distance, so D_tree wins
        assert torch.allclose(D_final, D_tree)

    def test_lambda_scales_spatial_distance(self):
        """Larger lambda = larger spatial distance = less shortcutting."""
        from data.spatial_adjacency import compute_graph_distance_matrix

        D_tree = torch.tensor([[0, 8], [8, 0]], dtype=torch.float32)
        contact = torch.tensor([[0.0, 0.3], [0.3, 0.0]], dtype=torch.float32)

        D_small_lambda = compute_graph_distance_matrix(
            D_tree, contact, lambda_=0.5, epsilon=0.01
        )
        D_large_lambda = compute_graph_distance_matrix(
            D_tree, contact, lambda_=5.0, epsilon=0.01
        )

        # Smaller lambda → smaller spatial distance → more shortcutting
        assert D_small_lambda[0, 1].item() < D_large_lambda[0, 1].item()

    def test_print_example_distances(self):
        """Print a readable example for visual inspection."""
        from data.spatial_adjacency import compute_graph_distance_matrix

        D_tree = torch.tensor([
            [0, 2, 4, 9, 10],
            [2, 0, 4, 9, 10],
            [4, 4, 0, 9, 10],
            [9, 9, 9, 0, 4],
            [10, 10, 10, 4, 0],
        ], dtype=torch.float32)

        contact = torch.tensor([
            [0.0,  0.0,  0.0,  0.5,  0.0],   # organ0: 50% contact with organ3
            [0.0,  0.0,  0.0,  0.3,  0.0],   # organ1: 30% contact with organ3
            [0.0,  0.0,  0.0,  0.0,  0.0],   # organ2: no spatial contacts
            [0.02, 0.01, 0.0,  0.0,  0.0],   # organ3: tiny contact with 0,1
            [0.0,  0.0,  0.0,  0.0,  0.0],   # organ4: no spatial contacts
        ], dtype=torch.float32)

        D_final = compute_graph_distance_matrix(D_tree, contact, lambda_=1.0, epsilon=0.01)
        D_diff = D_tree - D_final

        print("\n=== Graph Distance Fusion Example ===")
        print(f"D_tree:\n{D_tree}")
        print(f"Contact:\n{contact}")
        print(f"D_final:\n{D_final.round(decimals=2)}")
        print(f"D_diff (shortened by):\n{D_diff.round(decimals=2)}")
```

**Step 2: Run test to verify it fails**

Run: `cd /home/comp/25481568/code/HyperBody && python -m pytest tests/hyperbolic/test_spatial_adjacency.py::TestComputeGraphDistanceMatrix -v`
Expected: FAIL — `ImportError: cannot import name 'compute_graph_distance_matrix'`

**Step 3: Implement `compute_graph_distance_matrix()`**

Append to `data/spatial_adjacency.py`:

```python
def compute_graph_distance_matrix(
    D_tree: torch.Tensor,
    contact_matrix: torch.Tensor,
    lambda_: float = 1.0,
    epsilon: float = 0.01,
) -> torch.Tensor:
    """
    Fuse tree distance with spatial contact into a graph distance matrix.

    Formula:
        D_final(u, v) = min(D_tree(u, v), lambda / (Contact(u, v) + epsilon))

    The result is asymmetric because Contact(u, v) != Contact(v, u).

    Args:
        D_tree:          (C, C) symmetric tree distance matrix
        contact_matrix:  (C, C) asymmetric contact ratio matrix, values in [0, 1]
        lambda_:         scale factor for spatial distance (default 1.0)
        epsilon:         prevents division by zero (default 0.01)

    Returns:
        D_final: (C, C) float32 asymmetric distance matrix, diagonal = 0
    """
    D_spatial = lambda_ / (contact_matrix + epsilon)    # (C, C)
    D_final = torch.min(D_tree, D_spatial)              # (C, C) element-wise min
    D_final.fill_diagonal_(0)
    return D_final
```

**Step 4: Run test to verify it passes**

Run: `cd /home/comp/25481568/code/HyperBody && python -m pytest tests/hyperbolic/test_spatial_adjacency.py::TestComputeGraphDistanceMatrix -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add data/spatial_adjacency.py tests/hyperbolic/test_spatial_adjacency.py
git commit -m "feat: add compute_graph_distance_matrix() with per-pair min fusion"
```

---

## Task 4: Config + train.py Integration

**Files:**
- Modify: `config.py:44` — add new config fields
- Modify: `train.py:23,381-406` — add `"graph"` branch
- Create: `configs/LR-Curriculum-GraphDistance.yaml`
- Modify: `tests/hyperbolic/test_spatial_adjacency.py` — add integration test

### Step 1: Write the failing integration test

```python
# Append to tests/hyperbolic/test_spatial_adjacency.py

class TestGraphDistanceIntegration:
    """Integration test: graph distance matrix works with LorentzTreeRankingLoss."""

    def test_asymmetric_matrix_works_with_loss(self):
        """
        LorentzTreeRankingLoss should accept an asymmetric D_final matrix
        and produce a valid scalar loss with gradients.
        """
        from models.hyperbolic.lorentz_loss import LorentzTreeRankingLoss

        num_classes = 5
        embed_dim = 8

        # Create an asymmetric distance matrix (as D_final would be)
        D_final = torch.tensor([
            [0, 4, 10, 2, 8],
            [4, 0, 8, 6, 10],
            [10, 8, 0, 3, 4],
            [5, 9, 7, 0, 6],   # row 3 != col 3 → asymmetric
            [8, 10, 4, 6, 0],
        ], dtype=torch.float32)

        loss_fn = LorentzTreeRankingLoss(
            tree_dist_matrix=D_final,
            margin=0.1,
            curv=1.0,
            num_samples_per_class=16,
            num_negatives=3,
        )
        loss_fn.set_epoch(10, 100)  # past warmup

        # Fake input
        B, C, D, H, W = 1, embed_dim, 4, 4, 4
        voxel_emb = torch.randn(B, C, D, H, W, requires_grad=True)
        labels = torch.randint(0, num_classes, (B, D, H, W))
        label_emb = torch.randn(num_classes, C, requires_grad=True)

        loss = loss_fn(voxel_emb, labels, label_emb)

        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"

        loss.backward()
        assert voxel_emb.grad is not None, "Should have gradients"

    def test_asymmetric_sampling_differs_by_anchor(self):
        """
        With asymmetric D_final, sampling weights for class A as anchor
        should differ from class B as anchor for the same pair.
        """
        from data.spatial_adjacency import compute_graph_distance_matrix

        D_tree = torch.tensor([[0, 10], [10, 0]], dtype=torch.float32)
        contact = torch.tensor([[0.0, 0.5], [0.02, 0.0]], dtype=torch.float32)

        D_final = compute_graph_distance_matrix(D_tree, contact)

        # D_final[0,1] should be small (class 0 has 50% contact with class 1)
        # D_final[1,0] should stay ~10 (class 1 has only 2% contact with class 0)
        assert D_final[0, 1].item() < 5.0, "High contact should shorten distance"
        assert D_final[1, 0].item() > 5.0, "Low contact should not shorten much"
```

**Step 2: Run test to verify it fails**

Run: `cd /home/comp/25481568/code/HyperBody && python -m pytest tests/hyperbolic/test_spatial_adjacency.py::TestGraphDistanceIntegration -v`
Expected: PASS (these tests use existing code + new functions already implemented).

> Note: These tests verify compatibility and should already pass if Tasks 1-3 are done. If they pass, continue to the config/train.py changes.

**Step 3: Add config fields to `config.py`**

Add after line 44 (`hyp_distance_mode`):

```python
    # Spatial adjacency (for hyp_distance_mode="graph")
    spatial_dilation_radius: int = 2       # cube dilation radius in voxels
    spatial_lambda: float = 1.0            # scale factor for spatial edge distance
    spatial_epsilon: float = 0.01          # prevents division by zero in contact->distance
    spatial_contact_matrix: str = ""       # path to precomputed contact_matrix.pt (empty = compute)
```

**Step 4: Add `"graph"` branch to `train.py`**

At line 23, add import:
```python
from data.spatial_adjacency import compute_contact_matrix_from_dataset, compute_graph_distance_matrix
```

Replace lines 381-406 with:

```python
    # Hyperbolic ranking loss (with Curriculum Negative Mining)
    # Choose loss class based on hyp_distance_mode config
    if cfg.hyp_distance_mode == "tree":
        # Tree-based negative sampling: uses precomputed tree distances
        tree_dist_matrix = compute_tree_distance_matrix(cfg.tree_file, class_names)
        hyp_criterion = LorentzTreeRankingLoss(
            tree_dist_matrix=tree_dist_matrix,
            margin=cfg.hyp_margin,
            curv=cfg.hyp_curv,
            num_samples_per_class=cfg.hyp_samples_per_class,
            num_negatives=cfg.hyp_num_negatives,
            t_start=cfg.hyp_t_start,
            t_end=cfg.hyp_t_end,
            warmup_epochs=cfg.hyp_warmup_epochs,
        )
        logger.info(f"Using LorentzTreeRankingLoss (tree distance mode)")
    elif cfg.hyp_distance_mode == "graph":
        # Graph-based: tree distance + spatial adjacency edges
        D_tree = compute_tree_distance_matrix(cfg.tree_file, class_names)

        # Load or compute contact matrix
        if cfg.spatial_contact_matrix and os.path.exists(cfg.spatial_contact_matrix):
            contact_matrix = torch.load(cfg.spatial_contact_matrix, weights_only=True)
            logger.info(f"Loaded contact matrix from {cfg.spatial_contact_matrix}")
        else:
            logger.info("Computing contact matrix from training set GT...")
            contact_matrix = compute_contact_matrix_from_dataset(
                train_dataset,
                num_classes=cfg.num_classes,
                dilation_radius=cfg.spatial_dilation_radius,
            )
            # Cache for future runs
            cache_path = os.path.join(cfg.checkpoint_dir, "contact_matrix.pt")
            os.makedirs(cfg.checkpoint_dir, exist_ok=True)
            torch.save(contact_matrix, cache_path)
            logger.info(f"Saved contact matrix to {cache_path}")

        graph_dist_matrix = compute_graph_distance_matrix(
            D_tree, contact_matrix,
            lambda_=cfg.spatial_lambda,
            epsilon=cfg.spatial_epsilon,
        )
        logger.info(
            f"Graph distance matrix: {(D_tree - graph_dist_matrix > 0).sum().item()} "
            f"pairs shortened by spatial adjacency"
        )

        hyp_criterion = LorentzTreeRankingLoss(
            tree_dist_matrix=graph_dist_matrix,
            margin=cfg.hyp_margin,
            curv=cfg.hyp_curv,
            num_samples_per_class=cfg.hyp_samples_per_class,
            num_negatives=cfg.hyp_num_negatives,
            t_start=cfg.hyp_t_start,
            t_end=cfg.hyp_t_end,
            warmup_epochs=cfg.hyp_warmup_epochs,
        )
        logger.info(f"Using LorentzTreeRankingLoss (graph distance mode)")
    else:
        # Default: Hyperbolic distance-based negative sampling
        hyp_criterion = LorentzRankingLoss(
            margin=cfg.hyp_margin,
            curv=cfg.hyp_curv,
            num_samples_per_class=cfg.hyp_samples_per_class,
            num_negatives=cfg.hyp_num_negatives,
            t_start=cfg.hyp_t_start,
            t_end=cfg.hyp_t_end,
            warmup_epochs=cfg.hyp_warmup_epochs,
        )
        logger.info(f"Using LorentzRankingLoss (hyperbolic distance mode)")
```

**Step 5: Create config file `configs/LR-Curriculum-GraphDistance.yaml`**

Copy `configs/LR-Curriculum-TreeDistance.yaml` and modify:

```yaml
# Changed fields only:
hyp_distance_mode: "graph"              # Use graph distance (tree + spatial adjacency)
spatial_dilation_radius: 2              # cube dilation radius
spatial_lambda: 1.0                     # spatial edge scale factor
spatial_epsilon: 0.01                   # prevents division by zero
spatial_contact_matrix: ""              # empty = compute from dataset and cache

checkpoint_dir: "checkpoints/LR-Curriculum-GraphDistance"
log_dir: "runs/LR-Curriculum-GraphDistance"
```

**Step 6: Commit**

```bash
git add config.py train.py configs/LR-Curriculum-GraphDistance.yaml tests/hyperbolic/test_spatial_adjacency.py
git commit -m "feat: integrate graph distance mode into config and train.py"
```

---

## Task 5: Visualizations

**Files:**
- Create: `tests/hyperbolic/test_spatial_adjacency_visual.py`

All outputs go to `docs/visualizations/spatial_adjacency/`.

### Step 1: Write visualization test

This task computes a contact matrix from **real data** (a small subset) and produces 3 visualizations.

```python
# tests/hyperbolic/test_spatial_adjacency_visual.py
"""
Visual validation of spatial adjacency graph.

Produces:
  - docs/visualizations/spatial_adjacency/contact_matrix_heatmap.html
  - docs/visualizations/spatial_adjacency/distance_diff_heatmap.html
  - docs/visualizations/spatial_adjacency/sampling_shift_plot.html

Run: pytest tests/hyperbolic/test_spatial_adjacency_visual.py -v -s
"""
import json
import os

import pytest
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

OUT_DIR = "docs/visualizations/spatial_adjacency"


@pytest.fixture(scope="module")
def class_names():
    with open("Dataset/dataset_info.json") as f:
        return json.load(f)["class_names"]


@pytest.fixture(scope="module")
def tree_path():
    return "Dataset/tree.json"


@pytest.fixture(scope="module")
def contact_and_trees(class_names, tree_path):
    """Compute contact matrix from a small subset of training data."""
    from data.spatial_adjacency import compute_contact_matrix_from_dataset
    from data.spatial_adjacency import compute_graph_distance_matrix
    from data.organ_hierarchy import compute_tree_distance_matrix
    from data.dataset import HyperBodyDataset

    dataset = HyperBodyDataset(
        data_dir="Dataset/voxel_data",
        split_file="Dataset/dataset_split.json",
        split="train",
        volume_size=(144, 128, 268),
    )

    # Use subset for speed (first 50 samples)
    from torch.utils.data import Subset
    subset = Subset(dataset, range(min(50, len(dataset))))

    num_classes = len(class_names)
    contact = compute_contact_matrix_from_dataset(
        subset, num_classes=num_classes, dilation_radius=2
    )
    D_tree = compute_tree_distance_matrix(tree_path, class_names)
    D_final = compute_graph_distance_matrix(D_tree, contact, lambda_=1.0, epsilon=0.01)

    return contact, D_tree, D_final


class TestVisualization1ContactHeatmap:
    def test_generate_contact_matrix_heatmap(self, contact_and_trees, class_names):
        """Generate interactive heatmap of asymmetric contact matrix."""
        contact, _, _ = contact_and_trees
        os.makedirs(OUT_DIR, exist_ok=True)

        fig = go.Figure(data=go.Heatmap(
            z=contact.numpy(),
            x=class_names,
            y=class_names,
            colorscale='Hot',
            reversescale=True,
            text=np.round(contact.numpy(), 3),
            texttemplate="%{text}",
            hovertemplate="From: %{y}<br>To: %{x}<br>Contact: %{z:.4f}<extra></extra>",
        ))
        fig.update_layout(
            title="Asymmetric Contact Matrix: Contact(row→col)",
            xaxis_title="Target Organ (v)",
            yaxis_title="Source Organ (u) [dilated]",
            width=1400, height=1200,
        )
        fig.write_html(os.path.join(OUT_DIR, "contact_matrix_heatmap.html"))


class TestVisualization2DistanceDiff:
    def test_generate_distance_diff_heatmap(self, contact_and_trees, class_names):
        """Generate D_diff = D_tree - D_final showing spatial shortcuts."""
        _, D_tree, D_final = contact_and_trees
        D_diff = D_tree - D_final
        os.makedirs(OUT_DIR, exist_ok=True)

        fig = go.Figure(data=go.Heatmap(
            z=D_diff.numpy(),
            x=class_names,
            y=class_names,
            colorscale='RdBu',
            zmid=0,
            text=np.round(D_diff.numpy(), 2),
            texttemplate="%{text}",
            hovertemplate="From: %{y}<br>To: %{x}<br>Shortened by: %{z:.2f}<extra></extra>",
        ))
        fig.update_layout(
            title="Distance Difference: D_tree - D_final (positive = shortened by spatial edge)",
            xaxis_title="Target Organ (v)",
            yaxis_title="Source Organ (u)",
            width=1400, height=1200,
        )
        fig.write_html(os.path.join(OUT_DIR, "distance_diff_heatmap.html"))


class TestVisualization3SamplingShift:
    def test_generate_sampling_shift_plot(self, contact_and_trees, class_names):
        """
        Shift Plot: for selected anchor organs, compare normalized
        sampling probability P(v|u) between tree-only and graph distance.
        """
        _, D_tree, D_final = contact_and_trees
        os.makedirs(OUT_DIR, exist_ok=True)

        anchors = ["rib_left_1", "liver", "spine", "gallbladder"]
        temperature = 0.5  # representative mid-training temperature

        fig = make_subplots(
            rows=len(anchors), cols=1,
            subplot_titles=[f"Anchor: {a}" for a in anchors],
            vertical_spacing=0.06,
        )

        for row_idx, anchor_name in enumerate(anchors):
            u = class_names.index(anchor_name)

            # Tree-only probabilities
            w_tree = torch.exp(-D_tree[u] / temperature)
            w_tree[u] = 0  # mask self
            p_tree = w_tree / w_tree.sum()

            # Graph probabilities
            w_graph = torch.exp(-D_final[u] / temperature)
            w_graph[u] = 0
            p_graph = w_graph / w_graph.sum()

            # Sort by probability shift (graph - tree) descending
            shift = p_graph - p_tree
            sorted_idx = torch.argsort(shift, descending=True)

            # Show top 15 most affected classes
            top_k = 15
            top_idx = sorted_idx[:top_k]
            names_top = [class_names[i] for i in top_idx]

            fig.add_trace(go.Bar(
                name="Tree P(v|u)",
                x=names_top,
                y=p_tree[top_idx].numpy(),
                marker_color='steelblue',
                showlegend=(row_idx == 0),
            ), row=row_idx + 1, col=1)

            fig.add_trace(go.Bar(
                name="Graph P(v|u)",
                x=names_top,
                y=p_graph[top_idx].numpy(),
                marker_color='coral',
                showlegend=(row_idx == 0),
            ), row=row_idx + 1, col=1)

        fig.update_layout(
            title=f"Sampling Probability Shift (T={temperature})",
            barmode='group',
            height=400 * len(anchors),
            width=1200,
        )
        fig.write_html(os.path.join(OUT_DIR, "sampling_shift_plot.html"))
```

**Step 2: Run visualization test**

Run: `cd /home/comp/25481568/code/HyperBody && python -m pytest tests/hyperbolic/test_spatial_adjacency_visual.py -v -s`
Expected: 3 PASS, 3 HTML files generated in `docs/visualizations/spatial_adjacency/`

> **Note:** This requires real dataset files. If they are not available, the test will skip/fail with a file-not-found error — that's expected. The test is designed to run on the training machine.

**Step 3: Commit**

```bash
git add tests/hyperbolic/test_spatial_adjacency_visual.py
git commit -m "feat: add spatial adjacency visualizations (contact heatmap, D_diff, shift plot)"
```

---

## Task 6: Precomputation Script

**Files:**
- Create: `scripts/precompute_contact_matrix.py`

This is a standalone script that precomputes the contact matrix for the full training set and saves it as a `.pt` file. This avoids recomputation on every training run.

### Step 1: Write the script

```python
# scripts/precompute_contact_matrix.py
"""
Precompute spatial contact matrix from training set ground truth.

Usage:
    python scripts/precompute_contact_matrix.py \
        --output Dataset/contact_matrix.pt \
        --dilation-radius 2

Reads dataset paths from Dataset/dataset_split.json and Dataset/dataset_info.json.
"""
import argparse
import json
import os
import sys
import time

import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import HyperBodyDataset
from data.spatial_adjacency import compute_contact_matrix_from_dataset


def main():
    parser = argparse.ArgumentParser(description="Precompute spatial contact matrix")
    parser.add_argument("--output", type=str, default="Dataset/contact_matrix.pt",
                        help="Output path for contact_matrix.pt")
    parser.add_argument("--dilation-radius", type=int, default=2,
                        help="Cube dilation radius in voxels (default: 2)")
    parser.add_argument("--data-dir", type=str, default="Dataset/voxel_data")
    parser.add_argument("--split-file", type=str, default="Dataset/dataset_split.json")
    parser.add_argument("--dataset-info", type=str, default="Dataset/dataset_info.json")
    parser.add_argument("--volume-size", type=int, nargs=3, default=[144, 128, 268])
    args = parser.parse_args()

    with open(args.dataset_info) as f:
        info = json.load(f)
    class_names = info["class_names"]
    num_classes = len(class_names)
    print(f"Classes: {num_classes}")

    dataset = HyperBodyDataset(
        data_dir=args.data_dir,
        split_file=args.split_file,
        split="train",
        volume_size=tuple(args.volume_size),
    )
    print(f"Training samples: {len(dataset)}")

    print(f"Computing contact matrix (dilation_radius={args.dilation_radius})...")
    t0 = time.time()
    contact_matrix = compute_contact_matrix_from_dataset(
        dataset, num_classes=num_classes, dilation_radius=args.dilation_radius
    )
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")

    # Print stats
    nonzero = (contact_matrix > 0).sum().item()
    total = num_classes * num_classes - num_classes  # exclude diagonal
    print(f"Non-zero contacts: {nonzero}/{total} ({100*nonzero/total:.1f}%)")
    print(f"Max contact: {contact_matrix.max().item():.4f}")
    print(f"Mean non-zero contact: {contact_matrix[contact_matrix > 0].mean().item():.4f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(contact_matrix, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/precompute_contact_matrix.py
git commit -m "feat: add precompute_contact_matrix.py script"
```

---

## Task Summary

| Task | What | Test Count | Files |
|------|------|-----------|-------|
| 1 | `_compute_single_sample_overlap()` | 5 tests | `data/spatial_adjacency.py`, test file |
| 2 | `compute_contact_matrix_from_dataset()` | 3 tests | same files |
| 3 | `compute_graph_distance_matrix()` | 6 tests | same files |
| 4 | Config + train.py integration | 2 tests | `config.py`, `train.py`, config yaml |
| 5 | Visualizations | 3 viz tests | visual test file |
| 6 | Precomputation script | — | `scripts/precompute_contact_matrix.py` |

Total: **16 unit tests + 3 visual tests + 1 script**

### Dependency Order

```
Task 1 → Task 2 → Task 3 → Task 4
                           → Task 5 (needs Task 1-3)
                           → Task 6 (needs Task 2)
```

Tasks 4, 5, 6 are independent of each other (all depend on Tasks 1-3).

---

## Graph Mode Quickstart (Post-Upgrade)

After this upgrade, `train.py` in `hyp_distance_mode: "graph"` no longer computes
contact/graph distance matrices at runtime. It must load a precomputed
`graph_distance_matrix.pt`.

### Step 1: Precompute

```bash
python scripts/precompute_graph_distance.py \
  --output-dir Dataset \
  --tree-file Dataset/tree.json \
  --data-dir Dataset/voxel_data \
  --split-file Dataset/dataset_split.json \
  --dataset-info Dataset/dataset_info.json \
  --volume-size 144 128 268 \
  --dilation-radius 3 \
  --lambda 1.0 \
  --epsilon 0.01 \
  --class-batch-size 5 \
  --num-workers 16
```

Artifacts:
- `Dataset/contact_matrix.pt`
- `Dataset/graph_distance_matrix.pt`

### Step 2: Configure and Train

```yaml
hyp_distance_mode: "graph"
graph_distance_matrix: "Dataset/graph_distance_matrix.pt"
```

```bash
python train.py --config configs/LR-Curriculum-GraphDistance.yaml
```

### Common Errors

- `FileNotFoundError: Graph distance mode requires precomputed graph_distance_matrix`
  - `graph_distance_matrix` is empty or points to a missing file.
- `graph_distance_matrix shape (...) != (num_classes, num_classes)`
  - Precomputed file and current config use different class definitions.
