# Lorentz Hyperbolic Embedding Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Lorentz hyperbolic embeddings as auxiliary loss for organ hierarchy encoding in HyperBody.

**Architecture:** Decoder features feed both segmentation head (existing) and new hyperbolic head. Hyperbolic head projects to Lorentz manifold via 1x1 conv + exp_map0. LorentzRankingLoss pulls voxel embeddings toward their class embeddings.

**Tech Stack:** PyTorch, NumPy, Plotly (visualization)

---

## Task 1: Core Lorentz Math Operations

**Files:**
- Create: `models/hyperbolic/__init__.py`
- Create: `models/hyperbolic/lorentz_ops.py`
- Create: `tests/hyperbolic/__init__.py`
- Create: `tests/hyperbolic/test_lorentz_ops.py`

### Step 1.1: Create directory structure

```bash
mkdir -p models/hyperbolic tests/hyperbolic
touch models/hyperbolic/__init__.py tests/hyperbolic/__init__.py
```

### Step 1.2: Write failing tests for exp_map0 and log_map0

Create `tests/hyperbolic/test_lorentz_ops.py`:

```python
import torch
import pytest
import math


class TestExpLogMap:
    """Test exp_map0 and log_map0 operations."""

    def test_exp_log_inverse(self):
        """exp_map0 and log_map0 should be inverses."""
        from models.hyperbolic.lorentz_ops import exp_map0, log_map0

        torch.manual_seed(42)
        v = torch.randn(100, 32) * 0.5  # Tangent vectors
        x = exp_map0(v, curv=1.0)
        v_rec = log_map0(x, curv=1.0)
        assert torch.allclose(v, v_rec, atol=1e-5), f"Max diff: {(v - v_rec).abs().max()}"

    def test_exp_map_zero_vector(self):
        """Zero tangent vector should map to origin (zero spatial components)."""
        from models.hyperbolic.lorentz_ops import exp_map0

        v = torch.zeros(10, 32)
        x = exp_map0(v, curv=1.0)
        assert torch.allclose(x, torch.zeros_like(x), atol=1e-6)

    def test_exp_map_output_shape(self):
        """exp_map0 should preserve shape."""
        from models.hyperbolic.lorentz_ops import exp_map0

        v = torch.randn(5, 10, 32)
        x = exp_map0(v, curv=1.0)
        assert x.shape == v.shape

    def test_exp_map_large_norm_stability(self):
        """exp_map0 should handle large norm vectors without overflow."""
        from models.hyperbolic.lorentz_ops import exp_map0

        v = torch.randn(10, 32) * 10.0  # Large vectors
        x = exp_map0(v, curv=1.0)
        assert torch.isfinite(x).all(), "Output contains inf or nan"
```

### Step 1.3: Run tests to verify they fail

```bash
pytest tests/hyperbolic/test_lorentz_ops.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'models.hyperbolic.lorentz_ops'`

### Step 1.4: Implement exp_map0 and log_map0

Create `models/hyperbolic/lorentz_ops.py`:

```python
"""
Lorentz (hyperboloid) model operations for hyperbolic geometry.

The Lorentz model represents hyperbolic space as a hyperboloid in Minkowski space.
We store only spatial components; time component is computed as needed:
    x_time = sqrt(1/curv + ||x_space||^2)

Reference: HyperPath (models/lorentz.py)
"""
import math
import torch
from torch import Tensor


def exp_map0(v: Tensor, curv: float = 1.0, eps: float = 1e-7) -> Tensor:
    """
    Exponential map from tangent space at origin to Lorentz manifold.

    Args:
        v: Tangent vectors at origin [..., D]
        curv: Curvature (positive value for negative curvature -curv)
        eps: Small value for numerical stability

    Returns:
        Points on Lorentz manifold (spatial components only) [..., D]
    """
    # ||v|| scaled by sqrt(curv)
    v_norm = torch.norm(v, dim=-1, keepdim=True)
    rc_vnorm = math.sqrt(curv) * v_norm

    # Clamp sinh input to prevent overflow: asinh(2^15) ≈ 11.09
    sinh_input = torch.clamp(rc_vnorm, min=eps, max=math.asinh(2**15))

    # x = sinh(sqrt(c)*||v||) * v / (sqrt(c)*||v||)
    # For numerical stability, handle small norms specially
    scale = torch.sinh(sinh_input) / torch.clamp(rc_vnorm, min=eps)
    return scale * v


def log_map0(x: Tensor, curv: float = 1.0, eps: float = 1e-7) -> Tensor:
    """
    Logarithmic map from Lorentz manifold to tangent space at origin.

    Args:
        x: Points on Lorentz manifold (spatial components only) [..., D]
        curv: Curvature (positive value for negative curvature -curv)
        eps: Small value for numerical stability

    Returns:
        Tangent vectors at origin [..., D]
    """
    # Compute time component: x_time = sqrt(1/curv + ||x||^2)
    x_sqnorm = torch.sum(x**2, dim=-1, keepdim=True)
    x_time = torch.sqrt(1.0 / curv + x_sqnorm)

    # Distance from origin: acosh(sqrt(curv) * x_time)
    # Note: sqrt(curv) * x_time >= 1 always (equality at origin)
    acosh_input = math.sqrt(curv) * x_time
    distance = torch.acosh(torch.clamp(acosh_input, min=1.0 + eps))

    # v = distance * x / ||x||
    x_norm = torch.norm(x, dim=-1, keepdim=True)
    scale = distance / torch.clamp(x_norm, min=eps)

    # Handle origin case: when x ≈ 0, return 0
    return torch.where(x_norm > eps, scale * x, torch.zeros_like(x))
```

### Step 1.5: Run tests to verify they pass

```bash
pytest tests/hyperbolic/test_lorentz_ops.py::TestExpLogMap -v
```

Expected: PASS

### Step 1.6: Write failing tests for distance functions

Add to `tests/hyperbolic/test_lorentz_ops.py`:

```python
class TestDistanceFunctions:
    """Test pointwise_dist, pairwise_dist, and distance_to_origin."""

    def test_pointwise_dist_shape(self):
        """pointwise_dist should return element-wise distances."""
        from models.hyperbolic.lorentz_ops import exp_map0, pointwise_dist

        torch.manual_seed(42)
        v1 = torch.randn(100, 32) * 0.5
        v2 = torch.randn(100, 32) * 0.5
        x1 = exp_map0(v1)
        x2 = exp_map0(v2)

        dist = pointwise_dist(x1, x2)
        assert dist.shape == (100,), f"Expected (100,), got {dist.shape}"

    def test_pointwise_dist_non_negative(self):
        """Distances should be non-negative."""
        from models.hyperbolic.lorentz_ops import exp_map0, pointwise_dist

        torch.manual_seed(42)
        x1 = exp_map0(torch.randn(50, 32) * 0.5)
        x2 = exp_map0(torch.randn(50, 32) * 0.5)

        dist = pointwise_dist(x1, x2)
        assert (dist >= -1e-6).all(), f"Negative distance found: {dist.min()}"

    def test_pointwise_dist_symmetry(self):
        """Distance should be symmetric: d(x,y) == d(y,x)."""
        from models.hyperbolic.lorentz_ops import exp_map0, pointwise_dist

        torch.manual_seed(42)
        x1 = exp_map0(torch.randn(50, 32) * 0.5)
        x2 = exp_map0(torch.randn(50, 32) * 0.5)

        d_xy = pointwise_dist(x1, x2)
        d_yx = pointwise_dist(x2, x1)
        assert torch.allclose(d_xy, d_yx, atol=1e-6)

    def test_pointwise_dist_self_zero(self):
        """Distance to self should be zero."""
        from models.hyperbolic.lorentz_ops import exp_map0, pointwise_dist

        torch.manual_seed(42)
        x = exp_map0(torch.randn(50, 32) * 0.5)
        dist = pointwise_dist(x, x)
        assert torch.allclose(dist, torch.zeros_like(dist), atol=1e-5)

    def test_triangle_inequality(self):
        """Triangle inequality: d(x,z) <= d(x,y) + d(y,z)."""
        from models.hyperbolic.lorentz_ops import exp_map0, pointwise_dist

        torch.manual_seed(42)
        x = exp_map0(torch.randn(50, 32) * 0.5)
        y = exp_map0(torch.randn(50, 32) * 0.5)
        z = exp_map0(torch.randn(50, 32) * 0.5)

        d_xy = pointwise_dist(x, y)
        d_yz = pointwise_dist(y, z)
        d_xz = pointwise_dist(x, z)

        assert (d_xz <= d_xy + d_yz + 1e-5).all(), "Triangle inequality violated"

    def test_pairwise_dist_shape(self):
        """pairwise_dist should return [N, M] matrix."""
        from models.hyperbolic.lorentz_ops import exp_map0, pairwise_dist

        torch.manual_seed(42)
        x = exp_map0(torch.randn(10, 32) * 0.5)
        y = exp_map0(torch.randn(20, 32) * 0.5)

        dist = pairwise_dist(x, y)
        assert dist.shape == (10, 20), f"Expected (10, 20), got {dist.shape}"

    def test_distance_to_origin_shape(self):
        """distance_to_origin should reduce last dimension."""
        from models.hyperbolic.lorentz_ops import exp_map0, distance_to_origin

        torch.manual_seed(42)
        x = exp_map0(torch.randn(5, 10, 32) * 0.5)

        dist = distance_to_origin(x)
        assert dist.shape == (5, 10), f"Expected (5, 10), got {dist.shape}"

    def test_origin_has_zero_distance(self):
        """Origin should have zero distance from origin."""
        from models.hyperbolic.lorentz_ops import distance_to_origin

        x = torch.zeros(10, 32)  # Origin in spatial components
        dist = distance_to_origin(x)
        assert torch.allclose(dist, torch.zeros_like(dist), atol=1e-5)
```

### Step 1.7: Run tests to verify they fail

```bash
pytest tests/hyperbolic/test_lorentz_ops.py::TestDistanceFunctions -v
```

Expected: FAIL with `ImportError: cannot import name 'pointwise_dist'`

### Step 1.8: Implement distance functions

Add to `models/hyperbolic/lorentz_ops.py`:

```python
def pointwise_dist(x: Tensor, y: Tensor, curv: float = 1.0, eps: float = 1e-7) -> Tensor:
    """
    Element-wise geodesic distance between corresponding points.

    Args:
        x: Points on Lorentz manifold [..., D]
        y: Points on Lorentz manifold [..., D] (same shape as x)
        curv: Curvature
        eps: Small value for numerical stability

    Returns:
        Geodesic distances [...] (last dimension reduced)
    """
    # Compute time components
    x_time = torch.sqrt(1.0 / curv + torch.sum(x**2, dim=-1))
    y_time = torch.sqrt(1.0 / curv + torch.sum(y**2, dim=-1))

    # Lorentz inner product: <x,y>_L = x_s · y_s - x_t * y_t
    spatial_inner = torch.sum(x * y, dim=-1)
    lorentz_inner = spatial_inner - x_time * y_time

    # Distance: acosh(-curv * <x,y>_L) / sqrt(curv)
    acosh_input = -curv * lorentz_inner
    distance = torch.acosh(torch.clamp(acosh_input, min=1.0 + eps)) / math.sqrt(curv)

    return distance


def pairwise_dist(x: Tensor, y: Tensor, curv: float = 1.0, eps: float = 1e-7) -> Tensor:
    """
    All-pairs geodesic distance between two sets of points.

    Args:
        x: Points on Lorentz manifold [N, D]
        y: Points on Lorentz manifold [M, D]
        curv: Curvature
        eps: Small value for numerical stability

    Returns:
        Distance matrix [N, M]
    """
    # Compute time components
    x_time = torch.sqrt(1.0 / curv + torch.sum(x**2, dim=-1, keepdim=True))  # [N, 1]
    y_time = torch.sqrt(1.0 / curv + torch.sum(y**2, dim=-1, keepdim=True))  # [M, 1]

    # Lorentz inner product: <x,y>_L = x_s @ y_s.T - x_t @ y_t.T
    spatial_inner = x @ y.T  # [N, M]
    time_inner = x_time @ y_time.T  # [N, M]
    lorentz_inner = spatial_inner - time_inner

    # Distance: acosh(-curv * <x,y>_L) / sqrt(curv)
    acosh_input = -curv * lorentz_inner
    distance = torch.acosh(torch.clamp(acosh_input, min=1.0 + eps)) / math.sqrt(curv)

    return distance


def distance_to_origin(x: Tensor, curv: float = 1.0, eps: float = 1e-7) -> Tensor:
    """
    Geodesic distance from origin for each point.

    Args:
        x: Points on Lorentz manifold [..., D]
        curv: Curvature
        eps: Small value for numerical stability

    Returns:
        Distances from origin [...]
    """
    # Time component of x
    x_time = torch.sqrt(1.0 / curv + torch.sum(x**2, dim=-1))

    # Origin time component: sqrt(1/curv)
    origin_time = math.sqrt(1.0 / curv)

    # Lorentz inner product with origin: 0 - x_t * origin_t = -x_t * origin_t
    lorentz_inner = -x_time * origin_time

    # Distance
    acosh_input = -curv * lorentz_inner
    distance = torch.acosh(torch.clamp(acosh_input, min=1.0 + eps)) / math.sqrt(curv)

    return distance
```

### Step 1.9: Run all distance tests

```bash
pytest tests/hyperbolic/test_lorentz_ops.py::TestDistanceFunctions -v
```

Expected: PASS

### Step 1.10: Add lorentz_to_poincare for visualization

Add to `models/hyperbolic/lorentz_ops.py`:

```python
def lorentz_to_poincare(x: Tensor, curv: float = 1.0) -> Tensor:
    """
    Project Lorentz spatial components to Poincare disk.

    Useful for 2D visualization of hyperbolic embeddings.

    Args:
        x: Points on Lorentz manifold (spatial components) [..., D]
        curv: Curvature

    Returns:
        Points in Poincare disk [..., D] (||p|| < 1)
    """
    x_time = torch.sqrt(1.0 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    return x / (1.0 + x_time)
```

### Step 1.11: Update __init__.py exports

Update `models/hyperbolic/__init__.py`:

```python
from models.hyperbolic.lorentz_ops import (
    exp_map0,
    log_map0,
    pointwise_dist,
    pairwise_dist,
    distance_to_origin,
    lorentz_to_poincare,
)

__all__ = [
    "exp_map0",
    "log_map0",
    "pointwise_dist",
    "pairwise_dist",
    "distance_to_origin",
    "lorentz_to_poincare",
]
```

### Step 1.12: Run all tests

```bash
pytest tests/hyperbolic/test_lorentz_ops.py -v
```

Expected: All PASS

### Step 1.13: Commit

```bash
git add models/hyperbolic/ tests/hyperbolic/
git commit -m "feat(hyperbolic): add Lorentz math operations

- exp_map0: tangent space -> Lorentz manifold
- log_map0: Lorentz manifold -> tangent space
- pointwise_dist: O(K) element-wise geodesic distance
- pairwise_dist: O(N*M) all-pairs distance
- distance_to_origin: for hierarchy visualization
- lorentz_to_poincare: for Poincare disk projection

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Organ Hierarchy Parser

**Files:**
- Create: `data/organ_hierarchy.py`
- Create: `tests/hyperbolic/test_organ_hierarchy.py`

### Step 2.1: Write failing tests

Create `tests/hyperbolic/test_organ_hierarchy.py`:

```python
import pytest
import json
import os


class TestOrganHierarchy:
    """Test organ hierarchy parsing."""

    @pytest.fixture
    def tree_path(self):
        return "Dataset/tree.json"

    @pytest.fixture
    def class_names(self):
        with open("Dataset/dataset_info.json") as f:
            return json.load(f)["class_names"]

    def test_load_organ_hierarchy_returns_dict(self, tree_path, class_names):
        """Should return a dict mapping class_idx -> depth."""
        from data.organ_hierarchy import load_organ_hierarchy

        depths = load_organ_hierarchy(tree_path, class_names)
        assert isinstance(depths, dict)

    def test_all_classes_have_depth(self, tree_path, class_names):
        """Every class should have a depth assigned."""
        from data.organ_hierarchy import load_organ_hierarchy

        depths = load_organ_hierarchy(tree_path, class_names)
        for idx, name in enumerate(class_names):
            assert idx in depths, f"Class {idx} ({name}) has no depth"

    def test_depths_are_positive(self, tree_path, class_names):
        """All depths should be positive integers."""
        from data.organ_hierarchy import load_organ_hierarchy

        depths = load_organ_hierarchy(tree_path, class_names)
        for idx, depth in depths.items():
            assert isinstance(depth, int), f"Depth for class {idx} is not int"
            assert depth >= 1, f"Depth for class {idx} is < 1"

    def test_rib_deeper_than_skeletal_system(self, tree_path, class_names):
        """rib_left_1 should be deeper than structures closer to root."""
        from data.organ_hierarchy import load_organ_hierarchy

        depths = load_organ_hierarchy(tree_path, class_names)

        # rib_left_1 is at depth 6 (human_body > skeletal_system > axial_skeleton >
        # thoracic_cage > ribs > ribs_left > rib_left_1)
        rib_idx = class_names.index("rib_left_1")
        spine_idx = class_names.index("spine")

        # spine is at depth 3, rib_left_1 is at depth 6
        assert depths[rib_idx] > depths[spine_idx], \
            f"rib_left_1 depth {depths[rib_idx]} should be > spine depth {depths[spine_idx]}"

    def test_siblings_have_same_depth(self, tree_path, class_names):
        """Sibling organs should have the same depth."""
        from data.organ_hierarchy import load_organ_hierarchy

        depths = load_organ_hierarchy(tree_path, class_names)

        # kidney_left and kidney_right are siblings
        left_idx = class_names.index("kidney_left")
        right_idx = class_names.index("kidney_right")

        assert depths[left_idx] == depths[right_idx], \
            f"kidney_left depth {depths[left_idx]} != kidney_right depth {depths[right_idx]}"
```

### Step 2.2: Run tests to verify they fail

```bash
pytest tests/hyperbolic/test_organ_hierarchy.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'data.organ_hierarchy'`

### Step 2.3: Implement organ_hierarchy.py

Create `data/organ_hierarchy.py`:

```python
"""
Parse organ hierarchy tree to extract class depths.

The tree.json defines anatomical hierarchy:
- human_body (root, depth 0)
  - skeletal_system (depth 1)
    - axial_skeleton (depth 2)
      ...
        - rib_left_1 (leaf, depth 6)
"""
import json
from typing import Dict, List, Optional


def _find_depth_recursive(
    tree: dict,
    target_name: str,
    current_depth: int = 0
) -> Optional[int]:
    """
    Recursively search for target_name in tree and return its depth.

    Args:
        tree: Dictionary representing the hierarchy subtree
        target_name: Class name to find
        current_depth: Current depth in traversal

    Returns:
        Depth if found, None otherwise
    """
    for key, value in tree.items():
        if isinstance(value, str):
            # Leaf node: value is the class name
            if value == target_name:
                return current_depth + 1
        elif isinstance(value, dict):
            # Intermediate node: recurse
            result = _find_depth_recursive(value, target_name, current_depth + 1)
            if result is not None:
                return result
    return None


def load_organ_hierarchy(tree_path: str, class_names: List[str]) -> Dict[int, int]:
    """
    Parse tree.json and compute depth for each class.

    Args:
        tree_path: Path to tree.json
        class_names: List of class names (index = class_idx)

    Returns:
        Dictionary mapping class_idx -> depth
    """
    with open(tree_path, "r") as f:
        tree = json.load(f)

    depths = {}
    for idx, name in enumerate(class_names):
        depth = _find_depth_recursive(tree, name, current_depth=0)
        if depth is None:
            # Default depth for classes not in tree (shouldn't happen)
            depth = 1
        depths[idx] = depth

    return depths


def get_depth_stats(depths: Dict[int, int]) -> Dict[str, int]:
    """
    Get statistics about depth distribution.

    Args:
        depths: Dictionary mapping class_idx -> depth

    Returns:
        Dictionary with min_depth, max_depth, unique_depths
    """
    depth_values = list(depths.values())
    return {
        "min_depth": min(depth_values),
        "max_depth": max(depth_values),
        "unique_depths": len(set(depth_values)),
    }
```

### Step 2.4: Run tests

```bash
pytest tests/hyperbolic/test_organ_hierarchy.py -v
```

Expected: All PASS

### Step 2.5: Commit

```bash
git add data/organ_hierarchy.py tests/hyperbolic/test_organ_hierarchy.py
git commit -m "feat(data): add organ hierarchy parser

Parse Dataset/tree.json to compute class -> depth mapping for
hierarchy-aware hyperbolic embedding initialization.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Label Embedding Module

**Files:**
- Create: `models/hyperbolic/label_embedding.py`
- Create: `tests/hyperbolic/test_label_embedding.py`

### Step 3.1: Write failing tests

Create `tests/hyperbolic/test_label_embedding.py`:

```python
import torch
import pytest
import json


class TestLorentzLabelEmbedding:
    """Test LorentzLabelEmbedding module."""

    @pytest.fixture
    def class_depths(self):
        """Load real class depths from dataset."""
        from data.organ_hierarchy import load_organ_hierarchy
        with open("Dataset/dataset_info.json") as f:
            class_names = json.load(f)["class_names"]
        return load_organ_hierarchy("Dataset/tree.json", class_names)

    def test_output_shape(self, class_depths):
        """Output should be [num_classes, embed_dim]."""
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding

        emb = LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=class_depths
        )
        out = emb()
        assert out.shape == (70, 32), f"Expected (70, 32), got {out.shape}"

    def test_output_is_on_manifold(self, class_depths):
        """Output should be valid Lorentz points (finite values)."""
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding

        emb = LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=class_depths
        )
        out = emb()
        assert torch.isfinite(out).all(), "Output contains inf or nan"

    def test_deeper_organs_farther_from_origin(self, class_depths):
        """Deeper organs should be initialized farther from origin."""
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding
        from models.hyperbolic.lorentz_ops import distance_to_origin

        torch.manual_seed(42)
        emb = LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=class_depths,
            min_radius=0.1,
            max_radius=2.0
        )
        out = emb()
        distances = distance_to_origin(out)

        # Find a shallow and deep class
        min_depth = min(class_depths.values())
        max_depth = max(class_depths.values())

        shallow_idx = [i for i, d in class_depths.items() if d == min_depth][0]
        deep_idx = [i for i, d in class_depths.items() if d == max_depth][0]

        assert distances[deep_idx] > distances[shallow_idx], \
            f"Deep class dist {distances[deep_idx]:.4f} should be > shallow {distances[shallow_idx]:.4f}"

    def test_gradient_flow(self, class_depths):
        """Gradients should flow through the embedding."""
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding

        emb = LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=class_depths
        )
        out = emb()
        loss = out.sum()
        loss.backward()

        # Check tangent_embeddings has gradients
        assert emb.tangent_embeddings.grad is not None
        assert (emb.tangent_embeddings.grad != 0).any()

    def test_different_seeds_different_directions(self, class_depths):
        """Different random seeds should give different initial directions."""
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding

        torch.manual_seed(42)
        emb1 = LorentzLabelEmbedding(num_classes=70, embed_dim=32, class_depths=class_depths)

        torch.manual_seed(123)
        emb2 = LorentzLabelEmbedding(num_classes=70, embed_dim=32, class_depths=class_depths)

        # Directions should differ
        assert not torch.allclose(emb1.tangent_embeddings, emb2.tangent_embeddings)
```

### Step 3.2: Run tests to verify they fail

```bash
pytest tests/hyperbolic/test_label_embedding.py -v
```

Expected: FAIL with `ModuleNotFoundError`

### Step 3.3: Implement LorentzLabelEmbedding

Create `models/hyperbolic/label_embedding.py`:

```python
"""
Learnable label embeddings in Lorentz (hyperbolic) space.

Embeddings are stored as tangent vectors at the origin and mapped to the
Lorentz manifold via exp_map0 during forward pass.
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional

from models.hyperbolic.lorentz_ops import exp_map0


class LorentzLabelEmbedding(nn.Module):
    """
    Learnable class embeddings in Lorentz hyperbolic space.

    Stores tangent vectors at origin, applies exp_map0 in forward pass.
    Initialization uses hierarchy depth: deeper classes start farther from origin.
    """

    def __init__(
        self,
        num_classes: int = 70,
        embed_dim: int = 32,
        curv: float = 1.0,
        class_depths: Optional[Dict[int, int]] = None,
        min_radius: float = 0.1,
        max_radius: float = 2.0,
    ):
        """
        Args:
            num_classes: Number of classes
            embed_dim: Embedding dimension
            curv: Fixed curvature (positive value for negative curvature -curv)
            class_depths: Dict mapping class_idx -> hierarchy depth
            min_radius: Tangent norm for shallowest classes
            max_radius: Tangent norm for deepest classes
        """
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.curv = curv

        # Initialize tangent vectors
        tangent_vectors = self._init_tangent_vectors(
            num_classes, embed_dim, class_depths, min_radius, max_radius
        )
        self.tangent_embeddings = nn.Parameter(tangent_vectors)

    def _init_tangent_vectors(
        self,
        num_classes: int,
        embed_dim: int,
        class_depths: Optional[Dict[int, int]],
        min_radius: float,
        max_radius: float,
    ) -> Tensor:
        """
        Initialize tangent vectors with hierarchy-aware norms.

        Args:
            num_classes: Number of classes
            embed_dim: Embedding dimension
            class_depths: Dict mapping class_idx -> hierarchy depth
            min_radius: Tangent norm for shallowest classes
            max_radius: Tangent norm for deepest classes

        Returns:
            Tensor of shape [num_classes, embed_dim]
        """
        tangent_vectors = torch.zeros(num_classes, embed_dim)

        if class_depths is None:
            # Fallback: uniform random initialization
            tangent_vectors = torch.randn(num_classes, embed_dim) * 0.5
            return tangent_vectors

        # Get depth range
        depths = list(class_depths.values())
        min_depth = min(depths)
        max_depth = max(depths)
        depth_range = max_depth - min_depth
        if depth_range == 0:
            depth_range = 1  # Avoid division by zero

        for class_idx in range(num_classes):
            depth = class_depths.get(class_idx, min_depth)

            # Normalize depth to [0, 1]
            normalized_depth = (depth - min_depth) / depth_range

            # Compute tangent norm based on depth
            tangent_norm = min_radius + (max_radius - min_radius) * normalized_depth

            # Random unit direction
            direction = torch.randn(embed_dim)
            direction = direction / direction.norm()

            # Tangent vector = direction * norm
            tangent_vectors[class_idx] = direction * tangent_norm

        return tangent_vectors

    def forward(self) -> Tensor:
        """
        Get Lorentz embeddings for all classes.

        Returns:
            Tensor of shape [num_classes, embed_dim] in Lorentz space
        """
        return exp_map0(self.tangent_embeddings, self.curv)

    def get_embedding(self, class_idx: int) -> Tensor:
        """Get Lorentz embedding for a single class."""
        tangent = self.tangent_embeddings[class_idx]
        return exp_map0(tangent.unsqueeze(0), self.curv).squeeze(0)
```

### Step 3.4: Run tests

```bash
pytest tests/hyperbolic/test_label_embedding.py -v
```

Expected: All PASS

### Step 3.5: Update exports

Add to `models/hyperbolic/__init__.py`:

```python
from models.hyperbolic.label_embedding import LorentzLabelEmbedding
```

Update `__all__`:

```python
__all__ = [
    "exp_map0",
    "log_map0",
    "pointwise_dist",
    "pairwise_dist",
    "distance_to_origin",
    "lorentz_to_poincare",
    "LorentzLabelEmbedding",
]
```

### Step 3.6: Commit

```bash
git add models/hyperbolic/label_embedding.py models/hyperbolic/__init__.py tests/hyperbolic/test_label_embedding.py
git commit -m "feat(hyperbolic): add LorentzLabelEmbedding

Learnable class embeddings in Lorentz space with hierarchy-aware
initialization. Deeper organs start farther from origin.
"
```

---

## Task 4: Projection Head Module

**Files:**
- Create: `models/hyperbolic/projection_head.py`
- Create: `tests/hyperbolic/test_projection_head.py`

### Step 4.1: Write failing tests

Create `tests/hyperbolic/test_projection_head.py`:

```python
import torch
import pytest


class TestLorentzProjectionHead:
    """Test LorentzProjectionHead module."""

    def test_output_shape(self):
        """Output should preserve spatial dimensions."""
        from models.hyperbolic.projection_head import LorentzProjectionHead

        head = LorentzProjectionHead(in_channels=32, embed_dim=32)
        x = torch.randn(2, 32, 16, 12, 8)  # [B, C, H, W, D]

        out = head(x)
        assert out.shape == (2, 32, 16, 12, 8), f"Expected (2, 32, 16, 12, 8), got {out.shape}"

    def test_output_is_finite(self):
        """Output should not contain inf or nan."""
        from models.hyperbolic.projection_head import LorentzProjectionHead

        head = LorentzProjectionHead(in_channels=32, embed_dim=32)
        x = torch.randn(2, 32, 8, 6, 4)

        out = head(x)
        assert torch.isfinite(out).all(), "Output contains inf or nan"

    def test_different_embed_dim(self):
        """Should work with different embedding dimensions."""
        from models.hyperbolic.projection_head import LorentzProjectionHead

        head = LorentzProjectionHead(in_channels=64, embed_dim=16)
        x = torch.randn(2, 64, 8, 6, 4)

        out = head(x)
        assert out.shape == (2, 16, 8, 6, 4)

    def test_gradient_flow(self):
        """Gradients should flow through the head."""
        from models.hyperbolic.projection_head import LorentzProjectionHead

        head = LorentzProjectionHead(in_channels=32, embed_dim=32)
        x = torch.randn(2, 32, 4, 4, 4, requires_grad=True)

        out = head(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert head.conv.weight.grad is not None
```

### Step 4.2: Run tests to verify they fail

```bash
pytest tests/hyperbolic/test_projection_head.py -v
```

Expected: FAIL

### Step 4.3: Implement LorentzProjectionHead

Create `models/hyperbolic/projection_head.py`:

```python
"""
Projection head for mapping decoder features to Lorentz hyperbolic space.
"""
import torch
import torch.nn as nn
from torch import Tensor

from models.hyperbolic.lorentz_ops import exp_map0


class LorentzProjectionHead(nn.Module):
    """
    Projects 3D feature maps to Lorentz hyperbolic space.

    Architecture: 1x1x1 Conv3d -> exp_map0
    """

    def __init__(
        self,
        in_channels: int = 32,
        embed_dim: int = 32,
        curv: float = 1.0,
    ):
        """
        Args:
            in_channels: Number of input channels from decoder
            embed_dim: Embedding dimension in Lorentz space
            curv: Fixed curvature
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.curv = curv

        self.conv = nn.Conv3d(in_channels, embed_dim, kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Project decoder features to Lorentz space.

        Args:
            x: Decoder features [B, C, H, W, D]

        Returns:
            Lorentz embeddings [B, embed_dim, H, W, D]
        """
        # 1x1x1 convolution
        x = self.conv(x)  # [B, embed_dim, H, W, D]

        # Permute for exp_map0: [B, H, W, D, embed_dim]
        x = x.permute(0, 2, 3, 4, 1)

        # Map to Lorentz manifold
        x = exp_map0(x, self.curv)

        # Permute back: [B, embed_dim, H, W, D]
        x = x.permute(0, 4, 1, 2, 3)

        return x
```

### Step 4.4: Run tests

```bash
pytest tests/hyperbolic/test_projection_head.py -v
```

Expected: All PASS

### Step 4.5: Update exports

Add to `models/hyperbolic/__init__.py`:

```python
from models.hyperbolic.projection_head import LorentzProjectionHead
```

Update `__all__`.

### Step 4.6: Commit

```bash
git add models/hyperbolic/projection_head.py models/hyperbolic/__init__.py tests/hyperbolic/test_projection_head.py
git commit -m "feat(hyperbolic): add LorentzProjectionHead

1x1x1 conv + exp_map0 to project decoder features to Lorentz space.
"
```

---

## Task 5: Lorentz Ranking Loss

**Files:**
- Create: `models/hyperbolic/lorentz_loss.py`
- Create: `tests/hyperbolic/test_lorentz_loss.py`

### Step 5.1: Write failing tests

Create `tests/hyperbolic/test_lorentz_loss.py`:

```python
import torch
import pytest


class TestLorentzRankingLoss:
    """Test LorentzRankingLoss module."""

    def test_output_is_scalar(self):
        """Loss should return a scalar."""
        from models.hyperbolic.lorentz_loss import LorentzRankingLoss
        from models.hyperbolic.lorentz_ops import exp_map0

        loss_fn = LorentzRankingLoss(margin=0.1, num_samples_per_class=8, num_negatives=4)

        # Create fake data
        voxel_emb = exp_map0(torch.randn(2, 32, 4, 4, 4) * 0.3)
        labels = torch.randint(0, 70, (2, 4, 4, 4))
        label_emb = exp_map0(torch.randn(70, 32) * 0.5)

        loss = loss_fn(voxel_emb, labels, label_emb)
        assert loss.dim() == 0, f"Expected scalar, got shape {loss.shape}"

    def test_loss_is_non_negative(self):
        """Loss should be non-negative (triplet margin loss)."""
        from models.hyperbolic.lorentz_loss import LorentzRankingLoss
        from models.hyperbolic.lorentz_ops import exp_map0

        loss_fn = LorentzRankingLoss(margin=0.1, num_samples_per_class=8, num_negatives=4)

        voxel_emb = exp_map0(torch.randn(2, 32, 4, 4, 4) * 0.3)
        labels = torch.randint(0, 70, (2, 4, 4, 4))
        label_emb = exp_map0(torch.randn(70, 32) * 0.5)

        loss = loss_fn(voxel_emb, labels, label_emb)
        assert loss >= 0, f"Loss should be >= 0, got {loss}"

    def test_gradient_flow_to_voxel_emb(self):
        """Gradients should flow to voxel embeddings."""
        from models.hyperbolic.lorentz_loss import LorentzRankingLoss
        from models.hyperbolic.lorentz_ops import exp_map0

        loss_fn = LorentzRankingLoss(margin=0.1, num_samples_per_class=8, num_negatives=4)

        tangent = torch.randn(2, 32, 4, 4, 4) * 0.3
        tangent.requires_grad = True
        voxel_emb = exp_map0(tangent)

        labels = torch.randint(0, 70, (2, 4, 4, 4))
        label_emb = exp_map0(torch.randn(70, 32) * 0.5)

        loss = loss_fn(voxel_emb, labels, label_emb)
        loss.backward()

        assert tangent.grad is not None, "No gradient for voxel tangent vectors"

    def test_gradient_flow_to_label_emb(self):
        """Gradients should flow to label embeddings."""
        from models.hyperbolic.lorentz_loss import LorentzRankingLoss
        from models.hyperbolic.lorentz_ops import exp_map0

        loss_fn = LorentzRankingLoss(margin=0.1, num_samples_per_class=8, num_negatives=4)

        voxel_emb = exp_map0(torch.randn(2, 32, 4, 4, 4) * 0.3)
        labels = torch.randint(0, 70, (2, 4, 4, 4))

        label_tangent = torch.randn(70, 32) * 0.5
        label_tangent.requires_grad = True
        label_emb = exp_map0(label_tangent)

        loss = loss_fn(voxel_emb, labels, label_emb)
        loss.backward()

        assert label_tangent.grad is not None, "No gradient for label tangent vectors"

    def test_handles_single_class_batch(self):
        """Should handle batches with only one class present."""
        from models.hyperbolic.lorentz_loss import LorentzRankingLoss
        from models.hyperbolic.lorentz_ops import exp_map0

        loss_fn = LorentzRankingLoss(margin=0.1, num_samples_per_class=8, num_negatives=4)

        voxel_emb = exp_map0(torch.randn(1, 32, 4, 4, 4) * 0.3)
        labels = torch.zeros(1, 4, 4, 4, dtype=torch.long)  # All class 0
        label_emb = exp_map0(torch.randn(70, 32) * 0.5)

        # Should not crash, may return 0 if no negatives available
        loss = loss_fn(voxel_emb, labels, label_emb)
        assert torch.isfinite(loss), "Loss is not finite"
```

### Step 5.2: Run tests to verify they fail

```bash
pytest tests/hyperbolic/test_lorentz_loss.py -v
```

Expected: FAIL

### Step 5.3: Implement LorentzRankingLoss

Create `models/hyperbolic/lorentz_loss.py`:

```python
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
```

### Step 5.4: Run tests

```bash
pytest tests/hyperbolic/test_lorentz_loss.py -v
```

Expected: All PASS

### Step 5.5: Update exports

Add to `models/hyperbolic/__init__.py`:

```python
from models.hyperbolic.lorentz_loss import LorentzRankingLoss
```

### Step 5.6: Commit

```bash
git add models/hyperbolic/lorentz_loss.py models/hyperbolic/__init__.py tests/hyperbolic/test_lorentz_loss.py
git commit -m "feat(hyperbolic): add LorentzRankingLoss

Triplet margin loss in Lorentz space with configurable num_negatives
per anchor. Uses pointwise_dist for O(K) efficiency.
"
```

---

## Task 6: Modify UNet3D for Feature Output

**Files:**
- Modify: `models/unet3d.py`
- Create: `tests/test_unet3d_features.py`

### Step 6.1: Write failing test

Create `tests/test_unet3d_features.py`:

```python
import torch
import pytest


class TestUNet3DFeatures:
    """Test UNet3D return_features functionality."""

    def test_default_returns_only_logits(self):
        """By default, forward should return only logits."""
        from models.unet3d import UNet3D

        model = UNet3D(in_channels=1, num_classes=70, base_channels=32)
        x = torch.randn(1, 1, 32, 24, 32)

        out = model(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape[1] == 70

    def test_return_features_gives_tuple(self):
        """With return_features=True, should return (logits, features)."""
        from models.unet3d import UNet3D

        model = UNet3D(in_channels=1, num_classes=70, base_channels=32)
        x = torch.randn(1, 1, 32, 24, 32)

        out = model(x, return_features=True)
        assert isinstance(out, tuple)
        assert len(out) == 2

    def test_features_shape_matches_output(self):
        """Features should have same spatial dimensions as output."""
        from models.unet3d import UNet3D

        model = UNet3D(in_channels=1, num_classes=70, base_channels=32)
        x = torch.randn(1, 1, 32, 24, 32)

        logits, features = model(x, return_features=True)

        # Features are d2 with base_channels
        assert features.shape[0] == 1
        assert features.shape[1] == 32  # base_channels
        assert features.shape[2:] == logits.shape[2:]

    def test_backward_compatible(self):
        """Existing code using model(x) should still work."""
        from models.unet3d import UNet3D

        model = UNet3D(in_channels=1, num_classes=70, base_channels=32)
        x = torch.randn(1, 1, 16, 12, 16)

        # Old usage
        logits = model(x)
        assert logits.shape == (1, 70, 16, 12, 16)
```

### Step 6.2: Run test to verify it fails

```bash
pytest tests/test_unet3d_features.py -v
```

Expected: FAIL on `test_return_features_gives_tuple`

### Step 6.3: Modify UNet3D

Edit `models/unet3d.py`, modify the `forward` method:

```python
def forward(self, x: torch.Tensor, return_features: bool = False):
    """
    Args:
        x: Input tensor of shape (B, 1, H, W, D)
        return_features: If True, return (logits, decoder_features)

    Returns:
        If return_features=False: logits of shape (B, num_classes, H, W, D)
        If return_features=True: (logits, d2) where d2 is (B, base_channels, H, W, D)
    """
    # Encoder path with skip connections
    e1 = self.enc1(x)   # (B, 32, H, W, D)
    e2 = self.enc2(e1)  # (B, 64, H/2, W/2, D/2)
    e3 = self.enc3(e2)  # (B, 128, H/4, W/4, D/4)
    e4 = self.enc4(e3)  # (B, 256, H/8, W/8, D/8)

    # Dense Bottleneck
    b = self.bottleneck(e4)  # (B, 384, H/8, W/8, D/8)

    # Decoder path with skip connections
    d4 = self.dec4(b, e3)   # (B, 128, H/4, W/4, D/4)
    d3 = self.dec3(d4, e2)  # (B, 64, H/2, W/2, D/2)
    d2 = self.dec2(d3, e1)  # (B, 32, H, W, D)

    # Final output
    out = self.final(d2)  # (B, num_classes, H, W, D)

    if return_features:
        return out, d2
    return out
```

### Step 6.4: Run tests

```bash
pytest tests/test_unet3d_features.py -v
```

Expected: All PASS

### Step 6.5: Commit

```bash
git add models/unet3d.py tests/test_unet3d_features.py
git commit -m "feat(unet3d): add return_features option

Allow returning decoder features (d2) for hyperbolic projection head.
Backward compatible - default behavior unchanged.
"
```

---

## Task 7: BodyNet Wrapper

**Files:**
- Create: `models/body_net.py`
- Create: `tests/test_body_net.py`

### Step 7.1: Write failing tests

Create `tests/test_body_net.py`:

```python
import torch
import pytest
import json


class TestBodyNet:
    """Test BodyNet model wrapper."""

    @pytest.fixture
    def class_depths(self):
        from data.organ_hierarchy import load_organ_hierarchy
        with open("Dataset/dataset_info.json") as f:
            class_names = json.load(f)["class_names"]
        return load_organ_hierarchy("Dataset/tree.json", class_names)

    def test_output_is_tuple(self, class_depths):
        """Forward should return (logits, voxel_emb, label_emb)."""
        from models.body_net import BodyNet

        model = BodyNet(
            num_classes=70,
            base_channels=32,
            embed_dim=32,
            class_depths=class_depths
        )
        x = torch.randn(1, 1, 32, 24, 32)

        out = model(x)
        assert isinstance(out, tuple)
        assert len(out) == 3

    def test_logits_shape(self, class_depths):
        """Logits should have correct shape."""
        from models.body_net import BodyNet

        model = BodyNet(
            num_classes=70,
            base_channels=32,
            embed_dim=32,
            class_depths=class_depths
        )
        x = torch.randn(1, 1, 32, 24, 32)

        logits, _, _ = model(x)
        assert logits.shape == (1, 70, 32, 24, 32)

    def test_voxel_emb_shape(self, class_depths):
        """Voxel embeddings should have correct shape."""
        from models.body_net import BodyNet

        model = BodyNet(
            num_classes=70,
            base_channels=32,
            embed_dim=32,
            class_depths=class_depths
        )
        x = torch.randn(1, 1, 32, 24, 32)

        _, voxel_emb, _ = model(x)
        assert voxel_emb.shape == (1, 32, 32, 24, 32)

    def test_label_emb_shape(self, class_depths):
        """Label embeddings should have correct shape."""
        from models.body_net import BodyNet

        model = BodyNet(
            num_classes=70,
            base_channels=32,
            embed_dim=32,
            class_depths=class_depths
        )
        x = torch.randn(1, 1, 32, 24, 32)

        _, _, label_emb = model(x)
        assert label_emb.shape == (70, 32)

    def test_gradient_flow(self, class_depths):
        """Gradients should flow through all outputs."""
        from models.body_net import BodyNet

        model = BodyNet(
            num_classes=70,
            base_channels=32,
            embed_dim=32,
            class_depths=class_depths
        )
        x = torch.randn(1, 1, 16, 12, 16, requires_grad=True)

        logits, voxel_emb, label_emb = model(x)
        loss = logits.sum() + voxel_emb.sum() + label_emb.sum()
        loss.backward()

        assert x.grad is not None
```

### Step 7.2: Run tests to verify they fail

```bash
pytest tests/test_body_net.py -v
```

Expected: FAIL

### Step 7.3: Implement BodyNet

Create `models/body_net.py`:

```python
"""
BodyNet: UNet3D with hyperbolic embedding head.

Combines segmentation and hyperbolic geometry for hierarchy-aware
organ segmentation.
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional, Tuple

from models.unet3d import UNet3D
from models.hyperbolic.projection_head import LorentzProjectionHead
from models.hyperbolic.label_embedding import LorentzLabelEmbedding


class BodyNet(nn.Module):
    """
    UNet3D with Lorentz hyperbolic embedding branch.

    Returns:
        - logits: Segmentation logits [B, num_classes, H, W, D]
        - voxel_emb: Lorentz voxel embeddings [B, embed_dim, H, W, D]
        - label_emb: Lorentz class embeddings [num_classes, embed_dim]
    """

    def __init__(
        self,
        # UNet3D params
        in_channels: int = 1,
        num_classes: int = 70,
        base_channels: int = 32,
        growth_rate: int = 32,
        dense_layers: int = 4,
        bn_size: int = 4,
        # Hyperbolic params
        embed_dim: int = 32,
        curv: float = 1.0,
        class_depths: Optional[Dict[int, int]] = None,
        min_radius: float = 0.1,
        max_radius: float = 2.0,
    ):
        """
        Args:
            in_channels: Input channels for UNet3D
            num_classes: Number of segmentation classes
            base_channels: Base channels for UNet3D
            growth_rate: Dense block growth rate
            dense_layers: Number of dense layers
            bn_size: Bottleneck size multiplier
            embed_dim: Hyperbolic embedding dimension
            curv: Hyperbolic curvature
            class_depths: Dict mapping class_idx -> hierarchy depth
            min_radius: Min tangent norm for label embedding init
            max_radius: Max tangent norm for label embedding init
        """
        super().__init__()

        # Segmentation backbone
        self.unet = UNet3D(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            growth_rate=growth_rate,
            dense_layers=dense_layers,
            bn_size=bn_size,
        )

        # Hyperbolic projection head (from decoder features to Lorentz space)
        self.hyp_head = LorentzProjectionHead(
            in_channels=base_channels,
            embed_dim=embed_dim,
            curv=curv,
        )

        # Learnable class embeddings in Lorentz space
        self.label_emb = LorentzLabelEmbedding(
            num_classes=num_classes,
            embed_dim=embed_dim,
            curv=curv,
            class_depths=class_depths,
            min_radius=min_radius,
            max_radius=max_radius,
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass.

        Args:
            x: Input volume [B, in_channels, H, W, D]

        Returns:
            Tuple of:
                - logits: [B, num_classes, H, W, D]
                - voxel_emb: [B, embed_dim, H, W, D] in Lorentz space
                - label_emb: [num_classes, embed_dim] in Lorentz space
        """
        # Get segmentation logits and decoder features
        logits, d2 = self.unet(x, return_features=True)

        # Project decoder features to Lorentz space
        voxel_emb = self.hyp_head(d2)

        # Get class embeddings in Lorentz space
        label_emb = self.label_emb()

        return logits, voxel_emb, label_emb
```

### Step 7.4: Run tests

```bash
pytest tests/test_body_net.py -v
```

Expected: All PASS

### Step 7.5: Update models __init__.py

Add to `models/__init__.py`:

```python
from models.body_net import BodyNet
```

### Step 7.6: Commit

```bash
git add models/body_net.py models/__init__.py tests/test_body_net.py
git commit -m "feat(models): add BodyNet wrapper

Combines UNet3D segmentation with Lorentz hyperbolic embedding branch.
Returns (logits, voxel_emb, label_emb) tuple.
"
```

---

## Task 8: Update Config

**Files:**
- Modify: `config.py`

### Step 8.1: Add hyperbolic config fields

Edit `config.py`:

```python
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    # Data
    data_dir: str = "Dataset/voxel_data"
    split_file: str = "Dataset/dataset_split.json"
    tree_file: str = "Dataset/tree.json"
    dataset_info_file: str = "Dataset/dataset_info.json"
    num_classes: int = 70
    voxel_size: float = 4.0
    volume_size: Tuple[int, int, int] = (144, 128, 268)  # X, Y, Z

    # Model
    in_channels: int = 1
    base_channels: int = 32
    num_levels: int = 4

    # Dense Bottleneck
    growth_rate: int = 32      # channels added per layer
    dense_layers: int = 4      # number of dense layers
    bn_size: int = 4           # 1x1x1 compression factor

    # Hyperbolic
    hyp_embed_dim: int = 32
    hyp_curv: float = 1.0
    hyp_weight: float = 0.05      # Loss weight
    hyp_margin: float = 0.1       # Triplet margin
    hyp_samples_per_class: int = 64
    hyp_num_negatives: int = 8    # Negative classes per anchor
    hyp_min_radius: float = 0.1   # Shallow organ init norm
    hyp_max_radius: float = 2.0   # Deep organ init norm

    # Training
    batch_size: int = 4  # per GPU
    num_workers: int = 0
    epochs: int = 120
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # AMP
    use_amp: bool = True

    # Loss
    ce_weight: float = 0.5
    dice_weight: float = 0.5

    # LR scheduler
    lr_patience: int = 10
    lr_factor: float = 0.5

    # Checkpoint
    checkpoint_dir: str = "checkpoints"
    save_every: int = 10
    log_dir: str = "runs"

    # GPU
    gpu_ids: List[int] = field(default_factory=lambda: [0, 1])

    # Resume
    resume: str = ""
```

### Step 8.2: Commit

```bash
git add config.py
git commit -m "feat(config): add hyperbolic configuration fields

Add hyp_* fields for Lorentz hyperbolic embedding configuration.
"
```

---

## Task 9: Update train.py

**Files:**
- Modify: `train.py`

### Step 9.1: Update imports and model creation

Edit `train.py`:

Add imports:
```python
import json
from data.organ_hierarchy import load_organ_hierarchy
from models.body_net import BodyNet
from models.hyperbolic.lorentz_loss import LorentzRankingLoss
```

### Step 9.2: Update model creation in main()

Replace UNet3D creation with BodyNet:

```python
# Load organ hierarchy for hyperbolic embeddings
with open(cfg.dataset_info_file) as f:
    class_names = json.load(f)["class_names"]
class_depths = load_organ_hierarchy(cfg.tree_file, class_names)

# Model
logger.info("Creating model...")
model = BodyNet(
    in_channels=cfg.in_channels,
    num_classes=cfg.num_classes,
    base_channels=cfg.base_channels,
    growth_rate=cfg.growth_rate,
    dense_layers=cfg.dense_layers,
    bn_size=cfg.bn_size,
    embed_dim=cfg.hyp_embed_dim,
    curv=cfg.hyp_curv,
    class_depths=class_depths,
    min_radius=cfg.hyp_min_radius,
    max_radius=cfg.hyp_max_radius,
)
```

### Step 9.3: Add hyperbolic loss criterion

After CombinedLoss creation:
```python
# Hyperbolic ranking loss
hyp_criterion = LorentzRankingLoss(
    margin=cfg.hyp_margin,
    curv=cfg.hyp_curv,
    num_samples_per_class=cfg.hyp_samples_per_class,
    num_negatives=cfg.hyp_num_negatives,
)
```

### Step 9.4: Update train_one_epoch signature and logic

Update the function to accept and use hyperbolic loss:

```python
def train_one_epoch(model, loader, seg_criterion, hyp_criterion, hyp_weight, optimizer, device, grad_clip, epoch=0, scaler=None):
    """Train for one epoch.

    Args:
        model: The model to train (BodyNet)
        loader: DataLoader for training data
        seg_criterion: Segmentation loss function (CombinedLoss)
        hyp_criterion: Hyperbolic loss function (LorentzRankingLoss)
        hyp_weight: Weight for hyperbolic loss
        optimizer: Optimizer
        device: Device to use
        grad_clip: Gradient clipping value (0 to disable)
        epoch: Current epoch number (used for DistributedSampler shuffling)
        scaler: Optional GradScaler for AMP training (None to disable AMP)

    Returns:
        Tuple of (avg_total_loss, avg_seg_loss, avg_hyp_loss)
    """
    model.train()

    if hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(epoch)

    total_loss_sum = 0.0
    seg_loss_sum = 0.0
    hyp_loss_sum = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="  Train", leave=False, disable=not is_main_process())
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                logits, voxel_emb, label_emb = model(inputs)
                seg_loss = seg_criterion(logits, targets)
                hyp_loss = hyp_criterion(voxel_emb, targets, label_emb)
                total_loss = seg_loss + hyp_weight * hyp_loss

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, voxel_emb, label_emb = model(inputs)
            seg_loss = seg_criterion(logits, targets)
            hyp_loss = hyp_criterion(voxel_emb, targets, label_emb)
            total_loss = seg_loss + hyp_weight * hyp_loss

            total_loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss_sum += total_loss.item()
        seg_loss_sum += seg_loss.item()
        hyp_loss_sum += hyp_loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{total_loss.item():.4f}", seg=f"{seg_loss.item():.4f}", hyp=f"{hyp_loss.item():.4f}")

    n = max(num_batches, 1)
    return total_loss_sum / n, seg_loss_sum / n, hyp_loss_sum / n
```

### Step 9.5: Update validate function

```python
@torch.no_grad()
def validate(model, loader, seg_criterion, hyp_criterion, hyp_weight, metric, device):
    """Validate and compute metrics.

    Returns:
        Tuple of (val_total_loss, val_seg_loss, val_hyp_loss, dice_per_class, mean_dice).
    """
    model.eval()
    metric.reset()
    total_loss_sum = 0.0
    seg_loss_sum = 0.0
    hyp_loss_sum = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="  Val  ", leave=False, disable=not is_main_process())
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits, voxel_emb, label_emb = model(inputs)
        seg_loss = seg_criterion(logits, targets)
        hyp_loss = hyp_criterion(voxel_emb, targets, label_emb)
        total_loss = seg_loss + hyp_weight * hyp_loss

        total_loss_sum += total_loss.item()
        seg_loss_sum += seg_loss.item()
        hyp_loss_sum += hyp_loss.item()
        num_batches += 1

        metric.update(logits, targets)
        pbar.set_postfix(loss=f"{total_loss.item():.4f}")

    if is_distributed():
        metric.sync_across_processes()

    n = max(num_batches, 1)
    dice_per_class, mean_dice, _ = metric.compute()

    return total_loss_sum / n, seg_loss_sum / n, hyp_loss_sum / n, dice_per_class, mean_dice
```

### Step 9.6: Update training loop calls

Update the calls in main():

```python
# Train
train_total, train_seg, train_hyp = train_one_epoch(
    model, train_loader, criterion, hyp_criterion, cfg.hyp_weight,
    optimizer, device, cfg.grad_clip, epoch=epoch, scaler=scaler
)

# Validate
val_total, val_seg, val_hyp, dice_per_class, mean_dice = validate(
    model, val_loader, criterion, hyp_criterion, cfg.hyp_weight, metric, device
)
```

### Step 9.7: Update TensorBoard logging

```python
if writer:
    writer.add_scalar("Loss/train_total", train_total, epoch)
    writer.add_scalar("Loss/train_seg", train_seg, epoch)
    writer.add_scalar("Loss/train_hyp", train_hyp, epoch)
    writer.add_scalar("Loss/val_total", val_total, epoch)
    writer.add_scalar("Loss/val_seg", val_seg, epoch)
    writer.add_scalar("Loss/val_hyp", val_hyp, epoch)
    writer.add_scalar("Dice/mean", mean_dice, epoch)
    writer.add_scalar("LR", current_lr, epoch)
    # ... rest of logging
```

### Step 9.8: Update epoch summary logging

```python
logger.info(
    f"  Train: total={train_total:.4f} seg={train_seg:.4f} hyp={train_hyp:.4f} | "
    f"Val: total={val_total:.4f} seg={val_seg:.4f} hyp={val_hyp:.4f} | "
    f"Dice: {mean_dice:.4f} (best: {best_dice:.4f})"
    f"{' *' if is_best else ''}"
)
```

### Step 9.9: Commit

```bash
git add train.py
git commit -m "feat(train): integrate hyperbolic loss

- Use BodyNet instead of UNet3D
- Add LorentzRankingLoss with hyp_weight=0.05
- Track seg_loss and hyp_loss separately
- Update TensorBoard logging for hyperbolic metrics
"
```

---

## Task 10: Visualization Tests

**Files:**
- Create: `tests/hyperbolic/test_visualization.py`

### Step 10.1: Create visualization tests

Create `tests/hyperbolic/test_visualization.py`:

```python
import torch
import pytest
import json
import os


class TestVisualization:
    """Test hyperbolic embedding visualization."""

    @pytest.fixture
    def class_depths(self):
        from data.organ_hierarchy import load_organ_hierarchy
        with open("Dataset/dataset_info.json") as f:
            class_names = json.load(f)["class_names"]
        return load_organ_hierarchy("Dataset/tree.json", class_names)

    @pytest.fixture
    def class_names(self):
        with open("Dataset/dataset_info.json") as f:
            return json.load(f)["class_names"]

    @pytest.fixture
    def output_dir(self):
        path = "docs/visualizations/hyperbolic"
        os.makedirs(path, exist_ok=True)
        return path

    def test_poincare_disk_visualization(self, class_depths, class_names, output_dir):
        """Visualize label embeddings in Poincare disk."""
        import plotly.express as px
        import pandas as pd
        from sklearn.decomposition import PCA
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding
        from models.hyperbolic.lorentz_ops import lorentz_to_poincare, distance_to_origin

        torch.manual_seed(42)
        emb = LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=class_depths
        )
        label_emb = emb()  # [70, 32]

        # Project to Poincare disk
        poincare_emb = lorentz_to_poincare(label_emb)  # [70, 32]

        # Use PCA to reduce to 2D for visualization
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(poincare_emb.detach().numpy())

        # Get distances for coloring
        distances = distance_to_origin(label_emb).detach().numpy()

        # Create dataframe
        df = pd.DataFrame({
            "x": coords_2d[:, 0],
            "y": coords_2d[:, 1],
            "class_name": class_names,
            "depth": [class_depths[i] for i in range(70)],
            "distance": distances,
        })

        # Create plot
        fig = px.scatter(
            df, x="x", y="y",
            color="depth",
            hover_data=["class_name", "distance"],
            title="Label Embeddings in Poincare Disk (PCA 2D)",
            color_continuous_scale="Viridis"
        )

        output_path = os.path.join(output_dir, "label_emb_poincare.html")
        fig.write_html(output_path)
        assert os.path.exists(output_path)

    def test_distance_matrix_heatmap(self, class_depths, class_names, output_dir):
        """Visualize pairwise distance matrix."""
        import plotly.express as px
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding
        from models.hyperbolic.lorentz_ops import pairwise_dist

        torch.manual_seed(42)
        emb = LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=class_depths
        )
        label_emb = emb()  # [70, 32]

        # Compute pairwise distances
        dist_matrix = pairwise_dist(label_emb, label_emb).detach().numpy()

        # Create heatmap
        fig = px.imshow(
            dist_matrix,
            x=class_names,
            y=class_names,
            color_continuous_scale="Blues",
            title="Pairwise Lorentz Distances Between Classes"
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            width=1200,
            height=1000
        )

        output_path = os.path.join(output_dir, "class_distance_matrix.html")
        fig.write_html(output_path)
        assert os.path.exists(output_path)

    def test_tsne_visualization(self, class_depths, class_names, output_dir):
        """Visualize label embeddings using t-SNE."""
        import plotly.express as px
        import pandas as pd
        from sklearn.manifold import TSNE
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding
        from models.hyperbolic.lorentz_ops import distance_to_origin

        torch.manual_seed(42)
        emb = LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=class_depths
        )
        label_emb = emb().detach().numpy()  # [70, 32]

        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=15)
        coords_2d = tsne.fit_transform(label_emb)

        # Get organ system from class name (simplified)
        def get_system(name):
            if "rib" in name or "spine" in name or "skull" in name or "sternum" in name or \
               "scapula" in name or "clavicula" in name or "humerus" in name or \
               "hip" in name or "femur" in name or "costal" in name:
                return "skeletal"
            elif "gluteus" in name or "autochthon" in name or "iliopsoas" in name:
                return "muscular"
            elif "kidney" in name or "bladder" in name or "adrenal" in name:
                return "urinary"
            elif "liver" in name or "stomach" in name or "pancreas" in name or \
                 "gallbladder" in name or "esophagus" in name or "bowel" in name or \
                 "duodenum" in name or "colon" in name:
                return "digestive"
            elif "heart" in name:
                return "cardiovascular"
            elif "brain" in name or "spinal_cord" in name:
                return "nervous"
            elif "lung" in name or "trachea" in name:
                return "respiratory"
            else:
                return "other"

        # Create dataframe
        df = pd.DataFrame({
            "x": coords_2d[:, 0],
            "y": coords_2d[:, 1],
            "class_name": class_names,
            "system": [get_system(name) for name in class_names],
            "depth": [class_depths[i] for i in range(70)],
        })

        # Create plot
        fig = px.scatter(
            df, x="x", y="y",
            color="system",
            symbol="system",
            hover_data=["class_name", "depth"],
            title="Label Embeddings t-SNE (colored by organ system)"
        )

        output_path = os.path.join(output_dir, "label_emb_tsne.html")
        fig.write_html(output_path)
        assert os.path.exists(output_path)
```

### Step 10.2: Run visualization tests

```bash
pytest tests/hyperbolic/test_visualization.py -v
```

Expected: All PASS (creates HTML files in `docs/visualizations/hyperbolic/`)

### Step 10.3: Commit

```bash
git add tests/hyperbolic/test_visualization.py
git commit -m "feat(tests): add hyperbolic visualization tests

Generate interactive Plotly visualizations:
- Poincare disk projection
- Distance matrix heatmap
- t-SNE colored by organ system

Output to docs/visualizations/hyperbolic/

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 11: Final Integration Test

**Files:**
- Create: `tests/test_integration.py`

### Step 11.1: Create integration test

Create `tests/test_integration.py`:

```python
import torch
import pytest
import json


class TestIntegration:
    """End-to-end integration tests."""

    @pytest.fixture
    def class_depths(self):
        from data.organ_hierarchy import load_organ_hierarchy
        with open("Dataset/dataset_info.json") as f:
            class_names = json.load(f)["class_names"]
        return load_organ_hierarchy("Dataset/tree.json", class_names)

    def test_full_forward_backward(self, class_depths):
        """Test complete forward and backward pass."""
        from models.body_net import BodyNet
        from models.losses import CombinedLoss
        from models.hyperbolic.lorentz_loss import LorentzRankingLoss

        # Create model
        model = BodyNet(
            num_classes=70,
            base_channels=32,
            embed_dim=32,
            class_depths=class_depths
        )

        # Create losses
        seg_criterion = CombinedLoss(num_classes=70)
        hyp_criterion = LorentzRankingLoss(margin=0.1, num_negatives=4)

        # Create fake batch
        x = torch.randn(2, 1, 32, 24, 32)
        targets = torch.randint(0, 70, (2, 32, 24, 32))

        # Forward
        logits, voxel_emb, label_emb = model(x)

        # Compute losses
        seg_loss = seg_criterion(logits, targets)
        hyp_loss = hyp_criterion(voxel_emb, targets, label_emb)
        total_loss = seg_loss + 0.05 * hyp_loss

        # Backward
        total_loss.backward()

        # Check gradients exist
        assert model.unet.enc1.block[0].weight.grad is not None
        assert model.hyp_head.conv.weight.grad is not None
        assert model.label_emb.tangent_embeddings.grad is not None

    def test_training_step_decreases_loss(self, class_depths):
        """Verify that a training step decreases loss."""
        from models.body_net import BodyNet
        from models.losses import CombinedLoss
        from models.hyperbolic.lorentz_loss import LorentzRankingLoss
        import torch.optim as optim

        torch.manual_seed(42)

        model = BodyNet(
            num_classes=70,
            base_channels=16,  # Smaller for speed
            embed_dim=16,
            class_depths=class_depths
        )
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        seg_criterion = CombinedLoss(num_classes=70)
        hyp_criterion = LorentzRankingLoss(margin=0.1, num_negatives=4)

        # Fixed input/target
        x = torch.randn(1, 1, 16, 12, 16)
        targets = torch.randint(0, 70, (1, 16, 12, 16))

        # Initial loss
        logits, voxel_emb, label_emb = model(x)
        loss_before = seg_criterion(logits, targets) + 0.05 * hyp_criterion(voxel_emb, targets, label_emb)

        # Training step
        optimizer.zero_grad()
        logits, voxel_emb, label_emb = model(x)
        loss = seg_criterion(logits, targets) + 0.05 * hyp_criterion(voxel_emb, targets, label_emb)
        loss.backward()
        optimizer.step()

        # Loss after
        logits, voxel_emb, label_emb = model(x)
        loss_after = seg_criterion(logits, targets) + 0.05 * hyp_criterion(voxel_emb, targets, label_emb)

        assert loss_after < loss_before, f"Loss did not decrease: {loss_before:.4f} -> {loss_after:.4f}"
```

### Step 11.2: Run integration tests

```bash
pytest tests/test_integration.py -v
```

Expected: All PASS

### Step 11.3: Run all tests

```bash
pytest tests/ -v --ignore=tests/test_dataset.py --ignore=tests/test_voxelizer.py
```

(Ignore data-dependent tests that require actual dataset)

### Step 11.4: Commit

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for hyperbolic training

Verify full forward/backward pass and loss decrease with training.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Summary

**Total Tasks:** 11
**New Files:** 14
**Modified Files:** 4

**Implementation Order:**
1. lorentz_ops.py (core math)
2. organ_hierarchy.py (hierarchy parsing)
3. label_embedding.py (class embeddings)
4. projection_head.py (feature projection)
5. lorentz_loss.py (ranking loss)
6. unet3d.py modification (return_features)
7. body_net.py (model wrapper)
8. config.py (new fields)
9. train.py (integration)
10. visualization tests
11. integration tests

**Run Full Test Suite:**
```bash
pytest tests/hyperbolic/ tests/test_body_net.py tests/test_unet3d_features.py tests/test_integration.py -v
```
