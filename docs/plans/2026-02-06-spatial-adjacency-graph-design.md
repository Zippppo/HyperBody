# Spatial Adjacency Graph for Curriculum Negative Mining

**Date:** 2026-02-06
**Status:** Design approved, ready for implementation

## Problem

The current tree-distance-based curriculum learning defines "hard negatives" purely by anatomical hierarchy distance. This misses a critical source of segmentation errors: **cross-system spatial neighbors**.

Example: `rib_left_1` and `lung` are physically adjacent (50% surface contact), but their tree distance is 9 (different body systems). The current system almost never samples `lung` as a hard negative for `rib_left_1`, yet boundary confusion between ribs and lungs is a primary source of segmentation errors (ribs predicted several times larger than GT).

## Solution

Extend the anatomical tree into a **directed graph** by adding spatial adjacency edges computed from training set ground truth labels. These edges act as "shortcuts" that reduce distance between physically adjacent but hierarchically distant organs.

### Core Formula

$$D_{final}(u, v) = \min \left( D_{tree}(u, v), \quad \lambda \cdot \frac{1}{\text{Contact}(u, v) + \epsilon} \right)$$

- `D_tree(u, v)`: Original symmetric tree distance (LCA-based)
- `Contact(u, v)`: Asymmetric contact ratio (fraction of organ u's surface touching organ v)
- `lambda`: Scale factor controlling spatial edge strength (default: 1.0)
- `epsilon`: Prevents division by zero (default: 0.01)

**Key property:** `D_final` is **asymmetric** — `D[rib, lung] != D[lung, rib]`, reflecting that small organs are more confused with large neighbors than vice versa.

**Why per-pair min instead of shortest path:** Full graph shortest path would create a "hub" problem — large organs (lung, liver) touching many structures would become transit stations, artificially shortening distances between unrelated organ pairs. Per-pair min only affects directly adjacent organs.

## Part 1: Spatial Contact Matrix Computation

### Definition

For organ u and organ v:

```
Contact(u, v) = (dilated_mask_u AND mask_v).sum() / mask_u.sum()
```

Where `dilated_mask_u` is organ u's mask after 3D dilation with radius 2.

### Vectorized Algorithm (One-Pass)

For each training sample:

```
1. labels [D,H,W] -> one_hot [C,1,D,H,W]    (F.one_hot + reshape, float32)
2. dilated = F.max_pool3d(one_hot, kernel_size=5, stride=1, padding=2)
3. squeeze -> dilated [C,D,H,W], original [C,D,H,W]
4. overlap = einsum('cdhw, kdhw -> ck')       -> [C, C]  (all pairs at once)
5. volume = original.sum(dim=[1,2,3])         -> [C]
```

Global accumulation across all training samples:

```
global_overlap += overlap
global_volume += volume
contact_matrix = global_overlap / global_volume.unsqueeze(1).clamp(min=1)
fill_diagonal_(0)
```

### Implementation Notes

- Use **cube structuring element** via `F.max_pool3d` (kernel=2r+1, stride=1, padding=r)
- Dilation radius: 2 voxels (kernel_size=5)
- One-hot encoding: `F.one_hot(labels, C).permute(3,0,1,2).unsqueeze(1).float()`
- Memory: 70 classes x 128^3 volume ~ 588MB float32 (acceptable, batch classes if needed)
- Skip classes with zero volume in a sample
- Output: `[C, C]` asymmetric matrix, diagonal = 0, values in [0, 1]

## Part 2: Distance Matrix Fusion

### Computation

```python
D_spatial = lambda_ / (contact_matrix + epsilon)   # [C, C]
D_final = torch.min(D_tree, D_spatial)              # [C, C] element-wise min
D_final.fill_diagonal_(0)                           # self-distance = 0
```

### Parameter Intuition (lambda=1.0)

| Contact(u,v) | Spatial Distance | Comparable Tree Level |
|---|---|---|
| 0.50 (50%) | 2.0 | Sibling (rib_left_1 vs rib_left_2) |
| 0.25 (25%) | 4.0 | Cousin (rib_left_1 vs rib_right_1) |
| 0.10 (10%) | 10.0 | Cross-system (no shortcut effect) |
| 0.02 (2%) | 50.0 | No effect (spatial >> tree) |
| 0 (none) | ~100 | No effect |

### Asymmetry Verification

The existing `LorentzTreeRankingLoss` indexes by anchor class:

```python
tree_dists = self.tree_dist_matrix[sampled_classes]  # [K, num_classes]
```

This naturally handles asymmetry: row u gives distances from anchor u to all negatives v. No code changes needed in the Loss class.

## Part 3: Visualization & Validation

All outputs to `docs/visualizations/spatial_adjacency/`.

### Visualization 1: Contact Matrix Heatmap
- 70x70 asymmetric `Contact(u, v)` heatmap
- Rows/columns grouped by body system
- Interactive HTML (plotly)

### Visualization 2: Distance Difference Map
- `D_diff = D_tree - D_final` (highlights where spatial edges shorten distance)
- Positive values = shortened by spatial adjacency
- Zero = unchanged

### Visualization 3: Sampling Probability Shift Plot
- For representative anchors (rib_left_1, liver, spine, gallbladder)
- Compute normalized sampling probability:
  $$P(v|u) = \frac{\exp(-D(u,v)/T)}{\sum_k \exp(-D(u,k)/T)}$$
- Side-by-side bar chart: tree-only vs graph probabilities
- Core evidence that spatial edges redirect hard negative sampling

## Part 4: Code Integration

### New Files
- `data/spatial_adjacency.py` — Contact matrix computation + distance fusion

### Modified Files
- `data/organ_hierarchy.py` — New `compute_graph_distance_matrix()` calling spatial_adjacency
- `train.py` — Support `hyp_distance_mode: "graph"`
- `configs/` — New config for graph mode

### New Config Parameters

```yaml
hyp_distance_mode: "graph"
spatial_dilation_radius: 2
spatial_lambda: 1.0
spatial_epsilon: 0.01
spatial_contact_matrix: null   # Optional: path to precomputed .pt file
```

### Precomputation
- Contact matrix computed once from training GT, saved as `.pt` file
- If `spatial_contact_matrix` path provided, load directly; otherwise compute and cache
- Avoids re-scanning entire dataset on every training run

### Loss Class Impact
- `LorentzTreeRankingLoss` requires **zero modifications**
- `D_final` is passed as the `tree_dist_matrix` parameter
- All changes are in data preprocessing, not in the training loop

## Literature Context

No existing work combines all of these elements:
- NonAdjLoss (Ganaye et al., MedIA 2019): organ adjacency for segmentation constraints, not ranking loss
- CLIP-Driven Universal Model: label embeddings capturing anatomy, not spatial contact
- SGCL: proximity-aware contrastive learning on graphs, not medical anatomy

The **asymmetric directed edges** from contact ratios and their use in **ranking loss negative sampling** appear to be novel contributions.
