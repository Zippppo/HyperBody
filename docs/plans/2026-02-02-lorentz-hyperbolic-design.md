# Lorentz Hyperbolic Embedding for HyperBody

## Overview

Add Lorentz (hyperboloid) model hyperbolic embeddings to HyperBody as an **auxiliary loss** alongside existing Dice+CE segmentation losses. The hyperbolic geometry encodes the anatomical organ hierarchy, encouraging the model to learn structure-aware representations.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Approach | Clean-slate, adapted to HyperBody | No existing Poincaré to migrate |
| Integration | Auxiliary loss from decoder features | Shared encoder, minimal disruption |
| Projection | 1x1 Conv (32->32) before exp_map0 | Decouples seg/hyp feature spaces |
| Embed dim | 32 (same as decoder channels) | Sufficient for 70 classes |
| Class exclusion | None (all 70 classes) | Class 0 (inside_body_empty) is valid |
| Loss weight | hyp_weight=0.05 | Additive, CE/Dice unchanged at 0.5/0.5 |
| Triplet margin | 0.1 | Standard value |
| Sampling | Random, 64 voxels per class | Simple MVP, hard mining later |
| Curvature | Fixed curv=1.0 | Not learnable in MVP |
| Distance fn | pointwise_dist (O(K)) | NOT pairwise_dist (O(K^2)) for loss |

## Architecture

```
                         UNet3D
  Encoder -> Bottleneck -> Decoder
                             |
                      d2: [B,32,H,W,D]
                             |
                +------------+-------------+
                |                          |
         Seg Head                  Hyperbolic Head
         Conv1x1 -> 70            Conv1x1(32->32)
                |                      |
                v                  exp_map0
         [B,70,H,W,D]                 |
          logits                       v
                |                [B,32,H,W,D]
                |               Lorentz embeddings
                v                      v
        Dice + CE Loss       LorentzRankingLoss
        weight: 0.5+0.5       weight: 0.05
                |                      |
                +----------+-----------+
                           v
                       Total Loss

total_loss = 0.5*ce + 0.5*dice + 0.05*lorentz_ranking
```

## File Structure

### New files

| File | Purpose |
|------|---------|
| `models/hyperbolic/__init__.py` | Export public API |
| `models/hyperbolic/lorentz_ops.py` | Core Lorentz math functions |
| `models/hyperbolic/projection_head.py` | LorentzProjectionHead |
| `models/hyperbolic/label_embedding.py` | LorentzLabelEmbedding |
| `models/hyperbolic/lorentz_loss.py` | LorentzRankingLoss |
| `models/body_net.py` | BodyNet wrapper (UNet3D + hyperbolic) |
| `data/organ_hierarchy.py` | Parse tree.json -> class depth mapping |
| `tests/hyperbolic/test_lorentz_ops.py` | Math operation tests |
| `tests/hyperbolic/test_label_embedding.py` | Embedding tests |
| `tests/hyperbolic/test_lorentz_loss.py` | Loss function tests |
| `tests/hyperbolic/test_visualization.py` | Visualization tests |

### Modified files

| File | Change |
|------|--------|
| `models/unet3d.py` | Add `return_features` parameter to forward() |
| `config.py` | Add hyperbolic config fields |
| `train.py` | Integrate BodyNet + LorentzRankingLoss |

## Component Details

### 1. lorentz_ops.py - Core Math

```python
def exp_map0(v: Tensor, curv=1.0, eps=1e-7) -> Tensor:
    """Tangent space -> Lorentz manifold. [.., D] -> [.., D]"""

def log_map0(x: Tensor, curv=1.0, eps=1e-7) -> Tensor:
    """Lorentz manifold -> Tangent space. [.., D] -> [.., D]"""

def pointwise_dist(x: Tensor, y: Tensor, curv=1.0, eps=1e-7) -> Tensor:
    """Element-wise geodesic distance. [.., D], [.., D] -> [..]"""

def pairwise_dist(x: Tensor, y: Tensor, curv=1.0, eps=1e-7) -> Tensor:
    """All-pairs geodesic distance. [N, D], [M, D] -> [N, M]"""

def distance_to_origin(x: Tensor, curv=1.0, eps=1e-7) -> Tensor:
    """Distance from origin. [.., D] -> [..]"""

def lorentz_to_poincare(x: Tensor, curv=1.0) -> Tensor:
    """Project to Poincare disk for visualization. [.., D] -> [.., D]"""
```

Numerical stability:
- sinh input clamped to [eps, asinh(2^15)]
- acosh input clamped to min=1+eps
- All norms clamped to min=eps

### 2. organ_hierarchy.py - Hierarchy Parsing

```python
def load_organ_hierarchy(tree_path: str, class_names: list[str]) -> dict[int, int]:
    """Parse Dataset/tree.json, return {class_idx: depth} mapping."""
```

Depth example from tree.json:
- human_body (depth 0, root, not a class)
- skeletal_system (depth 1, intermediate)
- axial_skeleton (depth 2, intermediate)
- thoracic_cage (depth 3, intermediate)
- ribs (depth 4, intermediate)
- ribs_left (depth 5, intermediate)
- rib_left_1 (depth 6, leaf) -> class index 23

### 3. label_embedding.py - LorentzLabelEmbedding

```python
class LorentzLabelEmbedding(nn.Module):
    def __init__(self, num_classes=70, embed_dim=32, curv=1.0,
                 class_depths=None, min_radius=0.1, max_radius=2.0):
        # Learnable tangent vectors: nn.Parameter([num_classes, embed_dim])
        # Initialized by hierarchy depth

    def forward(self) -> Tensor:
        """Returns Lorentz embeddings [num_classes, embed_dim]"""
        return exp_map0(self.tangent_embeddings, self.curv)
```

Initialization: `tangent_norm = min_radius + (max_radius - min_radius) * (depth - min_depth) / (max_depth - min_depth)`

### 4. projection_head.py - LorentzProjectionHead

```python
class LorentzProjectionHead(nn.Module):
    def __init__(self, in_channels=32, embed_dim=32, curv=1.0):
        self.conv = nn.Conv3d(in_channels, embed_dim, kernel_size=1)
        self.curv = curv

    def forward(self, x):
        # x: [B, C, H, W, D]
        x = self.conv(x)                      # [B, 32, H, W, D]
        x = x.permute(0, 2, 3, 4, 1)          # [B, H, W, D, 32]
        x = exp_map0(x, self.curv)             # Map to Lorentz
        x = x.permute(0, 4, 1, 2, 3)          # [B, 32, H, W, D]
        return x
```

### 5. lorentz_loss.py - LorentzRankingLoss

```python
class LorentzRankingLoss(nn.Module):
    def __init__(self, margin=0.1, curv=1.0, num_samples_per_class=64):
        ...

    def forward(self, voxel_emb, labels, label_emb):
        """
        voxel_emb:  [B, 32, H, W, D] - Lorentz voxel embeddings
        labels:     [B, H, W, D]     - ground truth
        label_emb:  [70, 32]         - Lorentz class embeddings

        Steps:
        1. Reshape voxel_emb to [N, 32], labels to [N]
        2. Sample K voxels per class present in batch
        3. For each sampled voxel:
           - anchor   = voxel embedding
           - positive = label_emb[true_class]
           - negative = label_emb[random_other_class]
        4. d_pos = pointwise_dist(anchors, positives)   # [K]
           d_neg = pointwise_dist(anchors, negatives)   # [K]
        5. loss = mean(max(0, margin + d_pos - d_neg))
        """
```

AMP: Force float32 for all distance computations.

### 6. body_net.py - Model Wrapper

```python
class BodyNet(nn.Module):
    def __init__(self, num_classes, base_channels, embed_dim, curv, class_depths, ...):
        self.unet = UNet3D(...)
        self.hyp_head = LorentzProjectionHead(base_channels, embed_dim, curv)
        self.label_emb = LorentzLabelEmbedding(num_classes, embed_dim, curv, class_depths)

    def forward(self, x):
        logits, d2 = self.unet(x, return_features=True)
        voxel_emb = self.hyp_head(d2)
        label_emb = self.label_emb()
        return logits, voxel_emb, label_emb
```

### 7. config.py - New Fields

```python
# Hyperbolic
hyp_embed_dim: int = 32
hyp_curv: float = 1.0
hyp_weight: float = 0.05
hyp_margin: float = 0.1
hyp_samples_per_class: int = 64
hyp_min_radius: float = 0.1
hyp_max_radius: float = 2.0
```

### 8. train.py - Integration

```python
total_loss = seg_loss + cfg.hyp_weight * hyp_loss
# = (0.5*ce + 0.5*dice) + 0.05 * lorentz_ranking
```

## Testing

### Unit Tests (tests/hyperbolic/)

**test_lorentz_ops.py:**
- exp_map0 / log_map0 inverse property
- Distance non-negativity
- Distance symmetry (pointwise)
- Triangle inequality
- Numerical stability at large norms
- Zero vector handling

**test_label_embedding.py:**
- Output shape [num_classes, embed_dim]
- Hierarchy depth ordering (deeper organs farther from origin)
- Gradient flow through embeddings

**test_lorentz_loss.py:**
- Loss outputs scalar
- Gradient flows to both voxel_emb and label_emb
- Perfect alignment produces low loss

### Visualization (docs/visualizations/hyperbolic/)

All outputs as interactive Plotly HTML:

1. **label_emb_tsne.html** - t-SNE of 70 class embeddings, colored by organ system
2. **label_emb_poincare.html** - Poincare disk projection (shallow=center, deep=edge)
3. **class_distance_matrix.html** - Pairwise distance heatmap with hierarchical clustering
4. **hierarchy_tree.html** - Tree with distance_to_origin annotations (sunburst/treemap)
5. **embedding_evolution.html** - Embedding movement during training (optional)

## Implementation Order

```
1. lorentz_ops.py          (core math, no dependencies)
   ↓
2. organ_hierarchy.py      (parse tree.json, no dependencies)
   ↓
3. label_embedding.py      (depends on 1, 2)
   ↓
4. projection_head.py      (depends on 1)
   ↓
5. lorentz_loss.py         (depends on 1)
   ↓
6. body_net.py + unet3d.py (depends on 3, 4)
   ↓
7. config.py + train.py    (depends on 5, 6)
   ↓
8. Tests + Visualization
```

## Configuration (MVP)

```yaml
hyperbolic:
  embed_dim: 32
  curv: 1.0             # Fixed, not learnable
  min_radius: 0.1       # Shallow organ init
  max_radius: 2.0       # Deep organ init
  loss:
    margin: 0.1
    weight: 0.05
    samples_per_class: 64
```
