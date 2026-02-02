# PaSCo-Body Hyperbolic Embedding Integration Plan

## Overview

将Hyperbolic Visual Embedding Learning（Liu et al. CVPR 2020）融入PaSCo-Body，使模型能够学习器官标签的层级语义结构。原文位于：ref_articles/Liu_Hyperbolic_Visual_Embedding_Learning_for_Zero-Shot_Recognition_CVPR_2020_paper.txt

**核心思想**：双曲空间（Poincaré Ball）天然适合表示层级结构——父类靠近圆心，子类靠近边界，同级类别相互远离。

**重要设计决策**：完全忽略class 0（体外点/outside_body），因为它提供的语义信息极少但数量极大，会严重影响hyperbolic loss的训练效率。

**与论文的差异说明**：
1. **Ranking Loss 变体**：论文原版是 `d(pos_label, voxel) - d(pos_label, neg_label)`，我们使用更标准的 triplet loss 形式 `d(voxel, pos_label) - d(voxel, neg_label)`，这在实践中更有效。
2. **简化 Möbius Transformation**：论文使用完整的 Möbius transformation layer，我们在 MVP 中使用 Conv3D + Exponential Map 简化版本。
3. **省略 Poincaré GloVe**：论文组合了 hierarchy embedding 和 semantic embedding，我们仅使用 hierarchy-based 初始化，对于器官分割任务已足够。

---

## 1. Dataset Class Mapping (72 classes)

根据 `Dataset/dataset_info.json`，完整的类别映射如下：

| ID | Name | Category |
|----|------|----------|
| 0 | outside_body | Background (IGNORED) |
| 1 | inside_body_empty | Internal cavity |
| 2 | liver | Visceral - Digestive |
| 3 | spleen | Visceral - Lymphatic |
| 4 | kidney_left | Visceral - Urinary |
| 5 | kidney_right | Visceral - Urinary |
| 6 | stomach | Visceral - Digestive |
| 7 | pancreas | Visceral - Digestive |
| 8 | gallbladder | Visceral - Digestive |
| 9 | urinary_bladder | Visceral - Urinary |
| 10 | prostate | Visceral - Reproductive |
| 11 | heart | Cardiovascular |
| 12 | brain | Neural |
| 13 | thyroid_gland | Endocrine |
| 14 | spinal_cord | Neural |
| 15 | lung | Respiratory |
| 16 | esophagus | Digestive - GI tract |
| 17 | trachea | Respiratory |
| 18 | small_bowel | Digestive - GI tract |
| 19 | duodenum | Digestive - GI tract |
| 20 | colon | Digestive - GI tract |
| 21 | adrenal_gland_left | Endocrine |
| 22 | adrenal_gland_right | Endocrine |
| 23 | spine | Skeletal - Axial |
| 24-35 | rib_left_1 to rib_left_12 | Skeletal - Ribs |
| 36-47 | rib_right_1 to rib_right_12 | Skeletal - Ribs |
| 48 | skull | Skeletal - Axial |
| 49 | sternum | Skeletal - Axial |
| 50 | costal_cartilages | Skeletal - Axial |
| 51 | scapula_left | Skeletal - Appendicular |
| 52 | scapula_right | Skeletal - Appendicular |
| 53 | clavicula_left | Skeletal - Appendicular |
| 54 | clavicula_right | Skeletal - Appendicular |
| 55 | humerus_left | Skeletal - Appendicular |
| 56 | humerus_right | Skeletal - Appendicular |
| 57 | hip_left | Skeletal - Appendicular |
| 58 | hip_right | Skeletal - Appendicular |
| 59 | femur_left | Skeletal - Appendicular |
| 60 | femur_right | Skeletal - Appendicular |
| 61 | gluteus_maximus_left | Muscular - Gluteal |
| 62 | gluteus_maximus_right | Muscular - Gluteal |
| 63 | gluteus_medius_left | Muscular - Gluteal |
| 64 | gluteus_medius_right | Muscular - Gluteal |
| 65 | gluteus_minimus_left | Muscular - Gluteal |
| 66 | gluteus_minimus_right | Muscular - Gluteal |
| 67 | autochthon_left | Muscular - Back |
| 68 | autochthon_right | Muscular - Back |
| 69 | iliopsoas_left | Muscular - Hip flexor |
| 70 | iliopsoas_right | Muscular - Hip flexor |
| 71 | rectum | Digestive - GI tract |

**注意**：实际使用71个类别（0-70），但数据集定义了72个（0-71）。需要确认 `N_CLASSES` 的实际值。

---

## 2. Organ Hierarchy Tree Structure

```
body (root, depth=0)
├── inside_body_empty (1, depth=1)
│
├── visceral_system (depth=1)
│   ├── digestive_accessory (depth=2)
│   │   ├── liver (2, depth=3)
│   │   ├── gallbladder (8, depth=3)
│   │   └── pancreas (7, depth=3)
│   │
│   ├── digestive_gi_tract (depth=2)
│   │   ├── stomach (6, depth=3)
│   │   ├── esophagus (16, depth=3)
│   │   ├── small_bowel (18, depth=3)
│   │   ├── duodenum (19, depth=3)
│   │   ├── colon (20, depth=3)
│   │   └── rectum (71, depth=3)
│   │
│   ├── urinary (depth=2)
│   │   ├── kidney_left (4, depth=3)
│   │   ├── kidney_right (5, depth=3)
│   │   └── urinary_bladder (9, depth=3)
│   │
│   ├── lymphatic (depth=2)
│   │   └── spleen (3, depth=3)
│   │
│   └── reproductive (depth=2)
│       └── prostate (10, depth=3)
│
├── cardiovascular_respiratory (depth=1)
│   ├── cardiovascular (depth=2)
│   │   └── heart (11, depth=3)
│   └── respiratory (depth=2)
│       ├── lung (15, depth=3)
│       └── trachea (17, depth=3)
│
├── neural_system (depth=1)
│   ├── brain (12, depth=2)
│   └── spinal_cord (14, depth=2)
│
├── endocrine_system (depth=1)
│   ├── thyroid_gland (13, depth=2)
│   ├── adrenal_gland_left (21, depth=2)
│   └── adrenal_gland_right (22, depth=2)
│
├── skeletal_system (depth=1)
│   ├── axial_skeleton (depth=2)
│   │   ├── skull (48, depth=3)
│   │   ├── spine (23, depth=3)
│   │   ├── sternum (49, depth=3)
│   │   ├── costal_cartilages (50, depth=3)
│   │   └── ribs (depth=3)
│   │       ├── rib_left_1..12 (24-35, depth=4)
│   │       └── rib_right_1..12 (36-47, depth=4)
│   │
│   └── appendicular_skeleton (depth=2)
│       ├── shoulder_girdle (depth=3)
│       │   ├── scapula_left (51, depth=4)
│       │   ├── scapula_right (52, depth=4)
│       │   ├── clavicula_left (53, depth=4)
│       │   └── clavicula_right (54, depth=4)
│       ├── upper_limb (depth=3)
│       │   ├── humerus_left (55, depth=4)
│       │   └── humerus_right (56, depth=4)
│       ├── pelvic_girdle (depth=3)
│       │   ├── hip_left (57, depth=4)
│       │   └── hip_right (58, depth=4)
│       └── lower_limb (depth=3)
│           ├── femur_left (59, depth=4)
│           └── femur_right (60, depth=4)
│
└── muscular_system (depth=1)
    ├── gluteal_muscles (depth=2)
    │   ├── gluteus_maximus_left (61, depth=3)
    │   ├── gluteus_maximus_right (62, depth=3)
    │   ├── gluteus_medius_left (63, depth=3)
    │   ├── gluteus_medius_right (64, depth=3)
    │   ├── gluteus_minimus_left (65, depth=3)
    │   └── gluteus_minimus_right (66, depth=3)
    ├── back_muscles (depth=2)
    │   ├── autochthon_left (67, depth=3)
    │   └── autochthon_right (68, depth=3)
    └── hip_flexors (depth=2)
        ├── iliopsoas_left (69, depth=3)
        └── iliopsoas_right (70, depth=3)
```

---

## 3. Mathematical Foundations

### 3.1 Poincaré Ball Definition

Poincaré Ball 是一个n维单位球内的流形：
```
D^n = {x ∈ R^n : ||x|| < 1}
```

Riemannian度量：
```
g_x^D = λ_x^2 * g^E,  where λ_x = 2 / (1 - ||x||^2)
```

### 3.2 Exponential Map (Eq. 1 from paper)

将欧几里得空间的切向量投影到Poincaré Ball（从原点出发）：

```
exp_0(v) = tanh(||v||) * v / ||v||
```

**PyTorch实现**：
```python
def exp_map_zero(v, eps=1e-5):
    """
    Exponential map from origin to Poincaré ball.
    Args:
        v: Tensor of shape (..., d) in tangent space at origin
    Returns:
        Tensor in Poincaré ball with ||result|| < 1
    """
    v_norm = torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=eps)
    return torch.tanh(v_norm) * v / v_norm
```

### 3.3 Poincaré Distance (Eq. 3 from paper)

两点间的双曲距离：

```
d_D(x, y) = arccosh(1 + 2 * ||x - y||^2 / ((1 - ||x||^2) * (1 - ||y||^2)))
```

**PyTorch实现**：
```python
def poincare_distance(x, y, eps=1e-5):
    """
    Poincaré distance between two points in the ball.
    Args:
        x, y: Tensors of shape (..., d)
    Returns:
        Distance tensor of shape (...)
    """
    diff_norm_sq = torch.sum((x - y) ** 2, dim=-1)
    x_norm_sq = torch.sum(x ** 2, dim=-1)
    y_norm_sq = torch.sum(y ** 2, dim=-1)

    numerator = 2 * diff_norm_sq
    denominator = (1 - x_norm_sq) * (1 - y_norm_sq)

    # Clamp for numerical stability
    arg = 1 + numerator / torch.clamp(denominator, min=eps)
    dist = torch.acosh(torch.clamp(arg, min=1.0 + eps))
    return dist
```

### 3.4 Project to Ball

确保点严格在球内：

```python
def project_to_ball(x, eps=1e-5):
    """
    Project points to inside the Poincaré ball (norm < 1).
    """
    max_norm = 1 - eps
    norm = torch.norm(x, dim=-1, keepdim=True)
    return x * torch.clamp(max_norm / norm, max=1.0)
```

### 3.5 Ranking Loss (Triplet Loss 变体)

**论文原版 (Eq. 11)**：
```
L = max(0, δ + d_D(t_cI, h_I) - d_D(t_cI, t_cI^-))
```
含义：正样本标签与voxel的距离应小于正样本标签与负样本标签的距离。

**我们的实现（标准 Triplet Loss 形式）**：
```
L = max(0, δ + d_D(h_I, t_cI) - d_D(h_I, t_cI^-))
```
含义：voxel与正样本标签的距离应小于voxel与负样本标签的距离。

其中：
- `h_I`: voxel的hyperbolic embedding
- `t_cI`: 对应标签的hyperbolic embedding（正样本）
- `t_cI^-`: 随机负样本的标签embedding
- `δ`: margin（论文使用1.0，我们使用0.1作为辅助loss）

**选择理由**：标准 triplet loss 形式更直观，且在实践中表现稳定。两种形式都能使 voxel embedding 靠近正确的标签 embedding。

---

## 4. Architecture Diagram

```
Input: Binary Occupancy Grid [B, 1, H, W, D]
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│              DenseUNet3D (Modified)                      │
│                                                          │
│  Encoder → Bottleneck → Decoder                         │
│                              │                           │
│                    decoder_features (d1)                 │
│                         [B, 32, H, W, D]                 │
│                              │                           │
│                    ┌─────────┴─────────┐                 │
│                    ▼                   ▼                 │
│            ┌─────────────┐     ┌──────────────────┐     │
│            │  out_conv   │     │ HyperbolicHead   │     │
│            │  32 → 72    │     │ 32 → 32 (hyp)    │     │
│            └──────┬──────┘     └────────┬─────────┘     │
└───────────────────┼─────────────────────┼───────────────┘
                    │                     │
                    ▼                     ▼
              logits [B,72,H,W,D]   voxel_emb [B,32,H,W,D]
                    │                     │ (in Poincaré Ball)
                    │                     │
                    ▼                     ▼
              CE Loss              ┌──────────────────┐
              (all classes)        │ Label Embeddings │
                    │              │ [72, 32] (hyp)   │
                    │              │ (class 0 ignored)│
                    │              └────────┬─────────┘
                    │                       │
                    │              Poincaré Distance + Rank Loss
                    │              (ONLY for classes 1-71)
                    │                       │
                    └───────────┬───────────┘
                                ▼
                    Total Loss = CE + λ * Hyperbolic Loss
```

---

## 5. Implementation Files

### 5.1 New Files to Create (7 files)

#### File 1: `pasco/data/body/organ_hierarchy.py`

```python
"""
Organ class hierarchy definition for hyperbolic embedding.
Based on anatomical structure of human body organs.
"""

# Complete class names from dataset_info.json
CLASS_NAMES = [
    "outside_body",          # 0 - IGNORED in hyperbolic loss
    "inside_body_empty",     # 1
    "liver",                 # 2
    "spleen",                # 3
    "kidney_left",           # 4
    "kidney_right",          # 5
    "stomach",               # 6
    "pancreas",              # 7
    "gallbladder",           # 8
    "urinary_bladder",       # 9
    "prostate",              # 10
    "heart",                 # 11
    "brain",                 # 12
    "thyroid_gland",         # 13
    "spinal_cord",           # 14
    "lung",                  # 15
    "esophagus",             # 16
    "trachea",               # 17
    "small_bowel",           # 18
    "duodenum",              # 19
    "colon",                 # 20
    "adrenal_gland_left",    # 21
    "adrenal_gland_right",   # 22
    "spine",                 # 23
    "rib_left_1",            # 24
    "rib_left_2",            # 25
    "rib_left_3",            # 26
    "rib_left_4",            # 27
    "rib_left_5",            # 28
    "rib_left_6",            # 29
    "rib_left_7",            # 30
    "rib_left_8",            # 31
    "rib_left_9",            # 32
    "rib_left_10",           # 33
    "rib_left_11",           # 34
    "rib_left_12",           # 35
    "rib_right_1",           # 36
    "rib_right_2",           # 37
    "rib_right_3",           # 38
    "rib_right_4",           # 39
    "rib_right_5",           # 40
    "rib_right_6",           # 41
    "rib_right_7",           # 42
    "rib_right_8",           # 43
    "rib_right_9",           # 44
    "rib_right_10",          # 45
    "rib_right_11",          # 46
    "rib_right_12",          # 47
    "skull",                 # 48
    "sternum",               # 49
    "costal_cartilages",     # 50
    "scapula_left",          # 51
    "scapula_right",         # 52
    "clavicula_left",        # 53
    "clavicula_right",       # 54
    "humerus_left",          # 55
    "humerus_right",         # 56
    "hip_left",              # 57
    "hip_right",             # 58
    "femur_left",            # 59
    "femur_right",           # 60
    "gluteus_maximus_left",  # 61
    "gluteus_maximus_right", # 62
    "gluteus_medius_left",   # 63
    "gluteus_medius_right",  # 64
    "gluteus_minimus_left",  # 65
    "gluteus_minimus_right", # 66
    "autochthon_left",       # 67
    "autochthon_right",      # 68
    "iliopsoas_left",        # 69
    "iliopsoas_right",       # 70
    "rectum",                # 71
]

N_CLASSES = len(CLASS_NAMES)  # 72

# Hierarchy tree structure
# Each node has: name, class_id (if leaf), children (if internal)
ORGAN_HIERARCHY = {
    "name": "body",
    "children": [
        {"name": "inside_body_empty", "class_id": 1},
        {
            "name": "visceral_system",
            "children": [
                {
                    "name": "digestive_accessory",
                    "children": [
                        {"name": "liver", "class_id": 2},
                        {"name": "gallbladder", "class_id": 8},
                        {"name": "pancreas", "class_id": 7},
                    ]
                },
                {
                    "name": "digestive_gi_tract",
                    "children": [
                        {"name": "stomach", "class_id": 6},
                        {"name": "esophagus", "class_id": 16},
                        {"name": "small_bowel", "class_id": 18},
                        {"name": "duodenum", "class_id": 19},
                        {"name": "colon", "class_id": 20},
                        {"name": "rectum", "class_id": 71},
                    ]
                },
                {
                    "name": "urinary",
                    "children": [
                        {"name": "kidney_left", "class_id": 4},
                        {"name": "kidney_right", "class_id": 5},
                        {"name": "urinary_bladder", "class_id": 9},
                    ]
                },
                {"name": "spleen", "class_id": 3},
                {"name": "prostate", "class_id": 10},
            ]
        },
        {
            "name": "cardiovascular_respiratory",
            "children": [
                {"name": "heart", "class_id": 11},
                {"name": "lung", "class_id": 15},
                {"name": "trachea", "class_id": 17},
            ]
        },
        {
            "name": "neural_system",
            "children": [
                {"name": "brain", "class_id": 12},
                {"name": "spinal_cord", "class_id": 14},
            ]
        },
        {
            "name": "endocrine_system",
            "children": [
                {"name": "thyroid_gland", "class_id": 13},
                {"name": "adrenal_gland_left", "class_id": 21},
                {"name": "adrenal_gland_right", "class_id": 22},
            ]
        },
        {
            "name": "skeletal_system",
            "children": [
                {
                    "name": "axial_skeleton",
                    "children": [
                        {"name": "skull", "class_id": 48},
                        {"name": "spine", "class_id": 23},
                        {"name": "sternum", "class_id": 49},
                        {"name": "costal_cartilages", "class_id": 50},
                        {
                            "name": "ribs_left",
                            "children": [
                                {"name": f"rib_left_{i}", "class_id": 23 + i}
                                for i in range(1, 13)
                            ]
                        },
                        {
                            "name": "ribs_right",
                            "children": [
                                {"name": f"rib_right_{i}", "class_id": 35 + i}
                                for i in range(1, 13)
                            ]
                        },
                    ]
                },
                {
                    "name": "appendicular_skeleton",
                    "children": [
                        {
                            "name": "shoulder_girdle",
                            "children": [
                                {"name": "scapula_left", "class_id": 51},
                                {"name": "scapula_right", "class_id": 52},
                                {"name": "clavicula_left", "class_id": 53},
                                {"name": "clavicula_right", "class_id": 54},
                            ]
                        },
                        {
                            "name": "upper_limb",
                            "children": [
                                {"name": "humerus_left", "class_id": 55},
                                {"name": "humerus_right", "class_id": 56},
                            ]
                        },
                        {
                            "name": "pelvic_girdle",
                            "children": [
                                {"name": "hip_left", "class_id": 57},
                                {"name": "hip_right", "class_id": 58},
                            ]
                        },
                        {
                            "name": "lower_limb",
                            "children": [
                                {"name": "femur_left", "class_id": 59},
                                {"name": "femur_right", "class_id": 60},
                            ]
                        },
                    ]
                },
            ]
        },
        {
            "name": "muscular_system",
            "children": [
                {
                    "name": "gluteal_muscles",
                    "children": [
                        {"name": "gluteus_maximus_left", "class_id": 61},
                        {"name": "gluteus_maximus_right", "class_id": 62},
                        {"name": "gluteus_medius_left", "class_id": 63},
                        {"name": "gluteus_medius_right", "class_id": 64},
                        {"name": "gluteus_minimus_left", "class_id": 65},
                        {"name": "gluteus_minimus_right", "class_id": 66},
                    ]
                },
                {
                    "name": "back_muscles",
                    "children": [
                        {"name": "autochthon_left", "class_id": 67},
                        {"name": "autochthon_right", "class_id": 68},
                    ]
                },
                {
                    "name": "hip_flexors",
                    "children": [
                        {"name": "iliopsoas_left", "class_id": 69},
                        {"name": "iliopsoas_right", "class_id": 70},
                    ]
                },
            ]
        },
    ]
}


def get_class_depths(hierarchy=None, current_depth=0):
    """
    Recursively compute depth for each class_id in the hierarchy.

    Returns:
        dict: {class_id: depth} mapping
    """
    if hierarchy is None:
        hierarchy = ORGAN_HIERARCHY

    depths = {}

    if "class_id" in hierarchy:
        depths[hierarchy["class_id"]] = current_depth

    if "children" in hierarchy:
        for child in hierarchy["children"]:
            child_depths = get_class_depths(child, current_depth + 1)
            depths.update(child_depths)

    return depths


def get_max_depth(hierarchy=None):
    """Get the maximum depth in the hierarchy tree."""
    depths = get_class_depths(hierarchy)
    return max(depths.values()) if depths else 0


# Pre-computed class depths for efficiency
CLASS_DEPTHS = get_class_depths()
MAX_DEPTH = get_max_depth()


if __name__ == "__main__":
    # Test the hierarchy
    print(f"Number of classes: {N_CLASSES}")
    print(f"Max depth: {MAX_DEPTH}")
    print(f"\nClass depths:")
    for cls_id, depth in sorted(CLASS_DEPTHS.items()):
        print(f"  {cls_id:2d}: {CLASS_NAMES[cls_id]:25s} (depth={depth})")
```

#### File 2: `pasco/models/hyperbolic/__init__.py`

```python
"""
Hyperbolic embedding modules for PaSCo-Body.
"""

from .poincare_ops import exp_map_zero, poincare_distance, project_to_ball
from .label_embedding import HyperbolicLabelEmbedding
from .projection_head import HyperbolicProjectionHead

__all__ = [
    "exp_map_zero",
    "poincare_distance",
    "project_to_ball",
    "HyperbolicLabelEmbedding",
    "HyperbolicProjectionHead",
]
```

#### File 3: `pasco/models/hyperbolic/poincare_ops.py`

```python
"""
Poincaré ball operations for hyperbolic embeddings.
Based on: Liu et al. "Hyperbolic Visual Embedding Learning for Zero-Shot Recognition" CVPR 2020
"""

import torch


def exp_map_zero(v, eps=1e-5):
    """
    Exponential map from origin to Poincaré ball.

    Formula (Eq. 1 from paper):
        exp_0(v) = tanh(||v||) * v / ||v||

    Args:
        v: Tensor of shape (..., d) in tangent space at origin (Euclidean)
        eps: Small value for numerical stability

    Returns:
        Tensor in Poincaré ball with ||result|| < 1
    """
    v_norm = torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=eps)
    return torch.tanh(v_norm) * v / v_norm


def poincare_distance(x, y, eps=1e-5):
    """
    Poincaré distance between two points in the ball.

    Formula (Eq. 3 from paper):
        d_D(x, y) = arccosh(1 + 2 * ||x - y||^2 / ((1 - ||x||^2) * (1 - ||y||^2)))

    Args:
        x, y: Tensors of shape (..., d) in Poincaré ball
        eps: Small value for numerical stability

    Returns:
        Distance tensor of shape (...)
    """
    diff_norm_sq = torch.sum((x - y) ** 2, dim=-1)
    x_norm_sq = torch.sum(x ** 2, dim=-1)
    y_norm_sq = torch.sum(y ** 2, dim=-1)

    numerator = 2 * diff_norm_sq
    denominator = (1 - x_norm_sq) * (1 - y_norm_sq)

    # Clamp for numerical stability
    arg = 1 + numerator / torch.clamp(denominator, min=eps)
    dist = torch.acosh(torch.clamp(arg, min=1.0 + eps))

    return dist


def project_to_ball(x, eps=1e-5):
    """
    Project points to strictly inside the Poincaré ball (||x|| < 1).

    Args:
        x: Tensor of shape (..., d)
        eps: Margin from boundary

    Returns:
        Projected tensor with ||result|| < 1 - eps
    """
    max_norm = 1 - eps
    norm = torch.norm(x, dim=-1, keepdim=True)
    # Only scale down if norm >= max_norm
    scale = torch.clamp(max_norm / norm, max=1.0)
    return x * scale


if __name__ == "__main__":
    # Unit tests
    print("Testing Poincaré operations...")

    # Test exp_map_zero
    v = torch.randn(10, 32)
    h = exp_map_zero(v)
    assert (h.norm(dim=-1) < 1).all(), "exp_map_zero: points should be inside ball"
    print("  exp_map_zero: PASS")

    # Test poincare_distance
    d = poincare_distance(h[:5], h[5:])
    assert (d >= 0).all(), "poincare_distance: should be non-negative"
    d_sym = poincare_distance(h[5:], h[:5])
    assert torch.allclose(d, d_sym, atol=1e-5), "poincare_distance: should be symmetric"
    print("  poincare_distance: PASS")

    # Test project_to_ball
    x_outside = torch.randn(10, 32) * 2  # Some points likely outside
    p = project_to_ball(x_outside)
    assert (p.norm(dim=-1) < 1).all(), "project_to_ball: should project inside"
    print("  project_to_ball: PASS")

    print("All tests passed!")
```

#### File 4: `pasco/models/hyperbolic/label_embedding.py`

```python
"""
Hyperbolic label embedding module.
Embeds class labels in Poincaré ball based on hierarchy structure.
"""

import torch
import torch.nn as nn

from .poincare_ops import project_to_ball


class HyperbolicLabelEmbedding(nn.Module):
    """
    Learnable hyperbolic embeddings for organ classes.

    Embeddings are initialized based on hierarchy depth:
    - Root/shallow classes are near the center (small radius)
    - Deep/leaf classes are near the boundary (large radius)

    IMPORTANT: 由于 embeddings 是可学习参数，梯度更新可能使其移出 Poincaré ball。
    因此在 forward() 中始终调用 project_to_ball() 确保 embeddings 保持在球内。

    Args:
        n_classes: Number of classes (72 for body dataset)
        embed_dim: Embedding dimension (default 32)
        ignore_class: Class to ignore in loss (default 0 = outside_body)
        min_radius: Minimum radius for embeddings (default 0.1)
        max_radius: Maximum radius for embeddings (default 0.8)
    """

    def __init__(
        self,
        n_classes=72,
        embed_dim=32,
        ignore_class=0,
        min_radius=0.1,
        max_radius=0.8,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.embed_dim = embed_dim
        self.ignore_class = ignore_class
        self.min_radius = min_radius
        self.max_radius = max_radius

        # Learnable embeddings for all classes
        self.embeddings = nn.Parameter(torch.zeros(n_classes, embed_dim))

        # Initialize based on hierarchy
        self._init_from_hierarchy()

    def _init_from_hierarchy(self):
        """Initialize embeddings based on hierarchy depth."""
        from pasco.data.body.organ_hierarchy import CLASS_DEPTHS, MAX_DEPTH

        with torch.no_grad():
            for class_id in range(self.n_classes):
                if class_id == self.ignore_class:
                    # Ignored class at origin (won't be used)
                    self.embeddings.data[class_id] = 0
                else:
                    # Get depth (default to max if not in hierarchy)
                    depth = CLASS_DEPTHS.get(class_id, MAX_DEPTH)

                    # Compute radius based on depth
                    # Deeper = larger radius (closer to boundary)
                    if MAX_DEPTH > 0:
                        radius = self.min_radius + (self.max_radius - self.min_radius) * (depth / MAX_DEPTH)
                    else:
                        radius = (self.min_radius + self.max_radius) / 2

                    # Random direction on unit sphere
                    direction = torch.randn(self.embed_dim)
                    direction = direction / direction.norm()

                    # Set embedding
                    self.embeddings.data[class_id] = radius * direction

            # Ensure all embeddings are inside the ball
            self.embeddings.data = project_to_ball(self.embeddings.data)

    def forward(self, class_indices=None):
        """
        Get embeddings for given class indices.

        Args:
            class_indices: Optional tensor of class indices. If None, return all embeddings.

        Returns:
            Embeddings tensor (always projected to ensure inside Poincaré ball)
        """
        # 始终投影以确保 embeddings 在球内（梯度更新可能使其移出）
        embeddings = project_to_ball(self.embeddings)

        if class_indices is None:
            return embeddings
        return embeddings[class_indices]

    def project_embeddings(self):
        """
        Manually project embeddings back to Poincaré ball.
        Call this after optimizer.step() if not using forward() projection.
        """
        with torch.no_grad():
            self.embeddings.data = project_to_ball(self.embeddings.data)


if __name__ == "__main__":
    # Test
    print("Testing HyperbolicLabelEmbedding...")

    emb = HyperbolicLabelEmbedding(n_classes=72, embed_dim=32, ignore_class=0)
    e = emb()

    assert e.shape == (72, 32), f"Expected (72, 32), got {e.shape}"
    assert (e[1:].norm(dim=-1) < 1).all(), "Non-ignored embeddings should be inside ball"
    assert e[0].norm() < 1e-5, "Ignored class should be at origin"

    print(f"  Shape: {e.shape}")
    print(f"  Norms range: [{e[1:].norm(dim=-1).min():.3f}, {e[1:].norm(dim=-1).max():.3f}]")

    # Test that gradient updates are handled
    e.sum().backward()
    print("  Backward: PASS")
    print("  PASS")
```

#### File 5: `pasco/models/hyperbolic/projection_head.py`

```python
"""
Hyperbolic projection head.
Projects CNN features to Poincaré ball.
"""

import torch
import torch.nn as nn

from .poincare_ops import exp_map_zero, project_to_ball


class HyperbolicProjectionHead(nn.Module):
    """
    Projects dense 3D features to Poincaré ball.

    Architecture:
        1. 1x1 Conv3D to project to embedding dimension
        2. Exponential map to project to Poincaré ball
        3. Project to ball (ensure inside boundary)

    Args:
        in_channels: Input feature channels (default 32, decoder output)
        embed_dim: Hyperbolic embedding dimension (default 32)
    """

    def __init__(self, in_channels=32, embed_dim=32):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Projection layer (1x1 conv)
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=1, bias=True)

        # Initialize projection weights
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, features):
        """
        Project features to Poincaré ball.

        Args:
            features: [B, C, H, W, D] decoder features in Euclidean space

        Returns:
            embeddings: [B, embed_dim, H, W, D] in Poincaré ball
        """
        # Project to embedding dimension
        x = self.proj(features)  # [B, embed_dim, H, W, D]

        # Reshape for Poincaré operations: [B, H, W, D, embed_dim]
        x = x.permute(0, 2, 3, 4, 1)

        # Project to Poincaré ball
        x = exp_map_zero(x)
        x = project_to_ball(x)

        # Back to [B, embed_dim, H, W, D]
        x = x.permute(0, 4, 1, 2, 3)

        return x


if __name__ == "__main__":
    # Test
    print("Testing HyperbolicProjectionHead...")

    head = HyperbolicProjectionHead(in_channels=32, embed_dim=32)

    # Dummy input
    features = torch.randn(2, 32, 16, 16, 16)
    out = head(features)

    assert out.shape == (2, 32, 16, 16, 16), f"Expected (2, 32, 16, 16, 16), got {out.shape}"

    # Check all points are inside the ball
    out_flat = out.permute(0, 2, 3, 4, 1).reshape(-1, 32)
    norms = out_flat.norm(dim=-1)
    assert (norms < 1).all(), "All embeddings should be inside Poincaré ball"

    print(f"  Output shape: {out.shape}")
    print(f"  Norms range: [{norms.min():.3f}, {norms.max():.3f}]")
    print("  PASS")
```

#### File 6: `pasco/loss/hyperbolic_loss.py`

```python
"""
Hyperbolic ranking loss for organ segmentation.
Based on: Liu et al. "Hyperbolic Visual Embedding Learning for Zero-Shot Recognition" CVPR 2020
"""

import torch
import torch.nn as nn

from pasco.models.hyperbolic.poincare_ops import poincare_distance


class HyperbolicRankingLoss(nn.Module):
    """
    Margin-based ranking loss in Poincaré ball.

    Loss (Eq. 11 from paper):
        L = max(0, margin + d(h_I, t_pos) - d(h_I, t_neg))

    where:
        - h_I: voxel embedding in Poincaré ball
        - t_pos: positive label embedding (ground truth)
        - t_neg: negative label embedding (random sample)
        - d(): Poincaré distance

    Args:
        margin: Ranking loss margin (default 0.1)
        ignore_classes: List of classes to ignore (default [0, 255])
    """

    def __init__(self, margin=0.1, ignore_classes=None):
        super().__init__()
        self.margin = margin
        self.ignore_classes = set(ignore_classes) if ignore_classes else {0, 255}

    def forward(self, voxel_embeddings, labels, label_embeddings):
        """
        Compute hyperbolic ranking loss.

        Args:
            voxel_embeddings: [B, D, H, W, Z] hyperbolic voxel embeddings
            labels: [B, H, W, Z] ground truth labels
            label_embeddings: [N_classes, D] hyperbolic label embeddings

        Returns:
            Scalar loss value
        """
        B, D, H, W, Z = voxel_embeddings.shape
        N_classes = label_embeddings.shape[0]
        device = voxel_embeddings.device

        # Flatten spatial dimensions
        # [B, D, H, W, Z] -> [B, H, W, Z, D] -> [B*H*W*Z, D]
        voxel_emb = voxel_embeddings.permute(0, 2, 3, 4, 1).reshape(-1, D)
        labels_flat = labels.reshape(-1)

        # Create valid mask: exclude ignored classes
        valid_mask = torch.ones_like(labels_flat, dtype=torch.bool)
        for ignore_cls in self.ignore_classes:
            valid_mask &= (labels_flat != ignore_cls)

        # Filter to valid voxels only
        voxel_emb = voxel_emb[valid_mask]
        labels_valid = labels_flat[valid_mask]

        if voxel_emb.shape[0] == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Get positive embeddings
        pos_emb = label_embeddings[labels_valid]  # [N_valid, D]

        # Compute positive distances
        d_pos = poincare_distance(voxel_emb, pos_emb)  # [N_valid]

        # Sample random negatives (different from positive class)
        # Create list of valid classes for negative sampling
        valid_classes = [i for i in range(N_classes) if i not in self.ignore_classes]
        valid_classes_tensor = torch.tensor(valid_classes, device=device)

        # Random indices into valid_classes
        rand_idx = torch.randint(0, len(valid_classes), (voxel_emb.shape[0],), device=device)
        neg_classes = valid_classes_tensor[rand_idx]

        # Ensure negative != positive
        same_mask = (neg_classes == labels_valid)
        if same_mask.any():
            # Shift by 1 within valid classes
            neg_classes[same_mask] = valid_classes_tensor[(rand_idx[same_mask] + 1) % len(valid_classes)]

        neg_emb = label_embeddings[neg_classes]  # [N_valid, D]

        # Compute negative distances
        d_neg = poincare_distance(voxel_emb, neg_emb)  # [N_valid]

        # Ranking loss: max(0, margin + d_pos - d_neg)
        loss = torch.relu(self.margin + d_pos - d_neg).mean()

        return loss


if __name__ == "__main__":
    # Test
    print("Testing HyperbolicRankingLoss...")

    loss_fn = HyperbolicRankingLoss(margin=0.1, ignore_classes=[0, 255])

    # Dummy inputs
    B, D, H, W, Z = 2, 32, 8, 8, 8
    N_classes = 72

    voxel_emb = torch.randn(B, D, H, W, Z) * 0.5  # Keep inside ball roughly
    voxel_emb = voxel_emb / voxel_emb.norm(dim=1, keepdim=True).clamp(min=1.0)  # Normalize

    labels = torch.randint(0, N_classes, (B, H, W, Z))

    label_emb = torch.randn(N_classes, D) * 0.5
    label_emb = label_emb / label_emb.norm(dim=1, keepdim=True).clamp(min=1.0)

    loss = loss_fn(voxel_emb, labels, label_emb)

    print(f"  Loss value: {loss.item():.4f}")
    assert loss.requires_grad, "Loss should require grad"

    # Test backward
    loss.backward()
    print("  Backward: PASS")
    print("  PASS")
```

#### File 7: `pasco/models/body_net_hyperbolic.py`

```python
"""
BodyNet with hyperbolic embedding learning.
Extends BodyNet to add hyperbolic projection head and ranking loss.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl

from pasco.models.body_net import BodyNet
from pasco.models.hyperbolic import HyperbolicLabelEmbedding, HyperbolicProjectionHead
from pasco.loss.hyperbolic_loss import HyperbolicRankingLoss


class BodyNetHyperbolic(BodyNet):
    """
    BodyNet with hyperbolic embedding learning for hierarchical organ segmentation.

    Extends BodyNet by adding:
        - HyperbolicProjectionHead: projects decoder features to Poincaré ball
        - HyperbolicLabelEmbedding: organ class embeddings in Poincaré ball
        - HyperbolicRankingLoss: ranking loss based on Poincaré distance

    Total loss = CE_loss + hyperbolic_weight * hyperbolic_loss

    Args:
        n_classes: Number of classes (72 for body dataset)
        embed_dim: Hyperbolic embedding dimension (default 32)
        hyperbolic_weight: Weight for hyperbolic loss (default 0.1)
        margin: Ranking loss margin (default 0.1)
        **kwargs: Additional arguments passed to BodyNet
    """

    def __init__(
        self,
        n_classes=72,
        embed_dim=32,
        hyperbolic_weight=0.1,
        margin=0.1,
        **kwargs
    ):
        super().__init__(n_classes=n_classes, **kwargs)

        self.embed_dim = embed_dim
        self.hyperbolic_weight = hyperbolic_weight

        # Get base_channels from kwargs (default 32)
        base_channels = kwargs.get('base_channels', 32)

        # Hyperbolic projection head
        self.hyp_head = HyperbolicProjectionHead(
            in_channels=base_channels,
            embed_dim=embed_dim
        )

        # Hyperbolic label embeddings
        self.label_emb = HyperbolicLabelEmbedding(
            n_classes=n_classes,
            embed_dim=embed_dim,
            ignore_class=0  # Ignore outside_body
        )

        # Hyperbolic ranking loss
        self.hyp_loss_fn = HyperbolicRankingLoss(
            margin=margin,
            ignore_classes=[0, 255]  # Ignore outside_body and invalid
        )

    def forward(self, x):
        """Forward pass returning logits only (for inference)."""
        return self.model(x)

    def forward_with_hyperbolic(self, x):
        """
        Forward pass returning both logits and hyperbolic embeddings.

        Returns:
            logits: [B, n_classes, H, W, D]
            voxel_embeddings: [B, embed_dim, H, W, D] in Poincaré ball
        """
        # Get logits and decoder features
        # Note: self.model is DenseUNet3D or DenseUNet3DLight, defined in BodyNet.__init__
        logits, decoder_features = self.model.forward_with_features(x)

        # Project to hyperbolic space
        voxel_embeddings = self.hyp_head(decoder_features)

        return logits, voxel_embeddings

    def training_step(self, batch, batch_idx):
        """Training step with combined CE and hyperbolic loss."""
        occupancy = batch["occupancy"]  # [B, 1, H, W, D]
        labels = batch["labels"]        # [B, H, W, D]

        # Forward pass with hyperbolic embeddings
        logits, voxel_emb = self.forward_with_hyperbolic(occupancy)

        # Cross-entropy loss (on all classes)
        ce_loss = self.compute_loss(logits, labels)

        # Hyperbolic ranking loss (excluding class 0)
        label_emb = self.label_emb()  # [n_classes, embed_dim]
        hyp_loss = self.hyp_loss_fn(voxel_emb, labels, label_emb)

        # Total loss
        total_loss = ce_loss + self.hyperbolic_weight * hyp_loss

        # Compute accuracy
        pred = logits.argmax(dim=1)
        valid_mask = labels != self.ignore_index
        accuracy = (pred[valid_mask] == labels[valid_mask]).float().mean()

        # Log metrics
        self.log("train/loss", total_loss, prog_bar=True, sync_dist=True)
        self.log("train/ce_loss", ce_loss, sync_dist=True)
        self.log("train/hyp_loss", hyp_loss, sync_dist=True)
        self.log("train/accuracy", accuracy, prog_bar=True, sync_dist=True)

        return total_loss

    def on_validation_epoch_start(self):
        """Reset IoU accumulators at the start of validation."""
        self.val_iou_sum = None
        self.val_iou_count = None

    def validation_step(self, batch, batch_idx):
        """Validation step with combined losses."""
        occupancy = batch["occupancy"]
        labels = batch["labels"]

        # Forward pass with hyperbolic embeddings
        logits, voxel_emb = self.forward_with_hyperbolic(occupancy)

        # Losses
        ce_loss = self.compute_loss(logits, labels)
        label_emb = self.label_emb()
        hyp_loss = self.hyp_loss_fn(voxel_emb, labels, label_emb)
        total_loss = ce_loss + self.hyperbolic_weight * hyp_loss

        # Compute IoU (same as parent)
        pred = logits.argmax(dim=1)
        iou_per_class, valid_mask = self.compute_iou(pred, labels, self.n_classes)

        # Accumulate IoU
        if self.val_iou_sum is None:
            self.val_iou_sum = torch.zeros(self.n_classes, device=self.device)
            self.val_iou_count = torch.zeros(self.n_classes, device=self.device)

        self.val_iou_sum += iou_per_class
        self.val_iou_count += valid_mask.float()

        # Log losses
        self.log("val/loss", total_loss, prog_bar=True, sync_dist=True)
        self.log("val/ce_loss", ce_loss, sync_dist=True)
        self.log("val/hyp_loss", hyp_loss, sync_dist=True)

        return {"loss": total_loss, "iou": iou_per_class, "valid": valid_mask}


if __name__ == "__main__":
    # Test
    print("Testing BodyNetHyperbolic...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BodyNetHyperbolic(
        n_classes=72,
        base_channels=32,
        embed_dim=32,
        hyperbolic_weight=0.1,
        margin=0.1
    ).to(device)

    # Dummy batch
    batch = {
        "occupancy": torch.randn(1, 1, 32, 32, 32).to(device),
        "labels": torch.randint(0, 72, (1, 32, 32, 32)).to(device)
    }

    # Test forward
    logits = model(batch["occupancy"])
    print(f"  Logits shape: {logits.shape}")

    # Test forward_with_hyperbolic
    logits, voxel_emb = model.forward_with_hyperbolic(batch["occupancy"])
    print(f"  Voxel embeddings shape: {voxel_emb.shape}")

    # Check embeddings are inside Poincaré ball
    voxel_emb_flat = voxel_emb.permute(0, 2, 3, 4, 1).reshape(-1, 32)
    norms = voxel_emb_flat.norm(dim=-1)
    assert (norms < 1).all(), "Voxel embeddings should be inside Poincaré ball"
    print(f"  Voxel embedding norms: [{norms.min():.3f}, {norms.max():.3f}]")

    # Test training step
    loss = model.training_step(batch, 0)
    print(f"  Training loss: {loss.item():.4f}")

    print("  PASS")
```

### 5.2 Files to Modify (2 files)

#### Modification 1: `pasco/models/dense_unet3d.py`

在 `DenseUNet3D` 类中添加 `forward_with_features` 方法（在 forward 方法后，约第157行）：

```python
def forward_with_features(self, x):
    """
    Forward pass that also returns decoder features.

    Args:
        x: [B, 1, H, W, D] input occupancy grid

    Returns:
        logits: [B, n_classes, H, W, D] class logits
        decoder_features: [B, base_channels, H, W, D] decoder output before final conv
    """
    # Initial
    e0 = self.init_conv(x)

    # Encoder
    e1 = self.enc1(e0)
    e2 = self.enc2(e1)
    e3 = self.enc3(e2)
    e4 = self.enc4(e3)

    # Bottleneck
    b = self.bottleneck(e4)

    # Decoder
    d4 = self.dec4(b, e3)
    d3 = self.dec3(d4, e2)
    d2 = self.dec2(d3, e1)
    d1 = self.dec1(d2, e0)

    # Output
    logits = self.out_conv(d1)

    return logits, d1  # Return both logits and decoder features
```

**同样为 `DenseUNet3DLight` 类添加此方法**（在其 forward 方法后）：

```python
def forward_with_features(self, x):
    """
    Forward pass that also returns decoder features.

    Args:
        x: [B, 1, H, W, D] input occupancy grid

    Returns:
        logits: [B, n_classes, H, W, D] class logits
        decoder_features: [B, base_channels, H, W, D] decoder output before final conv
    """
    # Initial
    e0 = self.init_conv(x)

    # Encoder
    e1 = self.enc1(e0)
    e2 = self.enc2(e1)
    e3 = self.enc3(e2)

    # Bottleneck
    b = self.bottleneck(e3)

    # Decoder
    d3 = self.dec3(b, e2)
    d2 = self.dec2(d3, e1)
    d1 = self.dec1(d2, e0)

    # Output
    logits = self.out_conv(d1)

    return logits, d1  # Return both logits and decoder features
```

**注意**：两个类的 encoder/decoder 层级不同：
- `DenseUNet3D`: 4 级 encoder/decoder (enc1-4, dec1-4)
- `DenseUNet3DLight`: 3 级 encoder/decoder (enc1-3, dec1-3)

#### Modification 2: `scripts/body/train_body.py`

添加命令行参数和条件模型创建：

```python
# 在 argparse 部分添加：
parser.add_argument("--use_hyperbolic", action="store_true",
                    help="Enable hyperbolic embedding learning")
parser.add_argument("--hyp_embed_dim", type=int, default=32,
                    help="Hyperbolic embedding dimension")
parser.add_argument("--hyp_weight", type=float, default=0.1,
                    help="Weight for hyperbolic loss")
parser.add_argument("--hyp_margin", type=float, default=0.1,
                    help="Margin for hyperbolic ranking loss")

# 在模型创建部分修改：
if args.use_hyperbolic:
    from pasco.models.body_net_hyperbolic import BodyNetHyperbolic
    model = BodyNetHyperbolic(
        n_classes=N_CLASSES,
        in_channels=1,
        base_channels=args.base_channels,
        lr=args.lr,
        weight_decay=args.weight_decay,
        class_weights=class_weights if args.use_class_weights else None,
        ignore_index=255,
        use_light_model=args.use_light_model,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        embed_dim=args.hyp_embed_dim,
        hyperbolic_weight=args.hyp_weight,
        margin=args.hyp_margin,
    )
else:
    model = BodyNet(...)  # Original code
```

---

## 6. Implementation Order

```
Step 1: pasco/data/body/organ_hierarchy.py        ← 无依赖，定义层级结构
Step 2: pasco/models/hyperbolic/__init__.py       ← 无依赖，模块初始化
Step 3: pasco/models/hyperbolic/poincare_ops.py   ← 无依赖，基础数学操作
Step 4: pasco/models/hyperbolic/label_embedding.py ← 依赖 Step 1, 3
Step 5: pasco/models/hyperbolic/projection_head.py ← 依赖 Step 3
Step 6: pasco/loss/hyperbolic_loss.py              ← 依赖 Step 3
Step 7: pasco/models/dense_unet3d.py (修改)        ← 无依赖，添加 forward_with_features
        - 为 DenseUNet3D 添加方法
        - 为 DenseUNet3DLight 添加方法
Step 8: pasco/models/body_net_hyperbolic.py        ← 依赖 Step 4, 5, 6, 7
Step 9: scripts/body/train_body.py (修改)          ← 依赖 Step 8
```

**验证顺序**：
```
1. 运行 python pasco/data/body/organ_hierarchy.py
2. 运行 python pasco/models/hyperbolic/poincare_ops.py
3. 运行 python pasco/models/hyperbolic/label_embedding.py
4. 运行 python pasco/models/hyperbolic/projection_head.py
5. 运行 python pasco/loss/hyperbolic_loss.py
6. 运行 python pasco/models/body_net_hyperbolic.py
7. 运行完整训练测试
```

---

## 7. MVP Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_classes` | 72 | Dataset has 72 classes (0-71) |
| `embed_dim` | 32 | Match decoder output channels |
| `margin` | 0.1 | Small margin for auxiliary loss |
| `hyp_weight` | 0.1 | Don't dominate CE loss |
| `ignore_classes` | [0, 255] | Skip background and invalid |
| `min_radius` | 0.1 | Shallow classes near center |
| `max_radius` | 0.8 | Deep classes near boundary |

**重要说明**：
1. `hyp_weight=0.1` 是保守的起始值，可根据实验结果调整（范围建议 0.01-0.5）
2. `margin=0.1` 较小是因为 hyperbolic loss 是辅助损失，不应过于激进
3. 所有 embeddings 通过 `project_to_ball()` 保持在 Poincaré ball 内部

---

## 8. Verification Commands

```bash
# Test Poincaré operations
python pasco/models/hyperbolic/poincare_ops.py

# Test label embedding
python pasco/models/hyperbolic/label_embedding.py

# Test projection head
python pasco/models/hyperbolic/projection_head.py

# Test hyperbolic loss
python pasco/loss/hyperbolic_loss.py

# Test full model
python pasco/models/body_net_hyperbolic.py

# Training verification (1 epoch)
python scripts/body/train_body.py \
    --dataset_root /path/to/data \
    --use_hyperbolic \
    --hyp_weight 0.1 \
    --max_epochs 1 \
    --batch_size 1 \
    --fast_dev_run
```

---

## 9. Changelog

- **v1.0**: Initial plan
- **v1.1**: Add class 0 removal, verify formulas
- **v1.2**: Complete implementation with full code, 72 classes support
- **v1.3**: Plan review and fixes
  - 明确 Ranking Loss 公式是论文的变体实现（标准 triplet loss 形式）
  - 在 `HyperbolicLabelEmbedding.forward()` 中添加周期性投影，确保 embeddings 在 ball 内
  - 添加 `project_embeddings()` 方法用于显式投影
  - 为 `DenseUNet3DLight` 添加 `forward_with_features` 方法
  - 在 `BodyNetHyperbolic` 中添加 `on_validation_epoch_start()` 重置 IoU 累积器
  - 添加 voxel embedding norm 检查到测试代码
  - 添加 `self.model` 引用的注释说明
  - 更新 MVP Configuration 添加重要说明

---

## 10. Future Improvements (Optional)

以下是可选的后续优化方向：

### 10.1 Möbius Transformation Layer

实现完整的 Möbius transformation（论文 Eq. 9-10）来替代简单的 Conv3D 投影：

```python
class MobiusLinear(nn.Module):
    """Möbius version of linear transformation."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # M⊗(x) = tanh(||Mx||/||x|| * arctanh(||x||)) * Mx/||Mx||
        x_norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-5)
        Mx = F.linear(x, self.weight)
        Mx_norm = Mx.norm(dim=-1, keepdim=True).clamp(min=1e-5)

        scale = torch.tanh(Mx_norm / x_norm * torch.atanh(x_norm.clamp(max=1-1e-5)))
        return scale * Mx / Mx_norm
```

### 10.2 Riemannian Optimizer

使用 `geoopt` 库的 Riemannian 优化器，更好地处理 Poincaré ball 上的优化：

```python
import geoopt

# 在 configure_optimizers 中：
manifold = geoopt.PoincareBall()
optimizer = geoopt.optim.RiemannianAdam([
    {'params': self.model.parameters()},
    {'params': self.hyp_head.parameters()},
    {'params': self.label_emb.embeddings, 'manifold': manifold},
], lr=self.lr)
```

### 10.3 Hard Negative Mining

改进负样本采样策略，使用 hard negative mining：

```python
def sample_hard_negatives(self, voxel_emb, pos_classes, label_emb, k=5):
    """Sample hard negatives: closest wrong classes."""
    # Compute distances to all classes
    all_dists = poincare_distance(voxel_emb.unsqueeze(1), label_emb.unsqueeze(0))

    # Mask out positive classes
    mask = torch.ones_like(all_dists, dtype=torch.bool)
    mask.scatter_(1, pos_classes.unsqueeze(1), False)

    # Get k closest wrong classes
    masked_dists = all_dists.masked_fill(~mask, float('inf'))
    hard_neg_indices = masked_dists.topk(k, dim=1, largest=False).indices

    return hard_neg_indices
```

### 10.4 Hierarchical Evaluation Metrics

添加论文中的层级评估指标，评估模型的鲁棒性：

```python
def hierarchical_accuracy(pred, target, parent_map):
    """
    Compute accuracy allowing parent class predictions.
    If pred != target but pred == parent(target), count as correct.
    """
    correct = (pred == target)
    parent_correct = (pred == parent_map[target])
    return (correct | parent_correct).float().mean()
```
