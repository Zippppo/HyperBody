# PaSCo Hyperbolic Module 迁移计划：Poincaré → Lorentz 模型

## 概述

将 PaSCo 项目的双曲嵌入从 **Poincaré 球模型** 迁移到 **Lorentz（双曲面）模型**，以获得更好的数值稳定性和计算效率。

**目标**: 快速跑起来的 MVP 版本，后续再添加高级功能。

### 迁移动机

| 特性 | Poincaré (当前) | Lorentz (目标) |
|------|----------------|----------------|
| **数值稳定性** | 边界处易不稳定 (‖x‖→1) | 无边界约束，更稳定 |
| **曲率** | 固定 (隐式 c=1) | 固定 curv=1.0 (MVP)，后续可扩展为可学习 |
| **距离计算** | 需要复杂的分母处理 | 直接 acosh 计算 |
| **梯度流** | 需要投影保持约束 | 约束自动满足 |

---

## 数学基础

### Lorentz 模型定义

- **空间**: (d+1) 维欧几里得空间中的双曲面 H^d
- **约束**: `x_time = sqrt(1/curv + ‖x_space‖²)`，其中 x_time 是时间分量，x_space 是空间分量
- **Lorentz 内积**: `⟨u,v⟩_L = ⟨u_s,v_s⟩_E - u_t·v_t`
- **曲率**: curv > 0 表示负曲率 -curv 的双曲空间

### 核心公式（与 HyperPath 一致）

> **参考**: `REF/ref_repos/HyperPath/models/lorentz.py`

#### 指数映射（切空间 → 双曲面）
```python
def exp_map0(x, curv=1.0, eps=1e-7):
    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)
    sinh_input = torch.clamp(rc_xnorm, min=eps, max=math.asinh(2**15))
    return torch.sinh(sinh_input) * x / torch.clamp(rc_xnorm, min=eps)
```

#### 对数映射（双曲面 → 切空间）
```python
def log_map0(x, curv=1.0, eps=1e-7):
    # 注意: 使用 1 + curv * ‖x‖² 而非 1/curv + ‖x‖²
    rc_x_time = torch.sqrt(1 + curv * torch.sum(x**2, dim=-1, keepdim=True))
    _distance0 = torch.acosh(torch.clamp(rc_x_time, min=1 + eps))
    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)
    return _distance0 * x / torch.clamp(rc_xnorm, min=eps)
```

#### 批量测地距离
```python
def pairwise_dist(x, y, curv=1.0, eps=1e-7):
    # 计算时间分量
    x_time = torch.sqrt(1/curv + torch.sum(x**2, dim=-1, keepdim=True))
    y_time = torch.sqrt(1/curv + torch.sum(y**2, dim=-1, keepdim=True))
    # Lorentz 内积
    xyl = x @ y.T - x_time @ y_time.T
    # 测地距离
    c_xyl = -curv * xyl
    return torch.acosh(torch.clamp(c_xyl, min=1 + eps)) / curv**0.5
```

#### 到原点距离
```python
def hyperbolic_distance_to_origin(x, curv=1.0, eps=1e-7):
    x_time = torch.sqrt(1/curv + torch.sum(x**2, dim=-1))
    origin_time = torch.sqrt(1/curv)
    c_xyl = -curv * (-x_time * origin_time)
    return torch.acosh(torch.clamp(c_xyl, min=1 + eps)) / curv**0.5
```

---

## 设计决策

### 存储策略
**只存储空间分量 (d维)**，时间分量按约束 `t = sqrt(1/curv + ‖s‖²)` 按需计算。

### 梯度处理
由于参数存储在切空间（欧几里得空间），**标准优化器可直接使用**，不需要 Riemannian 梯度处理。forward 时通过 `exp_map0` 映射到双曲面。

### 曲率策略（MVP 简化）

**MVP 阶段**: 使用固定曲率 `curv=1.0`，不可学习。

```python
# MVP: 固定曲率
CURV = 1.0  # 全局常量
```

## 实现计划

### 阶段 1: 核心数学操作 (lorentz_ops.py)

**创建文件**: `pasco/models/hyperbolic/lorentz_ops.py`

**参考**:
- `REF/ref_repos/HyperPath/models/lorentz.py` (数学操作)
- `REF/ref_repos/HyperPath/models/hypermil.py` (使用模式)

**需实现的函数**:

| 函数 | 功能 | 数值稳定性处理 |
|------|------|---------------|
| `exp_map0(v, curv)` | 切空间→双曲面 | sinh输入clamp到[eps, asinh(2^15)] |
| `log_map0(x, curv)` | 双曲面→切空间 | acosh输入clamp到min=1+eps |
| `pairwise_dist(x, y, curv)` | 批量测地距离 | acosh输入clamp |
| `hyperbolic_distance_to_origin(x, curv)` | 到原点距离 | 用于层级定位 |

**实现策略**: 直接从 HyperPath 复制并适配，保持数学公式一致。

---

### 阶段 2: 标签嵌入 (LorentzLabelEmbedding)

**修改文件**: `pasco/models/hyperbolic/label_embedding.py`

**替换类**: `HyperbolicLabelEmbedding` → `LorentzLabelEmbedding`

**关键改动**:

1. **固定曲率** (MVP简化):
```python
class LorentzLabelEmbedding(nn.Module):
    def __init__(self, n_classes=72, embed_dim=32, curv=1.0, ...):
        self.curv = curv  # 固定值，不可学习
```

2. **嵌入存储**: 存储切空间向量，forward时用exp_map0映射

3. **层级初始化**:
```python
def _init_embedding_by_depth(depth: int, max_depth: int, embed_dim: int,
                              min_radius: float = 0.1, max_radius: float = 2.0) -> Tensor:
    """初始化切空间向量，范数对应层级深度"""
    tangent_norm = min_radius + (max_radius - min_radius) * (depth / max_depth)
    direction = torch.randn(embed_dim)
    direction = direction / direction.norm()
    return direction * tangent_norm
```

4. **72类器官层级**:
   - Class 0 (outside_body): 零向量（原点）
   - Classes 1-71: 根据 `organ_hierarchy.py` 的深度初始化

---

### 阶段 3: 投影头 (LorentzProjectionHead)

**修改文件**: `pasco/models/hyperbolic/projection_head.py`

**替换类**: `HyperbolicProjectionHead` → `LorentzProjectionHead`

**关键改动** (MVP简化，移除可学习alpha):

```python
class LorentzProjectionHead(nn.Module):
    def __init__(self, in_channels, embed_dim, curv=1.0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, embed_dim, kernel_size=1)
        self.curv = curv  # 固定曲率

    def forward(self, x):
        # x: [B, C, H, W, D]
        x = self.conv(x)  # [B, embed_dim, H, W, D]
        x = x.permute(0, 2, 3, 4, 1)  # [B, H, W, D, embed_dim]
        x = exp_map0(x, self.curv)  # 映射到双曲面
        x = x.permute(0, 4, 1, 2, 3)  # [B, embed_dim, H, W, D]
        return x
```

**曲率参数**: 从构造函数传入（与 LabelEmbedding 保持一致）

---

### 阶段 4: 损失函数 (lorentz_loss.py)

**修改文件**: `pasco/loss/hyperbolic_loss.py` → 重命名为 `lorentz_loss.py`

#### LorentzRankingLoss
- 基于 Lorentz 距离的 triplet loss
- 公式: `max(0, margin + d(voxel, pos) - d(voxel, neg))`
- 使用 `pairwise_dist` 替换 `poincare_distance`

```python
class LorentzRankingLoss(nn.Module):
    def __init__(self, margin=0.1, curv=1.0, ignore_classes=None):
        super().__init__()
        self.margin = margin
        self.curv = curv
        self.ignore_classes = set(ignore_classes) if ignore_classes else {0, 255}

    def forward(self, voxel_embeddings, labels, label_embeddings):
        """
        Args:
            voxel_embeddings: [B, D, H, W, Z] Lorentz空间体素嵌入
            labels: [B, H, W, Z] 标签
            label_embeddings: [N_classes, D] Lorentz空间类别嵌入
        """
        # 使用 pairwise_dist 替换 poincare_distance
        ...
```

---

### 阶段 5: 模型集成 (BodyNetHyperbolic)

**修改文件**: `pasco/models/body_net_hyperbolic.py`

**替换组件**:
```python
# 固定曲率，所有组件共享
CURV = 1.0

# 旧
self.hyp_head = HyperbolicProjectionHead(...)
self.label_emb = HyperbolicLabelEmbedding(...)
self.hyp_loss_fn = HyperbolicRankingLoss(...)

# 新
self.hyp_head = LorentzProjectionHead(in_channels=32, embed_dim=32, curv=CURV)
self.label_emb = LorentzLabelEmbedding(n_classes=72, embed_dim=32, curv=CURV)
self.hyp_loss_fn = LorentzRankingLoss(margin=0.1, curv=CURV)
```

**训练循环更新**:
```python
def training_step(self, batch, batch_idx):
    ...
    label_emb = self.label_emb()  # 获取映射后的嵌入
    hyp_loss = self.hyp_loss_fn(voxel_emb, labels, label_emb)
    self.log("train/hyp_loss", hyp_loss)
```

---

### 阶段 6: 训练脚本更新

**修改文件**: `scripts/body/train_body.py`

**更新参数** (简化版):
```python
parser.add_argument('--hyp_embed_dim', type=int, default=32)
parser.add_argument('--hyp_weight', type=float, default=0.1)
parser.add_argument('--hyp_margin', type=float, default=0.1)
parser.add_argument('--hyp_curv', type=float, default=1.0)  # 固定曲率
# 移除 --hyp_learn_curv (MVP阶段不支持)
```

---

### 阶段 7: 测试

**修改文件**: `tests/hyperbolic/` 下的测试文件

**测试用例**:

1. **exp_map/log_map 互逆性**
```python
def test_exp_log_inverse():
    v = torch.randn(10, 32) * 0.5
    x = exp_map0(v, curv=1.0)
    v_rec = log_map0(x, curv=1.0)
    assert torch.allclose(v, v_rec, atol=1e-5)
```

2. **距离三角不等式**
```python
def test_triangle_inequality():
    v = torch.randn(3, 32) * 0.5
    x, y, z = exp_map0(v[0:1], 1.0), exp_map0(v[1:2], 1.0), exp_map0(v[2:3], 1.0)
    d_xy = pairwise_dist(x, y, 1.0).item()
    d_yz = pairwise_dist(y, z, 1.0).item()
    d_xz = pairwise_dist(x, z, 1.0).item()
    assert d_xz <= d_xy + d_yz + 1e-5
```

3. **距离对称性**
4. **距离非负性**
5. **数值稳定性（极值测试）**
6. **零向量处理**
7. **梯度流验证**
8. **与 HyperPath 参考实现对比**

---

## 文件变更清单

| 操作 | 文件路径 | 说明 |
|------|---------|------|
| **新建** | `pasco/models/hyperbolic/lorentz_ops.py` | Lorentz核心数学操作 |
| **重写** | `pasco/models/hyperbolic/label_embedding.py` | LorentzLabelEmbedding |
| **重写** | `pasco/models/hyperbolic/projection_head.py` | LorentzProjectionHead |
| **重写** | `pasco/loss/hyperbolic_loss.py` → `lorentz_loss.py` | Lorentz损失函数 |
| **修改** | `pasco/models/body_net_hyperbolic.py` | 集成Lorentz组件 |
| **修改** | `pasco/models/hyperbolic/__init__.py` | 更新导出 |
| **修改** | `pasco/loss/__init__.py` | 更新导出 |
| **修改** | `scripts/body/train_body.py` | 更新命令行参数 |
| **重写** | `tests/hyperbolic/test_stage2_poincare_ops.py` → `test_lorentz_ops.py` | 单元测试 |

---

## 配置参数 (MVP)

```yaml
hyperbolic:
  embed_dim: 32           # 空间分量维度
  curv: 1.0               # 固定曲率
  min_radius: 0.1         # 浅层类初始化范数
  max_radius: 2.0         # 深层类初始化范数
  loss:
    margin: 0.1
  hyperbolic_weight: 0.1  # 总体双曲损失权重
```

---

## 验证方法

1. **单元测试**: `pytest tests/hyperbolic/`
2. **集成测试**: 小数据集上端到端训练
3. **数值监控**: 观察梯度范数、嵌入分布

---

## 实施顺序

```
1. lorentz_ops.py (核心数学，必须首先实现)
   ↓
2. LorentzLabelEmbedding (依赖 lorentz_ops)
   ↓
3. LorentzProjectionHead (依赖 lorentz_ops)
   ↓
4. LorentzRankingLoss (依赖 lorentz_ops)
   ↓
5. BodyNetHyperbolic (集成所有组件)
   ↓
6. train_body.py (更新参数)
   ↓
7. 测试和验证
```

