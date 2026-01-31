# PaSCo-Body 双曲空间嵌入改进计划

## 背景与目标

**当前状态**: 已实现基础的Poincaré球嵌入，包括:
- `poincare_ops.py`: exp_map_zero, poincare_distance, project_to_ball
- `label_embedding.py`: 基于层级深度初始化的标签嵌入
- `hyperbolic_loss.py`: Triplet-style排名损失

**核心问题**: 层级约束仅在初始化时使用，训练过程中没有显式维护层级结构

**目标**: 通过"性价比"导向的改进策略，增强双曲空间的层级感知能力

---

## 改进计划总览

| Phase | 改进内容 | 难度 | 价值 | 时间估计 |
|-------|---------|------|------|---------|
| 1 | 可学习曲率 | 低 | 高 | 1天 |
| 2 | 不确定性正则化 | 低 | 高 | 0.5天 |
| 3 | 蕴含锥约束 | 中 | 最高 | 2-3天 |
| 4 | 层级感知负采样 | 低 | 高 | 0.5天 |
| 5 | Busemann边界原型 | 中 | 很高 | 2-3天 |

**推荐实施顺序**: 1 → 2 → 4 → 3 → 5 (快速收益优先)

---

## Phase 1: 可学习曲率

### 修改文件
- `pasco/models/hyperbolic/poincare_ops.py`
- `pasco/models/body_net_hyperbolic.py`
- `scripts/body/train_body.py`

### 实现要点

**1. 在 `poincare_ops.py` 添加带曲率的函数:**
```python
def mobius_add(x, y, c=1.0, eps=1e-5):
    """Möbius加法: x ⊕_c y"""
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2*c*xy + c*y2) * x + (1 - c*x2) * y
    denom = 1 + 2*c*xy + c*c*x2*y2
    return num / denom.clamp(min=eps)

def poincare_distance_c(x, y, c=1.0, eps=1e-5):
    """带曲率c的Poincaré距离"""
    sqrt_c = c ** 0.5
    diff = mobius_add(-x, y, c)
    diff_norm = diff.norm(dim=-1).clamp(min=eps)
    return (2 / sqrt_c) * torch.atanh(sqrt_c * diff_norm.clamp(max=1-eps))

def exp_map_zero_c(v, c=1.0, eps=1e-5):
    """带曲率c的指数映射"""
    sqrt_c = c ** 0.5
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=eps)
    return torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)
```

**2. 在 `body_net_hyperbolic.py` 添加可学习曲率:**
```python
class BodyNetHyperbolic(BodyNet):
    def __init__(self, ..., learnable_curvature=False, init_curvature=1.0):
        ...
        if learnable_curvature:
            self.curvature = nn.Parameter(torch.tensor(init_curvature))
        else:
            self.register_buffer('curvature', torch.tensor(init_curvature))

    @property
    def c(self):
        return self.curvature.clamp(0.1, 10.0)
```

**3. 训练参数:**
```
--learnable_curvature    # 启用可学习曲率
--init_curvature 1.0     # 初始曲率值
```

### 验证方法
- 监控 `curvature` 值在训练过程中的变化
- 对比固定曲率 vs 可学习曲率的 mIoU

---

## Phase 2: 不确定性正则化

### 修改文件
- `pasco/loss/hyperbolic_loss.py`
- `pasco/models/body_net_hyperbolic.py`
- `scripts/body/train_body.py`

### 实现要点

**1. 在 `hyperbolic_loss.py` 添加:**
```python
class HyperbolicUncertaintyLoss(nn.Module):
    """
    防止特征过度收敛到原点或边界
    L = mean((||z|| - target)^2)
    """
    def __init__(self, target_norm=0.5):
        super().__init__()
        self.target_norm = target_norm

    def forward(self, voxel_embeddings, valid_mask=None):
        # [B, D, H, W, Z] -> norms
        emb_flat = voxel_embeddings.permute(0, 2, 3, 4, 1).reshape(-1, D)
        if valid_mask is not None:
            emb_flat = emb_flat[valid_mask.reshape(-1)]
        norms = emb_flat.norm(dim=-1)
        return ((norms - self.target_norm) ** 2).mean()
```

**2. 训练参数:**
```
--uncertainty_weight 0.01    # 不确定性损失权重
--target_norm 0.5            # 目标norm值
```

### 损失组合
```
Total Loss = CE + λ_hyp * HypLoss + λ_unc * UncertaintyLoss
```

---

## Phase 3: 蕴含锥约束 (核心改进)

### 修改文件
- `pasco/data/body/organ_hierarchy.py` (添加parent_map)
- 新增 `pasco/models/hyperbolic/entailment_cone.py`
- `pasco/loss/hyperbolic_loss.py`
- `pasco/models/body_net_hyperbolic.py`

### 实现要点

**1. 在 `organ_hierarchy.py` 添加:**
```python
def build_parent_map(hierarchy=None, parent_id=None):
    """构建 {class_id: parent_class_id} 映射"""
    if hierarchy is None:
        hierarchy = ORGAN_HIERARCHY
    parent_map = {}
    current_id = hierarchy.get("class_id")

    if "children" in hierarchy:
        for child in hierarchy["children"]:
            child_id = child.get("class_id")
            if child_id is not None:
                parent_map[child_id] = current_id  # 叶子节点的父节点
            child_map = build_parent_map(child, current_id)
            parent_map.update(child_map)
    return parent_map

PARENT_MAP = build_parent_map()
```

**2. 新增 `entailment_cone.py`:**
```python
def cone_half_angle(p, K=0.1, c=1.0, eps=1e-5):
    """计算点p的锥半角: ω(p) = arcsin(2K / (√c * ||p||))"""
    sqrt_c = c ** 0.5
    p_norm = p.norm(dim=-1).clamp(min=eps)
    sin_omega = (2 * K / (sqrt_c * p_norm)).clamp(max=1.0)
    return torch.asin(sin_omega)

def angle_at_origin(u, v, eps=1e-5):
    """计算原点处u和v之间的夹角"""
    u_norm = u.norm(dim=-1).clamp(min=eps)
    v_norm = v.norm(dim=-1).clamp(min=eps)
    cos_angle = (u * v).sum(dim=-1) / (u_norm * v_norm)
    return torch.acos(cos_angle.clamp(-1+eps, 1-eps))

class HyperbolicEntailmentLoss(nn.Module):
    """
    蕴含锥损失: 强制子类落在父类锥内
    L = Σ max(0, θ(child, parent) - β * ω(parent))
    """
    def __init__(self, parent_map, K=0.1, beta=0.9, c=1.0):
        super().__init__()
        self.parent_map = parent_map  # {child_id: parent_id}
        self.K = K
        self.beta = beta
        self.c = c

    def forward(self, label_embeddings):
        loss = 0.0
        count = 0
        for child_id, parent_id in self.parent_map.items():
            if parent_id is None or parent_id >= label_embeddings.shape[0]:
                continue
            child_emb = label_embeddings[child_id]
            parent_emb = label_embeddings[parent_id]

            omega = cone_half_angle(parent_emb, K=self.K, c=self.c)
            theta = angle_at_origin(child_emb, parent_emb)

            violation = torch.relu(theta - self.beta * omega)
            loss = loss + violation
            count += 1

        return loss / max(count, 1)
```

**3. 训练参数:**
```
--entailment_weight 0.1    # 蕴含损失权重
--entailment_K 0.1         # Lipschitz常数，控制锥宽度
--entailment_beta 0.9      # 松弛因子
```

### 验证方法
- 可视化标签嵌入空间中的锥形区域
- 统计子类落在父类锥内的比例
- 比较加入蕴含损失前后的层级一致性

---

## Phase 4: 层级感知负采样

### 修改文件
- `pasco/data/body/organ_hierarchy.py`
- `pasco/loss/hyperbolic_loss.py`

### 实现要点

**1. 在 `organ_hierarchy.py` 添加:**
```python
def get_siblings(class_id):
    """获取同一父节点下的兄弟类别"""
    parent = PARENT_MAP.get(class_id)
    if parent is None:
        return []
    return [cid for cid, pid in PARENT_MAP.items() if pid == parent and cid != class_id]

def get_same_depth_classes(class_id):
    """获取同一深度的类别"""
    depth = CLASS_DEPTHS.get(class_id, 0)
    return [cid for cid, d in CLASS_DEPTHS.items() if d == depth and cid != class_id]

SIBLINGS_MAP = {cid: get_siblings(cid) for cid in CLASS_DEPTHS.keys()}
```

**2. 修改 `HyperbolicRankingLoss`:**
```python
class HyperbolicRankingLoss(nn.Module):
    def __init__(self, ..., negative_strategy='random', siblings_map=None):
        self.negative_strategy = negative_strategy
        self.siblings_map = siblings_map

    def sample_negatives(self, pos_labels):
        if self.negative_strategy == 'sibling':
            # 50%概率选择兄弟节点，50%随机
            ...
        elif self.negative_strategy == 'hard':
            # 选择距离正样本最近的负类
            ...
        else:
            # 原始随机采样
            ...
```

**3. 训练参数:**
```
--negative_strategy random|sibling|hard
```

---

## Phase 5: Busemann边界原型 (进阶)

### 修改文件
- `pasco/models/hyperbolic/poincare_ops.py`
- 新增 `pasco/models/hyperbolic/ideal_prototype.py`
- `pasco/models/body_net_hyperbolic.py`

### 实现要点

**1. 在 `poincare_ops.py` 添加:**
```python
def busemann_function(x, p, eps=1e-5):
    """
    Busemann函数: B_p(x) = log(||p - x||^2 / (1 - ||x||^2))
    p: 边界上的理想点 (||p|| = 1)
    x: 球内的点
    """
    diff_norm_sq = ((p - x) ** 2).sum(dim=-1)
    x_norm_sq = (x ** 2).sum(dim=-1).clamp(max=1-eps)
    return torch.log(diff_norm_sq / (1 - x_norm_sq + eps) + eps)
```

**2. 新增 `ideal_prototype.py`:**
```python
class IdealPrototypeEmbedding(nn.Module):
    """边界上的理想原型嵌入"""
    def __init__(self, n_classes, embed_dim, class_depths, max_depth):
        super().__init__()
        # 使用方向向量表示边界点
        self.directions = nn.Parameter(torch.randn(n_classes, embed_dim))
        # 可选: 基于层级初始化方向
        self._init_from_hierarchy(class_depths, max_depth)

    def forward(self):
        # 归一化到单位球面 (边界)
        return F.normalize(self.directions, dim=-1)
```

**3. 训练参数:**
```
--use_busemann           # 使用Busemann函数分类
--ideal_prototypes       # 使用边界原型
```

---

## 关键文件列表

| 文件路径 | 修改内容 |
|---------|---------|
| `pasco/models/hyperbolic/poincare_ops.py` | 添加带曲率操作、Busemann函数 |
| `pasco/models/hyperbolic/entailment_cone.py` | 新增蕴含锥计算 |
| `pasco/models/hyperbolic/ideal_prototype.py` | 新增边界原型 |
| `pasco/loss/hyperbolic_loss.py` | 添加不确定性损失、蕴含损失、层级负采样 |
| `pasco/data/body/organ_hierarchy.py` | 添加parent_map、siblings_map |
| `pasco/models/body_net_hyperbolic.py` | 集成所有新组件 |
| `scripts/body/train_body.py` | 添加新参数开关 |

---

## 验证计划

### 单元测试
```bash
# 每个新模块添加对应测试
pytest tests/hyperbolic/test_poincare_ops_c.py      # 带曲率操作
pytest tests/hyperbolic/test_entailment_cone.py     # 蕴含锥
pytest tests/hyperbolic/test_busemann.py            # Busemann函数
```

### 集成测试
```bash
# 快速验证训练流程
python scripts/body/train_body.py \
    --dataset_root Dataset/voxel_data \
    --batch_size 2 \
    --max_epochs 5 \
    --use_hyperbolic \
    --learnable_curvature \
    --uncertainty_weight 0.01 \
    --entailment_weight 0.1 \
    --gpuids 0
```

### 消融实验
```bash
# Baseline
python scripts/body/train_body.py --use_hyperbolic

# + 可学习曲率
python scripts/body/train_body.py --use_hyperbolic --learnable_curvature

# + 不确定性
python scripts/body/train_body.py --use_hyperbolic --uncertainty_weight 0.01

# + 蕴含锥
python scripts/body/train_body.py --use_hyperbolic --entailment_weight 0.1

# Full
python scripts/body/train_body.py --use_hyperbolic --learnable_curvature \
    --uncertainty_weight 0.01 --entailment_weight 0.1 --negative_strategy sibling
```

---

## 风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| 曲率发散 | clamp到[0.1, 10.0] |
| 蕴含损失过强 | 从小权重(0.01)开始，逐步增加 |
| 边界数值不稳定 | 添加eps保护，使用log1p |
| 训练不收敛 | 每个改进单独验证，保留回退开关 |

---

## 预期收益

1. **可学习曲率**: 让模型自适应最优空间几何
2. **不确定性正则化**: 防止特征collapse，改善泛化
3. **蕴含锥约束**: 直接解决层级约束问题，核心改进
4. **层级负采样**: 加速收敛，提升边界清晰度
5. **Busemann原型**: 更好的层级分类决策边界

综合预期: mIoU提升 2-5%，层级一致性显著改善
