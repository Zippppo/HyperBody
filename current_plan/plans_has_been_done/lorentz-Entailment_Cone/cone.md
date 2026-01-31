# Entailment Cone 功能实现计划

## 概述

为 PaSCo-Body 的 Hyperbolic 模块添加 Entailment Cone 功能，借鉴 HyperPath 的实现方式，利用器官层次树结构 (`ORGAN_HIERARCHY`) 建立蕴含关系约束。

---

## MVP 范围定义

```
┌─────────────────────────────────────────────┐
│           MVP 范围                           │
├─────────────────────────────────────────────┤
│ ✓ Label-to-Label Entailment Cone            │
│ ✓ 固定曲率 (CURV=1.0)                        │
│ ✓ 可学习 Embedding (层次深度初始化)           │
│ ✓ 三种损失: entail + contra + position      │
│                                              │
│ ✗ Voxel-to-Label Entailment (后续扩展)       │
│ ✗ 可学习曲率 (后续扩展)                       │
│ ✗ 预训练文本编码器如 CONCH (后续扩展)         │
└─────────────────────────────────────────────┘
```

---

## 关键文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `pasco/models/hyperbolic/lorentz_ops.py` | 扩展 | 添加 `half_aperture`, `oxy_angle` |
| `pasco/data/body/organ_hierarchy.py` | 扩展 | 添加层次关系提取工具函数 |
| `pasco/loss/entailment_cone_loss.py` | **新建** | EntailmentConeLoss 类 |
| `pasco/models/body_net_hyperbolic.py` | 修改 | 集成 entailment cone loss |
| `tests/hyperbolic/test_entailment_cone.py` | **新建** | 测试用例 |

---

## 任务 0: 架构决策记录 (ADR)

### 目标

记录关键架构决策，确保团队理解设计选择的理由。

### 决策 1: 标签 Embedding 方案

**问题**：HyperPath 使用 CONCH 预训练文本编码器生成标签嵌入，当前方案是否应该采用？

**决策**：MVP 阶段使用可学习 embedding，基于层次深度初始化


### 决策 2: 蕴含约束作用范围

**问题**：HyperPath 对视觉特征和语义特征都施加蕴含约束，当前应选择什么？

**决策**：MVP 只对 Label-to-Label 施加蕴含约束


### 决策 3: 曲率处理

**决策**：MVP 使用固定曲率 (CURV=1.0)

**后续扩展点**：
```python
# 可学习曲率的实现方式 (参考 HyperPath)
self.curv = nn.Parameter(torch.tensor(1.0).log())
# 使用时: curv = self.curv.exp()
```

需要同步修改：`exp_map0`, `pointwise_dist`, `half_aperture`, `oxy_angle`

---

## 任务 1: 扩展 lorentz_ops.py

### 目标

添加计算蕴含锥几何关系的两个核心函数：半圆锥角和外角。

### 需要添加的函数

#### 1.1 `half_aperture()` - 计算半圆锥角

**功能**：计算双曲空间中点形成的蕴含锥的"开口大小"

**几何直觉**：
```
                    ∧ (tangent direction)
                   /|\
                  / | \
                 /  |  \
                /   |   \
               / φ  |  φ \    ← φ 是半圆锥角 (half aperture)
              /     |     \
             /      |      \
            --------x--------  ← x 是锥的顶点
                    |
                    O (origin)

离原点越远的点，其蕴含锥越窄（φ 越小）
→ 更具体的概念蕴含更少的子概念
```

**输入契约**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `x` | `Tensor (B, D)` | 空间分量（不含时间分量） |
| `curv` | `float` 或 `Tensor` | 曲率，正值 |
| `min_radius` | `float` | 原点附近最小半径，默认 0.1 |
| `eps` | `float` | 数值稳定性参数，默认 1e-7 |

**输出契约**：
| 返回值 | 类型 | 范围 |
|--------|------|------|
| 半圆锥角 | `Tensor (B,)` | `(0, π/2)` |

**数学公式**：
```
φ(x) = arcsin(2 * min_radius / (||x||_space * √curv + eps))
```

**关键约束**：
- 当 `||x|| → 0` 时，arcsin 输入 clamp 到 `[-1+ε, 1-ε]`
- 输出应随 `||x||` 增大而单调递减

**参考实现**：HyperPath `REF/ref_repos/HyperPath/models/lorentz.py:157-183`

#### 1.2 `oxy_angle()` - 计算外角

**功能**：计算双曲三角形 O-x-y 中，在顶点 x 处测量的、y 相对于原点方向的外角

**几何直觉**：
```
        y
       /
      /
     /  θ_ext (外角，我们要计算的)
    x ←-------- tangent to geodesic from O
     \
      \
       \
        O (origin)

θ_ext = π - ∠Oxy (内角的补角)

当 θ_ext < φ(x) 时，y 在 x 的蕴含锥内
→ x 蕴含 y (x entails y)
```

**输入契约**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `x` | `Tensor (B, D)` | 第一批向量（空间分量） |
| `y` | `Tensor (B, D)` | 第二批向量（与 x 同形状） |
| `curv` | `float` 或 `Tensor` | 曲率 |
| `eps` | `float` | 数值稳定性参数，默认 1e-7 |

**输出契约**：
| 返回值 | 类型 | 范围 |
|--------|------|------|
| 外角 | `Tensor (B,)` | `(0, π)` |

**数学原理** (双曲余弦定理)：
```python
# 1. 从空间分量计算时间分量
x_time = sqrt(1/curv + ||x_space||²)
y_time = sqrt(1/curv + ||y_space||²)

# 2. 计算三边的双曲距离
d_Ox = hyperbolic_distance(O, x)
d_Oy = hyperbolic_distance(O, y)
d_xy = hyperbolic_distance(x, y)

# 3. 使用双曲余弦定理求内角
cos_angle = (cosh(d_Ox) * cosh(d_xy) - cosh(d_Oy)) / (sinh(d_Ox) * sinh(d_xy))
interior_angle = arccos(clamp(cos_angle, -1+eps, 1-eps))

# 4. 外角 = π - 内角
exterior_angle = π - interior_angle
```

**关键约束**：
- arccos 输入需要 clamp 保护
- 当 x 和 y 重合时，外角应接近 0（或返回 0）
- 当 y 在 O-x 连线上时，外角应为 0 或 π

**参考实现**：HyperPath `REF/ref_repos/HyperPath/models/lorentz.py:186-221`

### 与现有代码的关系

- 应与 `exp_map0`, `pointwise_dist` 等函数的参数命名保持一致
- 可复用 `hyperbolic_distance_to_origin` 计算 d_Ox, d_Oy
- 可复用 `pointwise_dist` 计算 d_xy

---

## 任务 2: 扩展 organ_hierarchy.py

### 目标

添加从 `ORGAN_HIERARCHY` 提取结构化关系对的工具函数，供 EntailmentConeLoss 使用。

### 前置知识：层次树结构

当前 `ORGAN_HIERARCHY` 是嵌套字典结构，包含：
- **叶子节点**：有 `class_id` 的实际器官类别
- **虚拟节点**：无 `class_id` 的中间分组节点（如 "urinary_system", "ribs_left"）

### 边界规则

```python
# 排除的 class_id（不参与蕴含关系）
EXCLUDED_CLASS_IDS = {0, 1}  # outside_body, inside_body_empty

# 有效 class_id 范围
VALID_CLASS_ID_RANGE = [2, NUM_CLASSES - 1]  # 包含两端
```

**排除理由**：
- `class_id=0` (outside_body): 是根节点/背景，不是解剖概念
- `class_id=1` (inside_body_empty): 是未标注区域，不是具体器官

### 需要添加的函数

#### 2.1 `get_parent_child_pairs()` - 获取父子蕴含关系

**功能**：提取所有应该满足蕴含关系的 (parent_id, child_id) 对

**输出契约**：
| 返回值 | 类型 | 说明 |
|--------|------|------|
| 关系对列表 | `List[Tuple[int, int]]` | 每个元素是 (parent_class_id, child_class_id) |

**虚拟节点处理规则**：
- 当路径为 A → B(虚拟) → C 时，应返回 (A, C)，跳过虚拟节点 B
- 只返回两端都有 `class_id` 且不在 `EXCLUDED_CLASS_IDS` 中的关系对

**预期数量**：80-120 对

**验证测试**：
```python
pairs = get_parent_child_pairs()
for parent_id, child_id in pairs:
    assert parent_id not in EXCLUDED_CLASS_IDS
    assert child_id not in EXCLUDED_CLASS_IDS
    assert parent_id != child_id
```

#### 2.2 `get_sibling_pairs()` - 获取兄弟互斥关系

**功能**：提取同一父节点下的兄弟类对，用于矛盾损失

**输出契约**：
| 返回值 | 类型 | 说明 |
|--------|------|------|
| 关系对列表 | `List[Tuple[int, int]]` | 每个元素是 (class_id_a, class_id_b)，其中 a < b |

**兄弟关系定义**：
- 共享同一直接父节点（含虚拟节点）的所有叶子节点互为兄弟
- 例如 "ribs_left" 下的 12 根肋骨互为兄弟

**预期数量**：150-200 对（主要来自肋骨 C(12,2)*2=132 对、椎骨、肌肉等）

#### 2.3 `get_ancestor_descendant_pairs()` - 获取祖孙蕴含关系

**功能**：提取所有祖先-后代关系对，包括跨多层级的关系

**输出契约**：
| 返回值 | 类型 | 说明 |
|--------|------|------|
| 关系对列表 | `List[Tuple[int, int]]` | 每个元素是 (ancestor_class_id, descendant_class_id) |

**与 2.1 的区别**：
- 2.1 只返回直接父子关系（相邻层级）
- 2.3 返回所有层级的祖孙关系（包含 2.1 的结果）

**数学关系**：`get_parent_child_pairs() ⊆ get_ancestor_descendant_pairs()`

### 通用约束

- 返回的所有 ID 必须在 `VALID_CLASS_ID_RANGE` 内
- 函数应该是幂等的（多次调用返回相同结果）
- 使用 `@lru_cache` 缓存结果（层次树不会变化）
- 排除 `EXCLUDED_CLASS_IDS` 中的类别

### 参考资源

- 当前层次树定义：`pasco/data/body/organ_hierarchy.py` 中的 `ORGAN_HIERARCHY`
- 类别映射：`CLASS_ID_TO_NAME`, `NAME_TO_CLASS_ID`

---

## 任务 3: 创建 EntailmentConeLoss

### 目标

实现向量化的蕴含锥损失模块，对训练速度影响控制在 <1%。

### 模块设计

#### 输入契约

| 参数 | 类型 | 说明 |
|------|------|------|
| `label_embeddings` | `Tensor (N, D)` | 所有类别的双曲嵌入 |
| `curv` | `float` | 曲率（固定值，不可学习） |

**`label_embeddings` 详细说明**：
```python
# 来源
label_embeddings = self.label_emb()  # 调用 LorentzLabelEmbedding.forward()

# 形状
# N = n_classes (72 for PaSCo-Body，包含 class 0 和 1)
# D = embed_dim (空间分量维度，不含时间分量)

# class 0 (outside_body) 和 class 1 (inside_body_empty) 的处理
# → 在关系对中自动排除，因为 get_*_pairs() 函数不返回这些 ID
# → 即使 embeddings 包含这些类，也不会参与损失计算
```

#### 输出契约

| 返回值 | 类型 | 说明 |
|--------|------|------|
| 损失字典 | `Dict[str, Tensor]` | 包含 `entail`, `contra`, `pos` 三个标量损失 |

### 三种损失组件

#### 3.1 蕴含损失 (entail_loss)

**目标**：祖先类应该蕴含后代类（后代在祖先的蕴含锥内）

**判断条件**：当 `θ(ancestor, descendant) < φ(ancestor)` 时满足蕴含

**损失设计** (HyperPath 风格)：
```python
# 对每对 (ancestor, descendant)
theta = oxy_angle(ancestor_emb, descendant_emb, curv)  # 外角
phi = half_aperture(ancestor_emb, curv)               # 半锥角

# 指数因子：严重违反时惩罚更大
factor = torch.exp(torch.clamp(theta / phi - 1, max=3.0))

# 基础惩罚：只有违反时才有损失
alpha = 1.0  # 可调节的容忍系数
base_penalty = torch.clamp(theta - alpha * phi, min=0)

# 最终损失
entail_loss = (factor * base_penalty).mean()
```

#### 3.2 矛盾损失 (contra_loss)

**目标**：兄弟类不应该互相蕴含（互相在对方的蕴含锥外）

**损失设计**：
```python
# 对每对 (sibling_a, sibling_b)
theta = oxy_angle(a_emb, b_emb, curv)
phi = half_aperture(a_emb, curv)

# 指数因子：不希望的蕴含关系
factor = torch.exp(torch.clamp(phi / theta - 1, max=3.0))

# 当 theta < phi 时产生惩罚
base_penalty = torch.clamp(phi - theta, min=0)

contra_loss = (factor * base_penalty).mean()
```

#### 3.3 位置损失 (pos_loss)

**目标**：祖先类应该比后代类更靠近原点（更一般的概念）

**损失设计**：
```python
# 对每对 (ancestor, descendant)
d_ancestor = hyperbolic_distance_to_origin(ancestor_emb, curv)
d_descendant = hyperbolic_distance_to_origin(descendant_emb, curv)

margin = 0.1  # 可配置
pos_loss = torch.clamp(d_ancestor + margin - d_descendant, min=0).mean()
```

### 关键设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 关系对存储 | `register_buffer` | 不占梯度显存，自动跟随设备 |
| 计算方式 | 向量化批量计算 | 避免 Python 循环，性能影响 <1% |
| 曲率处理 | 作为参数传入，不存储 | 与 BodyNetHyperbolic 的 CURV 保持同步 |
| 空关系对 | 返回 0 损失，不报错 | 支持测试时使用小层次树 |
| 使用哪种关系 | ancestor_descendant (非 parent_child) | 蕴含应跨所有层级 |

### 初始化流程

```python
def __init__(self, min_radius=0.1, margin=0.1):
    super().__init__()

    # 获取关系对
    entail_pairs = get_ancestor_descendant_pairs()  # 用于 entail_loss 和 pos_loss
    sibling_pairs = get_sibling_pairs()             # 用于 contra_loss

    # 转换为 tensor 并注册为 buffer
    if entail_pairs:
        entail_idx = torch.tensor(entail_pairs, dtype=torch.long)
        self.register_buffer('entail_parent_idx', entail_idx[:, 0])
        self.register_buffer('entail_child_idx', entail_idx[:, 1])
    else:
        self.register_buffer('entail_parent_idx', torch.empty(0, dtype=torch.long))
        self.register_buffer('entail_child_idx', torch.empty(0, dtype=torch.long))

    # 类似处理 sibling_pairs...
```


### 参考实现

- HyperPath 蕴含损失：`REF/ref_repos/HyperPath/models/hypermil.py:407-413`
- HyperPath 矛盾损失：`REF/ref_repos/HyperPath/models/hypermil.py:393-405`

---

## 任务 4: 集成到 BodyNetHyperbolic

### 目标

将 EntailmentConeLoss 集成到训练流程中，作为可选的正则化项。

### 需要修改的位置

#### 4.1 `__init__` 方法

**新增参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_entailment_cone` | `bool` | `True` | 是否启用蕴含锥损失 |
| `entailment_weight` | `float` | `0.1` | 总体权重 |
| `entail_loss_weight` | `float` | `1.0` | 蕴含损失子权重 |
| `contra_loss_weight` | `float` | `1.0` | 矛盾损失子权重 |
| `pos_loss_weight` | `float` | `0.5` | 位置损失子权重 |

**初始化逻辑**：
```python
if use_entailment_cone:
    self.entail_loss_fn = EntailmentConeLoss(min_radius=0.1)
else:
    self.entail_loss_fn = None
```

#### 4.2 `training_step` 方法

**修改内容**：
```python
# 在计算现有损失后
if self.entail_loss_fn is not None:
    label_emb = self.label_emb()  # (N, D)
    cone_losses = self.entail_loss_fn(label_emb, curv=CURV)

    entail_loss_total = (
        self.entail_loss_weight * cone_losses['entail'] +
        self.contra_loss_weight * cone_losses['contra'] +
        self.pos_loss_weight * cone_losses['pos']
    )

    total_loss = total_loss + self.entailment_weight * entail_loss_total
```

#### 4.3 日志记录

需要记录的指标：
- `train/cone_loss_total`: 加权后的总蕴含锥损失
- `train/cone_loss_entail`: 蕴含损失分量
- `train/cone_loss_contra`: 矛盾损失分量
- `train/cone_loss_pos`: 位置损失分量

### 与现有损失的关系

```
total_loss = ce_loss
           + hyperbolic_weight * hyp_loss      (voxel-to-label ranking)
           + entailment_weight * cone_loss     (label-to-label hierarchy) ← 新增
```

- `hyp_loss`: 作用于 (voxel_emb, label_emb) 对，拉近同类、推远异类
- `cone_loss`: 只作用于 label_emb 之间，约束层次结构

两者互补，不冲突。

### 损失调试指南

**推荐的调试顺序**：

1. **阶段 1: 只开启 entail_loss**
   ```python
   entail_loss_weight=1.0, contra_loss_weight=0.0, pos_loss_weight=0.0
   ```
   - 监控 `train/cone_loss_entail` 应随训练下降
   - 验证蕴含关系是否在学习

2. **阶段 2: 加入 pos_loss**
   ```python
   entail_loss_weight=1.0, contra_loss_weight=0.0, pos_loss_weight=0.5
   ```
   - 监控标签嵌入的范数分布
   - 验证祖先类是否更靠近原点

3. **阶段 3: 加入 contra_loss**
   ```python
   entail_loss_weight=1.0, contra_loss_weight=1.0, pos_loss_weight=0.5
   ```
   - 监控兄弟类之间的角度
   - 注意：过早加入可能导致训练不稳定

**正常值范围参考** (训练初期)：
| 损失分量 | 预期初始值 | 收敛后 |
|----------|------------|--------|
| entail_loss | 0.5-2.0 | <0.1 |
| contra_loss | 0.1-0.5 | <0.05 |
| pos_loss | 0.2-1.0 | <0.1 |

**异常情况处理**：
- 如果 `entail_loss` 不下降：检查 `half_aperture` 和 `oxy_angle` 的数值稳定性
- 如果 `contra_loss` 爆炸：降低 `contra_loss_weight` 或延迟开启
- 如果所有损失为 0：检查关系对是否正确生成

---

## 任务 5: 添加测试

### 目标

为新增功能添加单元测试和集成测试。

### 测试文件

新建 `tests/hyperbolic/test_entailment_cone.py`

### 测试分类

#### 5.1 lorentz_ops 新函数测试

| 测试项 | 验证内容 |
|--------|----------|
| `test_half_aperture_output_shape` | 输出形状为 `(B,)` |
| `test_half_aperture_output_range` | 输出在 `(0, π/2)` 范围内 |
| `test_half_aperture_monotonicity` | 范数越大，角度越小 |
| `test_half_aperture_numerical_stability` | 零向量、极大向量不报错/不产生 NaN |
| `test_half_aperture_gradient_flow` | 梯度能正确反传 |
| `test_oxy_angle_output_shape` | 输出形状为 `(B,)` |
| `test_oxy_angle_output_range` | 输出在 `(0, π)` 范围内 |
| `test_oxy_angle_same_point` | 当 x=y 时，角度接近 0 |
| `test_oxy_angle_gradient_flow` | 梯度能正确反传 |
| `test_oxy_angle_numerical_stability` | 边界情况不产生 NaN |

#### 5.2 hierarchy 工具函数测试

| 测试项 | 验证内容 |
|--------|----------|
| `test_parent_child_pairs_not_empty` | 返回非空列表 |
| `test_parent_child_pairs_valid_ids` | 所有 ID 在有效范围内，不含排除 ID |
| `test_parent_child_pairs_no_excluded` | 不包含 class_id 0 或 1 |
| `test_sibling_pairs_no_self_pair` | 不包含 (a, a) 形式的对 |
| `test_sibling_pairs_unique` | 只有 (a, b) 形式，a < b |
| `test_sibling_pairs_no_excluded` | 不包含排除的 class_id |
| `test_ancestor_descendant_includes_parent_child` | 祖孙关系包含父子关系 |
| `test_functions_are_cached` | 多次调用返回相同对象 (id 相等) |

#### 5.3 EntailmentConeLoss 测试

| 测试项 | 验证内容 |
|--------|----------|
| `test_forward_returns_dict` | 返回包含 `entail`, `contra`, `pos` 的字典 |
| `test_forward_all_losses_scalar` | 所有损失是 0 维 tensor |
| `test_forward_losses_non_negative` | 所有损失 >= 0 |
| `test_gradient_flows_to_embeddings` | 梯度能传播到 label_embeddings |
| `test_empty_pairs_no_error` | 空关系对时返回 0 损失，不报错 |
| `test_device_transfer` | `.to(device)` 后 buffer 正确转移 |
| `test_performance_benchmark` | 300 对关系的计算 < 2ms (可选，标记 slow) |

#### 5.4 集成测试

| 测试项 | 验证内容 |
|--------|----------|
| `test_body_net_hyperbolic_with_cone` | 启用时能正常完成一个 training_step |
| `test_body_net_hyperbolic_without_cone` | 禁用时与原有行为一致 |
| `test_end_to_end_gradient` | 从 total_loss 到 label_emb 的梯度路径完整 |
| `test_logging_keys_present` | 日志中包含所有 cone_loss 相关 key |

#### 5.5 数值稳定性测试

| 测试项 | 验证内容 |
|--------|----------|
| `test_curv_near_zero` | curv=0.01 时不产生 NaN |
| `test_embedding_near_origin` | 范数很小的嵌入不导致除零 |
| `test_embedding_far_from_origin` | 范数很大的嵌入不导致溢出 |
| `test_identical_embeddings` | 两个相同嵌入不导致 NaN |

### 回归测试

运行现有测试确保无回归：
```bash
pytest tests/hyperbolic/test_lorentz_ops.py -v
pytest tests/hyperbolic/test_label_embedding.py -v
```

---

## 实现顺序与依赖关系

```
任务 0 (ADR 文档) ← 已完成（本文档）
    ↓
任务 1 (lorentz_ops) ←──────┐
    ↓                       │ 可并行
任务 2 (hierarchy) ─────────┤
    ↓                       ↓
任务 3 (EntailmentConeLoss) ←┘
    ↓
任务 5.1-5.3 (基础测试)
    ↓
任务 4 (集成)
    ↓
任务 5.4-5.5 (集成测试 + 数值稳定性测试)
```

- 任务 1 和任务 2 可以并行开发
- 任务 3 依赖任务 1 和任务 2
- 任务 4 依赖任务 3
- 测试可以在对应任务完成后立即编写

---

## 预期关系对数量

| 关系类型 | 预计数量 | 主要来源 |
|----------|----------|----------|
| 父子对 | 80-120 | 器官系统 → 具体器官 |
| 兄弟对 | 150-200 | 肋骨(132)、椎骨、左右配对器官 |
| 祖孙对 | 100-150 | 包含跨层级关系 |

---

## 关键设计决策总结

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 标签 Embedding | 可学习 (非 CONCH) | 无额外依赖，层次结构已显式定义 |
| 蕴含约束范围 | Label-to-Label | 语义清晰，计算高效，不改变架构 |
| 基础实现风格 | HyperPath | 指数因子 + 多种损失更适合层次建模 |
| 层次定义方式 | 预定义语义树 | PaSCo 有明确的解剖学层次 |
| 计算方式 | 向量化 | 避免 Python 循环开销 |
| 曲率处理 | 固定值传入 | MVP 简化，后续可改为可学习 |
| 虚拟节点 | 跳过，建立祖孙关系 | 保持 class_id 的一致性 |

---

## 后续扩展点

以下功能不在 MVP 范围内，但架构已预留扩展空间：

### 1. 可学习曲率
```python
# 在 BodyNetHyperbolic 中
self.curv = nn.Parameter(torch.tensor(1.0).log())
# 调用时传入 curv=self.curv.exp()
```

### 2. Voxel-to-Label 蕴含
```python
# 在 EntailmentConeLoss 中新增方法
def voxel_label_entailment(self, voxel_emb, label_emb, labels, curv):
    # 对每个 voxel，其标签的祖先应该蕴含该 voxel
    pass
```

### 3. 预训练文本编码器
```python
# 替换 LorentzLabelEmbedding
class TextEncodedLabelEmbedding(nn.Module):
    def __init__(self, text_encoder, organ_names, curv):
        # 使用 RadBERT 等医学文本编码器
        pass
```

### 4. 动态层次关系
```python
# 支持运行时修改层次树
class DynamicEntailmentConeLoss(EntailmentConeLoss):
    def update_hierarchy(self, new_pairs):
        # 更新 buffer
        pass
```

---

## 参考资料

- **技术分析文档**: `current_plan/Cone_analysis.md`
- **HyperPath 代码**: `REF/ref_repos/HyperPath/models/lorentz.py`, `hypermil.py`
- **MERU 代码**: `meru/meru/lorentz.py`, `models.py`
- **当前层次定义**: `pasco/data/body/organ_hierarchy.py`
- **当前 Lorentz 实现**: `pasco/models/hyperbolic/lorentz_ops.py`
- **当前标签嵌入**: `pasco/models/hyperbolic/label_embedding.py`
