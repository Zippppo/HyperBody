# Virtual Node Embeddings for Entailment Cone Loss

> **文档协同说明**
> - 本文档：设计计划（Why & What）— 问题分析、设计决策、架构、参数
> - [vir_emb-strategy.md](./vir_emb-strategy.md)：TDD 执行策略（How）— 测试用例、执行步骤、检查点

---

## 1. Problem Statement

### 现状分析

当前 `ORGAN_HIERARCHY` 包含：

- **21 个虚拟节点**（无 `class_id`）：skeletal_system, ribs_left, visceral_system 等中间层级
- **72 个叶子节点**（有 `class_id` 0-71）：实际器官类别

**问题：** 所有有 `class_id` 的节点都是叶子节点，导致：

| 函数  | 当前返回 | 期望  |
| --- | --- | --- |
| `get_ancestor_descendant_pairs()` | 0 对 | >0 对 |
| `entail_loss` | 始终为 0 | 非零  |
| `pos_loss` | 始终为 0 | 非零  |
| `contra_loss` | 193 对（正常） | 不变  |

**根本原因：** 虚拟节点没有 embedding，无法参与 entailment cone 约束。

---

## 2. Solution Design

### 2.1 设计目标

1. 为虚拟节点创建 learnable embeddings
2. 虚拟节点参与 entailment cone 约束（但**不参与**分割任务）
3. 保持向后兼容（`include_virtual=False` 时行为不变）
4. 支持配置化控制

### 2.2 核心设计决策

#### 决策 1：ID 分配策略

**方案：** 虚拟节点 ID = `len(CLASS_NAMES) + offset`（动态计算，非硬编码）

```
Real classes:    0 ~ N-1        (N = len(CLASS_NAMES) = 72)
Virtual nodes:   N ~ N+M-1      (M = 虚拟节点数量 = 21)
Total:           N + M          (93 embeddings)
```

**理由：** 如果将来添加新器官类别，ID 自动适应，无需手动修改。

#### 决策 2：虚拟节点初始化策略

**目标：** 虚拟节点（更通用概念）应该具有**更宽的 entailment cone**。

**实现思路：**

- Lorentz 模型中，距离原点越近 → cone 半角越大 → 覆盖范围越广
- 虚拟节点初始化时使用**更小的 tangent norm**
- 初始化范围基于虚拟节点深度：root 最小，深层虚拟节点稍大

#### 决策 3：Embedding 获取接口

提供两个接口，分离不同用途：

| 接口  | 返回形状 | 用途  |
| --- | --- | --- |
| `get_real_embeddings()` | [N, D] | 分割任务（LorentzRankingLoss） |
| `get_all_embeddings()` | [N+M, D] | 层级约束（EntailmentConeLoss） |

#### 决策 4：Sibling pairs 保持不变

**只有 entail_pairs 扩展**，sibling_pairs 仍然只包含叶子节点。

**理由：** 虚拟节点之间的"兄弟"关系（如 skeletal_system vs visceral_system）是类别划分，不是语义上的互斥关系，不适合用 contradiction loss 约束。

---

## 3. Alternative Solutions Considered

| 方案  | 描述  | 优点  | 缺点  | 结论  |
| --- | --- | --- | --- | --- |
| **A. 独立虚拟节点 embedding（本方案）** | 为每个虚拟节点创建 learnable embedding | 完整层级约束，可独立学习 | 增加 21×D 参数 | ✅ 采用 |
| **B. 虚拟节点 = 子节点均值** | 动态计算，无额外参数 | 零参数增加 | 梯度复杂，无法独立学习虚拟节点位置 | ❌ 排除 |
| **C. 跳过虚拟节点，直接连接叶子** | 如 liver→body 直接变成 liver→(最近的有ID祖先) | 改动最小 | 当前所有叶子的父节点都是虚拟节点，无法实现 | ❌ 不可行 |
| **D. 软层级约束（深度正则项）** | 用深度差异作为 loss 项 | 实现简单 | 不如显式 cone 约束几何意义明确 | ❌ 排除 |

---

## 4. Risk Analysis

| 风险  | 可能性 | 影响  | 缓解措施 |
| --- | --- | --- | --- |
| **虚拟节点梯度不稳定** | 中   | 训练震荡 | 1. 虚拟节点使用较小学习率<br>2. 梯度裁剪 |
| **loss scale 突变** | 高   | 训练不稳定 | entail_loss 从 0→非零，需调整 `entailment_weight` |
| **现有 checkpoint 不兼容** | 高   | 无法 resume | 不用考虑，训练成本不高 |
| **参数量增加** | 低   | OOM | 仅增加 21×32=672 参数，可忽略 |
| **回归风险** | 中   | 原有功能失效 | `include_virtual=False` 时走原有代码路径 |

---

## 5. Implementation Plan

> **TDD 执行详情见 [vir_emb-strategy.md](./vir_emb-strategy.md)**
> 每个 Phase 遵循 RED → GREEN → REFACTOR 循环

| Phase | 模块 | 目标 | 前置依赖 |
|-------|------|------|----------|
| 1 | `organ_hierarchy.py` | 虚拟节点注册表 + 完整层级关系 | 无 |
| 2 | `label_embedding.py` | Embedding 支持虚拟节点 | Phase 1 |
| 3 | `entailment_cone_loss.py` | Loss 支持虚拟节点 | Phase 1, 2 |
| 4 | `body_net_hyperbolic.py` | 整合到训练流程 | Phase 1, 2, 3 |
| 5 | `train_body.py` | 配置化控制 | Phase 4 |
| 6 | 集成测试 | 端到端验证 + 回归测试 | 全部 |

---

### Phase 1: Hierarchy Extension (`organ_hierarchy.py`)

**目标：** 建立虚拟节点注册表和完整层级关系。

**新增内容：**

- `build_virtual_node_registry()`: 遍历 ORGAN_HIERARCHY，为虚拟节点分配 ID
- `get_full_parent_child_pairs()`: 包含虚拟节点的直接父子关系
- `get_full_ancestor_descendant_pairs()`: 包含虚拟节点的所有祖先-后代关系
- 模块级变量：`N_VIRTUAL_NODES`, `N_TOTAL_EMBEDDINGS`, `VIRTUAL_NODE_NAMES`, `VIRTUAL_NODE_DEPTHS`

---

### Phase 2: Embedding Extension (`label_embedding.py`)

**目标：** `LorentzLabelEmbedding` 支持虚拟节点。

**修改内容：**

- 新增参数 `include_virtual: bool = False`
- 当 `include_virtual=True` 时，`tangent_vectors` 形状为 `[N+M, D]`
- 虚拟节点初始化：tangent norm 更小，基于深度递增
- 新增方法：`get_real_embeddings()`, `get_all_embeddings()`

**向后兼容：**

- `include_virtual=False` 时行为完全不变
- 添加 `load_state_dict` 重载，支持从旧 checkpoint 加载（自动扩展）

---

### Phase 3: Loss Extension (`entailment_cone_loss.py`)

**目标：** `EntailmentConeLoss` 支持虚拟节点。

**修改内容：**

- 新增参数 `include_virtual: bool = False`
- 当 `include_virtual=True` 时，使用 `get_full_ancestor_descendant_pairs()`
- sibling_pairs 保持不变（仅叶子节点）

---

### Phase 4: Model Integration (`body_net_hyperbolic.py`)

**目标：** 整合虚拟节点到训练流程。

**修改内容：**

- 新增参数 `include_virtual_nodes: bool = True`
- 传递给 `LorentzLabelEmbedding` 和 `EntailmentConeLoss`
- `training_step` 中分离两种 embedding 用途

**数据流：**

```
training_step:
├── label_emb.get_real_embeddings() → [72, D] → LorentzRankingLoss
└── label_emb.get_all_embeddings()  → [93, D] → EntailmentConeLoss
```

---

### Phase 5: Configuration Support (`train_body.py`)

**目标：** 通过命令行/配置文件控制虚拟节点功能。

**新增参数：**

- `--include_virtual_nodes`: 是否启用虚拟节点（默认 True）

---

### Phase 6: Integration Testing

**目标：** 端到端验证 + 回归测试

- 1 epoch 训练 smoke test
- loss 曲线合理性验证
- `include_virtual=False` 回归测试

---

## 6. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LorentzLabelEmbedding                            │
│                      (include_virtual=True)                             │
├─────────────────────────────────────────────────────────────────────────┤
│  tangent_vectors [93, D]                                                │
│  ├── [0:72]  Real class embeddings (参与分割 + 层级约束)                │
│  └── [72:93] Virtual node embeddings (仅参与层级约束)                   │
├─────────────────────────────────────────────────────────────────────────┤
│  get_real_embeddings() ──────────────────┐                              │
│       │                                  │                              │
│       ▼                                  ▼                              │
│  [72, D]                          LorentzRankingLoss                    │
│                                   (voxel ↔ label 距离)                  │
├─────────────────────────────────────────────────────────────────────────┤
│  get_all_embeddings() ───────────────────┐                              │
│       │                                  │                              │
│       ▼                                  ▼                              │
│  [93, D]                          EntailmentConeLoss                    │
│                                   ├── entail_loss: virtual→leaf,        │
│                                   │                virtual→virtual      │
│                                   ├── contra_loss: leaf↔leaf            │
│                                   └── pos_loss: ancestor 更近原点       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Backward Compatibility

### Checkpoint Migration

```python
# 在 load_state_dict 中处理旧 checkpoint
def _migrate_checkpoint(state_dict):
    if 'tangent_vectors' in state_dict:
        old_shape = state_dict['tangent_vectors'].shape[0]
        if old_shape < self.n_total:
            # 扩展 embedding，虚拟节点用默认初始化
            expanded = self._init_virtual_tangent_vectors()
            expanded[:old_shape] = state_dict['tangent_vectors']
            state_dict['tangent_vectors'] = expanded
    return state_dict
```

### Feature Flag

所有新功能通过 `include_virtual` 参数控制：

- `False`：走原有代码路径，行为完全不变
- `True`：启用虚拟节点功能

---

## 8. Success Metrics

| 指标  | 目标  |
| --- | --- |
| `entail_loss` | 从 0 变为正值（训练初期约 0.1-1.0） |
| `pos_loss` | 从 0 变为正值 |
| 训练收敛 | loss 持续下降，无震荡 |
| 回归测试 | `include_virtual=False` 时 loss 与修改前一致 |

---

## 9. Files to Modify

| 文件  | 变更类型 | 变更摘要 |
| --- | --- | --- |
| `pasco/data/body/organ_hierarchy.py` | 扩展  | 虚拟节点注册 + full hierarchy 函数 |
| `pasco/models/hyperbolic/label_embedding.py` | 扩展  | include_virtual 支持 |
| `pasco/loss/entailment_cone_loss.py` | 扩展  | include_virtual 支持 |
| `pasco/models/body_net_hyperbolic.py` | 扩展  | 整合虚拟节点 |
| `scripts/body/train_body.py` | 扩展  | 配置参数 |

**测试文件：** 详见 [vir_emb-strategy.md](./vir_emb-strategy.md#phase-实现顺序)

---

## 10. Open Questions (已确定)

> 以下问题已基于代码分析确定方案：

### Q1: 虚拟节点学习率

**结论：先用统一学习率**

**当前实现分析：**
- 使用统一的 `AdamW` 优化器，lr=1e-4
- 已有 warmup (5 epochs) + cosine annealing 调度
- 无参数组差异化学习率机制

**理由：**
- AdamW 具有自适应学习率特性，会自动调整梯度较大/较小的参数
- 虚拟节点数量少（21个），梯度相对稳定
- 如果训练中出现不稳定，可以后期添加参数组差异化

**备选方案（如需差异化）：**
```python
# 修改 configure_optimizers:
param_groups = [
    {'params': self.label_embedding.tangent_vectors[:72], 'lr': self.lr},      # real
    {'params': self.label_embedding.tangent_vectors[72:], 'lr': self.lr * 0.1}, # virtual
]
```

---

### Q2: entailment_weight 调整

**结论：调整内部组件权重，保持总权重不变**

**当前配置：**
| 权重 | 当前值 | 新值 |
|------|--------|------|
| `entailment_weight` | 0.1 | 0.1 (不变) |
| `entail_loss_weight` | 1.0 | **0.1** |
| `contra_loss_weight` | 1.0 | 1.0 (不变) |
| `pos_loss_weight` | 0.5 | **0.1** |

**理由：**
- entail_loss 和 pos_loss 从 0 → 非零，是新增的 loss 组件
- 降低其权重，避免突然主导训练
- contra_loss 已正常工作（193对兄弟关系），保持不变
- 保持 `entailment_weight=0.1` 总权重不变，便于与之前实验对比

---

### Q3: 虚拟节点初始化范围

**结论：使用 `[0.05, 0.08]` 范围，基于深度线性插值**

**当前叶子节点配置：**
- `min_radius = 0.1`（浅层器官）
- `max_radius = 2.0`（深层器官）

**虚拟节点配置：**
```python
virtual_min = 0.05                     # 比叶子节点的 min_radius (0.1) 更小
virtual_max = min_radius * 0.8 = 0.08  # 确保在叶子节点范围之下
```

**理由：**
- 虚拟节点是中间层级（更通用的概念）
- 应该比其后代叶子节点**更靠近原点**
- 距离原点越近 → cone 半角越大 → 覆盖范围越广
- `[0.05, 0.08]` 与叶子节点 `[0.1, 2.0]` 不重叠，保证层级分离

**初始化实现：**
```python
def _init_virtual_tangent_vectors(self):
    virtual_min = 0.05
    virtual_max = self.min_radius * 0.8  # = 0.08

    for i, (name, depth) in enumerate(VIRTUAL_NODE_DEPTHS.items()):
        # 深度越深，范数越大（但仍小于叶子）
        norm = virtual_min + (virtual_max - virtual_min) * (depth / MAX_VIRTUAL_DEPTH)
        direction = torch.randn(self.embed_dim)
        direction = direction / direction.norm()
        self.tangent_vectors.data[self.n_classes + i] = norm * direction
```

**初始化示例：**
| 虚拟节点类型 | 深度 | tangent_norm |
|-------------|------|--------------|
| body (root) | 0 | 0.05 |
| skeletal_system | 1 | ~0.06 |
| ribs_left | 2 | ~0.07 |
| 最深虚拟节点 | MAX | 0.08 |

---

### 确定参数汇总

| 参数 | 值 | 来源 |
|------|-----|------|
| 虚拟节点学习率 | 1e-4 (统一) | Q1 |
| entailment_weight | 0.1 | Q2 |
| entail_loss_weight | 0.1 | Q2 |
| contra_loss_weight | 1.0 | Q2 |
| pos_loss_weight | 0.1 | Q2 |
| virtual_min_radius | 0.05 | Q3 |
| virtual_max_radius | 0.08 | Q3 |