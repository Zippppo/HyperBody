# Virtual Node Embeddings Implementation Strategy

## Overview

本文档是 [vir_emb.md](./vir_emb.md) 实现计划的**工作流指南**，严格遵循 TDD（测试驱动开发）流程。
所有测试在conda的pasco环境中进行。

---

## TDD 工作流原则

```
┌─────────────────────────────────────────────────────────────────┐
│                    TDD Red-Green-Refactor Cycle                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────┐      ┌─────────┐      ┌──────────┐               │
│   │  RED    │ ──▶  │  GREEN  │ ──▶  │ REFACTOR │ ──▶ 下一任务  │
│   │ 写测试  │      │ 写实现  │      │  优化    │               │
│   │ (失败)  │      │ (通过)  │      │  代码    │               │
│   └─────────┘      └─────────┘      └──────────┘               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**核心原则：**

1. **先写测试，后写实现** - 任何功能代码之前必须有对应测试
2. **最小实现** - 只写让测试通过的最少代码
3. **逐步推进** - 一次只实现一个测试用例
4. **持续验证** - 每次修改后运行测试

---

## Phase 实现顺序

| Phase | 模块 | 前置依赖 | 测试文件 |
|-------|------|----------|----------|
| 1 | `organ_hierarchy.py` | 无 | `test_organ_hierarchy_virtual.py` |
| 2 | `label_embedding.py` | Phase 1 | `test_label_embedding_virtual.py` |
| 3 | `entailment_cone_loss.py` | Phase 1, 2 | `test_entailment_cone_virtual.py` |
| 4 | `body_net_hyperbolic.py` | Phase 1, 2, 3 | `test_body_net_virtual.py` |
| 5 | `train_body.py` | Phase 4 | `test_train_config_virtual.py` |
| 6 | 集成测试 | 全部 | `test_virtual_integration.py` |

---

## Phase 1: Hierarchy Extension

### Step 1.1: 写测试 (RED)

**文件：** `tests/hyperbolic/Lorentz/test_organ_hierarchy_virtual.py`

```python
# 测试用例清单
def test_virtual_node_registry_exists():
    """验证 VIRTUAL_NODE_REGISTRY 存在且非空"""

def test_virtual_node_count():
    """验证虚拟节点数量 == 21"""

def test_virtual_node_ids_start_from_n():
    """验证虚拟节点 ID 从 len(CLASS_NAMES) 开始"""

def test_n_total_embeddings():
    """验证 N_TOTAL_EMBEDDINGS == 72 + 21 = 93"""

def test_get_full_ancestor_descendant_pairs_not_empty():
    """验证 get_full_ancestor_descendant_pairs() 返回 > 0 对"""

def test_full_pairs_contain_virtual_to_leaf():
    """验证包含 virtual→leaf 关系"""

def test_full_pairs_contain_virtual_to_virtual():
    """验证包含 virtual→virtual 关系"""

def test_virtual_node_depths():
    """验证 VIRTUAL_NODE_DEPTHS 正确记录深度"""
```

### Step 1.2: 运行测试验证失败

```bash
pytest tests/hyperbolic/Lorentz/test_organ_hierarchy_virtual.py -v
# 预期：全部 FAILED（函数/变量不存在）
```

### Step 1.3: 实现功能 (GREEN)

**文件：** `pasco/data/body/organ_hierarchy.py`

实现以下内容：
- `build_virtual_node_registry()` 函数
- `get_full_parent_child_pairs()` 函数
- `get_full_ancestor_descendant_pairs()` 函数
- 模块级变量：`N_VIRTUAL_NODES`, `N_TOTAL_EMBEDDINGS`, `VIRTUAL_NODE_NAMES`, `VIRTUAL_NODE_DEPTHS`

### Step 1.4: 运行测试验证通过

```bash
pytest tests/hyperbolic/Lorentz/test_organ_hierarchy_virtual.py -v
# 预期：全部 PASSED
```

### Step 1.5: 验收标准

- [ ] `N_VIRTUAL_NODES == 21`
- [ ] `N_TOTAL_EMBEDDINGS == 93`
- [ ] `len(get_full_ancestor_descendant_pairs()) > 0`
- [ ] 包含 virtual→leaf 和 virtual→virtual 关系

---

## Phase 2: Embedding Extension

### Step 2.1: 写测试 (RED)

**文件：** `tests/hyperbolic/Lorentz/test_label_embedding_virtual.py`

```python
# 测试用例清单
def test_embedding_shape_without_virtual():
    """include_virtual=False 时形状为 [72, D]"""

def test_embedding_shape_with_virtual():
    """include_virtual=True 时形状为 [93, D]"""

def test_get_real_embeddings_shape():
    """get_real_embeddings() 返回 [72, D]"""

def test_get_all_embeddings_shape():
    """get_all_embeddings() 返回 [93, D]"""

def test_virtual_embeddings_closer_to_origin():
    """虚拟节点 embedding 距原点更近（tangent norm 更小）"""

def test_virtual_init_by_depth():
    """浅层虚拟节点比深层虚拟节点距原点更近"""

def test_backward_compatibility_false():
    """include_virtual=False 时行为不变"""

def test_embeddings_on_lorentz_manifold():
    """所有 embedding 满足 Lorentz 约束 x₀² - ||x||² = 1"""
```

### Step 2.2: 运行测试验证失败

```bash
pytest tests/hyperbolic/Lorentz/test_label_embedding_virtual.py -v
# 预期：全部 FAILED
```

### Step 2.3: 实现功能 (GREEN)

**文件：** `pasco/models/hyperbolic/label_embedding.py`

修改 `LorentzLabelEmbedding` 类：
- 新增参数 `include_virtual: bool = False`
- 扩展 `tangent_vectors` 形状
- 实现虚拟节点初始化逻辑
- 新增 `get_real_embeddings()`, `get_all_embeddings()` 方法

### Step 2.4: 运行测试验证通过

```bash
pytest tests/hyperbolic/Lorentz/test_label_embedding_virtual.py -v
# 预期：全部 PASSED
```

### Step 2.5: 验收标准

- [ ] `include_virtual=True` 时 `forward()` 返回 `[93, D]`
- [ ] `get_real_embeddings()` 返回 `[72, D]`
- [ ] 虚拟节点 embedding 距原点更近

---

## Phase 3: Loss Extension

### Step 3.1: 写测试 (RED)

**文件：** `tests/hyperbolic/Lorentz/test_entailment_cone_virtual.py`

```python
# 测试用例清单
def test_entail_loss_zero_without_virtual():
    """include_virtual=False 时 entail_loss == 0"""

def test_entail_loss_nonzero_with_virtual():
    """include_virtual=True 时 entail_loss > 0"""

def test_pos_loss_nonzero_with_virtual():
    """include_virtual=True 时 pos_loss > 0"""

def test_contra_loss_unchanged():
    """contra_loss 在两种模式下行为一致"""

def test_sibling_pairs_only_leaves():
    """sibling_pairs 仍然只包含叶子节点"""

def test_entail_pairs_include_virtual():
    """include_virtual=True 时 entail_pairs 包含虚拟节点"""

def test_loss_backward_pass():
    """loss 可以正常反向传播"""
```

### Step 3.2: 运行测试验证失败

```bash
pytest tests/hyperbolic/Lorentz/test_entailment_cone_virtual.py -v
# 预期：全部 FAILED
```

### Step 3.3: 实现功能 (GREEN)

**文件：** `pasco/loss/entailment_cone_loss.py`

修改 `EntailmentConeLoss` 类：
- 新增参数 `include_virtual: bool = False`
- 使用 `get_full_ancestor_descendant_pairs()` 获取扩展对
- sibling_pairs 保持不变

### Step 3.4: 运行测试验证通过

```bash
pytest tests/hyperbolic/Lorentz/test_entailment_cone_virtual.py -v
# 预期：全部 PASSED
```

### Step 3.5: 验收标准

- [ ] `include_virtual=True` 时 `entail_loss > 0`
- [ ] `include_virtual=True` 时 `pos_loss > 0`
- [ ] `include_virtual=False` 时行为不变

---

## Phase 4: Model Integration

### Step 4.1: 写测试 (RED)

**文件：** `tests/hyperbolic/Lorentz/test_body_net_virtual.py`

```python
# 测试用例清单
def test_model_init_with_virtual():
    """模型可以用 include_virtual_nodes=True 初始化"""

def test_model_init_without_virtual():
    """模型可以用 include_virtual_nodes=False 初始化"""

def test_training_step_no_error():
    """training_step 无报错"""

def test_real_embeddings_for_ranking_loss():
    """LorentzRankingLoss 使用 [72, D] 的 real embeddings"""

def test_all_embeddings_for_entailment_loss():
    """EntailmentConeLoss 使用 [93, D] 的 all embeddings"""

def test_loss_components_logged():
    """loss 日志中 entail_loss 和 pos_loss 非零"""
```

### Step 4.2: 运行测试验证失败

```bash
pytest tests/hyperbolic/Lorentz/test_body_net_virtual.py -v
# 预期：全部 FAILED
```

### Step 4.3: 实现功能 (GREEN)

**文件：** `pasco/models/body_net_hyperbolic.py`

修改模型：
- 新增参数 `include_virtual_nodes: bool = True`
- 传递给 `LorentzLabelEmbedding` 和 `EntailmentConeLoss`
- `training_step` 中分离两种 embedding 用途

### Step 4.4: 运行测试验证通过

```bash
pytest tests/hyperbolic/Lorentz/test_body_net_virtual.py -v
# 预期：全部 PASSED
```

### Step 4.5: 验收标准

- [ ] 端到端训练无报错
- [ ] loss 日志中 entail_loss 和 pos_loss 非零

---

## Phase 5: Configuration Support

### Step 5.1: 写测试 (RED)

**文件：** `tests/hyperbolic/Lorentz/test_train_config_virtual.py`

```python
# 测试用例清单
def test_config_include_virtual_default_true():
    """默认 include_virtual_nodes=True"""

def test_config_include_virtual_false():
    """可以设置 include_virtual_nodes=False"""

def test_config_passed_to_model():
    """配置正确传递给模型"""
```

### Step 5.2-5.4: 实现并验证

**文件：** `scripts/body/train_body.py`

---

## Phase 6: 集成测试

### Step 6.1: 写集成测试

**文件：** `tests/hyperbolic/Lorentz/test_virtual_integration.py`

```python
# 集成测试用例
def test_1_epoch_training_smoke():
    """1 epoch 训练 smoke test"""

def test_loss_curve_reasonable():
    """loss 曲线合理（持续下降，无震荡）"""

def test_regression_without_virtual():
    """include_virtual=False 时 loss 与修改前一致"""
```

### Step 6.2: 运行全部测试

```bash
# 运行所有虚拟节点相关测试
pytest tests/hyperbolic/Lorentz/test_*virtual*.py -v

# 运行完整测试套件确保无回归
pytest tests/ -v --ignore=tests/integration
```

---

## 检查点清单

每个 Phase 完成后，勾选以下检查点：

### Phase 1 检查点
- [ ] 测试文件已创建
- [ ] 测试运行失败（RED）
- [ ] 功能已实现
- [ ] 测试运行通过（GREEN）
- [ ] 代码已 review

### Phase 2 检查点
- [ ] 测试文件已创建
- [ ] 测试运行失败（RED）
- [ ] 功能已实现
- [ ] 测试运行通过（GREEN）
- [ ] 代码已 review

### Phase 3 检查点
- [ ] 测试文件已创建
- [ ] 测试运行失败（RED）
- [ ] 功能已实现
- [ ] 测试运行通过（GREEN）
- [ ] 代码已 review

### Phase 4 检查点
- [ ] 测试文件已创建
- [ ] 测试运行失败（RED）
- [ ] 功能已实现
- [ ] 测试运行通过（GREEN）
- [ ] 代码已 review

### Phase 5 检查点
- [ ] 测试文件已创建
- [ ] 测试运行失败（RED）
- [ ] 功能已实现
- [ ] 测试运行通过（GREEN）
- [ ] 代码已 review

### Phase 6 检查点
- [ ] 集成测试通过
- [ ] 回归测试通过
- [ ] Smoke test 通过

---

## 常用命令

```bash
# 运行单个测试文件
pytest tests/hyperbolic/Lorentz/test_organ_hierarchy_virtual.py -v

# 运行单个测试函数
pytest tests/hyperbolic/Lorentz/test_organ_hierarchy_virtual.py::test_virtual_node_count -v

# 运行并显示 print 输出
pytest tests/hyperbolic/Lorentz/test_organ_hierarchy_virtual.py -v -s

# 运行测试覆盖率
pytest tests/hyperbolic/Lorentz/test_*virtual*.py --cov=pasco --cov-report=term-missing

# 快速失败模式（第一个失败就停止）
pytest tests/hyperbolic/Lorentz/test_organ_hierarchy_virtual.py -v -x
```

---

## 注意事项

1. **不要跳过 RED 阶段** - 必须先看到测试失败，确认测试逻辑正确
2. **每次只实现一个测试** - 避免过度实现
3. **保持测试独立** - 测试之间不应有依赖
4. **失败时回滚** - 如果实现导致其他测试失败，立即回滚
5. **先写简单测试** - 从最基础的功能开始

---

## 参考文档

- 实现计划：[vir_emb.md](./vir_emb.md)
- TDD 指南：[.claude/agents/tdd-guide.md](../.claude/agents/tdd-guide.md)
