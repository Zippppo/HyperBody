# Entailment Cone 测试驱动开发策略

## 概述

本文档定义 Entailment Cone 功能的 **测试驱动开发 (TDD)** 工作流，确保每个任务在实现前先编写测试，遵循 Red-Green-Refactor 循环。

---

## TDD 核心原则

```
┌─────────────────────────────────────────────────────────────┐
│                    TDD 循环 (Red-Green-Refactor)              │
├─────────────────────────────────────────────────────────────┤
│  1. RED:    编写失败的测试（定义预期行为）                      │
│  2. GREEN:  编写最小实现使测试通过                             │
│  3. REFACTOR: 重构代码，保持测试通过                           │
│  4. REPEAT: 重复循环直到任务完成                               │
└─────────────────────────────────────────────────────────────┘
```

### 质量门禁

| 检查点 | 要求 |
|--------|------|
| 测试覆盖率 | 每个任务 ≥ 80% |
| 测试独立性 | 测试间无共享状态 |
| 边界覆盖 | 必须包含 null/空值/极值测试 |
| 梯度验证 | 所有可微函数必须验证梯度流 |

---

## 任务执行顺序与依赖图

```
                    ┌──────────────────┐
                    │   任务 0 (ADR)    │  ✓ 已完成
                    │   cone.md        │
                    └────────┬─────────┘
                             │
            ┌────────────────┴────────────────┐
            │                                 │
            ▼                                 ▼
    ┌───────────────┐                 ┌───────────────┐
    │   任务 1       │                 │   任务 2       │
    │  lorentz_ops  │                 │   hierarchy   │
    │  (可并行)      │                 │   (可并行)     │
    └───────┬───────┘                 └───────┬───────┘
            │                                 │
            └────────────────┬────────────────┘
                             │
                             ▼
                    ┌───────────────────┐
                    │      任务 3        │
                    │ EntailmentConeLoss│
                    └────────┬──────────┘
                             │
                             ▼
                    ┌───────────────────┐
                    │      任务 4        │
                    │  BodyNet 集成     │
                    └────────┬──────────┘
                             │
                             ▼
                    ┌───────────────────┐
                    │      任务 5        │
                    │  集成测试 & 数值   │
                    │  稳定性测试        │
                    └───────────────────┘
```

---

## 详细工作流

### 阶段 1: 基础扩展 (任务 1 & 2 可并行)

---

#### 任务 1: 扩展 lorentz_ops.py

##### Step 1.1: 编写 `half_aperture` 测试 (RED)

**文件**: `tests/hyperbolic/test_entailment_cone.py`

```python
# ============================================================
# 测试 1.1: half_aperture 函数测试
# ============================================================

class TestHalfAperture:
    """半圆锥角计算测试"""

    def test_output_shape(self):
        """输出形状应为 (B,)"""
        x = torch.randn(10, 32) * 0.5
        phi = half_aperture(x, curv=1.0)
        assert phi.shape == (10,)

    def test_output_range(self):
        """输出应在 (0, π/2) 范围内"""
        x = torch.randn(100, 32) * 0.5
        phi = half_aperture(x, curv=1.0)
        assert (phi > 0).all()
        assert (phi < math.pi / 2).all()

    def test_monotonicity(self):
        """范数越大，半锥角越小（更具体的概念）"""
        direction = torch.randn(32)
        direction = direction / direction.norm()

        x_small = direction.unsqueeze(0) * 0.1
        x_large = direction.unsqueeze(0) * 2.0

        phi_small = half_aperture(x_small, curv=1.0).item()
        phi_large = half_aperture(x_large, curv=1.0).item()

        assert phi_small > phi_large, \
            f"Expected {phi_small} > {phi_large}"

    def test_numerical_stability_zero_vector(self):
        """零向量应返回有效值（不产生 NaN/Inf）"""
        x = torch.zeros(10, 32)
        phi = half_aperture(x, curv=1.0)
        assert torch.isfinite(phi).all()

    def test_numerical_stability_large_norm(self):
        """大范数输入应数值稳定"""
        x = torch.randn(10, 32) * 100
        phi = half_aperture(x, curv=1.0)
        assert torch.isfinite(phi).all()
        assert (phi > 0).all()

    def test_gradient_flow(self):
        """梯度应能正确反传"""
        x = (torch.randn(10, 32) * 0.5).requires_grad_(True)
        phi = half_aperture(x, curv=1.0)
        loss = phi.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_different_curvatures(self):
        """不同曲率应产生有效输出"""
        x = torch.randn(10, 32) * 0.5
        for curv in [0.5, 1.0, 2.0]:
            phi = half_aperture(x, curv=curv)
            assert torch.isfinite(phi).all()
            assert (phi > 0).all()
            assert (phi < math.pi / 2).all()
```

**运行测试**:
```bash
pytest tests/hyperbolic/test_entailment_cone.py::TestHalfAperture -v
# 预期: 全部 FAILED (函数未实现)
```

##### Step 1.2: 实现 `half_aperture` (GREEN)

**文件**: `pasco/models/hyperbolic/lorentz_ops.py`

```python
def half_aperture(
    x: Tensor,
    curv: Union[float, Tensor] = 1.0,
    min_radius: float = 0.1,
    eps: float = 1e-7
) -> Tensor:
    """
    计算双曲空间中点形成的蕴含锥的半圆锥角。

    Args:
        x: (B, D) 空间分量
        curv: 曲率
        min_radius: 原点附近最小半径
        eps: 数值稳定性参数

    Returns:
        (B,) 半圆锥角，范围 (0, π/2)
    """
    # 计算空间范数
    x_norm = torch.norm(x, dim=-1)  # [B]

    # arcsin 输入: 2 * min_radius / (||x|| * sqrt(curv))
    sin_input = 2 * min_radius / (x_norm * curv ** 0.5 + eps)

    # clamp 到有效范围
    sin_input = torch.clamp(sin_input, min=eps, max=1 - eps)

    return torch.asin(sin_input)
```

**运行测试**:
```bash
pytest tests/hyperbolic/test_entailment_cone.py::TestHalfAperture -v
# 预期: 全部 PASSED
```

##### Step 1.3: 编写 `oxy_angle` 测试 (RED)

```python
class TestOxyAngle:
    """外角计算测试"""

    def test_output_shape(self):
        """输出形状应为 (B,)"""
        x = torch.randn(10, 32) * 0.5
        y = torch.randn(10, 32) * 0.5
        theta = oxy_angle(x, y, curv=1.0)
        assert theta.shape == (10,)

    def test_output_range(self):
        """输出应在 (0, π) 范围内"""
        x = torch.randn(100, 32) * 0.5
        y = torch.randn(100, 32) * 0.5
        theta = oxy_angle(x, y, curv=1.0)
        assert (theta >= 0).all()
        assert (theta <= math.pi + 1e-5).all()

    def test_same_point_zero_angle(self):
        """x=y 时外角应接近 0"""
        x = torch.randn(10, 32) * 0.5
        theta = oxy_angle(x, x, curv=1.0)
        assert torch.allclose(theta, torch.zeros_like(theta), atol=1e-3)

    def test_gradient_flow(self):
        """梯度应能正确反传"""
        x = (torch.randn(10, 32) * 0.5).requires_grad_(True)
        y = (torch.randn(10, 32) * 0.5).requires_grad_(True)
        theta = oxy_angle(x, y, curv=1.0)
        loss = theta.sum()
        loss.backward()
        assert x.grad is not None
        assert y.grad is not None
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(y.grad).all()

    def test_numerical_stability_near_origin(self):
        """接近原点时应数值稳定"""
        x = torch.randn(10, 32) * 1e-6
        y = torch.randn(10, 32) * 0.5
        theta = oxy_angle(x, y, curv=1.0)
        assert torch.isfinite(theta).all()

    def test_numerical_stability_large_values(self):
        """大值输入应数值稳定"""
        x = torch.randn(10, 32) * 10
        y = torch.randn(10, 32) * 10
        theta = oxy_angle(x, y, curv=1.0)
        assert torch.isfinite(theta).all()
```

##### Step 1.4: 实现 `oxy_angle` (GREEN)

**运行测试**:
```bash
pytest tests/hyperbolic/test_entailment_cone.py::TestOxyAngle -v
# 预期: 全部 PASSED
```

##### Step 1.5: 验收检查点

```bash
# 运行任务 1 全部测试
pytest tests/hyperbolic/test_entailment_cone.py -k "HalfAperture or OxyAngle" -v

# 检查覆盖率
pytest tests/hyperbolic/test_entailment_cone.py --cov=pasco.models.hyperbolic.lorentz_ops --cov-report=term-missing

# 验收标准:
# ✓ 所有测试通过
# ✓ half_aperture 和 oxy_angle 覆盖率 ≥ 80%
# ✓ 无 NaN/Inf 警告
```

---

#### 任务 2: 扩展 organ_hierarchy.py

##### Step 2.1: 编写层次关系提取测试 (RED)

**文件**: `tests/hyperbolic/test_entailment_cone.py`

```python
# ============================================================
# 测试 2: hierarchy 工具函数测试
# ============================================================

class TestParentChildPairs:
    """父子蕴含关系测试"""

    def test_not_empty(self):
        """应返回非空列表"""
        pairs = get_parent_child_pairs()
        assert len(pairs) > 0

    def test_returns_list_of_tuples(self):
        """应返回 List[Tuple[int, int]]"""
        pairs = get_parent_child_pairs()
        assert isinstance(pairs, list)
        for pair in pairs:
            assert isinstance(pair, tuple)
            assert len(pair) == 2
            assert isinstance(pair[0], int)
            assert isinstance(pair[1], int)

    def test_valid_class_ids(self):
        """所有 ID 应在有效范围内"""
        pairs = get_parent_child_pairs()
        for parent_id, child_id in pairs:
            assert 2 <= parent_id < N_CLASSES
            assert 2 <= child_id < N_CLASSES

    def test_no_excluded_ids(self):
        """不应包含排除的 class_id (0, 1)"""
        pairs = get_parent_child_pairs()
        for parent_id, child_id in pairs:
            assert parent_id not in {0, 1}
            assert child_id not in {0, 1}

    def test_no_self_pairs(self):
        """不应有 (a, a) 形式的对"""
        pairs = get_parent_child_pairs()
        for parent_id, child_id in pairs:
            assert parent_id != child_id

    def test_expected_count_range(self):
        """数量应在预期范围内 (80-120)"""
        pairs = get_parent_child_pairs()
        assert 50 <= len(pairs) <= 200, f"Got {len(pairs)} pairs"

    def test_is_cached(self):
        """多次调用应返回相同对象"""
        pairs1 = get_parent_child_pairs()
        pairs2 = get_parent_child_pairs()
        assert pairs1 is pairs2  # 同一对象（缓存）


class TestSiblingPairs:
    """兄弟互斥关系测试"""

    def test_not_empty(self):
        """应返回非空列表"""
        pairs = get_sibling_pairs()
        assert len(pairs) > 0

    def test_unique_ordering(self):
        """应只有 (a, b) 形式，其中 a < b"""
        pairs = get_sibling_pairs()
        for a, b in pairs:
            assert a < b, f"Invalid ordering: ({a}, {b})"

    def test_no_excluded_ids(self):
        """不应包含排除的 class_id"""
        pairs = get_sibling_pairs()
        for a, b in pairs:
            assert a not in {0, 1}
            assert b not in {0, 1}

    def test_no_duplicates(self):
        """不应有重复对"""
        pairs = get_sibling_pairs()
        assert len(pairs) == len(set(pairs))


class TestAncestorDescendantPairs:
    """祖孙蕴含关系测试"""

    def test_includes_parent_child(self):
        """应包含所有父子关系"""
        parent_child = set(get_parent_child_pairs())
        ancestor_descendant = set(get_ancestor_descendant_pairs())
        assert parent_child.issubset(ancestor_descendant)

    def test_transitive_closure(self):
        """应包含跨层级关系"""
        pairs = get_ancestor_descendant_pairs()
        # 如果 A→B 和 B→C 存在，则 A→C 应该也存在
        # (完整的传递闭包验证)
        assert len(pairs) >= len(get_parent_child_pairs())
```

##### Step 2.2: 实现层次关系提取 (GREEN)

**文件**: `pasco/data/body/organ_hierarchy.py`

```python
from functools import lru_cache
from typing import List, Tuple

EXCLUDED_CLASS_IDS = {0, 1}  # outside_body, inside_body_empty

@lru_cache(maxsize=1)
def get_parent_child_pairs() -> List[Tuple[int, int]]:
    """提取所有父子蕴含关系对"""
    # ... 实现
    pass

@lru_cache(maxsize=1)
def get_sibling_pairs() -> List[Tuple[int, int]]:
    """提取所有兄弟互斥关系对"""
    # ... 实现
    pass

@lru_cache(maxsize=1)
def get_ancestor_descendant_pairs() -> List[Tuple[int, int]]:
    """提取所有祖孙蕴含关系对"""
    # ... 实现
    pass
```

##### Step 2.3: 验收检查点

```bash
# 运行任务 2 全部测试
pytest tests/hyperbolic/test_entailment_cone.py -k "ParentChild or Sibling or Ancestor" -v

# 验收标准:
# ✓ 所有测试通过
# ✓ 父子对数量在预期范围内
# ✓ 函数正确缓存
```

---

### 阶段 2: 核心实现 (任务 3)

##### Step 3.1: 编写 EntailmentConeLoss 测试 (RED)

```python
# ============================================================
# 测试 3: EntailmentConeLoss 测试
# ============================================================

class TestEntailmentConeLoss:
    """蕴含锥损失模块测试"""

    @pytest.fixture
    def loss_fn(self):
        return EntailmentConeLoss(min_radius=0.1, margin=0.1)

    @pytest.fixture
    def mock_embeddings(self):
        # 模拟 72 个类的嵌入
        return torch.randn(72, 32) * 0.5

    def test_forward_returns_dict(self, loss_fn, mock_embeddings):
        """应返回包含三种损失的字典"""
        losses = loss_fn(mock_embeddings, curv=1.0)
        assert isinstance(losses, dict)
        assert 'entail' in losses
        assert 'contra' in losses
        assert 'pos' in losses

    def test_forward_all_losses_scalar(self, loss_fn, mock_embeddings):
        """所有损失应是标量 tensor"""
        losses = loss_fn(mock_embeddings, curv=1.0)
        for name, loss in losses.items():
            assert loss.dim() == 0, f"{name} is not scalar"

    def test_forward_losses_non_negative(self, loss_fn, mock_embeddings):
        """所有损失应 >= 0"""
        losses = loss_fn(mock_embeddings, curv=1.0)
        for name, loss in losses.items():
            assert loss >= 0, f"{name} is negative: {loss}"

    def test_gradient_flows_to_embeddings(self, loss_fn):
        """梯度应传播到 embeddings"""
        emb = (torch.randn(72, 32) * 0.5).requires_grad_(True)
        losses = loss_fn(emb, curv=1.0)
        total = losses['entail'] + losses['contra'] + losses['pos']
        total.backward()
        assert emb.grad is not None
        assert torch.isfinite(emb.grad).all()

    def test_empty_pairs_no_error(self):
        """空关系对时应返回 0 损失"""
        # 使用 mock 的空关系对
        loss_fn = EntailmentConeLoss(min_radius=0.1)
        # 假设测试环境下关系对为空
        emb = torch.randn(5, 32) * 0.5
        losses = loss_fn(emb, curv=1.0)
        # 应该不报错，返回 0

    def test_device_transfer(self, loss_fn, mock_embeddings):
        """设备转移后 buffer 应正确转移"""
        if torch.cuda.is_available():
            loss_fn = loss_fn.cuda()
            emb = mock_embeddings.cuda()
            losses = loss_fn(emb, curv=1.0)
            assert all(l.device.type == 'cuda' for l in losses.values())


class TestEntailmentConeLossNumericalStability:
    """数值稳定性测试"""

    def test_curv_near_zero(self):
        """小曲率不应产生 NaN"""
        loss_fn = EntailmentConeLoss()
        emb = torch.randn(72, 32) * 0.5
        losses = loss_fn(emb, curv=0.01)
        for name, loss in losses.items():
            assert torch.isfinite(loss), f"{name} is NaN/Inf"

    def test_embedding_near_origin(self):
        """小范数嵌入不应导致除零"""
        loss_fn = EntailmentConeLoss()
        emb = torch.randn(72, 32) * 1e-6
        losses = loss_fn(emb, curv=1.0)
        for name, loss in losses.items():
            assert torch.isfinite(loss), f"{name} is NaN/Inf"

    def test_embedding_far_from_origin(self):
        """大范数嵌入不应导致溢出"""
        loss_fn = EntailmentConeLoss()
        emb = torch.randn(72, 32) * 100
        losses = loss_fn(emb, curv=1.0)
        for name, loss in losses.items():
            assert torch.isfinite(loss), f"{name} is NaN/Inf"

    def test_identical_embeddings(self):
        """相同嵌入不应导致 NaN"""
        loss_fn = EntailmentConeLoss()
        emb = torch.randn(1, 32).expand(72, -1).clone() * 0.5
        losses = loss_fn(emb, curv=1.0)
        for name, loss in losses.items():
            assert torch.isfinite(loss), f"{name} is NaN/Inf"
```

##### Step 3.2: 实现 EntailmentConeLoss (GREEN)

**文件**: `pasco/loss/entailment_cone_loss.py`

##### Step 3.3: 验收检查点

```bash
# 运行任务 3 全部测试
pytest tests/hyperbolic/test_entailment_cone.py -k "EntailmentConeLoss" -v

# 性能基准测试 (可选)
pytest tests/hyperbolic/test_entailment_cone.py -k "performance" -v --benchmark

# 验收标准:
# ✓ 所有测试通过
# ✓ 300 对关系计算 < 2ms
# ✓ 所有数值稳定性测试通过
```

---

### 阶段 3: 集成 (任务 4)

##### Step 4.1: 编写集成测试 (RED)

```python
# ============================================================
# 测试 4: BodyNetHyperbolic 集成测试
# ============================================================

class TestBodyNetHyperbolicWithCone:
    """BodyNetHyperbolic + EntailmentCone 集成测试"""

    @pytest.fixture
    def model_with_cone(self):
        return BodyNetHyperbolic(
            use_entailment_cone=True,
            entailment_weight=0.1,
        )

    @pytest.fixture
    def model_without_cone(self):
        return BodyNetHyperbolic(
            use_entailment_cone=False,
        )

    @pytest.fixture
    def mock_batch(self):
        return {
            'image': torch.randn(2, 1, 64, 64, 64),
            'label': torch.randint(0, 72, (2, 64, 64, 64)),
        }

    def test_training_step_with_cone(self, model_with_cone, mock_batch):
        """启用时能正常完成 training_step"""
        model_with_cone.train()
        loss = model_with_cone.training_step(mock_batch, 0)
        assert torch.isfinite(loss)

    def test_training_step_without_cone(self, model_without_cone, mock_batch):
        """禁用时应与原有行为一致"""
        model_without_cone.train()
        loss = model_without_cone.training_step(mock_batch, 0)
        assert torch.isfinite(loss)

    def test_gradient_path(self, model_with_cone, mock_batch):
        """从 total_loss 到 label_emb 的梯度路径应完整"""
        model_with_cone.train()
        loss = model_with_cone.training_step(mock_batch, 0)
        loss.backward()

        # 验证 label_emb 有梯度
        label_emb_grad = model_with_cone.label_emb.emb.weight.grad
        assert label_emb_grad is not None
        assert torch.isfinite(label_emb_grad).all()

    def test_logging_keys(self, model_with_cone, mock_batch):
        """日志应包含所有 cone_loss 相关 key"""
        model_with_cone.train()
        # 执行 training_step 并检查 logged 指标
        # (具体实现取决于 logging 机制)
```

##### Step 4.2: 实现集成 (GREEN)

**文件**: `pasco/models/body_net_hyperbolic.py`

##### Step 4.3: 验收检查点

```bash
# 运行任务 4 全部测试
pytest tests/hyperbolic/test_entailment_cone.py -k "BodyNetHyperbolic" -v

# 完整回归测试
pytest tests/hyperbolic/ -v

# 验收标准:
# ✓ 所有集成测试通过
# ✓ 现有测试无回归
# ✓ 日志正确记录
```

---

### 阶段 4: 最终验证 (任务 5)

##### Step 5.1: 运行完整测试套件

```bash
# 全部蕴含锥相关测试
pytest tests/hyperbolic/test_entailment_cone.py -v

# 全部双曲相关测试
pytest tests/hyperbolic/ -v

# 完整项目测试
pytest tests/ -v
```

##### Step 5.2: 覆盖率报告

```bash
pytest tests/hyperbolic/test_entailment_cone.py \
    --cov=pasco.models.hyperbolic.lorentz_ops \
    --cov=pasco.data.body.organ_hierarchy \
    --cov=pasco.loss.entailment_cone_loss \
    --cov=pasco.models.body_net_hyperbolic \
    --cov-report=html \
    --cov-report=term-missing

# 打开覆盖率报告
# open htmlcov/index.html
```

##### Step 5.3: 性能验证

```bash
# 基准测试
python -c "
import torch
import time
from pasco.loss.entailment_cone_loss import EntailmentConeLoss

loss_fn = EntailmentConeLoss()
emb = torch.randn(72, 32).cuda()
loss_fn = loss_fn.cuda()

# 预热
for _ in range(10):
    loss_fn(emb, curv=1.0)

# 计时
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(100):
    loss_fn(emb, curv=1.0)
torch.cuda.synchronize()
end = time.perf_counter()

print(f'Average time: {(end - start) / 100 * 1000:.2f}ms')
# 预期: < 2ms
"
```

---

## 完整执行时间线

```
Day 1: 任务 1 + 任务 2 (并行)
├── 上午: 编写 lorentz_ops 测试 (RED)
├── 下午: 实现 lorentz_ops 函数 (GREEN)
├── 上午: 编写 hierarchy 测试 (RED) [并行]
└── 下午: 实现 hierarchy 函数 (GREEN) [并行]

Day 2: 任务 3
├── 上午: 编写 EntailmentConeLoss 测试 (RED)
└── 下午: 实现 EntailmentConeLoss (GREEN)

Day 3: 任务 4 + 任务 5
├── 上午: 编写集成测试 (RED)
├── 下午: 集成到 BodyNetHyperbolic (GREEN)
└── 晚上: 运行完整测试套件，生成覆盖率报告
```

---

## 测试文件组织

```
tests/
└── hyperbolic/
    └── test_entailment_cone.py
        ├── TestHalfAperture          # 任务 1.1
        ├── TestOxyAngle              # 任务 1.2
        ├── TestParentChildPairs      # 任务 2.1
        ├── TestSiblingPairs          # 任务 2.2
        ├── TestAncestorDescendantPairs  # 任务 2.3
        ├── TestEntailmentConeLoss    # 任务 3
        ├── TestEntailmentConeLossNumericalStability  # 任务 3 (稳定性)
        └── TestBodyNetHyperbolicWithCone  # 任务 4
```

---

## 紧急回滚计划

如果某个任务导致现有功能回归：

```bash
# 1. 识别问题提交
git log --oneline -10

# 2. 回滚到上一个稳定状态
git revert HEAD

# 3. 运行测试验证
pytest tests/hyperbolic/ -v

# 4. 分析问题原因后重新实现
```

---

## 质量检查清单

### 每个任务完成前必须检查

- [ ] 所有新功能都有测试覆盖
- [ ] 测试先于实现编写
- [ ] 边界条件已测试 (null, empty, extreme)
- [ ] 梯度流已验证
- [ ] 数值稳定性已测试
- [ ] 无 lint 错误
- [ ] 现有测试无回归
- [ ] 覆盖率 ≥ 80%

### 最终发布前必须检查

- [ ] 完整测试套件通过
- [ ] 性能基准满足要求 (< 2ms)
- [ ] 文档已更新
- [ ] 代码已 review

---

## 参考资源

- 原始实现计划: `current_plan/cone.md`
- TDD 指南: `.claude/agents/tdd-guide.md`
- 现有测试模式: `tests/hyperbolic/test_lorentz_ops.py`
- HyperPath 参考: `REF/ref_repos/HyperPath/`
