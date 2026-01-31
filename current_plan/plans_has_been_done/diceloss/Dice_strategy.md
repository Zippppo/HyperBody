# 多分类Dice Loss实现 - TDD策略

## 概述

本文档定义了实现Dice Loss功能的TDD（测试驱动开发）流程。与 `Dice_loss.md` 配合使用：
- **Dice_loss.md**: 定义"做什么"（功能规格、代码结构）
- **Dice_strategy.md**: 定义"怎么做"（TDD流程、验证步骤）

---

## TDD核心原则

### Red-Green-Refactor 循环

```
┌─────────────────────────────────────────────────────────┐
│  RED: 写失败的测试 → GREEN: 最小实现 → REFACTOR: 优化  │
└─────────────────────────────────────────────────────────┘
```

**关键规则**:
1. **先写测试，后写代码** - 没有测试就不写实现
2. **每次只实现让测试通过的最小代码**
3. **测试通过后再优化**
4. **每个TDD循环独立且可验证**

---

## 实现阶段划分

### 阶段1: 核心Loss函数 (`multi_class_dice_loss`)

对应 `Dice_loss.md` 步骤1

#### TDD循环 1.1: 基本功能

**RED - 写测试** (`tests/loss/test_dice_loss.py`)
```python
import pytest
import torch
from pasco.loss.losses import multi_class_dice_loss

class TestMultiClassDiceLoss:
    """Test multi_class_dice_loss function."""

    def test_output_is_scalar(self):
        """Loss output should be a scalar tensor."""
        B, C, H, W, D = 2, 5, 8, 8, 8
        inputs = torch.randn(B, C, H, W, D)
        targets = torch.randint(0, C, (B, H, W, D))

        loss = multi_class_dice_loss(inputs, targets)

        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.dtype == torch.float32

    def test_loss_range(self):
        """Loss should be in range [0, 1]."""
        B, C, H, W, D = 2, 5, 8, 8, 8
        inputs = torch.randn(B, C, H, W, D)
        targets = torch.randint(0, C, (B, H, W, D))

        loss = multi_class_dice_loss(inputs, targets)

        assert 0.0 <= loss.item() <= 1.0, f"Loss {loss.item()} out of range"
```

**验证失败**
```bash
pytest tests/loss/test_dice_loss.py::TestMultiClassDiceLoss::test_output_is_scalar -v
# 预期: ImportError 或 AssertionError
```

**GREEN - 最小实现** (见 `Dice_loss.md` 步骤1代码)

**验证通过**
```bash
pytest tests/loss/test_dice_loss.py::TestMultiClassDiceLoss::test_output_is_scalar -v
pytest tests/loss/test_dice_loss.py::TestMultiClassDiceLoss::test_loss_range -v
```

---

#### TDD循环 1.2: 完美预测

**RED - 写测试**
```python
def test_perfect_prediction_near_zero_loss(self):
    """Perfect prediction should yield near-zero loss."""
    B, C, H, W, D = 1, 5, 4, 4, 4
    targets = torch.randint(1, C, (B, H, W, D))  # 避开ignore_index=0

    # 创建完美预测的logits
    inputs = torch.zeros(B, C, H, W, D)
    for b in range(B):
        for h in range(H):
            for w in range(W):
                for d in range(D):
                    inputs[b, targets[b, h, w, d], h, w, d] = 10.0

    loss = multi_class_dice_loss(inputs, targets, ignore_index=0)

    assert loss.item() < 0.1, f"Perfect prediction loss {loss.item()} too high"
```

**验证 → 实现 → 验证**

---

#### TDD循环 1.3: 梯度流

**RED - 写测试**
```python
def test_gradient_flows(self):
    """Gradients should flow back to inputs."""
    B, C, H, W, D = 2, 5, 8, 8, 8
    inputs = torch.randn(B, C, H, W, D, requires_grad=True)
    targets = torch.randint(0, C, (B, H, W, D))

    loss = multi_class_dice_loss(inputs, targets)
    loss.backward()

    assert inputs.grad is not None, "No gradient computed"
    assert not torch.isnan(inputs.grad).any(), "Gradient contains NaN"
    assert not torch.isinf(inputs.grad).any(), "Gradient contains Inf"
```

---

#### TDD循环 1.4: ignore_index

**RED - 写测试**
```python
def test_ignore_index_excluded(self):
    """Voxels with ignore_index should not contribute to loss."""
    B, C, H, W, D = 1, 5, 4, 4, 4

    # 全部设为ignore_index=0
    targets = torch.zeros(B, H, W, D, dtype=torch.long)
    inputs = torch.randn(B, C, H, W, D)

    loss = multi_class_dice_loss(inputs, targets, ignore_index=0)

    # 没有有效体素时loss应为0
    assert loss.item() == 0.0, f"Loss should be 0 when all ignored, got {loss.item()}"
```

---

#### TDD循环 1.5: class_weights

**RED - 写测试**
```python
def test_class_weights_applied(self):
    """Class weights should affect the loss value."""
    B, C, H, W, D = 2, 5, 8, 8, 8
    inputs = torch.randn(B, C, H, W, D)
    targets = torch.randint(1, C, (B, H, W, D))  # 避开class 0

    # 无权重
    loss_unweighted = multi_class_dice_loss(inputs, targets)

    # 有权重
    weights = torch.tensor([0.0, 1.0, 2.0, 0.5, 1.5])
    loss_weighted = multi_class_dice_loss(inputs, targets, class_weights=weights)

    assert loss_unweighted.item() != loss_weighted.item(), \
        "Weights should change loss value"
```

---

#### TDD循环 1.6: 数值稳定性

**RED - 写测试**
```python
def test_numerical_stability_large_logits(self):
    """Should handle large logit values without overflow."""
    B, C, H, W, D = 2, 5, 8, 8, 8
    inputs = torch.randn(B, C, H, W, D) * 100  # 大logits
    targets = torch.randint(0, C, (B, H, W, D))

    loss = multi_class_dice_loss(inputs, targets)

    assert not torch.isnan(loss), "Loss is NaN with large logits"
    assert not torch.isinf(loss), "Loss is Inf with large logits"

def test_numerical_stability_small_logits(self):
    """Should handle very small logit values."""
    B, C, H, W, D = 2, 5, 8, 8, 8
    inputs = torch.randn(B, C, H, W, D) * 1e-6  # 小logits
    targets = torch.randint(0, C, (B, H, W, D))

    loss = multi_class_dice_loss(inputs, targets)

    assert not torch.isnan(loss), "Loss is NaN with small logits"
```

---

#### TDD循环 1.7: 内存效率

**RED - 写测试**
```python
@pytest.mark.slow
def test_no_oom_realistic_size(self):
    """Should not OOM on realistic input sizes."""
    B, C, H, W, D = 2, 71, 128, 128, 256  # 实际大小

    # 分块创建避免测试本身OOM
    inputs = torch.randn(B, C, H, W, D, device='cuda' if torch.cuda.is_available() else 'cpu')
    targets = torch.randint(0, C, (B, H, W, D), device=inputs.device)

    # 应该能完成计算
    loss = multi_class_dice_loss(inputs, targets)

    assert loss is not None
```

---

### 阶段2: BodyNet集成

对应 `Dice_loss.md` 步骤2

#### TDD循环 2.1: compute_loss返回格式

**RED - 写测试** (`tests/models/test_body_net.py`)
```python
def test_compute_loss_returns_tuple_when_dice_enabled(self):
    """compute_loss should return (total, ce, dice) tuple when dice enabled."""
    model = BodyNet(n_classes=5, use_dice_loss=True, dice_weight=0.5)

    logits = torch.randn(2, 5, 8, 8, 8)
    labels = torch.randint(0, 5, (2, 8, 8, 8))

    result = model.compute_loss(logits, labels)

    assert isinstance(result, tuple), "Should return tuple when dice enabled"
    assert len(result) == 3, "Should return (total_loss, ce_loss, dice_loss)"

def test_compute_loss_returns_scalar_when_dice_disabled(self):
    """compute_loss should return scalar when dice disabled."""
    model = BodyNet(n_classes=5, use_dice_loss=False)

    logits = torch.randn(2, 5, 8, 8, 8)
    labels = torch.randint(0, 5, (2, 8, 8, 8))

    result = model.compute_loss(logits, labels)

    assert isinstance(result, torch.Tensor), "Should return tensor when dice disabled"
    assert result.dim() == 0, "Should return scalar"
```

---

#### TDD循环 2.2: training_step日志

**RED - 写测试**
```python
def test_training_step_logs_dice_loss(self, mocker):
    """training_step should log dice_loss when enabled."""
    model = BodyNet(n_classes=5, use_dice_loss=True)
    mock_log = mocker.patch.object(model, 'log')

    batch = {
        "occupancy": torch.randn(2, 1, 8, 8, 8),
        "labels": torch.randint(0, 5, (2, 8, 8, 8))
    }

    model.training_step(batch, 0)

    # 验证日志调用
    log_names = [call[0][0] for call in mock_log.call_args_list]
    assert "train/dice_loss" in log_names, "Should log train/dice_loss"
    assert "train/ce_loss" in log_names, "Should log train/ce_loss"
```

---

### 阶段3: BodyNetHyperbolic集成

对应 `Dice_loss.md` 步骤3

#### TDD循环 3.1: 参数传递

**RED - 写测试** (`tests/models/test_body_net_hyperbolic.py`)
```python
def test_dice_params_passed_to_parent(self):
    """Dice parameters should be passed to parent class."""
    model = BodyNetHyperbolic(
        n_classes=5,
        use_dice_loss=True,
        dice_weight=0.3
    )

    assert model.use_dice_loss == True
    assert model.dice_weight == 0.3
```

---

### 阶段4: CLI集成

对应 `Dice_loss.md` 步骤4

#### TDD循环 4.1: 参数解析

**RED - 写测试** (`tests/scripts/test_train_body_args.py`)
```python
def test_dice_arguments_parsed():
    """CLI should parse dice loss arguments."""
    import sys
    from unittest.mock import patch

    test_args = [
        "train_body.py",
        "--dataset_root", "Dataset/voxel_data",
        "--use_dice_loss",
        "--dice_weight", "0.3"
    ]

    with patch.object(sys, 'argv', test_args):
        # 导入并解析参数
        # 验证args.use_dice_loss == True
        # 验证args.dice_weight == 0.3
        pass
```

---

## 执行流程

### 核心Loss函数

```bash
# 1. 创建测试文件结构
mkdir -p tests/loss
touch tests/loss/__init__.py
touch tests/loss/test_dice_loss.py

# 2. TDD循环 1.1-1.6
# 每个循环: 写测试 → 运行失败 → 实现 → 运行通过

# 3. 验证全部测试
pytest tests/loss/test_dice_loss.py -v

# 4. 验证导入
python -c "from pasco.loss.losses import multi_class_dice_loss; print('OK')"
```

### 模型集成

```bash
# 1. BodyNet测试
pytest tests/models/test_body_net.py -v -k "dice"

# 2. BodyNetHyperbolic测试
pytest tests/models/test_body_net_hyperbolic.py -v -k "dice"

# 3. 集成验证
python -c "
from pasco.models.body_net import BodyNet
model = BodyNet(use_dice_loss=True)
print('BodyNet OK')
"
```

### CLI和端到端验证

```bash
# 1. CLI参数测试
pytest tests/scripts/test_train_body_args.py -v

# 2. 短期训练验证
python scripts/body/train_body.py \
    --dataset_root Dataset/voxel_data \
    --batch_size 2 \
    --max_epochs 1 \
    --use_dice_loss \
    --dice_weight 0.5 \
    --gpuids 0

# 3. 检查日志包含dice_loss
```

---

## 测试覆盖率要求

```bash
# 运行覆盖率检查
pytest tests/loss/test_dice_loss.py --cov=pasco.loss.losses --cov-report=term-missing

# 目标:
# - multi_class_dice_loss 函数 >= 90% 覆盖率
# - 所有分支路径已测试
```

---

## 检查清单

### Loss函数 (`multi_class_dice_loss`)
- [ ] test_output_is_scalar
- [ ] test_loss_range
- [ ] test_perfect_prediction_near_zero_loss
- [ ] test_gradient_flows
- [ ] test_ignore_index_excluded
- [ ] test_class_weights_applied
- [ ] test_numerical_stability_large_logits
- [ ] test_numerical_stability_small_logits

### BodyNet
- [ ] test_compute_loss_returns_tuple_when_dice_enabled
- [ ] test_compute_loss_returns_scalar_when_dice_disabled
- [ ] test_training_step_logs_dice_loss

### BodyNetHyperbolic
- [ ] test_dice_params_passed_to_parent
- [ ] test_training_step_logs_dice_when_enabled

### CLI
- [ ] test_dice_arguments_parsed
- [ ] test_config_saves_dice_params

---

## 与Dice_loss.md的映射

| Dice_loss.md 步骤 | TDD阶段 | 测试文件 |
|------------------|---------|---------|
| 步骤1: losses.py | 阶段1 | tests/loss/test_dice_loss.py |
| 步骤2: body_net.py | 阶段2 | tests/models/test_body_net.py |
| 步骤3: body_net_hyperbolic.py | 阶段3 | tests/models/test_body_net_hyperbolic.py |
| 步骤4: train_body.py | 阶段4 | tests/scripts/test_train_body_args.py |
| 步骤5: 单元测试 | 已整合到各阶段 | - |

---

## 注意事项

1. **严格遵循Red-Green-Refactor**: 每个测试必须先失败再通过
2. **测试独立性**: 每个测试可以独立运行
3. **增量提交**: 每完成一个TDD循环可以提交一次
4. **文档同步**: 实现过程中如发现计划需要调整，同步更新Dice_loss.md
