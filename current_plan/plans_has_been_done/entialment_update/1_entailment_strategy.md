# TDD实施策略：Hyperbolic模块分类输出

> 本文档是 `1_entailment.md` 的行动指南，严格遵循TDD流程。

---

## TDD原则

每个功能点必须遵循：

```
RED   → 编写失败的测试
GREEN → 编写最小实现使测试通过
REFACTOR → 优化代码，保持测试通过
```

**禁止事项**：
- 禁止在测试通过前编写额外功能
- 禁止跳过测试直接实现
- 禁止在GREEN阶段优化代码

---

## Phase 1: pairwise_dist_voxel 函数

### 1.1 创建测试文件

**文件**: `tests/test_pairwise_dist_voxel.py`

```bash
# 确认测试目录存在
ls tests/
```

### 1.2 RED: 编写测试

```python
# tests/test_pairwise_dist_voxel.py
import torch
import pytest

class TestPairwiseDistVoxel:
    """Test pairwise_dist_voxel function"""

    def test_output_shape(self):
        """输出形状应为 [B, N, H, W, Z]"""
        from pasco.models.hyperbolic.lorentz_ops import pairwise_dist_voxel

        B, D, H, W, Z = 2, 8, 4, 4, 4
        N = 70

        x = torch.randn(B, D, H, W, Z)
        y = torch.randn(N, D)

        result = pairwise_dist_voxel(x, y, curv=1.0)

        assert result.shape == (B, N, H, W, Z)

    def test_non_negative_distances(self):
        """测地距离应为非负"""
        from pasco.models.hyperbolic.lorentz_ops import pairwise_dist_voxel

        x = torch.randn(1, 8, 2, 2, 2)
        y = torch.randn(10, 8)

        result = pairwise_dist_voxel(x, y, curv=1.0)

        assert (result >= 0).all()

    def test_self_distance_zero(self):
        """相同点距离应接近0"""
        from pasco.models.hyperbolic.lorentz_ops import pairwise_dist_voxel

        D = 8
        point = torch.randn(1, D)
        x = point.view(1, D, 1, 1, 1)
        y = point.clone()

        result = pairwise_dist_voxel(x, y, curv=1.0)

        assert result.abs().max() < 1e-5

    def test_gradient_flow(self):
        """梯度应能正常传播"""
        from pasco.models.hyperbolic.lorentz_ops import pairwise_dist_voxel

        x = torch.randn(1, 8, 2, 2, 2, requires_grad=True)
        y = torch.randn(10, 8, requires_grad=True)

        result = pairwise_dist_voxel(x, y, curv=1.0)
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        assert y.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isnan(y.grad).any()

    def test_numerical_stability(self):
        """数值稳定性：大值输入不应产生nan/inf"""
        from pasco.models.hyperbolic.lorentz_ops import pairwise_dist_voxel

        x = torch.randn(1, 8, 2, 2, 2) * 10
        y = torch.randn(5, 8) * 10

        result = pairwise_dist_voxel(x, y, curv=1.0)

        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
```

### 1.3 运行测试（预期失败）

```bash
pytest tests/test_pairwise_dist_voxel.py -v
```

**预期输出**: `ImportError` 或 `AttributeError`（函数不存在）

### 1.4 GREEN: 最小实现

在 `pasco/models/hyperbolic/lorentz_ops.py` 添加函数（见技术计划）。

### 1.5 运行测试（预期通过）

```bash
pytest tests/test_pairwise_dist_voxel.py -v
```

### 1.6 REFACTOR

- 检查代码可读性
- 添加类型注解
- 确保测试仍然通过

---

## Phase 2: BodyNetHyperbolic 分类方法

### 2.1 RED: 编写测试

**文件**: `tests/test_hyperbolic_classification.py`

```python
# tests/test_hyperbolic_classification.py
import torch
import pytest

class TestHyperbolicClassification:
    """Test hyperbolic classification methods"""

    @pytest.fixture
    def model(self):
        from pasco.models.body_net_hyperbolic import BodyNetHyperbolic
        return BodyNetHyperbolic(n_classes=70, use_hyp_classification=True)

    def test_compute_hyperbolic_logits_shape(self, model):
        """compute_hyperbolic_logits输出形状正确"""
        voxel_emb = torch.randn(2, model.hyp_dim, 8, 8, 8)
        logits = model.compute_hyperbolic_logits(voxel_emb)

        assert logits.shape == (2, 70, 8, 8, 8)

    def test_forward_with_all_logits(self, model):
        """forward_with_all_logits返回三个输出"""
        x = torch.randn(1, 1, 32, 32, 32)

        ce_logits, hyp_logits, voxel_emb = model.forward_with_all_logits(x)

        # Shape checks
        assert ce_logits.dim() == 5
        assert hyp_logits.dim() == 5
        assert voxel_emb.dim() == 5

        # Class dimension
        assert ce_logits.shape[1] == 70
        assert hyp_logits.shape[1] == 70

    def test_temperature_buffer_registered(self, model):
        """温度参数应注册为buffer"""
        assert hasattr(model, 'hyp_temperature')
        assert model.hyp_temperature.item() == pytest.approx(0.1)

    def test_logits_gradient_flow(self, model):
        """梯度应能从logits传播到label_emb"""
        voxel_emb = torch.randn(1, model.hyp_dim, 4, 4, 4)

        logits = model.compute_hyperbolic_logits(voxel_emb)
        loss = logits.sum()
        loss.backward()

        # label_emb应有梯度
        label_emb_weight = model.label_emb.get_real_embeddings()
        # 注意：需要检查label_emb的梯度机制
```

### 2.2 运行测试（预期失败）

```bash
pytest tests/test_hyperbolic_classification.py -v
```

### 2.3 GREEN: 实现

1. 在 `BodyNetHyperbolic.__init__` 添加参数
2. 实现 `compute_hyperbolic_logits` 方法
3. 实现 `forward_with_all_logits` 方法

### 2.4 运行测试（预期通过）

```bash
pytest tests/test_hyperbolic_classification.py -v
```

---

## Phase 3: 推理方法

### 3.1 RED: 编写测试

**追加到** `tests/test_hyperbolic_classification.py`：

```python
class TestPredictionMethods:
    """Test prediction methods"""

    @pytest.fixture
    def model(self):
        from pasco.models.body_net_hyperbolic import BodyNetHyperbolic
        return BodyNetHyperbolic(n_classes=70, use_hyp_classification=True)

    def test_predict_returns_four_outputs(self, model):
        """predict返回4个输出"""
        x = torch.randn(1, 1, 32, 32, 32)

        result = model.predict(x)

        assert len(result) == 4
        ce_pred, hyp_pred, ce_logits, hyp_logits = result

    def test_predict_shapes(self, model):
        """predict输出形状正确"""
        B, H, W, Z = 2, 32, 32, 32
        x = torch.randn(B, 1, H, W, Z)

        ce_pred, hyp_pred, ce_logits, hyp_logits = model.predict(x)

        assert ce_pred.shape == (B, H, W, Z)
        assert hyp_pred.shape == (B, H, W, Z)

    def test_predict_values_in_range(self, model):
        """预测值应在[0, n_classes)范围内"""
        x = torch.randn(1, 1, 16, 16, 16)

        ce_pred, hyp_pred, _, _ = model.predict(x)

        assert ce_pred.min() >= 0
        assert ce_pred.max() < 70
        assert hyp_pred.min() >= 0
        assert hyp_pred.max() < 70

    def test_predict_hyperbolic_only(self, model):
        """predict_hyperbolic仅返回hyp预测"""
        x = torch.randn(1, 1, 16, 16, 16)

        result = model.predict_hyperbolic(x)

        assert result.shape == (1, 16, 16, 16)
        assert result.min() >= 0
        assert result.max() < 70
```

### 3.2 GREEN: 实现

实现 `predict` 和 `predict_hyperbolic` 方法。

---

## Phase 4: CLI参数

### 4.1 RED: 编写测试

**文件**: `tests/test_cli_args.py`

```python
# tests/test_cli_args.py
import subprocess
import pytest

class TestCLIArguments:
    """Test CLI argument parsing"""

    def test_help_includes_hyp_classification(self):
        """--help应显示use_hyp_classification参数"""
        result = subprocess.run(
            ['python', 'scripts/body/train_body.py', '--help'],
            capture_output=True, text=True
        )
        assert '--use_hyp_classification' in result.stdout

    def test_help_includes_hyp_temperature(self):
        """--help应显示hyp_temperature参数"""
        result = subprocess.run(
            ['python', 'scripts/body/train_body.py', '--help'],
            capture_output=True, text=True
        )
        assert '--hyp_temperature' in result.stdout
```

### 4.2 GREEN: 实现

在 `scripts/body/train_body.py` 添加参数。

---

## 执行检查清单

每个Phase完成后，勾选对应项：

### Phase 1: pairwise_dist_voxel
- [ ] 测试文件已创建
- [ ] RED: 测试运行失败（函数不存在）
- [ ] GREEN: 函数实现，测试通过
- [ ] REFACTOR: 代码优化，测试仍通过

### Phase 2: 分类方法
- [ ] 测试文件已创建
- [ ] RED: 测试运行失败
- [ ] GREEN: 方法实现，测试通过
- [ ] REFACTOR: 代码优化

### Phase 3: 推理方法
- [ ] 测试已追加
- [ ] RED: 测试运行失败
- [ ] GREEN: 方法实现，测试通过

### Phase 4: CLI参数
- [ ] 测试文件已创建
- [ ] RED: 测试运行失败
- [ ] GREEN: 参数添加，测试通过

### 最终验证
- [ ] `pytest tests/ -v` 全部通过
- [ ] 覆盖率 >= 80%
- [ ] 集成测试（技术计划中的验证命令）通过

---

## 命令速查

```bash
# 运行所有相关测试
pytest tests/test_pairwise_dist_voxel.py tests/test_hyperbolic_classification.py tests/test_cli_args.py -v

# 检查覆盖率
pytest tests/ --cov=pasco.models.hyperbolic --cov=pasco.models.body_net_hyperbolic --cov-report=term-missing

# 集成验证
python -c "
from pasco.models.body_net_hyperbolic import BodyNetHyperbolic
import torch

model = BodyNetHyperbolic(n_classes=70, use_hyp_classification=True)
x = torch.randn(1, 1, 32, 32, 32)
ce_logits, hyp_logits, voxel_emb = model.forward_with_all_logits(x)
print(f'CE logits: {ce_logits.shape}')
print(f'Hyp logits: {hyp_logits.shape}')
"
```
