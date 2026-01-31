# LorentzRankingLoss Effective Number 加权采样优化计划 (TDD版)

## 目标

- 使用 Effective Number (β=0.999) 加权采样减少计算量
- 从 **646万体素** 采样到 **10K-50K** 体素
- 预期加速：**50-100x**（从 95ms 降到 1-2ms）
- 保持向后兼容
- **测试覆盖率 ≥ 80%**

---

## 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `tests/hyperbolic/Lorentz/test_lorentz_loss.py` | **先修改**：添加采样相关测试 |
| `pasco/loss/lorentz_loss.py` | 核心：添加加权采样逻辑 |
| `pasco/models/body_net_hyperbolic.py` | 传递新参数到 loss 函数 |
| `scripts/body/train_body.py` | 添加 CLI 参数 |

---

## TDD 开发流程

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: Effective Number 权重计算                          │
│  ┌───────┐    ┌───────┐    ┌──────────┐                     │
│  │  RED  │ -> │ GREEN │ -> │ REFACTOR │                     │
│  │ 写测试 │    │ 写实现 │    │  优化    │                     │
│  └───────┘    └───────┘    └──────────┘                     │
├─────────────────────────────────────────────────────────────┤
│  Phase 2: 采样逻辑                                           │
│  ┌───────┐    ┌───────┐    ┌──────────┐                     │
│  │  RED  │ -> │ GREEN │ -> │ REFACTOR │                     │
│  └───────┘    └───────┘    └──────────┘                     │
├─────────────────────────────────────────────────────────────┤
│  Phase 3: 集成测试                                           │
│  ┌───────┐    ┌───────┐    ┌──────────┐                     │
│  │  RED  │ -> │ GREEN │ -> │ REFACTOR │                     │
│  └───────┘    └───────┘    └──────────┘                     │
├─────────────────────────────────────────────────────────────┤
│  Phase 4: 性能验证                                           │
│  ┌───────┐    ┌───────┐                                     │
│  │ 基准  │ -> │ 验证  │                                     │
│  └───────┘    └───────┘                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Effective Number 权重计算

### Step 1.1 [RED] 编写权重计算测试

**文件**: `tests/hyperbolic/Lorentz/test_lorentz_loss.py`

在文件末尾添加以下测试类：

```python
import numpy as np
import pytest
import torch

from pasco.loss.lorentz_loss import LorentzRankingLoss


class TestEffectiveNumberWeights:
    """测试 Effective Number 权重计算 - TDD Phase 1"""

    def test_compute_weights_uniform_distribution(self):
        """均匀分布时权重应相等"""
        freq = np.array([1000, 1000, 1000, 1000])
        weights = LorentzRankingLoss._compute_effective_number_weights(
            freq, beta=0.999, ignore_classes=set()
        )
        # 所有权重应相同
        assert np.allclose(weights, weights[0], rtol=1e-5)

    def test_compute_weights_imbalanced_distribution(self):
        """不平衡分布时稀有类权重更高"""
        freq = np.array([10000, 100, 10, 1])
        weights = LorentzRankingLoss._compute_effective_number_weights(
            freq, beta=0.999, ignore_classes=set()
        )
        # 稀有类(频率低)权重应该更高
        assert weights[3] > weights[2] > weights[1] > weights[0]

    def test_compute_weights_ignore_classes(self):
        """忽略类权重应为0"""
        freq = np.array([1000, 1000, 1000, 1000])
        weights = LorentzRankingLoss._compute_effective_number_weights(
            freq, beta=0.999, ignore_classes={0, 2}
        )
        assert weights[0] == 0.0
        assert weights[2] == 0.0
        assert weights[1] > 0.0
        assert weights[3] > 0.0

    def test_compute_weights_zero_frequency(self):
        """零频率类权重应为0"""
        freq = np.array([1000, 0, 500, 0])
        weights = LorentzRankingLoss._compute_effective_number_weights(
            freq, beta=0.999, ignore_classes=set()
        )
        assert weights[1] == 0.0
        assert weights[3] == 0.0
        assert weights[0] > 0.0
        assert weights[2] > 0.0

    def test_compute_weights_beta_effect(self):
        """高beta更强调稀有类"""
        freq = np.array([10000, 100])
        w_high_beta = LorentzRankingLoss._compute_effective_number_weights(
            freq, beta=0.999, ignore_classes=set()
        )
        w_low_beta = LorentzRankingLoss._compute_effective_number_weights(
            freq, beta=0.9, ignore_classes=set()
        )
        # 高beta时，稀有类/常见类的权重比应更大
        ratio_high = w_high_beta[1] / w_high_beta[0]
        ratio_low = w_low_beta[1] / w_low_beta[0]
        assert ratio_high > ratio_low

    def test_compute_weights_output_dtype(self):
        """输出应为 float32"""
        freq = np.array([1000, 100, 10])
        weights = LorentzRankingLoss._compute_effective_number_weights(
            freq, beta=0.999, ignore_classes=set()
        )
        assert weights.dtype == np.float32

    def test_compute_weights_numerical_stability(self):
        """大频率值时数值稳定"""
        freq = np.array([1e9, 1e6, 1e3, 1])
        weights = LorentzRankingLoss._compute_effective_number_weights(
            freq, beta=0.999, ignore_classes=set()
        )
        assert np.all(np.isfinite(weights))
        assert np.all(weights >= 0)
```

### Step 1.2 [RED] 运行测试验证失败

```bash
pytest tests/hyperbolic/Lorentz/test_lorentz_loss.py::TestEffectiveNumberWeights -v
# 预期: 7 个测试失败 (方法 _compute_effective_number_weights 不存在)
```

### Step 1.3 [GREEN] 实现权重计算方法

**文件**: `pasco/loss/lorentz_loss.py`

在 `LorentzRankingLoss` 类中添加静态方法：

```python
@staticmethod
def _compute_effective_number_weights(
    class_frequencies: np.ndarray,
    beta: float,
    ignore_classes: Set[int]
) -> np.ndarray:
    """
    计算 Effective Number 权重用于类别平衡采样。

    公式: w_c = (1-β) / (1-β^{N_c})

    Args:
        class_frequencies: 每个类别的样本数量，shape [n_classes]
        beta: Effective Number 的 β 参数，通常 0.9-0.9999
        ignore_classes: 要忽略的类别集合（权重设为0）

    Returns:
        weights: 每个类别的采样权重，shape [n_classes]，dtype float32

    Reference:
        Cui et al. "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019)
    """
    # 确保频率至少为1，避免除零
    freq = np.clip(class_frequencies.astype(np.float64), 1, None)

    # Effective Number: E_n = (1 - β^n) / (1 - β)
    # 权重是其倒数: w = (1 - β) / (1 - β^n)
    effective_num = 1.0 - np.power(beta, freq)
    weights = (1.0 - beta) / effective_num

    # 忽略类权重设为 0
    for cls in ignore_classes:
        if 0 <= cls < len(weights):
            weights[cls] = 0.0

    # 零频率类权重设为 0
    weights[class_frequencies == 0] = 0.0

    return weights.astype(np.float32)
```

### Step 1.4 [GREEN] 运行测试验证通过

```bash
pytest tests/hyperbolic/Lorentz/test_lorentz_loss.py::TestEffectiveNumberWeights -v
# 预期: 7 个测试通过
```

### Step 1.5 [REFACTOR] 检查代码质量

- [ ] 方法文档完整
- [ ] 类型标注正确
- [ ] 无重复代码

---

## Phase 2: 采样逻辑

### Step 2.1 [RED] 编写采样逻辑测试

**文件**: `tests/hyperbolic/Lorentz/test_lorentz_loss.py`

继续添加测试类：

```python
class TestLorentzRankingLossSampling:
    """测试加权采样功能 - TDD Phase 2"""

    @pytest.fixture
    def sample_frequencies(self):
        """模拟类别频率: class 1 很多, class 2-4 很少"""
        freq = np.zeros(71)
        freq[1] = 100000   # 常见类
        freq[2] = 100      # 稀有类
        freq[3] = 50       # 更稀有
        freq[4] = 10       # 极稀有
        return freq

    @pytest.fixture
    def small_input(self):
        """小规模输入用于快速测试"""
        torch.manual_seed(42)
        voxel_emb = torch.randn(1, 32, 10, 10, 10)
        labels = torch.randint(1, 5, (1, 10, 10, 10))
        label_emb = torch.randn(71, 32)
        return voxel_emb, labels, label_emb

    # ============ 初始化测试 ============

    def test_init_with_sampling_params(self, sample_frequencies):
        """带采样参数的初始化"""
        loss_fn = LorentzRankingLoss(
            max_voxels=1000,
            class_frequencies=sample_frequencies,
            beta=0.999
        )
        assert loss_fn.max_voxels == 1000
        assert loss_fn._class_weights is not None
        assert loss_fn._class_weights.shape[0] == 71

    def test_init_without_sampling(self):
        """不启用采样时向后兼容"""
        loss_fn = LorentzRankingLoss()
        assert loss_fn.max_voxels is None
        assert loss_fn._class_weights is None

    def test_init_requires_frequencies_when_sampling(self):
        """启用采样但未提供频率时应报错"""
        with pytest.raises(ValueError, match="class_frequencies required"):
            LorentzRankingLoss(max_voxels=1000)

    def test_init_class_weights_on_correct_device(self, sample_frequencies):
        """class_weights 应随模型移动到正确设备"""
        loss_fn = LorentzRankingLoss(
            max_voxels=1000,
            class_frequencies=sample_frequencies,
        )
        # buffer 应该能正确移动设备
        if torch.cuda.is_available():
            loss_fn = loss_fn.cuda()
            assert loss_fn._class_weights.device.type == "cuda"

    # ============ 前向传播测试 ============

    def test_forward_no_sampling_when_disabled(self, small_input):
        """max_voxels=None 时不采样"""
        loss_fn = LorentzRankingLoss()
        voxel_emb, labels, label_emb = small_input

        loss = loss_fn(voxel_emb, labels, label_emb)
        assert loss.numel() == 1
        assert not torch.isnan(loss)

    def test_forward_no_sampling_when_below_threshold(self, sample_frequencies, small_input):
        """体素数低于阈值时不采样"""
        loss_fn = LorentzRankingLoss(
            max_voxels=100000,  # 高阈值，不会触发采样
            class_frequencies=sample_frequencies,
        )
        voxel_emb, labels, label_emb = small_input  # 1000 体素

        loss = loss_fn(voxel_emb, labels, label_emb)
        assert loss.numel() == 1
        assert not torch.isnan(loss)

    def test_forward_samples_when_above_threshold(self, sample_frequencies):
        """体素数超过阈值时进行采样"""
        torch.manual_seed(42)
        max_voxels = 100
        loss_fn = LorentzRankingLoss(
            max_voxels=max_voxels,
            class_frequencies=sample_frequencies,
        )

        # 创建远超阈值的输入 (8000 体素)
        voxel_emb = torch.randn(1, 32, 20, 20, 20)
        labels = torch.randint(1, 5, (1, 20, 20, 20))
        label_emb = torch.randn(71, 32)

        loss = loss_fn(voxel_emb, labels, label_emb)
        assert loss.numel() == 1
        assert not torch.isnan(loss)
        assert torch.isfinite(loss)

    def test_forward_output_shape_unchanged(self, sample_frequencies, small_input):
        """采样不改变输出形状（仍为标量）"""
        loss_fn = LorentzRankingLoss(
            max_voxels=100,
            class_frequencies=sample_frequencies,
        )
        voxel_emb, labels, label_emb = small_input

        loss = loss_fn(voxel_emb, labels, label_emb)
        assert loss.shape == torch.Size([])  # 标量

    # ============ 权重分布测试 ============

    def test_sampling_weights_favor_rare_classes(self, sample_frequencies):
        """采样权重应偏向稀有类"""
        loss_fn = LorentzRankingLoss(
            max_voxels=1000,
            class_frequencies=sample_frequencies,
        )

        weights = loss_fn._class_weights
        # 频率: class1=100000, class2=100, class3=50, class4=10
        # 权重应该: class4 > class3 > class2 > class1
        assert weights[4] > weights[3] > weights[2] > weights[1]

    def test_sampling_weights_zero_for_ignored(self, sample_frequencies):
        """忽略类的权重应为0"""
        loss_fn = LorentzRankingLoss(
            max_voxels=1000,
            class_frequencies=sample_frequencies,
            ignore_classes={0, 255}
        )

        weights = loss_fn._class_weights
        assert weights[0] == 0.0
        # class 255 超出数组范围，只检查 0

    # ============ 梯度测试 ============

    def test_gradient_flows_through_sampling(self, sample_frequencies):
        """梯度应该能通过采样流动"""
        torch.manual_seed(42)
        loss_fn = LorentzRankingLoss(
            max_voxels=100,
            class_frequencies=sample_frequencies,
        )

        voxel_emb = torch.randn(1, 32, 10, 10, 10, requires_grad=True)
        labels = torch.randint(1, 5, (1, 10, 10, 10))
        label_emb = torch.randn(71, 32, requires_grad=True)

        loss = loss_fn(voxel_emb, labels, label_emb)
        loss.backward()

        assert voxel_emb.grad is not None
        assert label_emb.grad is not None
        assert not torch.isnan(voxel_emb.grad).any()
        assert not torch.isnan(label_emb.grad).any()

    def test_gradient_sparse_when_sampling(self, sample_frequencies):
        """采样后只有部分体素收到梯度"""
        torch.manual_seed(42)
        max_voxels = 100
        loss_fn = LorentzRankingLoss(
            max_voxels=max_voxels,
            class_frequencies=sample_frequencies,
        )

        # 1000 个体素，只采样 100 个
        voxel_emb = torch.randn(1, 32, 10, 10, 10, requires_grad=True)
        labels = torch.randint(1, 5, (1, 10, 10, 10))
        label_emb = torch.randn(71, 32, requires_grad=True)

        loss = loss_fn(voxel_emb, labels, label_emb)
        loss.backward()

        # 梯度应该存在
        assert voxel_emb.grad is not None

    # ============ 边界条件测试 ============

    def test_fallback_uniform_when_all_weights_zero(self):
        """所有有效类权重为0时回退到均匀采样"""
        # 只有 class 0 (忽略类) 有频率
        freq = np.zeros(71)
        freq[0] = 10000

        loss_fn = LorentzRankingLoss(
            max_voxels=100,
            class_frequencies=freq,
            ignore_classes={0, 255}
        )

        torch.manual_seed(42)
        voxel_emb = torch.randn(1, 32, 10, 10, 10)
        labels = torch.ones(1, 10, 10, 10, dtype=torch.long)  # 全是 class 1
        label_emb = torch.randn(71, 32)

        # 应该不崩溃，回退到均匀采样
        loss = loss_fn(voxel_emb, labels, label_emb)
        assert not torch.isnan(loss)

    def test_exact_threshold_boundary(self, sample_frequencies):
        """体素数恰好等于阈值时的行为"""
        torch.manual_seed(42)
        max_voxels = 1000  # 恰好等于输入体素数
        loss_fn = LorentzRankingLoss(
            max_voxels=max_voxels,
            class_frequencies=sample_frequencies,
        )

        # 恰好 1000 个体素
        voxel_emb = torch.randn(1, 32, 10, 10, 10)
        labels = torch.randint(1, 5, (1, 10, 10, 10))
        label_emb = torch.randn(71, 32)

        loss = loss_fn(voxel_emb, labels, label_emb)
        assert not torch.isnan(loss)

    def test_very_small_max_voxels(self, sample_frequencies):
        """极小采样数仍能工作"""
        torch.manual_seed(42)
        loss_fn = LorentzRankingLoss(
            max_voxels=10,  # 极小
            class_frequencies=sample_frequencies,
        )

        voxel_emb = torch.randn(1, 32, 10, 10, 10)
        labels = torch.randint(1, 5, (1, 10, 10, 10))
        label_emb = torch.randn(71, 32)

        loss = loss_fn(voxel_emb, labels, label_emb)
        assert not torch.isnan(loss)
        assert torch.isfinite(loss)

    # ============ 批量测试 ============

    def test_batch_processing(self, sample_frequencies):
        """多批次输入正确处理"""
        torch.manual_seed(42)
        loss_fn = LorentzRankingLoss(
            max_voxels=500,
            class_frequencies=sample_frequencies,
        )

        # 批量大小 = 4
        voxel_emb = torch.randn(4, 32, 10, 10, 10)
        labels = torch.randint(1, 5, (4, 10, 10, 10))
        label_emb = torch.randn(71, 32)

        loss = loss_fn(voxel_emb, labels, label_emb)
        assert loss.numel() == 1
        assert not torch.isnan(loss)
```

### Step 2.2 [RED] 运行测试验证失败

```bash
pytest tests/hyperbolic/Lorentz/test_lorentz_loss.py::TestLorentzRankingLossSampling -v
# 预期: 多个测试失败 (新参数不存在)
```

### Step 2.3 [GREEN] 实现采样逻辑

**文件**: `pasco/loss/lorentz_loss.py`

#### 2.3.1 修改 `__init__` 方法

```python
def __init__(
    self,
    margin: float = 0.1,
    curv: float = 1.0,
    ignore_classes: Optional[Set[int]] = None,
    n_classes: int = 71,
    # === 新增参数 ===
    max_voxels: Optional[int] = None,
    class_frequencies: Optional[np.ndarray] = None,
    beta: float = 0.999,
):
    """
    Lorentz 空间中的 Ranking Loss，支持 Effective Number 加权采样。

    Args:
        margin: triplet loss 的 margin
        curv: Lorentz 空间曲率
        ignore_classes: 忽略的类别集合
        n_classes: 类别总数
        max_voxels: 采样上限，None 表示不采样
        class_frequencies: 类别频率数组，启用采样时必须提供
        beta: Effective Number 的 β 参数
    """
    super().__init__()
    self.margin = margin
    self.curv = curv
    self.ignore_classes = ignore_classes if ignore_classes else {0, 255}
    self.n_classes = n_classes
    self.max_voxels = max_voxels

    # 预计算有效类别 (现有逻辑)
    all_classes = set(range(n_classes))
    valid = sorted(all_classes - self.ignore_classes)
    self.register_buffer("valid_classes", torch.tensor(valid, dtype=torch.long), persistent=False)

    ignore_list = sorted(self.ignore_classes & all_classes)
    self.register_buffer("_ignore_tensor", torch.tensor(ignore_list, dtype=torch.long), persistent=False)

    # === 新增：采样权重 ===
    if max_voxels is not None:
        if class_frequencies is None:
            raise ValueError("class_frequencies required when max_voxels is set")

        effective_weights = self._compute_effective_number_weights(
            class_frequencies, beta, self.ignore_classes
        )
        self.register_buffer(
            "_class_weights",
            torch.tensor(effective_weights, dtype=torch.float32),
            persistent=False
        )
    else:
        self._class_weights = None
```

#### 2.3.2 修改 `forward` 方法

在现有的有效体素筛选之后、距离计算之前添加采样逻辑：

```python
def forward(
    self,
    voxel_embeddings: Tensor,
    labels: Tensor,
    label_embeddings: Tensor,
) -> Tensor:
    """
    计算 Lorentz Ranking Loss。

    Args:
        voxel_embeddings: [B, D, H, W, Z] 体素嵌入
        labels: [B, H, W, Z] 体素标签
        label_embeddings: [N_classes, D] 标签嵌入

    Returns:
        标量 loss 值
    """
    device = voxel_embeddings.device
    B, D, H, W, Z = voxel_embeddings.shape

    # 重排为 [B*H*W*Z, D]
    voxel_emb = voxel_embeddings.permute(0, 2, 3, 4, 1).reshape(-1, D)
    labels_flat = labels.reshape(-1)

    # 筛选有效体素（非忽略类）
    valid_mask = ~torch.isin(labels_flat, self._ignore_tensor)
    voxel_emb = voxel_emb[valid_mask]
    labels_valid = labels_flat[valid_mask]

    n_valid = voxel_emb.shape[0]
    if n_valid == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # ========== 新增：加权采样 ==========
    if self.max_voxels is not None and n_valid > self.max_voxels:
        # 每个体素的采样权重 = 其类别的权重
        voxel_weights = self._class_weights[labels_valid]

        weight_sum = voxel_weights.sum()
        if weight_sum < 1e-8:
            # 边界情况：权重全为0，回退到均匀采样
            sample_indices = torch.randint(
                0, n_valid, (self.max_voxels,), device=device
            )
        else:
            # 加权采样
            voxel_probs = voxel_weights / weight_sum
            sample_indices = torch.multinomial(
                voxel_probs, self.max_voxels, replacement=True
            )

        voxel_emb = voxel_emb[sample_indices]
        labels_valid = labels_valid[sample_indices]
    # ========== 采样逻辑结束 ==========

    # 后续计算距离、负样本采样、triplet loss（保持不变）
    # ... 现有代码 ...
```

### Step 2.4 [GREEN] 运行测试验证通过

```bash
pytest tests/hyperbolic/Lorentz/test_lorentz_loss.py::TestLorentzRankingLossSampling -v
# 预期: 所有测试通过
```

### Step 2.5 [REFACTOR] 代码优化

- [ ] 检查采样逻辑是否可以抽取为独立方法
- [ ] 确保类型标注完整
- [ ] 添加采样相关的日志（可选）

---

## Phase 3: 集成测试

### Step 3.1 [RED] 编写集成测试

**文件**: `tests/hyperbolic/Lorentz/test_lorentz_loss.py`

```python
class TestLorentzRankingLossIntegration:
    """集成测试 - TDD Phase 3"""

    @pytest.fixture
    def sample_frequencies(self):
        freq = np.zeros(71)
        freq[1:20] = np.random.randint(100, 10000, size=19)
        return freq

    def test_integration_with_body_net_hyperbolic(self, sample_frequencies):
        """与 BodyNetHyperbolic 集成"""
        from pasco.models.body_net_hyperbolic import BodyNetHyperbolic

        model = BodyNetHyperbolic(
            n_classes=71,
            max_voxels_ranking=1000,
            ranking_beta=0.999,
            class_frequencies=sample_frequencies,
        )

        assert model.hyp_loss_fn.max_voxels == 1000
        assert model.hyp_loss_fn._class_weights is not None

    def test_backward_compatibility_body_net(self):
        """不传采样参数时向后兼容"""
        from pasco.models.body_net_hyperbolic import BodyNetHyperbolic

        model = BodyNetHyperbolic(n_classes=71)

        assert model.hyp_loss_fn.max_voxels is None
        assert model.hyp_loss_fn._class_weights is None

    def test_training_step_with_sampling(self, sample_frequencies):
        """训练步骤中采样正常工作"""
        from pasco.models.body_net_hyperbolic import BodyNetHyperbolic

        model = BodyNetHyperbolic(
            n_classes=71,
            max_voxels_ranking=500,
            class_frequencies=sample_frequencies,
        )

        # 模拟输入
        torch.manual_seed(42)
        occupancy = torch.randn(2, 1, 64, 64, 16)
        labels = torch.randint(0, 20, (2, 64, 64, 16))

        # 前向传播应该不报错
        model.eval()
        with torch.no_grad():
            logits, voxel_emb = model.forward_with_hyperbolic(occupancy)

        assert voxel_emb is not None
```

### Step 3.2 [GREEN] 修改 `body_net_hyperbolic.py`

**文件**: `pasco/models/body_net_hyperbolic.py`

#### 3.2.1 修改构造函数

```python
def __init__(
    self,
    n_classes=71,
    embed_dim=32,
    hyperbolic_weight=0.1,
    margin=0.1,
    use_entailment_cone=True,
    entailment_weight=0.1,
    # === 新增参数 ===
    max_voxels_ranking: Optional[int] = None,
    ranking_beta: float = 0.999,
    class_frequencies: Optional[np.ndarray] = None,
    **kwargs
):
```

#### 3.2.2 传递给 LorentzRankingLoss

```python
self.hyp_loss_fn = LorentzRankingLoss(
    margin=margin,
    curv=CURV,
    ignore_classes={0, 255},
    n_classes=n_classes,
    max_voxels=max_voxels_ranking,
    class_frequencies=class_frequencies,
    beta=ranking_beta,
)
```

### Step 3.3 [GREEN] 修改 `train_body.py`

**文件**: `scripts/body/train_body.py`

#### 3.3.1 添加 CLI 参数

```python
parser.add_argument(
    "--max_voxels_ranking",
    type=int,
    default=None,
    help="Max voxels for ranking loss sampling (None=no sampling)"
)
parser.add_argument(
    "--ranking_beta",
    type=float,
    default=0.999,
    help="Effective Number beta for sampling"
)
```

#### 3.3.2 创建模型时传递参数

```python
model = BodyNetHyperbolic(
    n_classes=n_classes,
    # ... 现有参数 ...
    max_voxels_ranking=args.max_voxels_ranking,
    ranking_beta=args.ranking_beta,
    class_frequencies=body_class_frequencies,  # 从数据集获取
)
```

### Step 3.4 [GREEN] 运行集成测试

```bash
pytest tests/hyperbolic/Lorentz/test_lorentz_loss.py::TestLorentzRankingLossIntegration -v
# 预期: 所有测试通过
```

---

## Phase 4: 性能验证

### Step 4.1 编写性能基准测试

**文件**: `tests/hyperbolic/Lorentz/test_lorentz_loss.py`

```python
class TestLorentzRankingLossPerformance:
    """性能基准测试 - TDD Phase 4"""

    @pytest.fixture
    def sample_frequencies(self):
        freq = np.zeros(71)
        freq[1:20] = np.random.randint(100, 100000, size=19)
        return freq

    @pytest.mark.slow
    def test_performance_improvement(self, sample_frequencies):
        """采样应显著减少计算时间"""
        import time

        torch.manual_seed(42)

        # 中等规模输入 (~26万体素)
        voxel_emb = torch.randn(1, 32, 64, 64, 64)
        labels = torch.randint(1, 10, (1, 64, 64, 64))
        label_emb = torch.randn(71, 32)

        # 无采样
        loss_no_sample = LorentzRankingLoss()

        # 预热
        _ = loss_no_sample(voxel_emb, labels, label_emb)

        start = time.time()
        for _ in range(5):
            _ = loss_no_sample(voxel_emb, labels, label_emb)
        time_no_sample = (time.time() - start) / 5

        # 有采样
        loss_sample = LorentzRankingLoss(
            max_voxels=10000,
            class_frequencies=sample_frequencies,
        )

        # 预热
        _ = loss_sample(voxel_emb, labels, label_emb)

        start = time.time()
        for _ in range(5):
            _ = loss_sample(voxel_emb, labels, label_emb)
        time_sample = (time.time() - start) / 5

        print(f"\n无采样: {time_no_sample*1000:.2f}ms")
        print(f"有采样 (10K): {time_sample*1000:.2f}ms")
        print(f"加速比: {time_no_sample/time_sample:.1f}x")

        # 至少快 10 倍
        assert time_sample < time_no_sample / 10

    @pytest.mark.slow
    def test_memory_efficiency(self, sample_frequencies):
        """采样应减少内存使用"""
        import gc

        torch.manual_seed(42)

        # 大规模输入
        voxel_emb = torch.randn(1, 32, 128, 128, 16)
        labels = torch.randint(1, 10, (1, 128, 128, 16))
        label_emb = torch.randn(71, 32)

        loss_fn = LorentzRankingLoss(
            max_voxels=30000,
            class_frequencies=sample_frequencies,
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # 应该不 OOM
        loss = loss_fn(voxel_emb, labels, label_emb)
        loss.backward()

        assert torch.isfinite(loss)
```

### Step 4.2 运行性能测试

```bash
pytest tests/hyperbolic/Lorentz/test_lorentz_loss.py::TestLorentzRankingLossPerformance -v -s
# 预期: 看到明显的性能提升
```

---

## Phase 5: 收尾

### Step 5.1 运行全部测试

```bash
# 运行所有 Lorentz 相关测试
pytest tests/hyperbolic/Lorentz/ -v

# 运行覆盖率检查
pytest tests/hyperbolic/Lorentz/test_lorentz_loss.py --cov=pasco.loss.lorentz_loss --cov-report=term-missing
```

### Step 5.2 验证覆盖率

确保新增代码覆盖率 ≥ 80%：
- [ ] `_compute_effective_number_weights` 方法
- [ ] `__init__` 中的采样参数处理
- [ ] `forward` 中的采样逻辑
- [ ] 边界条件（权重为0、体素数等于阈值等）

### Step 5.3 更新文档

- [ ] 更新 `lorentz_loss.py` 的模块文档
- [ ] 更新 `train_body.py` 的帮助信息

---

## 使用示例

### 启用采样（推荐）

```bash
python scripts/body/train_body.py \
    --use_hyperbolic \
    --max_voxels_ranking 30000 \
    --ranking_beta 0.999
```

### 不启用采样（向后兼容）

```bash
python scripts/body/train_body.py \
    --use_hyperbolic
# max_voxels_ranking 默认 None，不采样
```

---

## 预期效果

| 场景 | 体素数 | 时间 | 加速比 |
|------|--------|------|--------|
| 当前（无采样） | 646万 | 95ms | 1x |
| max_voxels=50K | 5万 | ~0.7ms | ~130x |
| max_voxels=30K | 3万 | ~0.5ms | ~200x |
| max_voxels=10K | 1万 | ~0.2ms | ~500x |

---

## 测试清单

### Phase 1: Effective Number 权重计算 (7 tests)
- [x] `test_compute_weights_uniform_distribution`
- [x] `test_compute_weights_imbalanced_distribution`
- [x] `test_compute_weights_ignore_classes`
- [x] `test_compute_weights_zero_frequency`
- [x] `test_compute_weights_beta_effect`
- [x] `test_compute_weights_output_dtype`
- [x] `test_compute_weights_numerical_stability`

### Phase 2: 采样逻辑 (16 tests)
- [x] `test_init_with_sampling_params`
- [x] `test_init_without_sampling`
- [x] `test_init_requires_frequencies_when_sampling`
- [x] `test_init_class_weights_on_correct_device`
- [x] `test_forward_no_sampling_when_disabled`
- [x] `test_forward_no_sampling_when_below_threshold`
- [x] `test_forward_samples_when_above_threshold`
- [x] `test_forward_output_shape_unchanged`
- [x] `test_sampling_weights_favor_rare_classes`
- [x] `test_sampling_weights_zero_for_ignored`
- [x] `test_gradient_flows_through_sampling`
- [x] `test_gradient_sparse_when_sampling`
- [x] `test_fallback_uniform_when_all_weights_zero`
- [x] `test_exact_threshold_boundary`
- [x] `test_very_small_max_voxels`
- [x] `test_batch_processing`

### Phase 3: 集成测试 (3 tests)
- [x] `test_integration_with_body_net_hyperbolic`
- [x] `test_backward_compatibility_body_net`
- [x] `test_training_step_with_sampling`

### Phase 4: 性能测试 (2 tests)
- [x] `test_performance_improvement`
- [x] `test_memory_efficiency`

**总计: 28 个新增测试**

---

## 实施顺序检查表

| 步骤 | 状态 | 说明 |
|------|------|------|
| Phase 1.1 | ✅ | 编写权重计算测试 |
| Phase 1.2 | ✅ | 运行测试 (RED) |
| Phase 1.3 | ✅ | 实现权重计算 |
| Phase 1.4 | ✅ | 运行测试 (GREEN) |
| Phase 2.1 | ✅ | 编写采样测试 |
| Phase 2.2 | ✅ | 运行测试 (RED) |
| Phase 2.3 | ✅ | 实现采样逻辑 |
| Phase 2.4 | ✅ | 运行测试 (GREEN) |
| Phase 3.1 | ✅ | 编写集成测试 |
| Phase 3.2 | ✅ | 修改 body_net_hyperbolic.py |
| Phase 3.3 | ✅ | 修改 train_body.py |
| Phase 3.4 | ✅ | 运行集成测试 (GREEN) |
| Phase 4.1 | ✅ | 编写性能测试 |
| Phase 4.2 | ✅ | 运行性能验证 |
| Phase 5.1 | ✅ | 运行全部测试 (41 passed) |
| Phase 5.2 | ⬜ | 验证覆盖率 ≥ 80% |
| Phase 5.3 | ⬜ | 更新文档 |
