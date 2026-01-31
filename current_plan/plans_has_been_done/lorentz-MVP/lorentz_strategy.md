# PaSCo Lorentz 迁移 - TDD 实施策略

## 概述

本文档定义了从 Poincaré 到 Lorentz 模型迁移的**测试驱动开发（TDD）**具体流程。

### TDD 核心原则

```
阶段 N 完成 → 编写阶段 N+1 测试 → 实现阶段 N+1 → 运行测试 → 通过 → 下一阶段
```

### 测试文件结构

```
tests/hyperbolic/
├── test_lorentz_ops.py              # 阶段1: 核心数学操作
├── test_lorentz_label_embedding.py  # 阶段2: 标签嵌入
├── test_lorentz_projection_head.py  # 阶段3: 投影头
├── test_lorentz_loss.py             # 阶段4: 损失函数
├── test_lorentz_body_net.py         # 阶段5: 模型集成
└── test_lorentz_integration.py      # 阶段6: 端到端集成测试
```

---

## 阶段 1: 核心数学操作 (lorentz_ops.py)

### 1.1 先写测试

**文件**: `tests/hyperbolic/test_lorentz_ops.py`

```python
"""
Lorentz 模型核心数学操作测试

数学说明:
- exp_map0: 切空间 → 双曲面（空间分量）
- log_map0: 双曲面（空间分量）→ 切空间
- pairwise_dist: 双曲面上两点间的测地距离
- hyperbolic_distance_to_origin: 到双曲面原点的测地距离

注意: log_map0 内部使用 sqrt(1 + curv * ||x||^2) 计算距离，
      而 pairwise_dist 使用 sqrt(1/curv + ||x||^2) 计算时间分量。
      当 curv=1.0 时两者等价，其他曲率时需注意这一差异。
"""
import pytest
import torch
import math

from pasco.models.hyperbolic.lorentz_ops import (
    exp_map0, log_map0, pairwise_dist, hyperbolic_distance_to_origin
)


class TestImports:
    """导入测试 - 确保所有函数可导入"""

    def test_all_functions_importable(self):
        """所有必需函数应可导入"""
        assert callable(exp_map0)
        assert callable(log_map0)
        assert callable(pairwise_dist)
        assert callable(hyperbolic_distance_to_origin)


class TestExpMap0:
    """exp_map0: 切空间 → 双曲面"""

    def test_zero_vector_maps_to_origin(self):
        """零向量应映射到双曲面原点（空间分量为零）"""
        v = torch.zeros(10, 32)
        x = exp_map0(v, curv=1.0)
        # 原点的空间分量应该是零向量
        assert torch.allclose(x, torch.zeros_like(x), atol=1e-6)

    def test_output_shape(self):
        """输出形状应与输入一致"""
        v = torch.randn(5, 16)
        x = exp_map0(v, curv=1.0)
        assert x.shape == v.shape

    def test_output_dtype(self):
        """输出数据类型应与输入一致"""
        v = torch.randn(5, 16, dtype=torch.float32)
        x = exp_map0(v, curv=1.0)
        assert x.dtype == v.dtype

        v64 = torch.randn(5, 16, dtype=torch.float64)
        x64 = exp_map0(v64, curv=1.0)
        assert x64.dtype == v64.dtype

    def test_numerical_stability_large_input(self):
        """大输入值应数值稳定（不产生 inf/nan）"""
        v = torch.randn(10, 32) * 100
        x = exp_map0(v, curv=1.0)
        assert torch.isfinite(x).all(), "Large input caused inf/nan"

    def test_numerical_stability_small_input(self):
        """小输入值应数值稳定"""
        v = torch.randn(10, 32) * 1e-8
        x = exp_map0(v, curv=1.0)
        assert torch.isfinite(x).all(), "Small input caused inf/nan"

    def test_numerical_stability_extreme_values(self):
        """极端值混合测试"""
        v = torch.randn(10, 32)
        v[0] *= 1e3   # 很大
        v[1] *= 1e-10  # 很小
        x = exp_map0(v, curv=1.0)
        assert torch.isfinite(x).all(), "Extreme values caused inf/nan"

    def test_different_curvatures(self):
        """不同曲率应产生有效输出"""
        torch.manual_seed(42)
        v = torch.randn(10, 32) * 0.5
        for curv in [0.5, 1.0, 2.0]:
            x = exp_map0(v, curv=curv)
            assert torch.isfinite(x).all(), f"curv={curv} caused inf/nan"

    def test_batch_consistency(self):
        """批量计算与逐个计算结果应一致"""
        torch.manual_seed(42)
        v = torch.randn(10, 32) * 0.5
        batch_result = exp_map0(v, curv=1.0)
        single_results = torch.stack([
            exp_map0(v[i:i+1], curv=1.0).squeeze(0) for i in range(10)
        ])
        assert torch.allclose(batch_result, single_results, atol=1e-6)


class TestLogMap0:
    """log_map0: 双曲面 → 切空间"""

    def test_output_shape(self):
        """输出形状应与输入一致"""
        x = torch.randn(5, 16) * 0.5
        v = log_map0(x, curv=1.0)
        assert v.shape == x.shape

    def test_numerical_stability(self):
        """应数值稳定"""
        x = torch.randn(10, 32) * 0.5
        v = log_map0(x, curv=1.0)
        assert torch.isfinite(v).all()

    def test_origin_maps_to_zero(self):
        """原点（空间分量为零）应映射回零向量"""
        x = torch.zeros(10, 32)
        v = log_map0(x, curv=1.0)
        assert torch.allclose(v, torch.zeros_like(v), atol=1e-6)


class TestExpLogInverse:
    """exp_map0 和 log_map0 应互为逆运算"""

    def test_exp_then_log(self):
        """exp_map0 后 log_map0 应恢复原向量"""
        torch.manual_seed(42)
        v = torch.randn(10, 32) * 0.5
        x = exp_map0(v, curv=1.0)
        v_recovered = log_map0(x, curv=1.0)
        assert torch.allclose(v, v_recovered, atol=1e-5), \
            f"Max diff: {(v - v_recovered).abs().max()}"

    def test_inverse_different_curvatures(self):
        """不同曲率下互逆性应成立（仅在 curv=1.0 时精确）"""
        torch.manual_seed(42)
        v = torch.randn(10, 32) * 0.3
        # 注意: log_map0 使用 sqrt(1 + curv * ||x||^2)
        # 当 curv != 1.0 时，需要验证参考实现的数学定义
        for curv in [1.0]:  # MVP 阶段只测试 curv=1.0
            x = exp_map0(v, curv=curv)
            v_recovered = log_map0(x, curv=curv)
            assert torch.allclose(v, v_recovered, atol=1e-5), \
                f"curv={curv}, max diff: {(v - v_recovered).abs().max()}"

    def test_log_then_exp(self):
        """log_map0 后 exp_map0 应恢复原点"""
        torch.manual_seed(42)
        # 直接使用空间分量作为双曲面上的点
        x = torch.randn(10, 32) * 0.5
        v = log_map0(x, curv=1.0)
        x_recovered = exp_map0(v, curv=1.0)
        assert torch.allclose(x, x_recovered, atol=1e-5), \
            f"Max diff: {(x - x_recovered).abs().max()}"


class TestPairwiseDist:
    """pairwise_dist: 批量测地距离"""

    def test_output_shape(self):
        """输出形状应为 [N, M]"""
        x = torch.randn(5, 16) * 0.5
        y = torch.randn(7, 16) * 0.5
        dist = pairwise_dist(x, y, curv=1.0)
        assert dist.shape == (5, 7)

    def test_non_negative(self):
        """距离应非负"""
        torch.manual_seed(42)
        x = torch.randn(10, 32) * 0.5
        y = torch.randn(10, 32) * 0.5
        dist = pairwise_dist(x, y, curv=1.0)
        assert (dist >= -1e-6).all(), f"Negative distance: {dist.min()}"

    def test_symmetry(self):
        """d(x,y) = d(y,x)"""
        torch.manual_seed(42)
        x = torch.randn(5, 16) * 0.5
        y = torch.randn(7, 16) * 0.5
        dist_xy = pairwise_dist(x, y, curv=1.0)
        dist_yx = pairwise_dist(y, x, curv=1.0)
        assert torch.allclose(dist_xy, dist_yx.T, atol=1e-5)

    def test_self_distance_zero(self):
        """d(x,x) = 0"""
        torch.manual_seed(42)
        x = torch.randn(10, 32) * 0.5
        dist = pairwise_dist(x, x, curv=1.0)
        diagonal = torch.diag(dist)
        assert torch.allclose(diagonal, torch.zeros_like(diagonal), atol=1e-5), \
            f"Self distance not zero: {diagonal}"

    def test_triangle_inequality(self):
        """三角不等式: d(x,z) <= d(x,y) + d(y,z)"""
        torch.manual_seed(42)
        # 使用空间分量作为双曲面上的点
        x = torch.randn(1, 32) * 0.5
        y = torch.randn(1, 32) * 0.5
        z = torch.randn(1, 32) * 0.5
        d_xy = pairwise_dist(x, y, curv=1.0).item()
        d_yz = pairwise_dist(y, z, curv=1.0).item()
        d_xz = pairwise_dist(x, z, curv=1.0).item()
        assert d_xz <= d_xy + d_yz + 1e-5, \
            f"Triangle inequality violated: {d_xz} > {d_xy} + {d_yz}"

    def test_numerical_stability(self):
        """数值稳定性"""
        x = torch.randn(10, 32) * 0.5
        y = torch.randn(10, 32) * 0.5
        dist = pairwise_dist(x, y, curv=1.0)
        assert torch.isfinite(dist).all()

    def test_different_curvatures(self):
        """不同曲率应产生有效输出"""
        torch.manual_seed(42)
        x = torch.randn(5, 16) * 0.5
        y = torch.randn(5, 16) * 0.5
        for curv in [0.5, 1.0, 2.0]:
            dist = pairwise_dist(x, y, curv=curv)
            assert torch.isfinite(dist).all(), f"curv={curv} caused inf/nan"
            assert (dist >= -1e-6).all(), f"curv={curv} caused negative distance"


class TestHyperbolicDistanceToOrigin:
    """hyperbolic_distance_to_origin: 到原点距离"""

    def test_output_shape(self):
        """输出形状应为 [N]"""
        x = torch.randn(10, 32) * 0.5
        dist = hyperbolic_distance_to_origin(x, curv=1.0)
        assert dist.shape == (10,)

    def test_non_negative(self):
        """距离应非负"""
        x = torch.randn(10, 32) * 0.5
        dist = hyperbolic_distance_to_origin(x, curv=1.0)
        assert (dist >= -1e-6).all(), f"Negative distance: {dist.min()}"

    def test_zero_at_origin(self):
        """原点处距离为零"""
        x = torch.zeros(10, 32)
        dist = hyperbolic_distance_to_origin(x, curv=1.0)
        assert torch.allclose(dist, torch.zeros_like(dist), atol=1e-5)

    def test_larger_norm_larger_distance(self):
        """更大范数应对应更大距离"""
        torch.manual_seed(42)
        direction = torch.randn(32)
        direction = direction / direction.norm()

        x_small = direction.unsqueeze(0) * 0.1
        x_large = direction.unsqueeze(0) * 1.0

        dist_small = hyperbolic_distance_to_origin(x_small, curv=1.0).item()
        dist_large = hyperbolic_distance_to_origin(x_large, curv=1.0).item()
        assert dist_large > dist_small, \
            f"Expected {dist_large} > {dist_small}"

    def test_consistency_with_pairwise_dist(self):
        """与 pairwise_dist 到原点的距离一致"""
        torch.manual_seed(42)
        x = torch.randn(10, 32) * 0.5
        origin = torch.zeros(1, 32)

        dist_to_origin = hyperbolic_distance_to_origin(x, curv=1.0)
        dist_pairwise = pairwise_dist(x, origin, curv=1.0).squeeze(1)

        assert torch.allclose(dist_to_origin, dist_pairwise, atol=1e-5), \
            f"Max diff: {(dist_to_origin - dist_pairwise).abs().max()}"


class TestGradients:
    """梯度流验证"""

    def test_exp_map_gradient(self):
        """exp_map0 应有有效梯度"""
        v = torch.randn(10, 32, requires_grad=True) * 0.5
        x = exp_map0(v, curv=1.0)
        loss = x.sum()
        loss.backward()
        assert v.grad is not None, "No gradient computed"
        assert torch.isfinite(v.grad).all(), "Gradient contains inf/nan"

    def test_exp_map_gradient_magnitude(self):
        """exp_map0 梯度应在合理范围内"""
        v = torch.randn(10, 32, requires_grad=True) * 0.5
        x = exp_map0(v, curv=1.0)
        loss = x.sum()
        loss.backward()
        assert v.grad.abs().max() < 1e6, "Gradient explosion detected"

    def test_log_map_gradient(self):
        """log_map0 应有有效梯度"""
        x = torch.randn(10, 32, requires_grad=True) * 0.5
        v = log_map0(x, curv=1.0)
        loss = v.sum()
        loss.backward()
        assert x.grad is not None, "No gradient computed"
        assert torch.isfinite(x.grad).all(), "Gradient contains inf/nan"

    def test_pairwise_dist_gradient(self):
        """pairwise_dist 应有有效梯度"""
        x = torch.randn(5, 16, requires_grad=True) * 0.5
        y = torch.randn(7, 16, requires_grad=True) * 0.5
        dist = pairwise_dist(x, y, curv=1.0)
        loss = dist.sum()
        loss.backward()
        assert x.grad is not None, "No gradient for x"
        assert y.grad is not None, "No gradient for y"
        assert torch.isfinite(x.grad).all(), "x gradient contains inf/nan"
        assert torch.isfinite(y.grad).all(), "y gradient contains inf/nan"

    def test_distance_to_origin_gradient(self):
        """hyperbolic_distance_to_origin 应有有效梯度"""
        x = torch.randn(10, 32, requires_grad=True) * 0.5
        dist = hyperbolic_distance_to_origin(x, curv=1.0)
        loss = dist.sum()
        loss.backward()
        assert x.grad is not None, "No gradient computed"
        assert torch.isfinite(x.grad).all(), "Gradient contains inf/nan"
```

### 1.2 实现代码

**文件**: `pasco/models/hyperbolic/lorentz_ops.py`

参考 `REF/ref_repos/HyperPath/models/lorentz.py` 实现以下函数：
- `exp_map0(v, curv=1.0, eps=1e-7)`
- `log_map0(x, curv=1.0, eps=1e-7)`
- `pairwise_dist(x, y, curv=1.0, eps=1e-7)`
- `hyperbolic_distance_to_origin(x, curv=1.0, eps=1e-7)`

### 1.3 验收标准

```bash
pytest tests/hyperbolic/test_lorentz_ops.py -v
```

**必须全部通过才能进入阶段 2**

---

## 阶段 2: 标签嵌入 (LorentzLabelEmbedding)

### 2.1 先写测试

**文件**: `tests/hyperbolic/test_lorentz_label_embedding.py`

```python
"""
LorentzLabelEmbedding 测试

关键设计:
- 存储切空间向量作为可学习参数
- forward() 时通过 exp_map0 映射到双曲面
- Class 0 (outside_body) 初始化为零向量（原点）
- 其他类根据器官层级深度初始化
"""
import pytest
import torch
import torch.nn as nn

from pasco.models.hyperbolic.label_embedding import LorentzLabelEmbedding
from pasco.models.hyperbolic.lorentz_ops import hyperbolic_distance_to_origin


class TestLorentzLabelEmbedding:
    """LorentzLabelEmbedding 基本测试"""

    @pytest.fixture
    def embedding(self):
        torch.manual_seed(42)
        return LorentzLabelEmbedding(n_classes=72, embed_dim=32, curv=1.0)

    def test_init(self, embedding):
        """初始化应成功"""
        assert embedding is not None
        assert isinstance(embedding, nn.Module)

    def test_output_shape(self, embedding):
        """输出形状应为 [n_classes, embed_dim]"""
        emb = embedding()
        assert emb.shape == (72, 32)

    def test_output_finite(self, embedding):
        """输出应为有限值"""
        emb = embedding()
        assert torch.isfinite(emb).all()

    def test_class_0_at_origin(self, embedding):
        """Class 0 (outside_body) 应在原点附近"""
        emb = embedding()
        class_0_emb = emb[0]
        dist_to_origin = hyperbolic_distance_to_origin(
            class_0_emb.unsqueeze(0), curv=1.0
        )
        assert dist_to_origin.item() < 0.1, \
            f"Class 0 distance to origin: {dist_to_origin.item()}"

    def test_hierarchy_distance_ordering(self, embedding):
        """非零类别平均距离原点应比 Class 0 更远"""
        emb = embedding()
        dist = hyperbolic_distance_to_origin(emb, curv=1.0)
        dist_class_0 = dist[0].item()
        dist_others_mean = dist[1:].mean().item()
        assert dist_others_mean > dist_class_0, \
            f"Expected others mean ({dist_others_mean}) > class 0 ({dist_class_0})"

    def test_different_embed_dims(self):
        """不同嵌入维度应正常工作"""
        for dim in [16, 32, 64]:
            torch.manual_seed(42)
            emb_module = LorentzLabelEmbedding(n_classes=72, embed_dim=dim, curv=1.0)
            emb = emb_module()
            assert emb.shape == (72, dim)
            assert torch.isfinite(emb).all()


class TestLorentzLabelEmbeddingGradients:
    """梯度测试"""

    def test_gradient_flow(self):
        """应有有效梯度流"""
        torch.manual_seed(42)
        embedding = LorentzLabelEmbedding(n_classes=72, embed_dim=32, curv=1.0)
        emb = embedding()
        loss = emb.sum()
        loss.backward()

        # 检查内部参数有梯度
        has_params = False
        for param in embedding.parameters():
            has_params = True
            assert param.grad is not None, "Parameter has no gradient"
            assert torch.isfinite(param.grad).all(), "Gradient contains inf/nan"

        assert has_params, "Module has no learnable parameters"


class TestLorentzLabelEmbeddingIndexing:
    """索引访问测试"""

    def test_get_single_embedding(self):
        """应能获取单个类别嵌入"""
        torch.manual_seed(42)
        embedding = LorentzLabelEmbedding(n_classes=72, embed_dim=32, curv=1.0)
        emb = embedding()
        single = emb[5]
        assert single.shape == (32,)

    def test_get_multiple_embeddings(self):
        """应能获取多个类别嵌入"""
        torch.manual_seed(42)
        embedding = LorentzLabelEmbedding(n_classes=72, embed_dim=32, curv=1.0)
        emb = embedding()
        indices = torch.tensor([1, 5, 10])
        selected = emb[indices]
        assert selected.shape == (3, 32)

    def test_embeddings_differ_between_classes(self):
        """不同类别的嵌入应不同"""
        torch.manual_seed(42)
        embedding = LorentzLabelEmbedding(n_classes=72, embed_dim=32, curv=1.0)
        emb = embedding()
        # 检查几对不同类别的嵌入确实不同
        assert not torch.allclose(emb[1], emb[2])
        assert not torch.allclose(emb[10], emb[20])


class TestLorentzLabelEmbeddingCurvature:
    """曲率参数测试"""

    def test_curv_attribute(self):
        """曲率属性应正确存储"""
        emb = LorentzLabelEmbedding(n_classes=72, embed_dim=32, curv=1.0)
        assert hasattr(emb, 'curv')
        assert emb.curv == 1.0

    def test_different_curvatures(self):
        """不同曲率应正常工作（MVP 阶段固定为 1.0）"""
        # MVP 阶段只支持 curv=1.0
        torch.manual_seed(42)
        emb = LorentzLabelEmbedding(n_classes=72, embed_dim=32, curv=1.0)
        output = emb()
        assert torch.isfinite(output).all()
```

### 2.2 实现代码

**文件**: `pasco/models/hyperbolic/label_embedding.py`

实现 `LorentzLabelEmbedding` 类，关键点：
- 存储切空间向量
- forward 时用 `exp_map0` 映射
- Class 0 初始化为零向量
- 其他类根据层级深度初始化

### 2.3 验收标准

```bash
pytest tests/hyperbolic/test_lorentz_label_embedding.py -v
```

**必须全部通过才能进入阶段 3**

---

## 阶段 3: 投影头 (LorentzProjectionHead)

### 3.1 先写测试

**文件**: `tests/hyperbolic/test_lorentz_projection_head.py`

```python
"""
LorentzProjectionHead 测试

关键设计:
- 输入: [B, C, H, W, D] 欧几里得特征
- 输出: [B, embed_dim, H, W, D] Lorentz 空间嵌入
- 内部使用 exp_map0 将特征映射到双曲面
"""
import pytest
import torch
import torch.nn as nn

from pasco.models.hyperbolic.projection_head import LorentzProjectionHead


class TestLorentzProjectionHead:
    """LorentzProjectionHead 基本测试"""

    @pytest.fixture
    def head(self):
        torch.manual_seed(42)
        return LorentzProjectionHead(in_channels=64, embed_dim=32, curv=1.0)

    def test_init(self, head):
        """初始化应成功"""
        assert head is not None
        assert isinstance(head, nn.Module)

    def test_output_shape(self, head):
        """输出形状应正确"""
        x = torch.randn(2, 64, 8, 8, 8)  # [B, C, H, W, D]
        out = head(x)
        assert out.shape == (2, 32, 8, 8, 8), f"Got shape {out.shape}"

    def test_output_finite(self, head):
        """输出应为有限值"""
        torch.manual_seed(42)
        x = torch.randn(2, 64, 8, 8, 8)
        out = head(x)
        assert torch.isfinite(out).all()

    def test_different_spatial_sizes(self):
        """不同空间尺寸应正常工作"""
        torch.manual_seed(42)
        head = LorentzProjectionHead(in_channels=64, embed_dim=32, curv=1.0)

        for size in [(4, 4, 4), (8, 8, 8), (16, 16, 16)]:
            x = torch.randn(2, 64, *size)
            out = head(x)
            assert out.shape == (2, 32, *size)
            assert torch.isfinite(out).all()

    def test_batch_size_one(self):
        """单样本 batch 应正常工作"""
        torch.manual_seed(42)
        head = LorentzProjectionHead(in_channels=64, embed_dim=32, curv=1.0)
        x = torch.randn(1, 64, 8, 8, 8)
        out = head(x)
        assert out.shape == (1, 32, 8, 8, 8)


class TestLorentzProjectionHeadGradients:
    """梯度测试"""

    def test_gradient_flow(self):
        """应有有效梯度流"""
        torch.manual_seed(42)
        head = LorentzProjectionHead(in_channels=64, embed_dim=32, curv=1.0)
        x = torch.randn(2, 64, 4, 4, 4, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "No gradient for input"
        assert torch.isfinite(x.grad).all(), "Gradient contains inf/nan"

    def test_parameters_have_gradients(self):
        """模块参数应有梯度"""
        torch.manual_seed(42)
        head = LorentzProjectionHead(in_channels=64, embed_dim=32, curv=1.0)
        x = torch.randn(2, 64, 4, 4, 4)
        out = head(x)
        loss = out.sum()
        loss.backward()

        for name, param in head.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"inf/nan gradient in {name}"


class TestLorentzProjectionHeadNumerical:
    """数值稳定性测试"""

    def test_large_input(self):
        """大输入应稳定"""
        torch.manual_seed(42)
        head = LorentzProjectionHead(in_channels=64, embed_dim=32, curv=1.0)
        x = torch.randn(2, 64, 4, 4, 4) * 10
        out = head(x)
        assert torch.isfinite(out).all(), "Large input caused inf/nan"

    def test_small_input(self):
        """小输入应稳定"""
        torch.manual_seed(42)
        head = LorentzProjectionHead(in_channels=64, embed_dim=32, curv=1.0)
        x = torch.randn(2, 64, 4, 4, 4) * 0.001
        out = head(x)
        assert torch.isfinite(out).all(), "Small input caused inf/nan"

    def test_zero_input(self):
        """零输入应稳定"""
        torch.manual_seed(42)
        head = LorentzProjectionHead(in_channels=64, embed_dim=32, curv=1.0)
        x = torch.zeros(2, 64, 4, 4, 4)
        out = head(x)
        assert torch.isfinite(out).all(), "Zero input caused inf/nan"
```

### 3.2 实现代码

**文件**: `pasco/models/hyperbolic/projection_head.py`

实现 `LorentzProjectionHead` 类。

### 3.3 验收标准

```bash
pytest tests/hyperbolic/test_lorentz_projection_head.py -v
```

**必须全部通过才能进入阶段 4**

---

## 阶段 4: 损失函数 (LorentzRankingLoss)

### 4.1 先写测试

**文件**: `tests/hyperbolic/test_lorentz_loss.py`

```python
"""
LorentzRankingLoss 测试

损失函数: max(0, margin + d(voxel, pos) - d(voxel, neg))
- voxel: 体素嵌入
- pos: 正样本标签嵌入（ground truth）
- neg: 负样本标签嵌入（随机采样）
"""
import pytest
import torch
import torch.nn as nn

from pasco.loss.lorentz_loss import LorentzRankingLoss


class TestLorentzRankingLoss:
    """LorentzRankingLoss 基本测试"""

    @pytest.fixture
    def loss_fn(self):
        return LorentzRankingLoss(margin=0.1, curv=1.0, ignore_classes={0, 255})

    def test_init(self, loss_fn):
        """初始化应成功"""
        assert loss_fn is not None
        assert isinstance(loss_fn, nn.Module)
        assert loss_fn.margin == 0.1
        assert loss_fn.curv == 1.0

    def test_output_scalar(self, loss_fn):
        """输出应为标量"""
        torch.manual_seed(42)
        voxel_emb = torch.randn(2, 32, 4, 4, 4)  # [B, D, H, W, Z]
        labels = torch.randint(0, 72, (2, 4, 4, 4))  # [B, H, W, Z]
        label_emb = torch.randn(72, 32)  # [N_classes, D]
        loss = loss_fn(voxel_emb, labels, label_emb)
        assert loss.dim() == 0, f"Expected scalar, got dim {loss.dim()}"

    def test_output_non_negative(self, loss_fn):
        """损失应非负"""
        torch.manual_seed(42)
        voxel_emb = torch.randn(2, 32, 4, 4, 4)
        labels = torch.randint(1, 72, (2, 4, 4, 4))  # 排除 class 0
        label_emb = torch.randn(72, 32)
        loss = loss_fn(voxel_emb, labels, label_emb)
        assert loss.item() >= 0, f"Negative loss: {loss.item()}"

    def test_output_finite(self, loss_fn):
        """输出应为有限值"""
        torch.manual_seed(42)
        voxel_emb = torch.randn(2, 32, 4, 4, 4)
        labels = torch.randint(1, 72, (2, 4, 4, 4))
        label_emb = torch.randn(72, 32)
        loss = loss_fn(voxel_emb, labels, label_emb)
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"

    def test_ignore_classes(self, loss_fn):
        """忽略类别应被跳过"""
        torch.manual_seed(42)
        voxel_emb = torch.randn(2, 32, 4, 4, 4)
        labels = torch.zeros(2, 4, 4, 4, dtype=torch.long)  # 全部为 class 0
        label_emb = torch.randn(72, 32)
        loss = loss_fn(voxel_emb, labels, label_emb)
        # 全部被忽略，损失应为 0
        assert abs(loss.item()) < 1e-7, f"Expected 0, got {loss.item()}"

    def test_ignore_class_255(self, loss_fn):
        """Class 255 也应被忽略"""
        torch.manual_seed(42)
        voxel_emb = torch.randn(2, 32, 4, 4, 4)
        labels = torch.full((2, 4, 4, 4), 255, dtype=torch.long)
        label_emb = torch.randn(72, 32)
        loss = loss_fn(voxel_emb, labels, label_emb)
        assert abs(loss.item()) < 1e-7, f"Expected 0 for class 255, got {loss.item()}"


class TestLorentzRankingLossGradients:
    """梯度测试"""

    def test_gradient_to_voxel_embeddings(self):
        """体素嵌入应有梯度"""
        torch.manual_seed(42)
        loss_fn = LorentzRankingLoss(margin=0.1, curv=1.0)
        voxel_emb = torch.randn(2, 32, 4, 4, 4, requires_grad=True)
        labels = torch.randint(1, 72, (2, 4, 4, 4))
        label_emb = torch.randn(72, 32)
        loss = loss_fn(voxel_emb, labels, label_emb)
        loss.backward()
        assert voxel_emb.grad is not None, "No gradient for voxel embeddings"
        assert torch.isfinite(voxel_emb.grad).all(), "Gradient contains inf/nan"

    def test_gradient_to_label_embeddings(self):
        """标签嵌入应有梯度"""
        torch.manual_seed(42)
        loss_fn = LorentzRankingLoss(margin=0.1, curv=1.0)
        voxel_emb = torch.randn(2, 32, 4, 4, 4)
        labels = torch.randint(1, 72, (2, 4, 4, 4))
        label_emb = torch.randn(72, 32, requires_grad=True)
        loss = loss_fn(voxel_emb, labels, label_emb)
        loss.backward()
        assert label_emb.grad is not None, "No gradient for label embeddings"
        assert torch.isfinite(label_emb.grad).all(), "Gradient contains inf/nan"

    def test_gradient_when_no_valid_voxels(self):
        """当没有有效体素时梯度应安全"""
        torch.manual_seed(42)
        loss_fn = LorentzRankingLoss(margin=0.1, curv=1.0, ignore_classes={0})
        voxel_emb = torch.randn(2, 32, 4, 4, 4, requires_grad=True)
        labels = torch.zeros(2, 4, 4, 4, dtype=torch.long)  # 全部被忽略
        label_emb = torch.randn(72, 32, requires_grad=True)
        loss = loss_fn(voxel_emb, labels, label_emb)
        loss.backward()
        # 应该不会出错，梯度可能为 None 或零


class TestLorentzRankingLossNumerical:
    """数值稳定性测试"""

    def test_large_embeddings(self):
        """大嵌入值应稳定"""
        torch.manual_seed(42)
        loss_fn = LorentzRankingLoss(margin=0.1, curv=1.0)
        voxel_emb = torch.randn(2, 32, 4, 4, 4) * 10
        labels = torch.randint(1, 72, (2, 4, 4, 4))
        label_emb = torch.randn(72, 32) * 10
        loss = loss_fn(voxel_emb, labels, label_emb)
        assert torch.isfinite(loss), "Large embeddings caused inf/nan"

    def test_small_embeddings(self):
        """小嵌入值应稳定"""
        torch.manual_seed(42)
        loss_fn = LorentzRankingLoss(margin=0.1, curv=1.0)
        voxel_emb = torch.randn(2, 32, 4, 4, 4) * 0.01
        labels = torch.randint(1, 72, (2, 4, 4, 4))
        label_emb = torch.randn(72, 32) * 0.01
        loss = loss_fn(voxel_emb, labels, label_emb)
        assert torch.isfinite(loss), "Small embeddings caused inf/nan"

    def test_different_margins(self):
        """不同 margin 应正常工作"""
        torch.manual_seed(42)
        voxel_emb = torch.randn(2, 32, 4, 4, 4)
        labels = torch.randint(1, 72, (2, 4, 4, 4))
        label_emb = torch.randn(72, 32)

        for margin in [0.01, 0.1, 0.5, 1.0]:
            loss_fn = LorentzRankingLoss(margin=margin, curv=1.0)
            loss = loss_fn(voxel_emb, labels, label_emb)
            assert torch.isfinite(loss), f"margin={margin} caused inf/nan"
            assert loss.item() >= 0, f"margin={margin} caused negative loss"
```

### 4.2 实现代码

**文件**: `pasco/loss/lorentz_loss.py`

实现 `LorentzRankingLoss` 类。

### 4.3 验收标准

```bash
pytest tests/hyperbolic/test_lorentz_loss.py -v
```

**必须全部通过才能进入阶段 5**

---

## 阶段 5: 模型集成 (BodyNetHyperbolic)

### 5.1 先写测试

**文件**: `tests/hyperbolic/test_lorentz_body_net.py`

```python
"""
BodyNetHyperbolic Lorentz 集成测试

注意: 此测试适配现有 BodyNetHyperbolic 接口:
- forward(x) 返回 logits
- forward_with_hyperbolic(x) 返回 (logits, voxel_embeddings)
- 损失计算通过 hyp_loss_fn(voxel_emb, labels, label_emb) 直接调用
"""
import pytest
import torch
import torch.nn as nn

from pasco.models.body_net_hyperbolic import BodyNetHyperbolic


class TestBodyNetHyperbolicInit:
    """初始化测试"""

    def test_init_with_lorentz_components(self):
        """初始化应包含 Lorentz 组件"""
        torch.manual_seed(42)
        model = BodyNetHyperbolic(
            n_classes=72,
            base_channels=32,
            embed_dim=32,
            hyperbolic_weight=0.1,
            margin=0.1,
            use_light_model=True,
        )
        assert model is not None
        assert hasattr(model, 'hyp_head'), "Missing hyp_head"
        assert hasattr(model, 'label_emb'), "Missing label_emb"
        assert hasattr(model, 'hyp_loss_fn'), "Missing hyp_loss_fn"

    def test_hyperbolic_weight_stored(self):
        """双曲损失权重应正确存储"""
        model = BodyNetHyperbolic(
            n_classes=72,
            base_channels=32,
            embed_dim=32,
            hyperbolic_weight=0.2,
            margin=0.1,
            use_light_model=True,
        )
        assert model.hyperbolic_weight == 0.2


class TestBodyNetHyperbolicForward:
    """前向传播测试"""

    @pytest.fixture
    def model(self):
        torch.manual_seed(42)
        return BodyNetHyperbolic(
            n_classes=72,
            base_channels=32,
            embed_dim=32,
            hyperbolic_weight=0.1,
            margin=0.1,
            use_light_model=True,
        )

    def test_forward_returns_logits(self, model):
        """forward() 应返回 logits"""
        torch.manual_seed(42)
        x = torch.randn(1, 1, 32, 32, 32)
        out = model(x)
        assert out.shape[0] == 1
        assert out.shape[1] == 72  # n_classes

    def test_forward_with_hyperbolic_returns_tuple(self, model):
        """forward_with_hyperbolic() 应返回 (logits, embeddings)"""
        torch.manual_seed(42)
        x = torch.randn(1, 1, 32, 32, 32)
        logits, voxel_emb = model.forward_with_hyperbolic(x)

        # 检查 logits
        assert logits.shape[0] == 1
        assert logits.shape[1] == 72

        # 检查双曲嵌入
        assert voxel_emb.shape[0] == 1
        assert voxel_emb.shape[1] == 32  # embed_dim

    def test_embeddings_are_finite(self, model):
        """嵌入应为有限值"""
        torch.manual_seed(42)
        x = torch.randn(1, 1, 32, 32, 32)
        _, voxel_emb = model.forward_with_hyperbolic(x)
        assert torch.isfinite(voxel_emb).all()


class TestBodyNetHyperbolicLoss:
    """损失计算测试"""

    @pytest.fixture
    def model(self):
        torch.manual_seed(42)
        return BodyNetHyperbolic(
            n_classes=72,
            base_channels=32,
            embed_dim=32,
            hyperbolic_weight=0.1,
            margin=0.1,
            use_light_model=True,
        )

    def test_hyperbolic_loss_computation(self, model):
        """双曲损失计算应正确"""
        torch.manual_seed(42)
        x = torch.randn(1, 1, 16, 16, 16)
        labels = torch.randint(0, 72, (1, 16, 16, 16))

        logits, voxel_emb = model.forward_with_hyperbolic(x)
        label_emb = model.label_emb()
        hyp_loss = model.hyp_loss_fn(voxel_emb, labels, label_emb)

        assert torch.isfinite(hyp_loss), f"Loss is not finite: {hyp_loss}"
        assert hyp_loss.item() >= 0, f"Negative loss: {hyp_loss.item()}"

    def test_combined_loss_computation(self, model):
        """组合损失计算应正确"""
        torch.manual_seed(42)
        x = torch.randn(1, 1, 16, 16, 16)
        labels = torch.randint(1, 72, (1, 16, 16, 16))  # 排除 class 0

        logits, voxel_emb = model.forward_with_hyperbolic(x)

        # CE loss
        ce_loss = nn.functional.cross_entropy(logits, labels)

        # Hyperbolic loss
        label_emb = model.label_emb()
        hyp_loss = model.hyp_loss_fn(voxel_emb, labels, label_emb)

        # Combined
        total_loss = ce_loss + model.hyperbolic_weight * hyp_loss

        assert torch.isfinite(total_loss)
        assert total_loss.item() >= 0


class TestBodyNetHyperbolicGradients:
    """梯度测试"""

    def test_gradient_flow_through_hyperbolic_path(self):
        """梯度应流经双曲路径"""
        torch.manual_seed(42)
        model = BodyNetHyperbolic(
            n_classes=72,
            base_channels=32,
            embed_dim=32,
            hyperbolic_weight=0.1,
            margin=0.1,
            use_light_model=True,
        )
        model.train()

        x = torch.randn(1, 1, 16, 16, 16)
        labels = torch.randint(1, 72, (1, 16, 16, 16))

        logits, voxel_emb = model.forward_with_hyperbolic(x)
        ce_loss = nn.functional.cross_entropy(logits, labels)

        label_emb = model.label_emb()
        hyp_loss = model.hyp_loss_fn(voxel_emb, labels, label_emb)

        total_loss = ce_loss + 0.1 * hyp_loss
        total_loss.backward()

        # 检查关键组件有梯度
        hyp_head_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.hyp_head.parameters()
        )
        assert hyp_head_has_grad, "hyp_head has no gradients"

        label_emb_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.label_emb.parameters()
        )
        assert label_emb_has_grad, "label_emb has no gradients"

    def test_all_parameters_have_gradients(self):
        """所有可训练参数应有梯度"""
        torch.manual_seed(42)
        model = BodyNetHyperbolic(
            n_classes=72,
            base_channels=32,
            embed_dim=32,
            hyperbolic_weight=0.1,
            margin=0.1,
            use_light_model=True,
        )
        model.train()

        x = torch.randn(1, 1, 16, 16, 16)
        labels = torch.randint(1, 72, (1, 16, 16, 16))

        logits, voxel_emb = model.forward_with_hyperbolic(x)
        ce_loss = nn.functional.cross_entropy(logits, labels)
        label_emb = model.label_emb()
        hyp_loss = model.hyp_loss_fn(voxel_emb, labels, label_emb)
        total_loss = ce_loss + 0.1 * hyp_loss
        total_loss.backward()

        # 检查所有参数
        params_without_grad = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is None:
                params_without_grad.append(name)

        assert len(params_without_grad) == 0, \
            f"Parameters without gradient: {params_without_grad}"
```

### 5.2 实现代码

**文件**: `pasco/models/body_net_hyperbolic.py`

集成所有 Lorentz 组件，替换现有 Poincaré 组件。

### 5.3 验收标准

```bash
pytest tests/hyperbolic/test_lorentz_body_net.py -v
```

**必须全部通过才能进入阶段 6**

---

## 阶段 6: 端到端集成测试

### 6.1 先写测试

**文件**: `tests/hyperbolic/test_lorentz_integration.py`

```python
"""
Lorentz 模型端到端集成测试

测试目标:
1. 模型能在单个 batch 上过拟合
2. 训练过程中不出现 NaN
3. 层级结构在训练后保持
"""
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from pasco.models.body_net_hyperbolic import BodyNetHyperbolic
from pasco.models.hyperbolic.lorentz_ops import hyperbolic_distance_to_origin


class TestEndToEndTraining:
    """端到端训练测试"""

    def test_overfit_single_batch(self):
        """应能过拟合单个 batch"""
        # 固定随机种子确保可重复性
        torch.manual_seed(42)

        model = BodyNetHyperbolic(
            n_classes=72,
            base_channels=32,
            embed_dim=32,
            hyperbolic_weight=0.1,
            margin=0.1,
            use_light_model=True,
        )
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # 固定的 batch
        x = torch.randn(2, 1, 16, 16, 16)
        labels = torch.randint(1, 72, (2, 16, 16, 16))

        initial_loss = None
        final_loss = None

        # 训练 100 步
        for step in range(100):
            optimizer.zero_grad()

            logits, voxel_emb = model.forward_with_hyperbolic(x)
            ce_loss = nn.functional.cross_entropy(logits, labels)

            label_emb = model.label_emb()
            hyp_loss = model.hyp_loss_fn(voxel_emb, labels, label_emb)

            total_loss = ce_loss + 0.1 * hyp_loss

            if step == 0:
                initial_loss = total_loss.item()
            if step == 99:
                final_loss = total_loss.item()

            total_loss.backward()
            optimizer.step()

        # 损失应下降
        assert final_loss < initial_loss, \
            f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"

    def test_no_nan_during_training(self):
        """训练过程中不应出现 NaN"""
        torch.manual_seed(42)

        model = BodyNetHyperbolic(
            n_classes=72,
            base_channels=32,
            embed_dim=32,
            hyperbolic_weight=0.1,
            margin=0.1,
            use_light_model=True,
        )
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        for step in range(20):
            # 每步使用不同的随机数据
            torch.manual_seed(42 + step)
            x = torch.randn(2, 1, 16, 16, 16)
            labels = torch.randint(1, 72, (2, 16, 16, 16))

            optimizer.zero_grad()

            logits, voxel_emb = model.forward_with_hyperbolic(x)
            ce_loss = nn.functional.cross_entropy(logits, labels)

            label_emb = model.label_emb()
            hyp_loss = model.hyp_loss_fn(voxel_emb, labels, label_emb)

            total_loss = ce_loss + 0.1 * hyp_loss

            assert torch.isfinite(total_loss), f"Step {step}: Loss is NaN or Inf"

            total_loss.backward()

            # 检查梯度
            for name, param in model.named_parameters():
                if param.grad is not None:
                    assert torch.isfinite(param.grad).all(), \
                        f"Step {step}: NaN gradient in {name}"

            optimizer.step()

    def test_loss_decreases_over_time(self):
        """损失应随时间下降"""
        torch.manual_seed(42)

        model = BodyNetHyperbolic(
            n_classes=72,
            base_channels=32,
            embed_dim=32,
            hyperbolic_weight=0.1,
            margin=0.1,
            use_light_model=True,
        )
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # 固定数据
        x = torch.randn(2, 1, 16, 16, 16)
        labels = torch.randint(1, 72, (2, 16, 16, 16))

        losses = []
        for step in range(50):
            optimizer.zero_grad()

            logits, voxel_emb = model.forward_with_hyperbolic(x)
            ce_loss = nn.functional.cross_entropy(logits, labels)

            label_emb = model.label_emb()
            hyp_loss = model.hyp_loss_fn(voxel_emb, labels, label_emb)

            total_loss = ce_loss + 0.1 * hyp_loss
            losses.append(total_loss.item())

            total_loss.backward()
            optimizer.step()

        # 后半段平均损失应小于前半段
        first_half_mean = sum(losses[:25]) / 25
        second_half_mean = sum(losses[25:]) / 25
        assert second_half_mean < first_half_mean, \
            f"Loss not decreasing: first half {first_half_mean:.4f}, second half {second_half_mean:.4f}"


class TestHyperbolicEmbeddingQuality:
    """双曲嵌入质量测试"""

    def test_hierarchy_preserved_at_init(self):
        """初始化时层级结构应正确"""
        torch.manual_seed(42)
        model = BodyNetHyperbolic(
            n_classes=72,
            base_channels=32,
            embed_dim=32,
            hyperbolic_weight=0.1,
            margin=0.1,
            use_light_model=True,
        )

        # 获取标签嵌入
        label_emb = model.label_emb()

        # 计算到原点距离
        dist = hyperbolic_distance_to_origin(label_emb, curv=1.0)

        # Class 0 应最近原点
        assert dist[0] < dist[1:].min(), \
            f"Class 0 dist ({dist[0]:.4f}) not smallest (min other: {dist[1:].min():.4f})"

    def test_label_embeddings_differ(self):
        """标签嵌入应各不相同"""
        torch.manual_seed(42)
        model = BodyNetHyperbolic(
            n_classes=72,
            base_channels=32,
            embed_dim=32,
            hyperbolic_weight=0.1,
            margin=0.1,
            use_light_model=True,
        )

        label_emb = model.label_emb()

        # 检查部分类别的嵌入不同
        for i in range(1, 10):
            for j in range(i + 1, 11):
                assert not torch.allclose(label_emb[i], label_emb[j]), \
                    f"Class {i} and {j} have same embedding"

    def test_voxel_embeddings_are_finite(self):
        """体素嵌入应为有限值"""
        torch.manual_seed(42)
        model = BodyNetHyperbolic(
            n_classes=72,
            base_channels=32,
            embed_dim=32,
            hyperbolic_weight=0.1,
            margin=0.1,
            use_light_model=True,
        )
        model.eval()

        x = torch.randn(2, 1, 16, 16, 16)
        _, voxel_emb = model.forward_with_hyperbolic(x)

        assert torch.isfinite(voxel_emb).all(), "Voxel embeddings contain inf/nan"


class TestModuleImports:
    """模块导入测试"""

    def test_all_lorentz_modules_importable(self):
        """所有 Lorentz 模块应可导入"""
        from pasco.models.hyperbolic.lorentz_ops import (
            exp_map0, log_map0, pairwise_dist, hyperbolic_distance_to_origin
        )
        from pasco.models.hyperbolic.label_embedding import LorentzLabelEmbedding
        from pasco.models.hyperbolic.projection_head import LorentzProjectionHead
        from pasco.loss.lorentz_loss import LorentzRankingLoss

        assert callable(exp_map0)
        assert callable(log_map0)
        assert callable(pairwise_dist)
        assert callable(hyperbolic_distance_to_origin)
        assert LorentzLabelEmbedding is not None
        assert LorentzProjectionHead is not None
        assert LorentzRankingLoss is not None
```

### 6.2 实现代码

更新 `scripts/body/train_body.py` 的命令行参数（如需要）。

### 6.3 验收标准

```bash
pytest tests/hyperbolic/test_lorentz_integration.py -v
```

**全部通过即完成迁移**

---

## 执行流程图

```
┌─────────────────────────────────────────────────────────────┐
│                      开始迁移                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  阶段 1: lorentz_ops.py                                      │
│  ├── 1. 编写 test_lorentz_ops.py                            │
│  ├── 2. 运行测试 (应全部失败 - 导入错误)                      │
│  ├── 3. 实现 lorentz_ops.py                                  │
│  ├── 4. 运行测试                                             │
│  └── 5. 全部通过? ──NO──> 修复实现                           │
│              │                                               │
│             YES                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  阶段 2: LorentzLabelEmbedding                               │
│  ├── 1. 编写 test_lorentz_label_embedding.py                │
│  ├── 2. 运行测试 (应全部失败)                                │
│  ├── 3. 实现 LorentzLabelEmbedding                          │
│  ├── 4. 运行测试                                             │
│  └── 5. 全部通过? ──NO──> 修复实现                           │
│              │                                               │
│             YES                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  阶段 3: LorentzProjectionHead                               │
│  ├── 1. 编写 test_lorentz_projection_head.py                │
│  ├── 2. 运行测试 (应全部失败)                                │
│  ├── 3. 实现 LorentzProjectionHead                          │
│  ├── 4. 运行测试                                             │
│  └── 5. 全部通过? ──NO──> 修复实现                           │
│              │                                               │
│             YES                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  阶段 4: LorentzRankingLoss                                  │
│  ├── 1. 编写 test_lorentz_loss.py                           │
│  ├── 2. 运行测试 (应全部失败)                                │
│  ├── 3. 实现 LorentzRankingLoss                             │
│  ├── 4. 运行测试                                             │
│  └── 5. 全部通过? ──NO──> 修复实现                           │
│              │                                               │
│             YES                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  阶段 5: BodyNetHyperbolic 集成                              │
│  ├── 1. 编写 test_lorentz_body_net.py                       │
│  ├── 2. 运行测试 (应全部失败)                                │
│  ├── 3. 集成所有 Lorentz 组件                                │
│  ├── 4. 运行测试                                             │
│  └── 5. 全部通过? ──NO──> 修复实现                           │
│              │                                               │
│             YES                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  阶段 6: 端到端集成测试                                       │
│  ├── 1. 编写 test_lorentz_integration.py                    │
│  ├── 2. 运行测试 (应全部失败)                                │
│  ├── 3. 更新 train_body.py                                  │
│  ├── 4. 运行测试                                             │
│  └── 5. 全部通过? ──NO──> 修复实现                           │
│              │                                               │
│             YES                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      迁移完成!                                │
│  运行完整测试套件: pytest tests/hyperbolic/test_lorentz*.py -v│
└─────────────────────────────────────────────────────────────┘
```

---

## 测试命令汇总

```bash
# 阶段 1
pytest tests/hyperbolic/test_lorentz_ops.py -v

# 阶段 2
pytest tests/hyperbolic/test_lorentz_label_embedding.py -v

# 阶段 3
pytest tests/hyperbolic/test_lorentz_projection_head.py -v

# 阶段 4
pytest tests/hyperbolic/test_lorentz_loss.py -v

# 阶段 5
pytest tests/hyperbolic/test_lorentz_body_net.py -v

# 阶段 6 (端到端)
pytest tests/hyperbolic/test_lorentz_integration.py -v

# 完整 Lorentz 测试套件
pytest tests/hyperbolic/test_lorentz*.py -v

# 带覆盖率
pytest tests/hyperbolic/test_lorentz*.py -v --cov=pasco.models.hyperbolic --cov=pasco.loss
```

---

## 进度追踪

| 阶段 | 测试文件 | 测试状态 | 实现状态 |
|------|----------|----------|----------|
| 1 | test_lorentz_ops.py | [ ] | [ ] |
| 2 | test_lorentz_label_embedding.py | [ ] | [ ] |
| 3 | test_lorentz_projection_head.py | [ ] | [ ] |
| 4 | test_lorentz_loss.py | [ ] | [ ] |
| 5 | test_lorentz_body_net.py | [ ] | [ ] |
| 6 | test_lorentz_integration.py | [ ] | [ ] |

---

## 注意事项

1. **严格遵守 TDD 顺序**: 必须先写测试，测试失败后再实现
2. **不能跳过阶段**: 每个阶段测试全部通过才能进入下一阶段
3. **测试失败时**: 优先修复实现，不要修改测试（除非测试本身有误）
4. **参考实现**: 核心数学操作参考 `REF/ref_repos/HyperPath/models/lorentz.py`
5. **保持简单**: MVP 阶段使用固定曲率 curv=1.0，不添加额外功能
6. **随机种子**: 所有测试使用 `torch.manual_seed(42)` 确保可重复性
7. **接口兼容**: 测试适配现有 `BodyNetHyperbolic` 接口（`forward_with_hyperbolic`）

---

## 修订历史

| 版本 | 日期 | 修改内容 |
|------|------|---------|
| v1.0 | - | 初始版本 |
| v1.1 | - | 审查修复：添加导入测试、修复接口不匹配、增强数值稳定性测试、固定随机种子 |
