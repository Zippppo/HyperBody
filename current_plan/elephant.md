# 实践准则：文本嵌入方向初始化 (TDD驱动)

> **核心原则**：先写测试，后写代码。没有失败的测试，就没有新代码。

---

## TDD 工作流概览

```
┌─────────────────────────────────────────────────────────────────┐
│                    RED-GREEN-REFACTOR 循环                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐                  │
│   │   RED   │────▶│  GREEN  │────▶│ REFACTOR│────┐             │
│   │ 写测试  │     │ 写代码  │     │  重构   │    │             │
│   │ (失败)  │     │ (通过)  │     │ (优化)  │    │             │
│   └─────────┘     └─────────┘     └─────────┘    │             │
│        ▲                                          │             │
│        └──────────────────────────────────────────┘             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 第一阶段：测试脚手架 (RED)

### 1.1 创建测试文件

**文件路径**：`tests/hyperbolic/Lorentz/text_embedding/test_text_direction_init.py`

```bash
# 确认目录存在
mkdir -p tests/hyperbolic/Lorentz/text_embedding
touch tests/hyperbolic/Lorentz/text_embedding/__init__.py
```

### 1.2 先写测试，必须失败

```python
"""
测试文件：test_text_direction_init.py
目标：验证 Mode 3 (文本方向 + 深度范数) 初始化

测试编写顺序（由简到繁）：
1. 参数存在性测试
2. 互斥性测试
3. 几何约束测试
4. 语义保持测试
5. 边界情况测试
"""
import pytest
import torch

# 第一个测试：参数存在
class TestParameterExists:
    """测试 use_text_direction_init 参数存在且可用"""

    def test_parameter_defaults_to_false(self):
        """参数默认为 False"""
        # 这个测试现在应该 FAIL，因为参数还不存在
        from pasco.models.hyperbolic.label_embedding import LorentzLabelEmbedding
        model = LorentzLabelEmbedding(n_classes=70, embed_dim=32)
        assert hasattr(model, 'use_text_direction_init')
        assert model.use_text_direction_init is False
```

### 1.3 运行测试，确认失败

```bash
pytest tests/hyperbolic/Lorentz/text_embedding/test_text_direction_init.py -v
# 预期：FAILED (AttributeError: 'LorentzLabelEmbedding' has no attribute 'use_text_direction_init')
```

**重要**：如果测试没有失败，说明测试写错了！

---

## 第二阶段：最小实现 (GREEN)

### 2.1 只写让测试通过的最少代码

在 `pasco/models/hyperbolic/label_embedding.py` 中：

```python
class LorentzLabelEmbedding(nn.Module):
    def __init__(
        self,
        n_classes: int,
        embed_dim: int,
        # ... 现有参数 ...
        use_text_direction_init: bool = False,  # 新增参数
    ):
        super().__init__()
        self.use_text_direction_init = use_text_direction_init  # 存储参数
        # ... 其余代码不变 ...
```

### 2.2 再次运行测试

```bash
pytest tests/hyperbolic/Lorentz/text_embedding/test_text_direction_init.py -v
# 预期：PASSED
```

---

## 第三阶段：迭代开发 (重复 RED-GREEN)

### 测试清单（按顺序实现）

| 序号 | 测试名称 | 测试内容 | 优先级 |
|------|---------|---------|--------|
| 1 | `test_parameter_defaults_to_false` | 参数默认值 | P0 |
| 2 | `test_mutual_exclusion` | 与 use_text_embeddings 互斥 | P0 |
| 3 | `test_tangent_vectors_is_parameter` | tangent_vectors 是 nn.Parameter | P0 |
| 4 | `test_no_projector_created` | 不创建 projector | P0 |
| 5 | `test_class_zero_at_origin` | 类别 0 在原点 | P0 |
| 6 | `test_deeper_classes_have_larger_norms` | 深层类范数更大 | P1 |
| 7 | `test_norms_in_expected_range` | 范数在 [min_radius, max_radius] | P1 |
| 8 | `test_embeddings_on_hyperboloid` | exp_map0 后在双曲面上 | P1 |
| 9 | `test_similar_classes_have_similar_directions` | 相似类方向相似 | P2 |
| 10 | `test_gradient_flows` | 梯度可以流动 | P1 |
| 11 | `test_all_embedding_types` | sat/clip/biomedclip 都可用 | P2 |
| 12 | `test_pca_dimension_reduction` | PCA 降维正常工作 | P2 |

### 每个测试的实现模板

```python
class TestMutualExclusion:
    """测试互斥性"""

    def test_raises_when_both_enabled(self):
        """同时启用两种模式应该报错"""
        from pasco.models.hyperbolic.label_embedding import LorentzLabelEmbedding

        with pytest.raises(ValueError, match="mutually exclusive"):
            LorentzLabelEmbedding(
                n_classes=70,
                embed_dim=32,
                use_text_embeddings=True,
                use_text_direction_init=True,
            )
```

---

## 第四阶段：几何约束测试

### 4.1 类别 0 在原点

```python
class TestGeometricConstraints:
    """几何约束测试"""

    def test_class_zero_at_origin(self):
        """类别 0 的切向量应为零向量"""
        model = LorentzLabelEmbedding(
            n_classes=70,
            embed_dim=32,
            use_text_direction_init=True,
            text_embedding_type='sat',
        )

        # 类别 0 的切向量应该是零
        assert torch.allclose(
            model.tangent_vectors.data[0],
            torch.zeros(32),
            atol=1e-6
        )

    def test_deeper_classes_have_larger_norms(self):
        """更深的类别应该有更大的切向量范数"""
        from pasco.data.body.organ_hierarchy import CLASS_DEPTHS

        model = LorentzLabelEmbedding(
            n_classes=70,
            embed_dim=32,
            use_text_direction_init=True,
        )

        # 获取两个不同深度的类
        shallow_classes = [c for c, d in CLASS_DEPTHS.items() if d == 1 and c > 0]
        deep_classes = [c for c, d in CLASS_DEPTHS.items() if d == 5]

        if shallow_classes and deep_classes:
            shallow_norm = model.tangent_vectors.data[shallow_classes[0]].norm()
            deep_norm = model.tangent_vectors.data[deep_classes[0]].norm()
            assert deep_norm > shallow_norm
```

### 4.2 双曲面约束

```python
    def test_embeddings_on_hyperboloid(self):
        """exp_map0 后的嵌入应该在双曲面上"""
        model = LorentzLabelEmbedding(
            n_classes=70,
            embed_dim=32,
            use_text_direction_init=True,
        )

        embeddings = model()  # 调用 forward
        curv = model.curv

        # 计算时间分量
        x_time = torch.sqrt(1/curv + torch.sum(embeddings**2, dim=-1))

        # 检查 Lorentz 约束: -t^2 + ||x||^2 = -1/c
        lorentz_norm = -x_time**2 + torch.sum(embeddings**2, dim=-1)
        expected = -1/curv

        assert torch.allclose(lorentz_norm, torch.full_like(lorentz_norm, expected), atol=1e-5)
```

---

## 第五阶段：边界情况测试

```python
class TestEdgeCases:
    """边界情况测试"""

    def test_missing_class_depth_uses_max_depth(self):
        """未定义深度的类使用 MAX_DEPTH"""
        # 如果有新类别未在 CLASS_DEPTHS 中定义
        pass

    def test_pca_with_small_embed_dim(self):
        """小 embed_dim 的 PCA 降维"""
        model = LorentzLabelEmbedding(
            n_classes=70,
            embed_dim=8,  # 很小的维度
            use_text_direction_init=True,
        )
        assert model.tangent_vectors.shape == (70, 8)

    def test_gradient_flows_through_tangent_vectors(self):
        """梯度可以流经 tangent_vectors"""
        model = LorentzLabelEmbedding(
            n_classes=70,
            embed_dim=32,
            use_text_direction_init=True,
        )

        embeddings = model()
        loss = embeddings.sum()
        loss.backward()

        assert model.tangent_vectors.grad is not None
        assert not torch.all(model.tangent_vectors.grad == 0)
```

---

## 实施检查清单

### 每次代码修改前

- [ ] 测试文件存在
- [ ] 新测试已编写
- [ ] 测试运行失败（RED）

### 每次代码修改后

- [ ] 运行 `pytest tests/hyperbolic/Lorentz/text_embedding/test_text_direction_init.py -v`
- [ ] 所有测试通过（GREEN）
- [ ] 考虑是否需要重构（REFACTOR）

### 功能完成标准

- [ ] 所有 P0 测试通过
- [ ] 所有 P1 测试通过
- [ ] 覆盖率 >= 80%
- [ ] 无 linting 错误

---

## 运行命令速查

```bash
# 运行单个测试文件
pytest tests/hyperbolic/Lorentz/text_embedding/test_text_direction_init.py -v

# 运行特定测试类
pytest tests/hyperbolic/Lorentz/text_embedding/test_text_direction_init.py::TestGeometricConstraints -v

# 运行特定测试方法
pytest tests/hyperbolic/Lorentz/text_embedding/test_text_direction_init.py::TestGeometricConstraints::test_class_zero_at_origin -v

# 运行并显示覆盖率
pytest tests/hyperbolic/Lorentz/text_embedding/test_text_direction_init.py --cov=pasco.models.hyperbolic.label_embedding --cov-report=term-missing

# 失败时停止
pytest tests/hyperbolic/Lorentz/text_embedding/test_text_direction_init.py -x

# 显示打印输出
pytest tests/hyperbolic/Lorentz/text_embedding/test_text_direction_init.py -s
```

---

## 禁止事项

1. **禁止**：在测试通过前写更多实现代码
2. **禁止**：跳过失败的测试继续开发
3. **禁止**：一次性写完所有代码再写测试
4. **禁止**：修改测试来适应错误的实现
5. **禁止**：提交没有对应测试的代码

---

## 进度追踪

| 阶段 | 状态 | 测试数 | 通过数 |
|------|------|--------|--------|
| 参数存在性 | 待开始 | 1 | 0 |
| 互斥性检查 | 待开始 | 1 | 0 |
| 基础功能 | 待开始 | 3 | 0 |
| 几何约束 | 待开始 | 4 | 0 |
| 边界情况 | 待开始 | 3 | 0 |
| **总计** | - | **12** | **0** |

---

> **记住**：写测试不是负担，而是设计工具。测试定义了代码应该做什么，代码只是让测试通过的手段。
