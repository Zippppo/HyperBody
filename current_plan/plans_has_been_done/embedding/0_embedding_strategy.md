# TDD工作流程：Pre-trained Text Embeddings 实现策略

## 概述

本文档定义了实现 `0_embedding.md` 计划时必须严格遵守的TDD工作流程。

**核心原则**：测试先行，代码后写。每个功能点必须先有失败的测试，再有通过的实现。

---

## TDD循环

```
RED → GREEN → REFACTOR → VERIFY
 ↑__________________________|
```

| 阶段 | 描述 | 验证标准 |
|------|------|----------|
| RED | 写测试，测试必须失败 | `pytest` 显示 FAILED |
| GREEN | 写最小实现，测试通过 | `pytest` 显示 PASSED |
| REFACTOR | 重构代码，保持测试通过 | `pytest` 仍然 PASSED |
| VERIFY | 验证覆盖率 ≥80% | `pytest --cov` 显示覆盖率 |

---

## 实现阶段与测试策略

### Phase 1: TextEmbeddingProjector 模块

**目标文件**: `pasco/models/hyperbolic/text_projector.py`

#### Step 1.1: 测试模块结构 (RED)

```bash
# 先创建测试文件
touch tests/models/hyperbolic/test_text_projector.py
```

**测试内容**:
```python
# tests/models/hyperbolic/test_text_projector.py
import pytest
import torch

class TestTextEmbeddingProjector:
    """TextEmbeddingProjector 单元测试"""

    def test_init_default_params(self):
        """测试默认参数初始化"""
        from pasco.models.hyperbolic.text_projector import TextEmbeddingProjector

        proj = TextEmbeddingProjector()
        assert proj.text_dim == 768
        assert proj.embed_dim == 32

    def test_init_custom_params(self):
        """测试自定义参数初始化"""
        from pasco.models.hyperbolic.text_projector import TextEmbeddingProjector

        proj = TextEmbeddingProjector(text_dim=512, embed_dim=64, hidden_dim=128)
        assert proj.text_dim == 512
        assert proj.embed_dim == 64

    def test_forward_shape(self):
        """测试前向传播输出形状"""
        from pasco.models.hyperbolic.text_projector import TextEmbeddingProjector

        proj = TextEmbeddingProjector(text_dim=768, embed_dim=32)
        x = torch.randn(72, 768)
        out = proj(x)

        assert out.shape == (72, 32)

    def test_forward_biomedclip_shape(self):
        """测试 BioMedCLIP 维度 (512)"""
        from pasco.models.hyperbolic.text_projector import TextEmbeddingProjector

        proj = TextEmbeddingProjector(text_dim=512, embed_dim=32)
        x = torch.randn(72, 512)
        out = proj(x)

        assert out.shape == (72, 32)

    def test_output_layer_small_init(self):
        """测试输出层小初始化 (std≈0.02)"""
        from pasco.models.hyperbolic.text_projector import TextEmbeddingProjector

        proj = TextEmbeddingProjector()
        output_layer = proj.mlp[3]  # Last linear layer

        # 检查权重标准差在合理范围内
        std = output_layer.weight.std().item()
        assert 0.01 < std < 0.05, f"Expected std ~0.02, got {std}"

        # 检查偏置为零
        assert torch.allclose(output_layer.bias, torch.zeros_like(output_layer.bias))

    def test_gradient_flow(self):
        """测试梯度可以正常反向传播"""
        from pasco.models.hyperbolic.text_projector import TextEmbeddingProjector

        proj = TextEmbeddingProjector()
        x = torch.randn(10, 768, requires_grad=True)
        out = proj(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert proj.mlp[0].weight.grad is not None
```

**执行顺序**:
1. 运行测试 → 预期 FAILED (模块不存在)
2. 创建 `text_projector.py` 空文件 → FAILED (类不存在)
3. 添加类骨架 → FAILED (方法不存在)
4. 逐步实现直到 PASSED

#### Step 1.2: 验证阶段

```bash
# 运行测试
pytest tests/models/hyperbolic/test_text_projector.py -v

# 检查覆盖率
pytest tests/models/hyperbolic/test_text_projector.py --cov=pasco.models.hyperbolic.text_projector --cov-report=term-missing
```

**覆盖率要求**: ≥80%

---

### Phase 2: LorentzLabelEmbedding 修改

**目标文件**: `pasco/models/hyperbolic/label_embedding.py`

#### Step 2.1: 测试常量定义 (RED)

```python
# tests/models/hyperbolic/test_label_embedding_text.py
import pytest

class TestTextEmbeddingConstants:
    """测试文本嵌入相关常量"""

    def test_embedding_paths_defined(self):
        """测试嵌入路径常量已定义"""
        from pasco.models.hyperbolic.label_embedding import TEXT_EMBEDDING_PATHS

        assert "sat" in TEXT_EMBEDDING_PATHS
        assert "clip" in TEXT_EMBEDDING_PATHS
        assert "biomedclip" in TEXT_EMBEDDING_PATHS

    def test_embedding_dims_defined(self):
        """测试嵌入维度常量已定义"""
        from pasco.models.hyperbolic.label_embedding import TEXT_EMBEDDING_DIMS

        assert TEXT_EMBEDDING_DIMS["sat"] == 768
        assert TEXT_EMBEDDING_DIMS["clip"] == 768
        assert TEXT_EMBEDDING_DIMS["biomedclip"] == 512
```

#### Step 2.2: 测试初始化参数 (RED)

```python
class TestLorentzLabelEmbeddingTextInit:
    """测试 LorentzLabelEmbedding 文本嵌入初始化"""

    def test_new_params_exist(self):
        """测试新参数 use_text_embeddings, text_embedding_type 存在"""
        from pasco.models.hyperbolic.label_embedding import LorentzLabelEmbedding
        import inspect

        sig = inspect.signature(LorentzLabelEmbedding.__init__)
        params = list(sig.parameters.keys())

        assert "use_text_embeddings" in params
        assert "text_embedding_type" in params

    def test_default_values(self):
        """测试默认值为禁用状态"""
        from pasco.models.hyperbolic.label_embedding import LorentzLabelEmbedding

        emb = LorentzLabelEmbedding(n_classes=72)

        assert emb.use_text_embeddings == False
        assert not hasattr(emb, 'projector') or emb.projector is None
```

#### Step 2.3: 测试文本嵌入加载 (RED)

```python
class TestTextEmbeddingLoading:
    """测试文本嵌入加载功能"""

    @pytest.fixture
    def n_classes(self):
        return 72

    def test_load_sat_embeddings(self, n_classes):
        """测试加载 SAT 嵌入"""
        from pasco.models.hyperbolic.label_embedding import LorentzLabelEmbedding

        emb = LorentzLabelEmbedding(
            n_classes=n_classes,
            embed_dim=32,
            use_text_embeddings=True,
            text_embedding_type="sat",
        )

        assert hasattr(emb, 'text_embeddings')
        assert emb.text_embeddings.shape == (n_classes, 768)
        assert emb.text_embeddings.dtype == torch.float16

    def test_load_biomedclip_embeddings(self, n_classes):
        """测试加载 BioMedCLIP 嵌入 (512维)"""
        from pasco.models.hyperbolic.label_embedding import LorentzLabelEmbedding

        emb = LorentzLabelEmbedding(
            n_classes=n_classes,
            embed_dim=32,
            use_text_embeddings=True,
            text_embedding_type="biomedclip",
        )

        assert emb.text_embeddings.shape == (n_classes, 512)

    def test_projector_created(self, n_classes):
        """测试 Projector 已创建"""
        from pasco.models.hyperbolic.label_embedding import LorentzLabelEmbedding

        emb = LorentzLabelEmbedding(
            n_classes=n_classes,
            embed_dim=32,
            use_text_embeddings=True,
            text_embedding_type="sat",
        )

        assert hasattr(emb, 'projector')
        assert emb.projector is not None

    def test_invalid_embedding_type_raises(self, n_classes):
        """测试无效嵌入类型抛出异常"""
        from pasco.models.hyperbolic.label_embedding import LorentzLabelEmbedding

        with pytest.raises(ValueError, match="Unknown embedding type"):
            LorentzLabelEmbedding(
                n_classes=n_classes,
                use_text_embeddings=True,
                text_embedding_type="invalid",
            )

    def test_label_id_reordering(self, n_classes):
        """测试 label_id 重排序正确"""
        from pasco.models.hyperbolic.label_embedding import LorentzLabelEmbedding
        import torch

        emb = LorentzLabelEmbedding(
            n_classes=n_classes,
            embed_dim=32,
            use_text_embeddings=True,
            text_embedding_type="sat",
        )

        # 验证 text_embeddings[label_id] 对应正确的嵌入
        # 通过检查非零行来验证
        non_zero_rows = (emb.text_embeddings.abs().sum(dim=1) > 0).sum()
        assert non_zero_rows > 0, "Should have non-zero embeddings"
```

#### Step 2.4: 测试前向传播 (RED)

```python
class TestForwardWithTextEmbeddings:
    """测试使用文本嵌入的前向传播"""

    def test_forward_shape(self):
        """测试输出形状正确"""
        from pasco.models.hyperbolic.label_embedding import LorentzLabelEmbedding

        emb = LorentzLabelEmbedding(
            n_classes=72,
            embed_dim=32,
            use_text_embeddings=True,
            text_embedding_type="sat",
        )

        out = emb()
        assert out.shape == (72, 32)

    def test_class_0_at_origin(self):
        """测试 class 0 (ignore_class) 在原点"""
        from pasco.models.hyperbolic.label_embedding import LorentzLabelEmbedding
        from pasco.models.hyperbolic.lorentz import hyperbolic_distance_to_origin

        emb = LorentzLabelEmbedding(
            n_classes=72,
            embed_dim=32,
            use_text_embeddings=True,
            text_embedding_type="sat",
        )

        out = emb()
        dist_0 = hyperbolic_distance_to_origin(out[0:1], curv=1.0)

        assert dist_0.item() < 0.01, f"Class 0 should be at origin, got distance {dist_0.item()}"

    def test_projector_trainable(self):
        """测试 Projector 参数可训练"""
        from pasco.models.hyperbolic.label_embedding import LorentzLabelEmbedding

        emb = LorentzLabelEmbedding(
            n_classes=72,
            embed_dim=32,
            use_text_embeddings=True,
            text_embedding_type="sat",
        )

        trainable_params = [p for p in emb.projector.parameters() if p.requires_grad]
        assert len(trainable_params) > 0, "Projector should have trainable parameters"

    def test_backward_pass(self):
        """测试反向传播正常工作"""
        from pasco.models.hyperbolic.label_embedding import LorentzLabelEmbedding

        emb = LorentzLabelEmbedding(
            n_classes=72,
            embed_dim=32,
            use_text_embeddings=True,
            text_embedding_type="sat",
        )

        out = emb()
        loss = out.sum()
        loss.backward()

        # 检查 projector 有梯度
        assert emb.projector.mlp[0].weight.grad is not None

    def test_text_embeddings_not_trainable(self):
        """测试 text_embeddings buffer 不可训练"""
        from pasco.models.hyperbolic.label_embedding import LorentzLabelEmbedding

        emb = LorentzLabelEmbedding(
            n_classes=72,
            embed_dim=32,
            use_text_embeddings=True,
            text_embedding_type="sat",
        )

        assert not emb.text_embeddings.requires_grad
```

---

### Phase 3: BodyNetHyperbolic 集成

**目标文件**: `pasco/models/body_net_hyperbolic.py`

#### Step 3.1: 测试参数传递 (RED)

```python
# tests/models/test_body_net_hyperbolic_text.py
import pytest
import torch

class TestBodyNetHyperbolicTextEmbeddings:
    """测试 BodyNetHyperbolic 文本嵌入集成"""

    def test_new_params_accepted(self):
        """测试新参数被接受"""
        from pasco.models.body_net_hyperbolic import BodyNetHyperbolic

        # 不应抛出异常
        model = BodyNetHyperbolic(
            n_classes=72,
            use_text_embeddings=True,
            text_embedding_type="sat",
            use_light_model=True,
        )

        assert model.use_text_embeddings == True
        assert model.text_embedding_type == "sat"

    def test_label_emb_uses_text_embeddings(self):
        """测试 label_emb 正确使用文本嵌入"""
        from pasco.models.body_net_hyperbolic import BodyNetHyperbolic

        model = BodyNetHyperbolic(
            n_classes=72,
            use_text_embeddings=True,
            text_embedding_type="sat",
            use_light_model=True,
        )

        assert model.label_emb.use_text_embeddings == True
        assert hasattr(model.label_emb, 'projector')

    def test_virtual_nodes_disabled_with_text_embeddings(self):
        """测试使用文本嵌入时虚拟节点被禁用"""
        from pasco.models.body_net_hyperbolic import BodyNetHyperbolic

        model = BodyNetHyperbolic(
            n_classes=72,
            use_text_embeddings=True,
            text_embedding_type="sat",
            include_virtual_nodes=True,  # 尝试启用
            use_light_model=True,
        )

        # 应该被自动禁用
        assert model.label_emb.include_virtual == False

    def test_forward_with_text_embeddings(self):
        """测试前向传播正常工作"""
        from pasco.models.body_net_hyperbolic import BodyNetHyperbolic

        model = BodyNetHyperbolic(
            n_classes=72,
            use_text_embeddings=True,
            text_embedding_type="sat",
            use_light_model=True,
        )

        x = torch.randn(1, 1, 32, 32, 32)
        logits, emb = model.forward_with_hyperbolic(x)

        assert logits.shape[1] == 72  # n_classes
        assert emb.shape[-1] == 32  # embed_dim
```

---

### Phase 4: CLI 参数

**目标文件**: `scripts/body/train_body.py`

#### Step 4.1: 测试 argparse (RED)

```python
# tests/scripts/test_train_body_args.py
import pytest
import sys

class TestTrainBodyTextEmbeddingArgs:
    """测试 train_body.py 命令行参数"""

    def test_use_text_embeddings_flag(self):
        """测试 --use_text_embeddings 标志"""
        from scripts.body.train_body import parse_args

        args = parse_args([
            "--dataset_root", "Dataset/voxel_data",
            "--use_hyperbolic",
            "--use_text_embeddings",
        ])

        assert args.use_text_embeddings == True

    def test_text_embedding_type_default(self):
        """测试 --text_embedding_type 默认值"""
        from scripts.body.train_body import parse_args

        args = parse_args([
            "--dataset_root", "Dataset/voxel_data",
            "--use_hyperbolic",
            "--use_text_embeddings",
        ])

        assert args.text_embedding_type == "sat"

    def test_text_embedding_type_choices(self):
        """测试 --text_embedding_type 选项"""
        from scripts.body.train_body import parse_args

        for choice in ["sat", "clip", "biomedclip"]:
            args = parse_args([
                "--dataset_root", "Dataset/voxel_data",
                "--use_hyperbolic",
                "--use_text_embeddings",
                "--text_embedding_type", choice,
            ])
            assert args.text_embedding_type == choice

    def test_invalid_text_embedding_type_rejected(self):
        """测试无效类型被拒绝"""
        from scripts.body.train_body import parse_args

        with pytest.raises(SystemExit):
            parse_args([
                "--dataset_root", "Dataset/voxel_data",
                "--use_text_embeddings",
                "--text_embedding_type", "invalid",
            ])
```

---

## 测试执行顺序

按依赖关系排序的执行顺序：

```
Phase 1: TextEmbeddingProjector (无依赖)
   ↓
Phase 2: LorentzLabelEmbedding (依赖 Phase 1)
   ↓
Phase 3: BodyNetHyperbolic (依赖 Phase 2)
   ↓
Phase 4: CLI 参数 (依赖 Phase 3)
   ↓
Phase 5: 集成测试 (全部依赖)
```

---

## 集成测试 (Phase 5)

所有单元测试通过后，运行集成测试：

```python
# tests/integration/test_text_embedding_integration.py
import pytest
import torch

class TestTextEmbeddingIntegration:
    """端到端集成测试"""

    @pytest.mark.parametrize("embedding_type", ["sat", "clip", "biomedclip"])
    def test_full_forward_pass(self, embedding_type):
        """测试完整前向传播"""
        from pasco.models.body_net_hyperbolic import BodyNetHyperbolic

        model = BodyNetHyperbolic(
            n_classes=72,
            use_text_embeddings=True,
            text_embedding_type=embedding_type,
            use_light_model=True,
        )

        x = torch.randn(2, 1, 32, 32, 32)
        logits, emb = model.forward_with_hyperbolic(x)

        assert logits.shape == (2, 72, 32, 32, 32)
        assert not torch.isnan(logits).any()
        assert not torch.isnan(emb).any()

    def test_training_step(self):
        """测试训练步骤 (forward + backward)"""
        from pasco.models.body_net_hyperbolic import BodyNetHyperbolic

        model = BodyNetHyperbolic(
            n_classes=72,
            use_text_embeddings=True,
            text_embedding_type="sat",
            use_light_model=True,
        )

        x = torch.randn(1, 1, 32, 32, 32)
        logits, emb = model.forward_with_hyperbolic(x)

        # 模拟损失
        loss = logits.mean() + emb.mean()
        loss.backward()

        # 验证 projector 有梯度
        assert model.label_emb.projector.mlp[0].weight.grad is not None
```

---

## 验证检查清单

每个 Phase 完成后，必须满足以下条件：

### Phase 1 完成条件
- [ ] `test_text_projector.py` 所有测试 PASSED
- [ ] 覆盖率 ≥80%
- [ ] `python -c "from pasco.models.hyperbolic.text_projector import TextEmbeddingProjector"` 成功

### Phase 2 完成条件
- [ ] `test_label_embedding_text.py` 所有测试 PASSED
- [ ] 覆盖率 ≥80%
- [ ] 原有 `label_embedding.py` 测试仍然 PASSED (向后兼容)

### Phase 3 完成条件
- [ ] `test_body_net_hyperbolic_text.py` 所有测试 PASSED
- [ ] 覆盖率 ≥80%
- [ ] 原有 `body_net_hyperbolic.py` 测试仍然 PASSED

### Phase 4 完成条件
- [ ] `test_train_body_args.py` 所有测试 PASSED
- [ ] `python scripts/body/train_body.py --help` 显示新参数

### Phase 5 完成条件
- [ ] 集成测试 PASSED
- [ ] 手动训练测试成功：
  ```bash
  python scripts/body/train_body.py \
      --dataset_root Dataset/voxel_data \
      --use_hyperbolic \
      --use_text_embeddings \
      --text_embedding_type sat \
      --max_epochs 1
  ```

---

## 命令速查

```bash
# 激活环境
conda activate pasco

# 运行单个测试文件
pytest tests/models/hyperbolic/test_text_projector.py -v

# 运行带覆盖率
pytest tests/models/hyperbolic/test_text_projector.py --cov=pasco.models.hyperbolic.text_projector --cov-report=term-missing

# 运行所有文本嵌入相关测试
pytest tests/ -k "text" -v

# 运行集成测试
pytest tests/integration/test_text_embedding_integration.py -v

# 检查全局覆盖率
pytest tests/ --cov=pasco --cov-report=html
open htmlcov/index.html
```

---

## 失败处理协议

当测试失败时：

1. **不要修改测试来让它通过**（除非测试本身有错误）
2. 分析失败原因
3. 修改实现代码
4. 重新运行测试
5. 如果仍然失败，重复步骤 2-4

**禁止行为**：
- 跳过失败的测试
- 注释掉失败的断言
- 在没有测试的情况下提交代码
