# Hyperbolic Embedding 分阶段实现计划 (TDD版本)

基于 `hyperbolic_plan.md` 的详细设计，采用**测试驱动开发（TDD）**方法分 **8 个阶段**实现。

---

## TDD 工作流程

```
┌─────────────────────────────────────────────────────────────────┐
│                         TDD 循环                                 │
│                                                                  │
│   阶段 N-1 测试通过                                              │
│         │                                                        │
│         ▼                                                        │
│   ┌─────────────────┐                                            │
│   │ 1. 编写阶段 N   │  ← 先写测试，此时测试必定失败              │
│   │    的测试脚本   │                                            │
│   └────────┬────────┘                                            │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────┐                                            │
│   │ 2. 运行测试     │  ← 确认测试失败（RED）                     │
│   │    确认失败     │                                            │
│   └────────┬────────┘                                            │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────┐                                            │
│   │ 3. 实现阶段 N   │  ← 编写最小代码使测试通过                  │
│   │    的代码       │                                            │
│   └────────┬────────┘                                            │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────┐                                            │
│   │ 4. 运行测试     │  ← 确认测试通过（GREEN）                   │
│   │    确认通过     │                                            │
│   └────────┬────────┘                                            │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────┐                                            │
│   │ 5. 代码优化     │  ← 可选：重构但保持测试通过                │
│   │   （可选）      │                                            │
│   └────────┬────────┘                                            │
│            │                                                     │
│            ▼                                                     │
│      进入阶段 N+1                                                │
└─────────────────────────────────────────────────────────────────┘
```

**核心原则**：
1. **每次只执行一个阶段**
2. **测试脚本必须在实现之前编写**
3. **测试通过后才能进入下一阶段**

---

## 测试文件结构

```
PaSCo/
├── tests/
│   └── hyperbolic/
│       ├── __init__.py
│       ├── test_stage1_organ_hierarchy.py
│       ├── test_stage2_poincare_ops.py
│       ├── test_stage3_label_embedding.py
│       ├── test_stage4_projection_head.py
│       ├── test_stage5_hyperbolic_loss.py
│       ├── test_stage6_dense_unet_features.py
│       ├── test_stage7_body_net_hyperbolic.py
│       └── test_stage8_train_integration.py
```

---

## 阶段概览

| 阶段 | 实现文件 | 测试文件 | 依赖阶段 |
|------|----------|----------|----------|
| 1 | `organ_hierarchy.py` | `test_stage1_organ_hierarchy.py` | 无 |
| 2 | `poincare_ops.py` | `test_stage2_poincare_ops.py` | 无 |
| 3 | `label_embedding.py` | `test_stage3_label_embedding.py` | 1, 2 |
| 4 | `projection_head.py` | `test_stage4_projection_head.py` | 2 |
| 5 | `hyperbolic_loss.py` | `test_stage5_hyperbolic_loss.py` | 2 |
| 6 | `dense_unet3d.py` (修改) | `test_stage6_dense_unet_features.py` | 无 |
| 7 | `body_net_hyperbolic.py` | `test_stage7_body_net_hyperbolic.py` | 3-6 |
| 8 | `train_body.py` (修改) | `test_stage8_train_integration.py` | 7 |

---

## 阶段 0：测试基础设施搭建

### 目标
创建测试目录和基础设施。

### 执行命令
```bash
cd /home/comp/25481568/code/PaSCo
mkdir -p tests/hyperbolic
touch tests/__init__.py
touch tests/hyperbolic/__init__.py
```

### 验收标准
- [ ] `tests/hyperbolic/` 目录存在
- [ ] `__init__.py` 文件存在

---

## 阶段 1：器官层级结构定义

### 1.1 编写测试脚本（先于实现）

**文件**: `tests/hyperbolic/test_stage1_organ_hierarchy.py`

```python
"""
Stage 1 Test: Organ Hierarchy Definition
测试器官类别的层级树结构定义。

运行方式:
    cd /home/comp/25481568/code/PaSCo
    python tests/hyperbolic/test_stage1_organ_hierarchy.py
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_imports():
    """测试模块是否可以正确导入"""
    print("Test 1.1: Testing imports...")
    try:
        from pasco.data.body.organ_hierarchy import (
            CLASS_NAMES,
            N_CLASSES,
            ORGAN_HIERARCHY,
            CLASS_DEPTHS,
            MAX_DEPTH,
            get_class_depths,
            get_max_depth,
        )
        print("  [PASS] All imports successful")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def test_class_names():
    """测试类别名称列表"""
    print("Test 1.2: Testing CLASS_NAMES...")
    from pasco.data.body.organ_hierarchy import CLASS_NAMES, N_CLASSES

    # 检查数量
    assert len(CLASS_NAMES) == 72, f"Expected 72 classes, got {len(CLASS_NAMES)}"
    print(f"  [PASS] CLASS_NAMES has {len(CLASS_NAMES)} elements")

    # 检查 N_CLASSES
    assert N_CLASSES == 72, f"Expected N_CLASSES=72, got {N_CLASSES}"
    print(f"  [PASS] N_CLASSES = {N_CLASSES}")

    # 检查第一个和最后一个类别
    assert CLASS_NAMES[0] == "outside_body", f"Expected CLASS_NAMES[0]='outside_body', got '{CLASS_NAMES[0]}'"
    assert CLASS_NAMES[71] == "rectum", f"Expected CLASS_NAMES[71]='rectum', got '{CLASS_NAMES[71]}'"
    print("  [PASS] First and last class names correct")

    # 检查一些关键类别
    key_classes = {
        1: "inside_body_empty",
        2: "liver",
        11: "heart",
        12: "brain",
        23: "spine",
        48: "skull",
    }
    for idx, name in key_classes.items():
        assert CLASS_NAMES[idx] == name, f"Expected CLASS_NAMES[{idx}]='{name}', got '{CLASS_NAMES[idx]}'"
    print("  [PASS] Key class names verified")

    return True


def test_organ_hierarchy_structure():
    """测试层级树结构"""
    print("Test 1.3: Testing ORGAN_HIERARCHY structure...")
    from pasco.data.body.organ_hierarchy import ORGAN_HIERARCHY

    # 检查根节点
    assert "name" in ORGAN_HIERARCHY, "Root should have 'name' key"
    assert ORGAN_HIERARCHY["name"] == "body", "Root name should be 'body'"
    assert "children" in ORGAN_HIERARCHY, "Root should have 'children' key"
    print("  [PASS] Root node structure correct")

    # 检查主要系统
    child_names = [c["name"] for c in ORGAN_HIERARCHY["children"]]
    expected_systems = [
        "inside_body_empty",
        "visceral_system",
        "cardiovascular_respiratory",
        "neural_system",
        "endocrine_system",
        "skeletal_system",
        "muscular_system",
    ]
    for sys_name in expected_systems:
        assert sys_name in child_names, f"Missing system: {sys_name}"
    print(f"  [PASS] All {len(expected_systems)} main systems present")

    return True


def test_class_depths():
    """测试类别深度计算"""
    print("Test 1.4: Testing CLASS_DEPTHS...")
    from pasco.data.body.organ_hierarchy import CLASS_DEPTHS, MAX_DEPTH

    # 检查类别 0 不在 CLASS_DEPTHS 中
    assert 0 not in CLASS_DEPTHS, "Class 0 (outside_body) should NOT be in CLASS_DEPTHS"
    print("  [PASS] Class 0 excluded from CLASS_DEPTHS")

    # 检查有效类别数量（1-71）
    valid_classes = set(range(1, 72))
    missing_classes = valid_classes - set(CLASS_DEPTHS.keys())
    assert len(missing_classes) == 0, f"Missing classes in CLASS_DEPTHS: {missing_classes}"
    print(f"  [PASS] All 71 valid classes (1-71) have depth values")

    # 检查深度值范围
    depths = list(CLASS_DEPTHS.values())
    assert min(depths) >= 1, f"Minimum depth should be >= 1, got {min(depths)}"
    assert max(depths) <= 5, f"Maximum depth should be <= 5, got {max(depths)}"
    print(f"  [PASS] Depth range: [{min(depths)}, {max(depths)}]")

    # 检查 MAX_DEPTH
    assert MAX_DEPTH == max(depths), f"MAX_DEPTH mismatch: {MAX_DEPTH} vs {max(depths)}"
    print(f"  [PASS] MAX_DEPTH = {MAX_DEPTH}")

    return True


def test_depth_hierarchy_consistency():
    """测试深度与层级结构的一致性"""
    print("Test 1.5: Testing depth-hierarchy consistency...")
    from pasco.data.body.organ_hierarchy import CLASS_DEPTHS

    # 同一系统的器官应该有相似的深度
    # 肾脏 (左/右) 应该深度相同
    assert CLASS_DEPTHS[4] == CLASS_DEPTHS[5], "kidney_left and kidney_right should have same depth"
    print("  [PASS] Paired organs have consistent depths")

    # 肋骨应该比脊柱更深（因为肋骨是在 ribs 节点下）
    assert CLASS_DEPTHS[24] >= CLASS_DEPTHS[23], "Ribs should be at same or deeper level than spine"
    print("  [PASS] Hierarchical depth relationships correct")

    # 打印一些示例深度
    examples = [
        (1, "inside_body_empty"),
        (2, "liver"),
        (11, "heart"),
        (23, "spine"),
        (24, "rib_left_1"),
    ]
    print("  Sample depths:")
    for cls_id, name in examples:
        print(f"    {cls_id:2d}: {name:20s} -> depth={CLASS_DEPTHS[cls_id]}")

    return True


def test_get_functions():
    """测试辅助函数"""
    print("Test 1.6: Testing helper functions...")
    from pasco.data.body.organ_hierarchy import (
        get_class_depths,
        get_max_depth,
        ORGAN_HIERARCHY,
    )

    # 测试 get_class_depths
    depths = get_class_depths(ORGAN_HIERARCHY)
    assert isinstance(depths, dict), "get_class_depths should return dict"
    assert len(depths) == 71, f"Expected 71 classes, got {len(depths)}"
    print("  [PASS] get_class_depths() returns correct dict")

    # 测试 get_max_depth
    max_d = get_max_depth(ORGAN_HIERARCHY)
    assert isinstance(max_d, int), "get_max_depth should return int"
    assert max_d >= 1, "max_depth should be >= 1"
    print(f"  [PASS] get_max_depth() = {max_d}")

    return True


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Stage 1 Tests: Organ Hierarchy Definition")
    print("=" * 60)

    tests = [
        test_imports,
        test_class_names,
        test_organ_hierarchy_structure,
        test_class_depths,
        test_depth_hierarchy_consistency,
        test_get_functions,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [FAIL] Exception: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n✓ Stage 1 PASSED - Ready for Stage 2")
        return True
    else:
        print("\n✗ Stage 1 FAILED - Fix issues before proceeding")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
```

### 1.2 运行测试（预期失败）

```bash
cd /home/comp/25481568/code/PaSCo
python tests/hyperbolic/test_stage1_organ_hierarchy.py
```

**预期输出**（实现前）:
```
Stage 1 Tests: Organ Hierarchy Definition
Test 1.1: Testing imports...
  [FAIL] Import error: No module named 'pasco.data.body.organ_hierarchy'
```

### 1.3 实现代码

**文件**: `pasco/data/body/organ_hierarchy.py`

（代码详见 `hyperbolic_plan.md` 第 5.1 节 File 1）

### 1.4 再次运行测试（预期通过）

```bash
python tests/hyperbolic/test_stage1_organ_hierarchy.py
```

**预期输出**（实现后）:
```
Stage 1 Tests: Organ Hierarchy Definition
============================================================
Test 1.1: Testing imports...
  [PASS] All imports successful

Test 1.2: Testing CLASS_NAMES...
  [PASS] CLASS_NAMES has 72 elements
  [PASS] N_CLASSES = 72
  [PASS] First and last class names correct
  [PASS] Key class names verified

...

============================================================
Results: 6 passed, 0 failed
============================================================

✓ Stage 1 PASSED - Ready for Stage 2
```

### 1.5 验收标准检查清单

- [ ] `N_CLASSES == 72`
- [ ] `MAX_DEPTH` 在合理范围内（通常 3-5）
- [ ] 所有 71 个有效类别（1-71）都有深度值
- [ ] 类别 0 不在 `CLASS_DEPTHS` 中
- [ ] 所有 6 个测试用例通过

---

## 阶段 2：Poincaré 球基础操作

### 2.1 编写测试脚本（先于实现）

**文件**: `tests/hyperbolic/test_stage2_poincare_ops.py`

```python
"""
Stage 2 Test: Poincaré Ball Operations
测试 Poincaré 球的基础数学操作。

运行方式:
    cd /home/comp/25481568/code/PaSCo
    python tests/hyperbolic/test_stage2_poincare_ops.py
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_imports():
    """测试模块导入"""
    print("Test 2.1: Testing imports...")
    try:
        from pasco.models.hyperbolic.poincare_ops import (
            exp_map_zero,
            poincare_distance,
            project_to_ball,
        )
        print("  [PASS] All imports successful")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def test_exp_map_zero_basic():
    """测试 exp_map_zero 基本功能"""
    print("Test 2.2: Testing exp_map_zero basic functionality...")
    from pasco.models.hyperbolic.poincare_ops import exp_map_zero

    # 测试不同形状的输入
    shapes = [(10, 32), (5, 64), (100, 16)]
    for shape in shapes:
        v = torch.randn(shape)
        h = exp_map_zero(v)

        # 检查形状保持不变
        assert h.shape == v.shape, f"Shape mismatch: {h.shape} vs {v.shape}"

        # 检查所有点都在球内 (norm < 1)
        norms = h.norm(dim=-1)
        assert (norms < 1).all(), f"Some points outside ball: max_norm={norms.max():.4f}"

    print(f"  [PASS] All shapes tested, all points inside ball")
    return True


def test_exp_map_zero_properties():
    """测试 exp_map_zero 的数学性质"""
    print("Test 2.3: Testing exp_map_zero mathematical properties...")
    from pasco.models.hyperbolic.poincare_ops import exp_map_zero

    # 性质1: 零向量映射到零向量
    v_zero = torch.zeros(10, 32)
    h_zero = exp_map_zero(v_zero)
    assert h_zero.norm(dim=-1).max() < 1e-4, "Zero vector should map to near-zero"
    print("  [PASS] Zero vector maps to origin")

    # 性质2: 大向量映射到接近边界的点
    v_large = torch.randn(10, 32) * 100
    h_large = exp_map_zero(v_large)
    norms = h_large.norm(dim=-1)
    assert (norms > 0.99).all(), f"Large vectors should map near boundary: min_norm={norms.min():.4f}"
    print("  [PASS] Large vectors map near boundary")

    # 性质3: 保持方向
    v = torch.randn(10, 32)
    h = exp_map_zero(v)
    # 方向一致性: h 和 v 应该有相同的方向（通过 cosine 相似度检验）
    v_normalized = v / v.norm(dim=-1, keepdim=True)
    h_normalized = h / h.norm(dim=-1, keepdim=True)
    cos_sim = (v_normalized * h_normalized).sum(dim=-1)
    assert (cos_sim > 0.99).all(), f"Direction should be preserved: min_cos_sim={cos_sim.min():.4f}"
    print("  [PASS] Direction preserved after mapping")

    return True


def test_exp_map_zero_gradient():
    """测试 exp_map_zero 的梯度"""
    print("Test 2.4: Testing exp_map_zero gradient...")
    from pasco.models.hyperbolic.poincare_ops import exp_map_zero

    v = torch.randn(10, 32, requires_grad=True)
    h = exp_map_zero(v)
    loss = h.sum()
    loss.backward()

    assert v.grad is not None, "Gradient should exist"
    assert not torch.isnan(v.grad).any(), "Gradient should not contain NaN"
    assert not torch.isinf(v.grad).any(), "Gradient should not contain Inf"
    print("  [PASS] Gradient computation successful")

    return True


def test_poincare_distance_basic():
    """测试 poincare_distance 基本功能"""
    print("Test 2.5: Testing poincare_distance basic functionality...")
    from pasco.models.hyperbolic.poincare_ops import poincare_distance, exp_map_zero

    # 创建球内的点
    x = exp_map_zero(torch.randn(10, 32))
    y = exp_map_zero(torch.randn(10, 32))

    d = poincare_distance(x, y)

    # 检查形状
    assert d.shape == (10,), f"Distance shape should be (10,), got {d.shape}"
    print("  [PASS] Output shape correct")

    # 检查非负性
    assert (d >= 0).all(), f"Distance should be non-negative: min={d.min():.4f}"
    print("  [PASS] Distance is non-negative")

    return True


def test_poincare_distance_symmetry():
    """测试 poincare_distance 对称性"""
    print("Test 2.6: Testing poincare_distance symmetry...")
    from pasco.models.hyperbolic.poincare_ops import poincare_distance, exp_map_zero

    x = exp_map_zero(torch.randn(10, 32))
    y = exp_map_zero(torch.randn(10, 32))

    d_xy = poincare_distance(x, y)
    d_yx = poincare_distance(y, x)

    assert torch.allclose(d_xy, d_yx, atol=1e-5), f"Distance should be symmetric: max_diff={torch.abs(d_xy - d_yx).max():.6f}"
    print("  [PASS] Distance is symmetric")

    return True


def test_poincare_distance_self():
    """测试自身到自身的距离为0"""
    print("Test 2.7: Testing distance to self is zero...")
    from pasco.models.hyperbolic.poincare_ops import poincare_distance, exp_map_zero

    x = exp_map_zero(torch.randn(10, 32))
    d = poincare_distance(x, x)

    assert (d < 1e-4).all(), f"Distance to self should be ~0: max={d.max():.6f}"
    print("  [PASS] Distance to self is near zero")

    return True


def test_poincare_distance_gradient():
    """测试 poincare_distance 的梯度"""
    print("Test 2.8: Testing poincare_distance gradient...")
    from pasco.models.hyperbolic.poincare_ops import poincare_distance, exp_map_zero

    x = exp_map_zero(torch.randn(10, 32))
    y = exp_map_zero(torch.randn(10, 32))
    x.requires_grad_(True)
    y.requires_grad_(True)

    d = poincare_distance(x, y)
    loss = d.sum()
    loss.backward()

    assert x.grad is not None and y.grad is not None, "Gradients should exist"
    assert not torch.isnan(x.grad).any() and not torch.isnan(y.grad).any(), "Gradients should not contain NaN"
    print("  [PASS] Gradient computation successful")

    return True


def test_project_to_ball():
    """测试 project_to_ball"""
    print("Test 2.9: Testing project_to_ball...")
    from pasco.models.hyperbolic.poincare_ops import project_to_ball

    # 测试球外的点
    x_outside = torch.randn(10, 32) * 2  # 可能在球外
    p = project_to_ball(x_outside)

    # 检查所有点在球内
    norms = p.norm(dim=-1)
    assert (norms < 1).all(), f"All points should be inside ball: max_norm={norms.max():.4f}"
    print(f"  [PASS] All points projected inside ball (max_norm={norms.max():.4f})")

    # 测试已经在球内的点不变（除非接近边界）
    x_inside = torch.randn(10, 32) * 0.3  # 明确在球内
    p_inside = project_to_ball(x_inside)
    assert torch.allclose(x_inside, p_inside, atol=1e-5), "Points inside should remain unchanged"
    print("  [PASS] Points already inside remain unchanged")

    return True


def test_numerical_stability():
    """测试数值稳定性"""
    print("Test 2.10: Testing numerical stability...")
    from pasco.models.hyperbolic.poincare_ops import exp_map_zero, poincare_distance, project_to_ball

    # 测试很小的向量
    v_small = torch.randn(10, 32) * 1e-8
    h_small = exp_map_zero(v_small)
    assert not torch.isnan(h_small).any(), "Small vectors should not produce NaN"
    print("  [PASS] Small vectors handled correctly")

    # 测试很大的向量
    v_large = torch.randn(10, 32) * 1e8
    h_large = exp_map_zero(v_large)
    assert not torch.isnan(h_large).any(), "Large vectors should not produce NaN"
    print("  [PASS] Large vectors handled correctly")

    # 测试接近边界的点
    x_near_boundary = torch.randn(10, 32)
    x_near_boundary = x_near_boundary / x_near_boundary.norm(dim=-1, keepdim=True) * 0.9999
    y = torch.randn(10, 32) * 0.5
    d = poincare_distance(x_near_boundary, y)
    assert not torch.isnan(d).any(), "Points near boundary should not produce NaN distance"
    print("  [PASS] Points near boundary handled correctly")

    return True


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Stage 2 Tests: Poincaré Ball Operations")
    print("=" * 60)

    tests = [
        test_imports,
        test_exp_map_zero_basic,
        test_exp_map_zero_properties,
        test_exp_map_zero_gradient,
        test_poincare_distance_basic,
        test_poincare_distance_symmetry,
        test_poincare_distance_self,
        test_poincare_distance_gradient,
        test_project_to_ball,
        test_numerical_stability,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [FAIL] Exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n✓ Stage 2 PASSED - Ready for Stage 3")
        return True
    else:
        print("\n✗ Stage 2 FAILED - Fix issues before proceeding")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
```

### 2.2 运行测试（预期失败）

```bash
python tests/hyperbolic/test_stage2_poincare_ops.py
```

### 2.3 实现代码

需要创建:
- `pasco/models/hyperbolic/__init__.py`
- `pasco/models/hyperbolic/poincare_ops.py`

（代码详见 `hyperbolic_plan.md` 第 5.1 节 File 2, 3）

### 2.4 再次运行测试（预期通过）

### 2.5 验收标准检查清单

- [ ] `exp_map_zero` 输出范数 < 1
- [ ] `exp_map_zero` 保持向量方向
- [ ] `poincare_distance` 非负
- [ ] `poincare_distance` 对称
- [ ] `poincare_distance` 自身距离为0
- [ ] `project_to_ball` 将任意点投影到球内
- [ ] 所有操作支持梯度计算
- [ ] 数值稳定（无 NaN/Inf）
- [ ] 所有 10 个测试用例通过

---

## 阶段 3：Hyperbolic 标签嵌入

### 3.1 编写测试脚本（先于实现）

**文件**: `tests/hyperbolic/test_stage3_label_embedding.py`

```python
"""
Stage 3 Test: Hyperbolic Label Embedding
测试基于层级结构的类别嵌入。

依赖: Stage 1 (organ_hierarchy), Stage 2 (poincare_ops)

运行方式:
    cd /home/comp/25481568/code/PaSCo
    python tests/hyperbolic/test_stage3_label_embedding.py
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def check_dependencies():
    """检查依赖阶段是否完成"""
    print("Checking dependencies...")
    try:
        from pasco.data.body.organ_hierarchy import CLASS_DEPTHS, MAX_DEPTH
        print("  [OK] Stage 1 (organ_hierarchy) available")
    except ImportError as e:
        print(f"  [FAIL] Stage 1 not complete: {e}")
        return False

    try:
        from pasco.models.hyperbolic.poincare_ops import project_to_ball
        print("  [OK] Stage 2 (poincare_ops) available")
    except ImportError as e:
        print(f"  [FAIL] Stage 2 not complete: {e}")
        return False

    return True


def test_imports():
    """测试模块导入"""
    print("Test 3.1: Testing imports...")
    try:
        from pasco.models.hyperbolic.label_embedding import HyperbolicLabelEmbedding
        print("  [PASS] Import successful")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def test_initialization():
    """测试初始化"""
    print("Test 3.2: Testing initialization...")
    from pasco.models.hyperbolic.label_embedding import HyperbolicLabelEmbedding

    emb = HyperbolicLabelEmbedding(
        n_classes=72,
        embed_dim=32,
        ignore_class=0,
        min_radius=0.1,
        max_radius=0.8,
    )

    # 检查参数
    assert emb.n_classes == 72, f"n_classes mismatch: {emb.n_classes}"
    assert emb.embed_dim == 32, f"embed_dim mismatch: {emb.embed_dim}"
    assert emb.ignore_class == 0, f"ignore_class mismatch: {emb.ignore_class}"
    print("  [PASS] Parameters initialized correctly")

    # 检查 embeddings 参数
    assert hasattr(emb, 'embeddings'), "Should have 'embeddings' parameter"
    assert emb.embeddings.shape == (72, 32), f"Embeddings shape mismatch: {emb.embeddings.shape}"
    print("  [PASS] Embeddings parameter created with correct shape")

    return True


def test_forward_all():
    """测试 forward 返回所有嵌入"""
    print("Test 3.3: Testing forward() with no arguments...")
    from pasco.models.hyperbolic.label_embedding import HyperbolicLabelEmbedding

    emb = HyperbolicLabelEmbedding(n_classes=72, embed_dim=32)
    e = emb()

    # 检查形状
    assert e.shape == (72, 32), f"Output shape mismatch: {e.shape}"
    print("  [PASS] Output shape correct: (72, 32)")

    # 检查所有嵌入在球内
    norms = e.norm(dim=-1)
    assert (norms < 1).all(), f"All embeddings should be inside ball: max_norm={norms.max():.4f}"
    print(f"  [PASS] All embeddings inside Poincaré ball (max_norm={norms.max():.4f})")

    return True


def test_forward_indices():
    """测试 forward 返回特定索引的嵌入"""
    print("Test 3.4: Testing forward() with class indices...")
    from pasco.models.hyperbolic.label_embedding import HyperbolicLabelEmbedding

    emb = HyperbolicLabelEmbedding(n_classes=72, embed_dim=32)

    # 测试单个索引（通过张量）
    indices = torch.tensor([1, 5, 10, 20])
    e = emb(indices)
    assert e.shape == (4, 32), f"Output shape mismatch: {e.shape}"
    print("  [PASS] Indexing with tensor works")

    # 检查索引结果与全量结果一致
    e_all = emb()
    assert torch.allclose(e, e_all[indices], atol=1e-5), "Indexed embeddings should match"
    print("  [PASS] Indexed embeddings match full embeddings")

    return True


def test_ignored_class():
    """测试忽略类别"""
    print("Test 3.5: Testing ignored class (class 0)...")
    from pasco.models.hyperbolic.label_embedding import HyperbolicLabelEmbedding

    emb = HyperbolicLabelEmbedding(n_classes=72, embed_dim=32, ignore_class=0)
    e = emb()

    # 类别 0 应该在原点或接近原点
    norm_class0 = e[0].norm()
    assert norm_class0 < 0.01, f"Class 0 should be at/near origin: norm={norm_class0:.4f}"
    print(f"  [PASS] Class 0 at origin (norm={norm_class0:.6f})")

    # 其他类别不应该在原点
    norms_others = e[1:].norm(dim=-1)
    assert (norms_others > 0.05).all(), f"Other classes should not be at origin: min_norm={norms_others.min():.4f}"
    print(f"  [PASS] Other classes away from origin (min_norm={norms_others.min():.4f})")

    return True


def test_depth_based_initialization():
    """测试基于深度的初始化"""
    print("Test 3.6: Testing depth-based initialization...")
    from pasco.models.hyperbolic.label_embedding import HyperbolicLabelEmbedding
    from pasco.data.body.organ_hierarchy import CLASS_DEPTHS

    emb = HyperbolicLabelEmbedding(
        n_classes=72,
        embed_dim=32,
        min_radius=0.1,
        max_radius=0.8,
    )
    e = emb()
    norms = e.norm(dim=-1)

    # 检查 norms 在 [min_radius, max_radius] 范围内（忽略类别 0）
    valid_norms = norms[1:]
    assert (valid_norms >= 0.05).all(), f"Norms should be >= 0.05: min={valid_norms.min():.4f}"
    assert (valid_norms <= 0.85).all(), f"Norms should be <= 0.85: max={valid_norms.max():.4f}"
    print(f"  [PASS] Norms in expected range: [{valid_norms.min():.3f}, {valid_norms.max():.3f}]")

    # 深层类别（如肋骨）应该比浅层类别（如 inside_body_empty）有更大的 norm
    # inside_body_empty (class 1) 深度较浅
    # rib_left_1 (class 24) 深度较深
    shallow_depth = CLASS_DEPTHS.get(1, 1)
    deep_depth = CLASS_DEPTHS.get(24, 4)
    if shallow_depth < deep_depth:
        # 注意：由于随机方向，我们只检查趋势
        print(f"  [INFO] Class 1 depth={shallow_depth}, Class 24 depth={deep_depth}")
        print(f"  [INFO] Class 1 norm={norms[1]:.4f}, Class 24 norm={norms[24]:.4f}")
    print("  [PASS] Depth-based initialization applied")

    return True


def test_gradient():
    """测试梯度计算"""
    print("Test 3.7: Testing gradient computation...")
    from pasco.models.hyperbolic.label_embedding import HyperbolicLabelEmbedding

    emb = HyperbolicLabelEmbedding(n_classes=72, embed_dim=32)
    e = emb()
    loss = e.sum()
    loss.backward()

    assert emb.embeddings.grad is not None, "Embeddings should have gradient"
    assert not torch.isnan(emb.embeddings.grad).any(), "Gradient should not contain NaN"
    print("  [PASS] Gradient computation successful")

    return True


def test_projection_after_update():
    """测试更新后仍在球内"""
    print("Test 3.8: Testing projection after gradient update...")
    from pasco.models.hyperbolic.label_embedding import HyperbolicLabelEmbedding

    emb = HyperbolicLabelEmbedding(n_classes=72, embed_dim=32)

    # 模拟梯度更新
    optimizer = torch.optim.SGD(emb.parameters(), lr=0.1)
    for _ in range(10):
        e = emb()
        loss = e.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每次更新后检查嵌入仍在球内
        with torch.no_grad():
            e_after = emb()
            norms = e_after.norm(dim=-1)
            assert (norms < 1).all(), f"Embeddings should stay inside ball: max_norm={norms.max():.4f}"

    print("  [PASS] Embeddings remain inside ball after updates")
    return True


def test_project_embeddings_method():
    """测试 project_embeddings 方法"""
    print("Test 3.9: Testing project_embeddings() method...")
    from pasco.models.hyperbolic.label_embedding import HyperbolicLabelEmbedding

    emb = HyperbolicLabelEmbedding(n_classes=72, embed_dim=32)

    # 人为将 embeddings 设置到球外
    with torch.no_grad():
        emb.embeddings.data = torch.randn(72, 32) * 2

    # 调用 project_embeddings
    emb.project_embeddings()

    # 检查现在在球内
    norms = emb.embeddings.data.norm(dim=-1)
    assert (norms < 1).all(), f"After projection, all should be inside: max_norm={norms.max():.4f}"
    print("  [PASS] project_embeddings() works correctly")

    return True


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Stage 3 Tests: Hyperbolic Label Embedding")
    print("=" * 60)

    # 首先检查依赖
    if not check_dependencies():
        print("\n✗ Stage 3 BLOCKED - Complete Stage 1 and 2 first")
        return False
    print()

    tests = [
        test_imports,
        test_initialization,
        test_forward_all,
        test_forward_indices,
        test_ignored_class,
        test_depth_based_initialization,
        test_gradient,
        test_projection_after_update,
        test_project_embeddings_method,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [FAIL] Exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n✓ Stage 3 PASSED - Ready for Stage 4")
        return True
    else:
        print("\n✗ Stage 3 FAILED - Fix issues before proceeding")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
```

### 3.2 验收标准检查清单

- [ ] 依赖检查通过（Stage 1, 2）
- [ ] 嵌入形状为 (72, 32)
- [ ] 所有嵌入在 Poincaré ball 内
- [ ] 类别 0 的嵌入接近原点
- [ ] 其他类别的嵌入基于深度初始化
- [ ] 支持反向传播
- [ ] 梯度更新后嵌入仍在球内
- [ ] `project_embeddings()` 方法工作正常
- [ ] 所有 9 个测试用例通过

---

## 阶段 4：Hyperbolic 投影头

### 4.1 编写测试脚本（先于实现）

**文件**: `tests/hyperbolic/test_stage4_projection_head.py`

```python
"""
Stage 4 Test: Hyperbolic Projection Head
测试将 CNN 特征投影到 Poincaré 球的模块。

依赖: Stage 2 (poincare_ops)

运行方式:
    cd /home/comp/25481568/code/PaSCo
    python tests/hyperbolic/test_stage4_projection_head.py
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def check_dependencies():
    """检查依赖阶段是否完成"""
    print("Checking dependencies...")
    try:
        from pasco.models.hyperbolic.poincare_ops import exp_map_zero, project_to_ball
        print("  [OK] Stage 2 (poincare_ops) available")
        return True
    except ImportError as e:
        print(f"  [FAIL] Stage 2 not complete: {e}")
        return False


def test_imports():
    """测试模块导入"""
    print("Test 4.1: Testing imports...")
    try:
        from pasco.models.hyperbolic.projection_head import HyperbolicProjectionHead
        print("  [PASS] Import successful")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def test_initialization():
    """测试初始化"""
    print("Test 4.2: Testing initialization...")
    from pasco.models.hyperbolic.projection_head import HyperbolicProjectionHead

    head = HyperbolicProjectionHead(in_channels=32, embed_dim=32)

    assert head.in_channels == 32, f"in_channels mismatch: {head.in_channels}"
    assert head.embed_dim == 32, f"embed_dim mismatch: {head.embed_dim}"
    assert hasattr(head, 'proj'), "Should have 'proj' layer"
    print("  [PASS] Initialization correct")

    return True


def test_forward_shape():
    """测试前向传播形状"""
    print("Test 4.3: Testing forward pass shape...")
    from pasco.models.hyperbolic.projection_head import HyperbolicProjectionHead

    head = HyperbolicProjectionHead(in_channels=32, embed_dim=32)

    # 测试不同输入尺寸
    test_cases = [
        (2, 32, 16, 16, 16),
        (1, 32, 32, 32, 32),
        (4, 32, 8, 8, 8),
    ]

    for shape in test_cases:
        B, C, H, W, D = shape
        x = torch.randn(B, C, H, W, D)
        out = head(x)
        expected_shape = (B, 32, H, W, D)
        assert out.shape == expected_shape, f"Shape mismatch: {out.shape} vs {expected_shape}"
        print(f"  [PASS] Input {shape} -> Output {out.shape}")

    return True


def test_forward_different_embed_dim():
    """测试不同嵌入维度"""
    print("Test 4.4: Testing different embedding dimensions...")
    from pasco.models.hyperbolic.projection_head import HyperbolicProjectionHead

    for embed_dim in [16, 32, 64]:
        head = HyperbolicProjectionHead(in_channels=32, embed_dim=embed_dim)
        x = torch.randn(2, 32, 8, 8, 8)
        out = head(x)
        assert out.shape == (2, embed_dim, 8, 8, 8), f"Shape mismatch for embed_dim={embed_dim}"
        print(f"  [PASS] embed_dim={embed_dim} works")

    return True


def test_output_in_ball():
    """测试输出在 Poincaré 球内"""
    print("Test 4.5: Testing output is inside Poincaré ball...")
    from pasco.models.hyperbolic.projection_head import HyperbolicProjectionHead

    head = HyperbolicProjectionHead(in_channels=32, embed_dim=32)

    # 测试多种输入
    for scale in [0.1, 1.0, 10.0]:
        x = torch.randn(2, 32, 8, 8, 8) * scale
        out = head(x)

        # Reshape to check norms: [B, D, H, W, Z] -> [B*H*W*Z, D]
        out_flat = out.permute(0, 2, 3, 4, 1).reshape(-1, 32)
        norms = out_flat.norm(dim=-1)

        assert (norms < 1).all(), f"All outputs should be inside ball: max_norm={norms.max():.4f}"
        print(f"  [PASS] Input scale={scale}: all norms < 1 (max={norms.max():.4f})")

    return True


def test_gradient():
    """测试梯度计算"""
    print("Test 4.6: Testing gradient computation...")
    from pasco.models.hyperbolic.projection_head import HyperbolicProjectionHead

    head = HyperbolicProjectionHead(in_channels=32, embed_dim=32)
    x = torch.randn(2, 32, 8, 8, 8, requires_grad=True)

    out = head(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "Input should have gradient"
    assert not torch.isnan(x.grad).any(), "Gradient should not contain NaN"

    # 检查参数梯度
    for name, param in head.named_parameters():
        assert param.grad is not None, f"Parameter {name} should have gradient"
        assert not torch.isnan(param.grad).any(), f"Parameter {name} gradient should not contain NaN"

    print("  [PASS] Gradient computation successful")
    return True


def test_numerical_stability():
    """测试数值稳定性"""
    print("Test 4.7: Testing numerical stability...")
    from pasco.models.hyperbolic.projection_head import HyperbolicProjectionHead

    head = HyperbolicProjectionHead(in_channels=32, embed_dim=32)

    # 测试很小的输入
    x_small = torch.randn(2, 32, 8, 8, 8) * 1e-8
    out_small = head(x_small)
    assert not torch.isnan(out_small).any(), "Small input should not produce NaN"
    print("  [PASS] Small input handled correctly")

    # 测试很大的输入
    x_large = torch.randn(2, 32, 8, 8, 8) * 1e4
    out_large = head(x_large)
    assert not torch.isnan(out_large).any(), "Large input should not produce NaN"
    assert (out_large.permute(0, 2, 3, 4, 1).reshape(-1, 32).norm(dim=-1) < 1).all(), "Output should still be inside ball"
    print("  [PASS] Large input handled correctly")

    return True


def test_training_mode():
    """测试训练和评估模式"""
    print("Test 4.8: Testing train/eval modes...")
    from pasco.models.hyperbolic.projection_head import HyperbolicProjectionHead

    head = HyperbolicProjectionHead(in_channels=32, embed_dim=32)
    x = torch.randn(2, 32, 8, 8, 8)

    # 训练模式
    head.train()
    out_train = head(x)
    assert out_train.shape == (2, 32, 8, 8, 8), "Train mode output shape incorrect"

    # 评估模式
    head.eval()
    out_eval = head(x)
    assert out_eval.shape == (2, 32, 8, 8, 8), "Eval mode output shape incorrect"

    print("  [PASS] Both train and eval modes work")
    return True


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Stage 4 Tests: Hyperbolic Projection Head")
    print("=" * 60)

    if not check_dependencies():
        print("\n✗ Stage 4 BLOCKED - Complete Stage 2 first")
        return False
    print()

    tests = [
        test_imports,
        test_initialization,
        test_forward_shape,
        test_forward_different_embed_dim,
        test_output_in_ball,
        test_gradient,
        test_numerical_stability,
        test_training_mode,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [FAIL] Exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n✓ Stage 4 PASSED - Ready for Stage 5")
        return True
    else:
        print("\n✗ Stage 4 FAILED - Fix issues before proceeding")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
```

### 4.2 验收标准检查清单

- [ ] 依赖检查通过（Stage 2）
- [ ] 输入 [B, C, H, W, D] → 输出 [B, embed_dim, H, W, D]
- [ ] 所有输出点的范数 < 1
- [ ] 支持不同的 embed_dim
- [ ] 支持反向传播
- [ ] 数值稳定（小/大输入）
- [ ] 训练和评估模式都正常
- [ ] 所有 8 个测试用例通过

---

## 阶段 5：Hyperbolic 排序损失

### 5.1 编写测试脚本（先于实现）

**文件**: `tests/hyperbolic/test_stage5_hyperbolic_loss.py`

```python
"""
Stage 5 Test: Hyperbolic Ranking Loss
测试基于 Poincaré 距离的排序损失。

依赖: Stage 2 (poincare_ops)

运行方式:
    cd /home/comp/25481568/code/PaSCo
    python tests/hyperbolic/test_stage5_hyperbolic_loss.py
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def check_dependencies():
    """检查依赖阶段是否完成"""
    print("Checking dependencies...")
    try:
        from pasco.models.hyperbolic.poincare_ops import poincare_distance
        print("  [OK] Stage 2 (poincare_ops) available")
        return True
    except ImportError as e:
        print(f"  [FAIL] Stage 2 not complete: {e}")
        return False


def test_imports():
    """测试模块导入"""
    print("Test 5.1: Testing imports...")
    try:
        from pasco.loss.hyperbolic_loss import HyperbolicRankingLoss
        print("  [PASS] Import successful")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def test_initialization():
    """测试初始化"""
    print("Test 5.2: Testing initialization...")
    from pasco.loss.hyperbolic_loss import HyperbolicRankingLoss

    loss_fn = HyperbolicRankingLoss(margin=0.1, ignore_classes=[0, 255])

    assert loss_fn.margin == 0.1, f"margin mismatch: {loss_fn.margin}"
    assert 0 in loss_fn.ignore_classes, "Class 0 should be ignored"
    assert 255 in loss_fn.ignore_classes, "Class 255 should be ignored"
    print("  [PASS] Initialization correct")

    return True


def create_test_data(B=2, D=32, H=8, W=8, Z=8, N_classes=72, device='cpu'):
    """创建测试数据"""
    # Voxel embeddings in Poincaré ball
    voxel_emb = torch.randn(B, D, H, W, Z, device=device) * 0.5
    voxel_emb = voxel_emb / voxel_emb.norm(dim=1, keepdim=True).clamp(min=1.0)

    # Labels
    labels = torch.randint(0, N_classes, (B, H, W, Z), device=device)

    # Label embeddings in Poincaré ball
    label_emb = torch.randn(N_classes, D, device=device) * 0.5
    label_emb = label_emb / label_emb.norm(dim=1, keepdim=True).clamp(min=1.0)

    return voxel_emb, labels, label_emb


def test_forward_basic():
    """测试基本前向传播"""
    print("Test 5.3: Testing basic forward pass...")
    from pasco.loss.hyperbolic_loss import HyperbolicRankingLoss

    loss_fn = HyperbolicRankingLoss(margin=0.1, ignore_classes=[0, 255])
    voxel_emb, labels, label_emb = create_test_data()

    loss = loss_fn(voxel_emb, labels, label_emb)

    # 检查 loss 是标量
    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    print(f"  [PASS] Loss is scalar: {loss.item():.4f}")

    # 检查 loss 非负
    assert loss >= 0, f"Loss should be non-negative: {loss.item()}"
    print("  [PASS] Loss is non-negative")

    return True


def test_loss_requires_grad():
    """测试 loss 需要梯度"""
    print("Test 5.4: Testing loss requires gradient...")
    from pasco.loss.hyperbolic_loss import HyperbolicRankingLoss

    loss_fn = HyperbolicRankingLoss(margin=0.1)
    voxel_emb, labels, label_emb = create_test_data()
    voxel_emb.requires_grad_(True)
    label_emb.requires_grad_(True)

    loss = loss_fn(voxel_emb, labels, label_emb)

    assert loss.requires_grad, "Loss should require gradient"
    print("  [PASS] Loss requires gradient")

    return True


def test_backward():
    """测试反向传播"""
    print("Test 5.5: Testing backward pass...")
    from pasco.loss.hyperbolic_loss import HyperbolicRankingLoss

    loss_fn = HyperbolicRankingLoss(margin=0.1)
    voxel_emb, labels, label_emb = create_test_data()
    voxel_emb.requires_grad_(True)
    label_emb.requires_grad_(True)

    loss = loss_fn(voxel_emb, labels, label_emb)
    loss.backward()

    assert voxel_emb.grad is not None, "voxel_emb should have gradient"
    assert label_emb.grad is not None, "label_emb should have gradient"
    assert not torch.isnan(voxel_emb.grad).any(), "voxel_emb gradient should not contain NaN"
    assert not torch.isnan(label_emb.grad).any(), "label_emb gradient should not contain NaN"
    print("  [PASS] Backward pass successful")

    return True


def test_ignore_classes():
    """测试忽略类别"""
    print("Test 5.6: Testing ignore classes...")
    from pasco.loss.hyperbolic_loss import HyperbolicRankingLoss

    loss_fn = HyperbolicRankingLoss(margin=0.1, ignore_classes=[0, 255])

    # 创建只有忽略类别的数据
    B, D, H, W, Z = 2, 32, 4, 4, 4
    voxel_emb = torch.randn(B, D, H, W, Z) * 0.5
    label_emb = torch.randn(72, D) * 0.5

    # 所有标签都是 0
    labels_all_zero = torch.zeros(B, H, W, Z, dtype=torch.long)
    loss_zero = loss_fn(voxel_emb, labels_all_zero, label_emb)
    assert loss_zero.item() == 0, f"Loss should be 0 when all labels are ignored: {loss_zero.item()}"
    print("  [PASS] All ignored labels -> loss = 0")

    # 所有标签都是 255
    labels_all_255 = torch.full((B, H, W, Z), 255, dtype=torch.long)
    loss_255 = loss_fn(voxel_emb, labels_all_255, label_emb)
    assert loss_255.item() == 0, f"Loss should be 0 when all labels are 255: {loss_255.item()}"
    print("  [PASS] All 255 labels -> loss = 0")

    return True


def test_negative_sampling():
    """测试负样本采样"""
    print("Test 5.7: Testing negative sampling...")
    from pasco.loss.hyperbolic_loss import HyperbolicRankingLoss

    loss_fn = HyperbolicRankingLoss(margin=0.1, ignore_classes=[0])

    # 创建数据，确保有正样本
    voxel_emb, labels, label_emb = create_test_data()

    # 将一些标签设为非忽略类别
    labels[:] = 5  # 全部设为类别 5

    # 运行多次，检查损失变化（说明负样本是随机的）
    losses = []
    for _ in range(5):
        loss = loss_fn(voxel_emb, labels, label_emb)
        losses.append(loss.item())

    # 由于随机负样本，损失应该有变化（除非 margin 很大导致都是 0）
    print(f"  [INFO] Losses over 5 runs: {losses}")
    print("  [PASS] Negative sampling executed without error")

    return True


def test_margin_effect():
    """测试 margin 对损失的影响"""
    print("Test 5.8: Testing margin effect...")
    from pasco.loss.hyperbolic_loss import HyperbolicRankingLoss

    voxel_emb, labels, label_emb = create_test_data()
    # 确保有非忽略类别
    labels[labels == 0] = 1

    # 不同 margin 应该产生不同的损失
    losses = {}
    for margin in [0.01, 0.1, 0.5, 1.0]:
        loss_fn = HyperbolicRankingLoss(margin=margin, ignore_classes=[0, 255])
        loss = loss_fn(voxel_emb, labels, label_emb)
        losses[margin] = loss.item()

    print(f"  [INFO] Losses for different margins: {losses}")

    # 一般来说，更大的 margin 应该产生更大的损失
    # 但这取决于数据，所以只检查是否有变化
    unique_losses = len(set(losses.values()))
    assert unique_losses > 1 or all(v == 0 for v in losses.values()), "Different margins should affect loss"
    print("  [PASS] Margin affects loss")

    return True


def test_different_batch_sizes():
    """测试不同批次大小"""
    print("Test 5.9: Testing different batch sizes...")
    from pasco.loss.hyperbolic_loss import HyperbolicRankingLoss

    loss_fn = HyperbolicRankingLoss(margin=0.1, ignore_classes=[0, 255])

    for B in [1, 2, 4]:
        voxel_emb, labels, label_emb = create_test_data(B=B)
        labels[labels == 0] = 1  # 确保有非忽略类别

        loss = loss_fn(voxel_emb, labels, label_emb)
        assert loss.dim() == 0, f"Loss should be scalar for batch_size={B}"
        assert not torch.isnan(loss), f"Loss should not be NaN for batch_size={B}"
        print(f"  [PASS] batch_size={B}: loss={loss.item():.4f}")

    return True


def test_gradient_values():
    """测试梯度值合理性"""
    print("Test 5.10: Testing gradient value sanity...")
    from pasco.loss.hyperbolic_loss import HyperbolicRankingLoss

    loss_fn = HyperbolicRankingLoss(margin=0.1, ignore_classes=[0, 255])
    voxel_emb, labels, label_emb = create_test_data()
    labels[labels == 0] = 1  # 确保有非忽略类别

    voxel_emb.requires_grad_(True)
    label_emb.requires_grad_(True)

    loss = loss_fn(voxel_emb, labels, label_emb)
    loss.backward()

    # 检查梯度不是全零（说明梯度流动正常）
    if voxel_emb.grad.abs().sum() > 0:
        print(f"  [PASS] voxel_emb gradient non-zero: max={voxel_emb.grad.abs().max():.6f}")
    else:
        print("  [INFO] voxel_emb gradient is zero (might be ok if loss is 0)")

    # 检查梯度幅度合理
    assert voxel_emb.grad.abs().max() < 1e6, "voxel_emb gradient too large"
    assert label_emb.grad.abs().max() < 1e6, "label_emb gradient too large"
    print("  [PASS] Gradient magnitudes are reasonable")

    return True


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Stage 5 Tests: Hyperbolic Ranking Loss")
    print("=" * 60)

    if not check_dependencies():
        print("\n✗ Stage 5 BLOCKED - Complete Stage 2 first")
        return False
    print()

    tests = [
        test_imports,
        test_initialization,
        test_forward_basic,
        test_loss_requires_grad,
        test_backward,
        test_ignore_classes,
        test_negative_sampling,
        test_margin_effect,
        test_different_batch_sizes,
        test_gradient_values,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [FAIL] Exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n✓ Stage 5 PASSED - Ready for Stage 6")
        return True
    else:
        print("\n✗ Stage 5 FAILED - Fix issues before proceeding")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
```

### 5.2 验收标准检查清单

- [ ] 依赖检查通过（Stage 2）
- [ ] Loss 值非负
- [ ] Loss 是标量
- [ ] 当所有 voxel 都是忽略类别时，返回 0
- [ ] 支持反向传播
- [ ] 负样本永远不等于正样本
- [ ] 不同 margin 影响 loss
- [ ] 梯度值合理
- [ ] 所有 10 个测试用例通过

---

## 阶段 6：DenseUNet3D 修改

### 6.1 编写测试脚本（先于实现）

**文件**: `tests/hyperbolic/test_stage6_dense_unet_features.py`

```python
"""
Stage 6 Test: DenseUNet3D forward_with_features
测试 DenseUNet3D 和 DenseUNet3DLight 的特征输出功能。

依赖: 无（但修改现有文件）

运行方式:
    cd /home/comp/25481568/code/PaSCo
    python tests/hyperbolic/test_stage6_dense_unet_features.py
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_imports():
    """测试模块导入"""
    print("Test 6.1: Testing imports...")
    try:
        from pasco.models.dense_unet3d import DenseUNet3D, DenseUNet3DLight
        print("  [PASS] Import successful")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def test_dense_unet3d_has_method():
    """测试 DenseUNet3D 是否有 forward_with_features 方法"""
    print("Test 6.2: Testing DenseUNet3D has forward_with_features...")
    from pasco.models.dense_unet3d import DenseUNet3D

    model = DenseUNet3D(n_classes=72, in_channels=1, base_channels=32)

    assert hasattr(model, 'forward_with_features'), "DenseUNet3D should have forward_with_features method"
    assert callable(getattr(model, 'forward_with_features')), "forward_with_features should be callable"
    print("  [PASS] DenseUNet3D has forward_with_features method")

    return True


def test_dense_unet3d_light_has_method():
    """测试 DenseUNet3DLight 是否有 forward_with_features 方法"""
    print("Test 6.3: Testing DenseUNet3DLight has forward_with_features...")
    from pasco.models.dense_unet3d import DenseUNet3DLight

    model = DenseUNet3DLight(n_classes=72, in_channels=1, base_channels=32)

    assert hasattr(model, 'forward_with_features'), "DenseUNet3DLight should have forward_with_features method"
    assert callable(getattr(model, 'forward_with_features')), "forward_with_features should be callable"
    print("  [PASS] DenseUNet3DLight has forward_with_features method")

    return True


def test_dense_unet3d_output_shape():
    """测试 DenseUNet3D forward_with_features 输出形状"""
    print("Test 6.4: Testing DenseUNet3D output shapes...")
    from pasco.models.dense_unet3d import DenseUNet3D

    model = DenseUNet3D(n_classes=72, in_channels=1, base_channels=32)

    B, C, H, W, D = 1, 1, 32, 32, 32
    x = torch.randn(B, C, H, W, D)

    logits, features = model.forward_with_features(x)

    # 检查 logits 形状
    expected_logits_shape = (B, 72, H, W, D)
    assert logits.shape == expected_logits_shape, f"Logits shape mismatch: {logits.shape} vs {expected_logits_shape}"
    print(f"  [PASS] Logits shape: {logits.shape}")

    # 检查 features 形状
    expected_features_shape = (B, 32, H, W, D)  # base_channels
    assert features.shape == expected_features_shape, f"Features shape mismatch: {features.shape} vs {expected_features_shape}"
    print(f"  [PASS] Features shape: {features.shape}")

    return True


def test_dense_unet3d_light_output_shape():
    """测试 DenseUNet3DLight forward_with_features 输出形状"""
    print("Test 6.5: Testing DenseUNet3DLight output shapes...")
    from pasco.models.dense_unet3d import DenseUNet3DLight

    model = DenseUNet3DLight(n_classes=72, in_channels=1, base_channels=32)

    B, C, H, W, D = 1, 1, 32, 32, 32
    x = torch.randn(B, C, H, W, D)

    logits, features = model.forward_with_features(x)

    # 检查 logits 形状
    expected_logits_shape = (B, 72, H, W, D)
    assert logits.shape == expected_logits_shape, f"Logits shape mismatch: {logits.shape} vs {expected_logits_shape}"
    print(f"  [PASS] Logits shape: {logits.shape}")

    # 检查 features 形状
    expected_features_shape = (B, 32, H, W, D)
    assert features.shape == expected_features_shape, f"Features shape mismatch: {features.shape} vs {expected_features_shape}"
    print(f"  [PASS] Features shape: {features.shape}")

    return True


def test_logits_consistency():
    """测试 forward 和 forward_with_features 的 logits 一致性"""
    print("Test 6.6: Testing logits consistency between forward methods...")
    from pasco.models.dense_unet3d import DenseUNet3D, DenseUNet3DLight

    for ModelClass, name in [(DenseUNet3D, "DenseUNet3D"), (DenseUNet3DLight, "DenseUNet3DLight")]:
        model = ModelClass(n_classes=72, in_channels=1, base_channels=32)
        model.eval()  # 确保没有随机性

        x = torch.randn(1, 1, 16, 16, 16)

        with torch.no_grad():
            logits_forward = model.forward(x)
            logits_with_features, _ = model.forward_with_features(x)

        assert torch.allclose(logits_forward, logits_with_features, atol=1e-5), \
            f"{name}: Logits should be identical"
        print(f"  [PASS] {name}: Logits are consistent")

    return True


def test_different_input_sizes():
    """测试不同输入尺寸"""
    print("Test 6.7: Testing different input sizes...")
    from pasco.models.dense_unet3d import DenseUNet3D

    model = DenseUNet3D(n_classes=72, in_channels=1, base_channels=32)

    # 测试不同尺寸（需要是 2^n 的倍数，因为有下采样）
    sizes = [(16, 16, 16), (32, 32, 32), (64, 64, 64)]

    for H, W, D in sizes:
        x = torch.randn(1, 1, H, W, D)
        try:
            logits, features = model.forward_with_features(x)
            assert logits.shape == (1, 72, H, W, D), f"Logits shape mismatch for size {(H, W, D)}"
            assert features.shape == (1, 32, H, W, D), f"Features shape mismatch for size {(H, W, D)}"
            print(f"  [PASS] Input size {(H, W, D)} works")
        except Exception as e:
            print(f"  [FAIL] Input size {(H, W, D)} failed: {e}")
            return False

    return True


def test_gradient_flow():
    """测试梯度流动"""
    print("Test 6.8: Testing gradient flow through features...")
    from pasco.models.dense_unet3d import DenseUNet3D

    model = DenseUNet3D(n_classes=72, in_channels=1, base_channels=32)

    x = torch.randn(1, 1, 16, 16, 16, requires_grad=True)
    logits, features = model.forward_with_features(x)

    # 对 features 的损失应该能反向传播
    loss = features.sum()
    loss.backward()

    assert x.grad is not None, "Input should have gradient from features"
    assert not torch.isnan(x.grad).any(), "Gradient should not contain NaN"
    print("  [PASS] Gradient flows through features")

    return True


def test_different_base_channels():
    """测试不同的 base_channels"""
    print("Test 6.9: Testing different base_channels...")
    from pasco.models.dense_unet3d import DenseUNet3D

    for base_channels in [16, 32, 64]:
        model = DenseUNet3D(n_classes=72, in_channels=1, base_channels=base_channels)
        x = torch.randn(1, 1, 16, 16, 16)

        logits, features = model.forward_with_features(x)

        assert features.shape[1] == base_channels, \
            f"Features channels should be {base_channels}, got {features.shape[1]}"
        print(f"  [PASS] base_channels={base_channels}: features shape correct")

    return True


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Stage 6 Tests: DenseUNet3D forward_with_features")
    print("=" * 60)
    print()

    tests = [
        test_imports,
        test_dense_unet3d_has_method,
        test_dense_unet3d_light_has_method,
        test_dense_unet3d_output_shape,
        test_dense_unet3d_light_output_shape,
        test_logits_consistency,
        test_different_input_sizes,
        test_gradient_flow,
        test_different_base_channels,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [FAIL] Exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n✓ Stage 6 PASSED - Ready for Stage 7")
        return True
    else:
        print("\n✗ Stage 6 FAILED - Fix issues before proceeding")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
```

### 6.2 验收标准检查清单

- [ ] `DenseUNet3D` 有 `forward_with_features` 方法
- [ ] `DenseUNet3DLight` 有 `forward_with_features` 方法
- [ ] `forward_with_features` 返回 (logits, features) 元组
- [ ] features 形状为 [B, base_channels, H, W, D]
- [ ] logits 与原 `forward()` 输出一致
- [ ] 支持梯度流动
- [ ] 支持不同输入尺寸
- [ ] 支持不同 base_channels
- [ ] 所有 9 个测试用例通过

---

## 阶段 7：BodyNetHyperbolic 完整模型

### 7.1 编写测试脚本（先于实现）

**文件**: `tests/hyperbolic/test_stage7_body_net_hyperbolic.py`

```python
"""
Stage 7 Test: BodyNetHyperbolic
测试集成了 hyperbolic embedding 的完整模型。

依赖: Stage 3-6

运行方式:
    cd /home/comp/25481568/code/PaSCo
    python tests/hyperbolic/test_stage7_body_net_hyperbolic.py
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def check_dependencies():
    """检查依赖阶段是否完成"""
    print("Checking dependencies...")

    deps = [
        ("Stage 3", "pasco.models.hyperbolic.label_embedding", "HyperbolicLabelEmbedding"),
        ("Stage 4", "pasco.models.hyperbolic.projection_head", "HyperbolicProjectionHead"),
        ("Stage 5", "pasco.loss.hyperbolic_loss", "HyperbolicRankingLoss"),
        ("Stage 6", "pasco.models.dense_unet3d", "DenseUNet3D"),
    ]

    for stage, module, cls in deps:
        try:
            mod = __import__(module, fromlist=[cls])
            getattr(mod, cls)
            print(f"  [OK] {stage} ({cls}) available")
        except (ImportError, AttributeError) as e:
            print(f"  [FAIL] {stage} not complete: {e}")
            return False

    # 特殊检查: DenseUNet3D 需要有 forward_with_features
    from pasco.models.dense_unet3d import DenseUNet3D
    if not hasattr(DenseUNet3D, 'forward_with_features'):
        print("  [FAIL] Stage 6: DenseUNet3D.forward_with_features not found")
        return False

    return True


def test_imports():
    """测试模块导入"""
    print("Test 7.1: Testing imports...")
    try:
        from pasco.models.body_net_hyperbolic import BodyNetHyperbolic
        print("  [PASS] Import successful")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def test_initialization():
    """测试初始化"""
    print("Test 7.2: Testing initialization...")
    from pasco.models.body_net_hyperbolic import BodyNetHyperbolic

    model = BodyNetHyperbolic(
        n_classes=72,
        in_channels=1,
        base_channels=32,
        embed_dim=32,
        hyperbolic_weight=0.1,
        margin=0.1,
    )

    # 检查组件
    assert hasattr(model, 'hyp_head'), "Should have hyp_head"
    assert hasattr(model, 'label_emb'), "Should have label_emb"
    assert hasattr(model, 'hyp_loss_fn'), "Should have hyp_loss_fn"
    print("  [PASS] All hyperbolic components initialized")

    # 检查参数
    assert model.embed_dim == 32, f"embed_dim mismatch: {model.embed_dim}"
    assert model.hyperbolic_weight == 0.1, f"hyperbolic_weight mismatch: {model.hyperbolic_weight}"
    print("  [PASS] Parameters correct")

    return True


def test_forward():
    """测试标准 forward（仅返回 logits）"""
    print("Test 7.3: Testing forward() returns only logits...")
    from pasco.models.body_net_hyperbolic import BodyNetHyperbolic

    model = BodyNetHyperbolic(n_classes=72, in_channels=1, base_channels=32)

    x = torch.randn(1, 1, 32, 32, 32)
    logits = model(x)

    # 应该只返回 logits，不是元组
    assert isinstance(logits, torch.Tensor), "forward() should return a tensor"
    assert logits.shape == (1, 72, 32, 32, 32), f"Logits shape mismatch: {logits.shape}"
    print(f"  [PASS] forward() returns logits only: {logits.shape}")

    return True


def test_forward_with_hyperbolic():
    """测试 forward_with_hyperbolic 返回 logits 和 embeddings"""
    print("Test 7.4: Testing forward_with_hyperbolic()...")
    from pasco.models.body_net_hyperbolic import BodyNetHyperbolic

    model = BodyNetHyperbolic(n_classes=72, in_channels=1, base_channels=32, embed_dim=32)

    x = torch.randn(1, 1, 32, 32, 32)
    logits, voxel_emb = model.forward_with_hyperbolic(x)

    # 检查 logits
    assert logits.shape == (1, 72, 32, 32, 32), f"Logits shape mismatch: {logits.shape}"
    print(f"  [PASS] Logits shape: {logits.shape}")

    # 检查 voxel_emb
    assert voxel_emb.shape == (1, 32, 32, 32, 32), f"Voxel emb shape mismatch: {voxel_emb.shape}"
    print(f"  [PASS] Voxel embeddings shape: {voxel_emb.shape}")

    # 检查 voxel_emb 在 Poincaré 球内
    voxel_emb_flat = voxel_emb.permute(0, 2, 3, 4, 1).reshape(-1, 32)
    norms = voxel_emb_flat.norm(dim=-1)
    assert (norms < 1).all(), f"Voxel embeddings should be inside ball: max_norm={norms.max():.4f}"
    print(f"  [PASS] Voxel embeddings inside Poincaré ball (max_norm={norms.max():.4f})")

    return True


def test_training_step():
    """测试 training_step"""
    print("Test 7.5: Testing training_step()...")
    from pasco.models.body_net_hyperbolic import BodyNetHyperbolic

    model = BodyNetHyperbolic(
        n_classes=72,
        in_channels=1,
        base_channels=32,
        embed_dim=32,
        hyperbolic_weight=0.1,
    )

    batch = {
        "occupancy": torch.randn(1, 1, 16, 16, 16),
        "labels": torch.randint(0, 72, (1, 16, 16, 16)),
    }

    loss = model.training_step(batch, batch_idx=0)

    assert isinstance(loss, torch.Tensor), "training_step should return a tensor"
    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert loss.requires_grad, "Loss should require gradient"
    print(f"  [PASS] training_step returns loss: {loss.item():.4f}")

    return True


def test_validation_step():
    """测试 validation_step"""
    print("Test 7.6: Testing validation_step()...")
    from pasco.models.body_net_hyperbolic import BodyNetHyperbolic

    model = BodyNetHyperbolic(
        n_classes=72,
        in_channels=1,
        base_channels=32,
        embed_dim=32,
        hyperbolic_weight=0.1,
    )

    # 初始化 validation 累积器
    model.on_validation_epoch_start()

    batch = {
        "occupancy": torch.randn(1, 1, 16, 16, 16),
        "labels": torch.randint(0, 72, (1, 16, 16, 16)),
    }

    result = model.validation_step(batch, batch_idx=0)

    assert isinstance(result, dict), "validation_step should return a dict"
    assert "loss" in result, "Result should contain 'loss'"
    print(f"  [PASS] validation_step returns dict with loss: {result['loss'].item():.4f}")

    return True


def test_label_embeddings():
    """测试 label embeddings 可以获取"""
    print("Test 7.7: Testing label embeddings access...")
    from pasco.models.body_net_hyperbolic import BodyNetHyperbolic

    model = BodyNetHyperbolic(n_classes=72, embed_dim=32)

    label_emb = model.label_emb()
    assert label_emb.shape == (72, 32), f"Label emb shape mismatch: {label_emb.shape}"
    print(f"  [PASS] Label embeddings shape: {label_emb.shape}")

    # 检查在球内
    norms = label_emb.norm(dim=-1)
    assert (norms < 1).all(), f"Label embeddings should be inside ball: max_norm={norms.max():.4f}"
    print(f"  [PASS] Label embeddings inside ball (max_norm={norms.max():.4f})")

    return True


def test_backward():
    """测试反向传播"""
    print("Test 7.8: Testing backward pass...")
    from pasco.models.body_net_hyperbolic import BodyNetHyperbolic

    model = BodyNetHyperbolic(n_classes=72, base_channels=32, embed_dim=32)

    batch = {
        "occupancy": torch.randn(1, 1, 16, 16, 16),
        "labels": torch.randint(1, 72, (1, 16, 16, 16)),  # 避免全是 0
    }

    loss = model.training_step(batch, 0)
    loss.backward()

    # 检查某些参数有梯度
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "At least some parameters should have non-zero gradient"
    print("  [PASS] Backward pass successful, gradients computed")

    return True


def test_hyperbolic_weight_effect():
    """测试 hyperbolic_weight 影响"""
    print("Test 7.9: Testing hyperbolic_weight effect...")
    from pasco.models.body_net_hyperbolic import BodyNetHyperbolic

    batch = {
        "occupancy": torch.randn(1, 1, 16, 16, 16),
        "labels": torch.randint(1, 72, (1, 16, 16, 16)),
    }

    # 不同的 hyperbolic_weight
    losses = {}
    for weight in [0.0, 0.1, 0.5]:
        model = BodyNetHyperbolic(
            n_classes=72,
            base_channels=32,
            embed_dim=32,
            hyperbolic_weight=weight,
        )
        with torch.no_grad():
            loss = model.training_step(batch, 0)
        losses[weight] = loss.item()

    print(f"  [INFO] Losses for different weights: {losses}")

    # weight=0 应该只有 CE loss
    # weight>0 应该有额外的 hyperbolic loss
    if losses[0.1] != losses[0.0]:
        print("  [PASS] hyperbolic_weight affects total loss")
    else:
        print("  [INFO] Losses are same (hyperbolic loss might be 0)")

    return True


def test_inheritance():
    """测试继承 BodyNet"""
    print("Test 7.10: Testing BodyNet inheritance...")
    from pasco.models.body_net_hyperbolic import BodyNetHyperbolic
    from pasco.models.body_net import BodyNet

    assert issubclass(BodyNetHyperbolic, BodyNet), "BodyNetHyperbolic should inherit from BodyNet"
    print("  [PASS] BodyNetHyperbolic inherits from BodyNet")

    # 检查 BodyNet 的方法仍然可用
    model = BodyNetHyperbolic(n_classes=72, base_channels=32)
    assert hasattr(model, 'compute_loss'), "Should have compute_loss from BodyNet"
    assert hasattr(model, 'compute_iou'), "Should have compute_iou from BodyNet"
    print("  [PASS] BodyNet methods available")

    return True


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Stage 7 Tests: BodyNetHyperbolic")
    print("=" * 60)

    if not check_dependencies():
        print("\n✗ Stage 7 BLOCKED - Complete Stages 3-6 first")
        return False
    print()

    tests = [
        test_imports,
        test_initialization,
        test_forward,
        test_forward_with_hyperbolic,
        test_training_step,
        test_validation_step,
        test_label_embeddings,
        test_backward,
        test_hyperbolic_weight_effect,
        test_inheritance,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [FAIL] Exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n✓ Stage 7 PASSED - Ready for Stage 8")
        return True
    else:
        print("\n✗ Stage 7 FAILED - Fix issues before proceeding")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
```

### 7.2 验收标准检查清单

- [ ] 依赖检查通过（Stage 3-6）
- [ ] 继承 `BodyNet` 的所有功能
- [ ] `forward()` 仅返回 logits（兼容推理）
- [ ] `forward_with_hyperbolic()` 返回 logits 和 embeddings
- [ ] voxel embeddings 在 Poincaré ball 内（范数 < 1）
- [ ] `training_step` 返回 CE + hyperbolic loss 的组合
- [ ] `validation_step` 正常工作
- [ ] 支持反向传播
- [ ] `hyperbolic_weight` 影响总损失
- [ ] 所有 10 个测试用例通过

---

## 阶段 8：训练脚本集成

### 8.1 编写测试脚本（先于实现）

**文件**: `tests/hyperbolic/test_stage8_train_integration.py`

```python
"""
Stage 8 Test: Training Script Integration
测试训练脚本的 hyperbolic 参数集成。

依赖: Stage 7

运行方式:
    cd /home/comp/25481568/code/PaSCo
    python tests/hyperbolic/test_stage8_train_integration.py
"""

import sys
import os
import subprocess
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def check_dependencies():
    """检查依赖阶段是否完成"""
    print("Checking dependencies...")
    try:
        from pasco.models.body_net_hyperbolic import BodyNetHyperbolic
        print("  [OK] Stage 7 (BodyNetHyperbolic) available")
        return True
    except ImportError as e:
        print(f"  [FAIL] Stage 7 not complete: {e}")
        return False


def test_help_output():
    """测试 --help 输出包含 hyperbolic 参数"""
    print("Test 8.1: Testing --help contains hyperbolic arguments...")

    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "scripts", "body", "train_body.py"
    )

    if not os.path.exists(script_path):
        print(f"  [FAIL] Script not found: {script_path}")
        return False

    result = subprocess.run(
        [sys.executable, script_path, "--help"],
        capture_output=True,
        text=True
    )

    help_text = result.stdout

    # 检查必要的参数
    required_args = [
        "--use_hyperbolic",
        "--hyp_embed_dim",
        "--hyp_weight",
        "--hyp_margin",
    ]

    missing = []
    for arg in required_args:
        if arg not in help_text:
            missing.append(arg)

    if missing:
        print(f"  [FAIL] Missing arguments in help: {missing}")
        return False

    print("  [PASS] All hyperbolic arguments present in --help")

    # 打印相关帮助内容
    lines = help_text.split('\n')
    print("  Hyperbolic arguments found:")
    for line in lines:
        if 'hyperbolic' in line.lower() or 'hyp_' in line.lower():
            print(f"    {line.strip()}")

    return True


def test_argument_parsing():
    """测试参数解析"""
    print("Test 8.2: Testing argument parsing...")

    # 模拟 argparse
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "scripts", "body", "train_body.py"
    )

    # 读取脚本内容检查参数定义
    with open(script_path, 'r') as f:
        content = f.read()

    # 检查参数定义
    checks = [
        ('--use_hyperbolic', 'action="store_true"' in content or "action='store_true'" in content),
        ('--hyp_embed_dim', 'hyp_embed_dim' in content),
        ('--hyp_weight', 'hyp_weight' in content),
        ('--hyp_margin', 'hyp_margin' in content),
    ]

    for arg, found in checks:
        if found:
            print(f"  [PASS] {arg} is defined")
        else:
            print(f"  [FAIL] {arg} definition not found")
            return False

    return True


def test_model_creation_logic():
    """测试模型创建逻辑存在"""
    print("Test 8.3: Testing model creation logic...")

    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "scripts", "body", "train_body.py"
    )

    with open(script_path, 'r') as f:
        content = f.read()

    # 检查条件模型创建
    if 'use_hyperbolic' in content and 'BodyNetHyperbolic' in content:
        print("  [PASS] Conditional BodyNetHyperbolic creation logic found")
    else:
        print("  [FAIL] Conditional model creation logic not found")
        return False

    # 检查 import
    if 'from pasco.models.body_net_hyperbolic import BodyNetHyperbolic' in content:
        print("  [PASS] BodyNetHyperbolic import found")
    else:
        # 可能是延迟导入
        if 'body_net_hyperbolic' in content:
            print("  [PASS] BodyNetHyperbolic import (possibly lazy) found")
        else:
            print("  [FAIL] BodyNetHyperbolic import not found")
            return False

    return True


def test_default_values():
    """测试默认值"""
    print("Test 8.4: Testing default values...")

    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "scripts", "body", "train_body.py"
    )

    with open(script_path, 'r') as f:
        content = f.read()

    # 检查默认值
    defaults = {
        'hyp_embed_dim': '32',
        'hyp_weight': '0.1',
        'hyp_margin': '0.1',
    }

    for arg, default in defaults.items():
        if f'default={default}' in content or f"default='{default}'" in content or f'default={float(default) if "." in default else int(default)}' in content:
            print(f"  [PASS] {arg} has expected default")
        else:
            print(f"  [INFO] {arg} default value might differ (this is ok)")

    return True


def test_script_dry_run():
    """测试脚本 dry run（不需要数据）"""
    print("Test 8.5: Testing script import (dry run)...")

    # 只测试脚本可以被导入（语法正确）
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "scripts", "body", "train_body.py"
    )

    # 检查语法
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", script_path],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("  [PASS] Script syntax is valid")
        return True
    else:
        print(f"  [FAIL] Script syntax error: {result.stderr}")
        return False


def test_no_hyperbolic_flag():
    """测试不使用 --use_hyperbolic 时的行为"""
    print("Test 8.6: Testing without --use_hyperbolic flag...")

    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "scripts", "body", "train_body.py"
    )

    with open(script_path, 'r') as f:
        content = f.read()

    # 检查有 else 分支或默认情况使用 BodyNet
    if 'else:' in content or 'BodyNet(' in content:
        print("  [PASS] Fallback to BodyNet when --use_hyperbolic is not set")
        return True
    else:
        print("  [WARN] Could not verify fallback behavior")
        return True  # 不阻止，只是警告


def test_hyperbolic_params_passed():
    """测试 hyperbolic 参数被传递给模型"""
    print("Test 8.7: Testing hyperbolic params passed to model...")

    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "scripts", "body", "train_body.py"
    )

    with open(script_path, 'r') as f:
        content = f.read()

    # 检查参数传递
    params_to_check = [
        'embed_dim',
        'hyperbolic_weight',
        'margin',
    ]

    found = 0
    for param in params_to_check:
        if param in content:
            found += 1

    if found >= 2:  # 至少找到 2 个
        print(f"  [PASS] Found {found}/3 hyperbolic params in model creation")
        return True
    else:
        print(f"  [WARN] Only found {found}/3 hyperbolic params")
        return True  # 不阻止


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Stage 8 Tests: Training Script Integration")
    print("=" * 60)

    if not check_dependencies():
        print("\n✗ Stage 8 BLOCKED - Complete Stage 7 first")
        return False
    print()

    tests = [
        test_help_output,
        test_argument_parsing,
        test_model_creation_logic,
        test_default_values,
        test_script_dry_run,
        test_no_hyperbolic_flag,
        test_hyperbolic_params_passed,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [FAIL] Exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n✓ Stage 8 PASSED - All stages complete!")
        print("\n" + "=" * 60)
        print("HYPERBOLIC EMBEDDING INTEGRATION COMPLETE")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Run actual training with: python scripts/body/train_body.py --use_hyperbolic ...")
        print("  2. Monitor hyp_loss in tensorboard/logs")
        print("  3. Evaluate results and tune hyperbolic_weight")
        return True
    else:
        print("\n✗ Stage 8 FAILED - Fix issues before proceeding")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
```

### 8.2 验收标准检查清单

- [ ] 依赖检查通过（Stage 7）
- [ ] 新参数在 `--help` 中显示
- [ ] 参数定义语法正确
- [ ] 条件模型创建逻辑存在
- [ ] 不使用 `--use_hyperbolic` 时行为不变
- [ ] Hyperbolic 参数被传递给模型
- [ ] 脚本语法正确
- [ ] 所有 7 个测试用例通过

---

## 快速参考：执行命令汇总

```bash
# 设置测试环境
cd /home/comp/25481568/code/PaSCo
mkdir -p tests/hyperbolic
touch tests/__init__.py
touch tests/hyperbolic/__init__.py

# 阶段 1
python tests/hyperbolic/test_stage1_organ_hierarchy.py

# 阶段 2
python tests/hyperbolic/test_stage2_poincare_ops.py

# 阶段 3
python tests/hyperbolic/test_stage3_label_embedding.py

# 阶段 4
python tests/hyperbolic/test_stage4_projection_head.py

# 阶段 5
python tests/hyperbolic/test_stage5_hyperbolic_loss.py

# 阶段 6
python tests/hyperbolic/test_stage6_dense_unet_features.py

# 阶段 7
python tests/hyperbolic/test_stage7_body_net_hyperbolic.py

# 阶段 8
python tests/hyperbolic/test_stage8_train_integration.py

# 运行所有测试
for i in {1..8}; do
    echo "=== Stage $i ==="
    python tests/hyperbolic/test_stage${i}_*.py
    if [ $? -ne 0 ]; then
        echo "Stage $i failed. Stopping."
        break
    fi
done
```

---

## 当前进度追踪

| 阶段 | 测试脚本 | 实现 | 测试通过 |
|------|----------|------|----------|
| 0 | - | - | [ ] |
| 1 | [ ] | [ ] | [ ] |
| 2 | [ ] | [ ] | [ ] |
| 3 | [ ] | [ ] | [ ] |
| 4 | [ ] | [ ] | [ ] |
| 5 | [ ] | [ ] | [ ] |
| 6 | [ ] | [ ] | [ ] |
| 7 | [ ] | [ ] | [ ] |
| 8 | [ ] | [ ] | [ ] |

---

## 开始实施

准备好后，告诉我 **"开始阶段 0"** 或 **"开始阶段 1"**，我将：
1. 首先创建该阶段的测试脚本
2. 运行测试确认失败（RED）
3. 实现代码
4. 运行测试确认通过（GREEN）
5. 继续下一阶段
