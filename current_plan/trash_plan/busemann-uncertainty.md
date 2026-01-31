# Busemann函数整合到PaSCo Hyperbolic模块可行性分析

## 1. 研究背景总结

### 1.1 HIPO论文核心方法

**Ideal Prototypes（理想原型）**：
- 位于Poincaré球边界（||u|| = 1）的点，代表"无穷远处"的类别原型
- 公式：`I_n = {u ∈ R^n : u₁² + u₂² + ... + uₙ² = 1}`

**Busemann函数定义**：
```
B_u(z) = log(||u - z||² / (1 - ||z||²))
```
- 用于测量双曲空间内的点到边界上Ideal Prototype的"距离"
- 范围：从 -∞（靠近u）到 +∞（远离u）

**Uncertainty-aware Learning**：
- **关键洞察**：到原点的双曲距离可作为uncertainty度量
  - 靠近原点 → 高不确定性
  - 靠近边界 → 低不确定性（更接近某个Ideal Prototype）
- **Uncertainty Loss**: `L_uncertainty = d_B(z, O)` （到原点的双曲距离）
- **总Loss**: `L_HIPO = L_Busemann + φ·L_uncertainty + L_CE`

### 1.2 Multi-Prototype论文核心方法

**Horospherical Classification Layer**：
- Score函数：`ξ(x) = -B_p(x) + a`（负Busemann值+偏置）
- 多类概率：`P(y=k|x) = softmax(ξ_k(x))`
- Horospheres（等Busemann面）作为决策边界

**多原型扩展**：
```
ξ_k(x) = -1/m · Σ B_{p_{k,i}}(x) + a_k
```
允许每个类有多个原型，更好处理类内多模态分布。

### 1.3 PaSCo当前Hyperbolic模块

| 组件 | 文件路径 | 当前实现 |
|------|----------|----------|
| 几何操作 | `pasco/models/hyperbolic/lorentz_ops.py` | Lorentz模型（hyperboloid） |
| 原型嵌入 | `pasco/models/hyperbolic/label_embedding.py` | 基于层级深度初始化，可学习参数 |
| 投影头 | `pasco/models/hyperbolic/projection_head.py` | exp_map投影到双曲空间 |
| Loss | `pasco/loss/lorentz_loss.py` | Ranking Loss (margin-based) |
| 模型 | `pasco/models/body_net_hyperbolic.py` | CE Loss + Hyperbolic Ranking Loss |

**当前限制**：
- 没有显式uncertainty估计
- 原型在空间内部（不是边界上）
- 没有使用Busemann函数

---

## 2. 技术可行性分析

### 2.1 模型兼容性问题

**挑战**：PaSCo使用Lorentz模型，而Busemann函数通常在Poincaré模型中定义。

**解决方案**：

| 方案 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| A. Lorentz→Poincaré转换 | 在计算Busemann时转换坐标 | 复用现有代码 | 额外计算开销 |
| B. 实现Lorentz Busemann | 直接在Lorentz模型中定义Busemann | 高效、统一 | 需要数学推导 |
| C. 添加Poincaré分支 | 保留双模型支持 | 灵活 | 代码复杂度增加 |

**推荐方案B**：Lorentz模型中的Busemann函数

Lorentz模型中，Ideal Prototype位于光锥（lightcone）上，Busemann函数可定义为：
```python
def lorentz_busemann(z, u, curv=1.0):
    """
    z: 双曲空间中的点 (time, space)
    u: 光锥上的理想原型 (满足 ⟨u,u⟩_L = 0)
    """
    # Lorentz内积: ⟨x,y⟩_L = -x₀y₀ + x₁y₁ + ... + xₙyₙ
    lorentz_inner = -z[..., 0] * u[..., 0] + (z[..., 1:] * u[..., 1:]).sum(-1)
    return -torch.log(-lorentz_inner)
```

### 2.2 Ideal Prototypes设计

**在Lorentz模型中定义Ideal Prototypes**：
- 位于光锥上：`⟨u,u⟩_L = -u₀² + ||u_space||² = 0`
- 即：`u₀ = ||u_space||`

**初始化策略**：
1. 使用CLIP语义嵌入确定原型间的相对位置（参考HIPO）
2. 利用器官层级结构（已有CLASS_DEPTHS）
3. 相似器官原型靠近，不同器官原型正交

### 2.3 Uncertainty估计

**两种uncertainty度量**：

1. **到原点的距离** (HIPO方法)：
   ```python
   def hyperbolic_uncertainty(z, curv=1.0):
       return lorentz_distance_to_origin(z, curv)
   ```
   - 越靠近原点 → 越不确定

2. **到最近Ideal Prototype的Busemann值**：
   ```python
   def busemann_uncertainty(z, ideal_prototypes):
       busemann_values = lorentz_busemann(z, ideal_prototypes)
       return busemann_values.min(dim=-1)  # 最小值越小 → 越确定
   ```

---

## 3. 整合方案

### 3.1 新增组件

```
pasco/models/hyperbolic/
├── lorentz_ops.py           # 添加: lorentz_busemann()
├── ideal_prototype.py       # 新增: IdealPrototypeLayer (光锥上的原型)
├── busemann_classifier.py   # 新增: BusemannClassifier (基于Busemann的分类)
└── uncertainty.py           # 新增: UncertaintyEstimator

pasco/loss/
└── busemann_loss.py         # 新增: BusemannLoss, UncertaintyRegularizer
```

### 3.2 核心类设计

```python
class LorentzIdealPrototype(nn.Module):
    """光锥上的Ideal Prototypes"""
    def __init__(self, n_classes, embed_dim):
        # 只存储空间方向，time分量由约束确定
        self.directions = nn.Parameter(...)  # 在S^{d-1}上

    def forward(self):
        # 返回光锥上的点: (||dir||, dir)
        return ideal_prototypes

class BusemannClassifier(nn.Module):
    """基于Busemann函数的分类层"""
    def __init__(self, n_classes, embed_dim):
        self.ideal_prototypes = LorentzIdealPrototype(n_classes, embed_dim)
        self.biases = nn.Parameter(...)  # 每类的偏置 a_k

    def forward(self, z):
        # ξ_k(z) = -B_{p_k}(z) + a_k
        busemann = lorentz_busemann(z, self.ideal_prototypes())
        return -busemann + self.biases

class BusemannLoss(nn.Module):
    """Busemann对齐Loss + Uncertainty正则化"""
    def __init__(self, phi=0.5):
        self.phi = phi  # uncertainty正则化强度

    def forward(self, z, labels, ideal_prototypes):
        L_B = busemann_alignment_loss(z, labels, ideal_prototypes)
        L_unc = uncertainty_loss(z)
        return L_B + self.phi * L_unc
```

### 3.3 与现有系统整合

**BodyNetHyperbolic扩展**：
```python
class BodyNetHyperbolicBusemann(BodyNetHyperbolic):
    def __init__(self, ...):
        super().__init__(...)
        # 添加Busemann分类器
        self.busemann_classifier = BusemannClassifier(n_classes, embed_dim)
        self.busemann_loss = BusemannLoss(phi=0.5)

    def forward_with_uncertainty(self, x):
        logits, voxel_emb = self.forward_with_hyperbolic(x)
        uncertainty = compute_uncertainty(voxel_emb)
        return logits, voxel_emb, uncertainty
```

---

## 4. 潜在应用场景

| 应用 | 描述 | 价值 |
|------|------|------|
| **分割置信度** | 每个voxel的uncertainty估计 | 临床决策支持 |
| **OOD检测** | 检测非训练分布的器官 | 提高安全性 |
| **主动学习** | 基于uncertainty选择需要标注的样本 | 降低标注成本 |
| **多模态分割** | 多原型捕获器官形态变异 | 提高泛化能力 |
| **层级一致性** | Busemann结构保持器官层级 | 解剖学合理性 |

---

## 5. 实施路线图（聚焦：Uncertainty估计 + Lorentz Busemann）

### Phase 1: Lorentz Busemann数学基础
**目标**：在Lorentz模型中推导和实现Busemann函数

**任务**：
- [ ] 推导Lorentz模型中Ideal Point的定义（光锥上的点）
- [ ] 推导Lorentz Busemann函数公式
- [ ] 实现 `lorentz_busemann()` 在 `lorentz_ops.py`
- [ ] 编写数学正确性单元测试

**关键文件**：
- `pasco/models/hyperbolic/lorentz_ops.py` (修改)
- `tests/hyperbolic/test_lorentz_busemann.py` (新增)

### Phase 2: Ideal Prototype层实现
**目标**：在光锥上定义和初始化Ideal Prototypes

**任务**：
- [ ] 实现 `LorentzIdealPrototype` 类
- [ ] 基于器官层级结构初始化（复用CLASS_DEPTHS）
- [ ] 添加正交性约束（语义相似类靠近）
- [ ] 编写测试验证原型位于光锥上

**关键文件**：
- `pasco/models/hyperbolic/ideal_prototype.py` (新增)
- `tests/hyperbolic/test_ideal_prototype.py` (新增)

### Phase 3: Uncertainty估计模块
**目标**：实现基于Busemann的uncertainty量化

**任务**：
- [ ] 实现两种uncertainty度量：
  - 到原点的双曲距离（HIPO方法）
  - 到最近Ideal Prototype的Busemann值
- [ ] 实现 `UncertaintyEstimator` 类
- [ ] 添加uncertainty正则化Loss `L_uncertainty`
- [ ] 编写测试验证uncertainty范围和梯度

**关键文件**：
- `pasco/models/hyperbolic/uncertainty.py` (新增)
- `pasco/loss/uncertainty_loss.py` (新增)
- `tests/hyperbolic/test_uncertainty.py` (新增)

### Phase 4: 整合到BodyNetHyperbolic
**目标**：将uncertainty功能整合到现有模型

**任务**：
- [ ] 扩展 `BodyNetHyperbolic` 添加 `forward_with_uncertainty()`
- [ ] 添加可选的uncertainty正则化到训练loss
- [ ] 支持输出每个voxel的uncertainty map
- [ ] 编写集成测试

**关键文件**：
- `pasco/models/body_net_hyperbolic.py` (修改)
- `tests/hyperbolic/test_uncertainty_integration.py` (新增)

### Phase 5: 验证与应用
**目标**：验证功能正确性和实用价值

**任务**：
- [ ] 运行完整训练验证无NaN/Inf
- [ ] 可视化uncertainty map
- [ ] 验证uncertainty与分割错误的相关性（高uncertainty区域应对应错误预测）
- [ ] 简单OOD检测demo（可选）

---

## 6. 风险与注意事项

| 风险 | 缓解措施 |
|------|----------|
| Lorentz Busemann数学复杂 | 先实现Poincaré版本验证概念 |
| 性能可能下降 | 保留现有ranking loss作为备选 |
| 数值稳定性问题 | 添加裁剪和正则化 |
| 计算开销增加 | 仅在需要uncertainty时计算 |

---

## 7. 结论与下一步

### 可行性评估：高度可行 ✓

**核心发现**：
1. HIPO论文证明了Busemann函数用于uncertainty估计的有效性
2. 关键公式可直接在Lorentz模型中实现
3. 与PaSCo现有架构兼容，无需大规模重构

### 选定实施方向
- **目标**：Uncertainty估计（分割置信度、OOD检测潜力）
- **技术方案**：直接在Lorentz模型中实现Busemann函数

### 新增文件清单
```
pasco/models/hyperbolic/
├── ideal_prototype.py      # Ideal Prototypes (光锥上)
└── uncertainty.py          # Uncertainty估计器

pasco/loss/
└── uncertainty_loss.py     # Uncertainty正则化Loss

tests/hyperbolic/
├── test_lorentz_busemann.py
├── test_ideal_prototype.py
└── test_uncertainty.py
```

### 修改文件清单
```
pasco/models/hyperbolic/lorentz_ops.py   # 添加lorentz_busemann()
pasco/models/body_net_hyperbolic.py      # 添加uncertainty输出接口
```

### 验证方法
1. 单元测试验证Busemann函数数学正确性
2. 集成测试验证训练稳定性（无NaN）
3. 可视化验证uncertainty与分割错误的相关性
