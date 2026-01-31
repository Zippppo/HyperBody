# HyperPath vs PaSCo 双曲投影头对比分析

## 1. HyperPath 的投影流程（更完善）

HyperPath 使用**三阶段投影架构**：

```
原始特征 (B, D)
    ↓
1. Euclidean投影层（2层MLP: Linear → ReLU → Dropout → Linear）
    ↓
欧氏空间特征 (B, D)
    ↓
2. 缩放（x * exp(alpha)，alpha是可学习参数）
    ↓
3. exp_map0() 指数映射到双曲面
    ↓
双曲特征 (B, D)
```

### 关键代码（hypermil.py:130-139）

```python
def hyper_proj(self, x, alpha):
    # 第一步：缩放特征
    x_hp = x * alpha.exp()

    # 第二步：指数映射到双曲面
    with torch.autocast(x_hp.device.type, dtype=torch.float32):
        x_hp = lorentz.exp_map0(x_hp, self.curv.exp())

    return x_hp
```

## 2. HyperPath 的关键特性

| 特性 | 实现 | 作用 |
|------|------|------|
| 可学习曲率 | `self.curv = nn.Parameter(log(1.0))` | 自适应调整双曲空间曲率 |
| 可学习缩放 | `alpha = nn.Parameter(log(embed_dim^-0.5))` | 控制投影前特征范数 |
| 2层MLP投影 | `Linear→ReLU→Dropout→Linear` | 更强的特征变换能力 |
| 曲率约束 | `clamp(curv, min=log(0.1), max=log(10))` | 防止曲率过大/过小 |
| Alpha约束 | `clamp(alpha, max=0)` → `exp(alpha) ≤ 1` | 确保特征不会过度放大 |
| FP32投影 | `torch.autocast(..., dtype=torch.float32)` | 数值稳定性 |

## 3. PaSCo 当前 MVP 版本（简化）

```
原始特征 (B, C, H, W, D)
    ↓
1. 1x1 Conv3D（单层线性投影）
    ↓
欧氏空间特征 (B, embed_dim, H, W, D)
    ↓
2. exp_map0() 直接投影到双曲面
    ↓
双曲特征 (B, embed_dim, H, W, D)
```

### 当前代码（projection_head.py:114-136）

```python
def forward(self, features):
    x = self.conv(features)  # 仅单层1x1卷积
    x = x.permute(0, 2, 3, 4, 1)
    x = exp_map0(x, curv=self.curv)  # 直接投影，curv固定
    x = x.permute(0, 4, 1, 2, 3)
    return x
```

## 4. 主要差距

| 方面 | PaSCo MVP | HyperPath | 建议改进 |
|------|-----------|-----------|----------|
| 投影网络 | 单层1x1 Conv | 2层MLP + Dropout | 增加网络深度 |
| 曲率 | 固定 `curv=1.0` | 可学习 + 约束 | 改为可学习参数 |
| 缩放 | 无 | 可学习 alpha | 添加缩放参数 |
| 数值稳定性 | 基础 | FP32强制 + 约束 | 添加autocast |
| 参数初始化 | Xavier | `embed_dim^-0.5` | 优化初始化 |

## 5. exp_map0 实现对比

PaSCo 的 `lorentz_ops.py` 中的 `exp_map0` 实现与 HyperPath **完全一致**：

```python
# 两者都使用相同的数学公式：
# exp_map0(v) = sinh(sqrt(c) * ||v||) * v / (sqrt(c) * ||v||)
```

**核心区别在于投影头的预处理，而非 exp_map 本身。**

## 6. 升级建议

如果要升级 `LorentzProjectionHead`，可以参考 HyperPath 添加：

1. **可学习曲率**：`self.curv = nn.Parameter(torch.tensor(1.0).log())`
2. **可学习缩放**：`self.alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())`
3. **更深投影网络**：从单层 Conv 升级为 2 层 MLP
4. **参数约束**：在 forward 中添加 clamp
5. **FP32精度**：在 exp_map 处强制使用 float32

## 7. 参考文件位置

| 功能 | 文件路径 |
|------|----------|
| HyperPath 双曲数学 | `REF/ref_repos/HyperPath/models/lorentz.py` |
| HyperPath 投影头 | `REF/ref_repos/HyperPath/models/hypermil.py` |
| PaSCo Lorentz操作 | `pasco/models/hyperbolic/lorentz_ops.py` |
| PaSCo 投影头 | `pasco/models/hyperbolic/projection_head.py` |
