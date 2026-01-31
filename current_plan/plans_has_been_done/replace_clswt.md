# 计划：用有效样本数方法替换交叉熵类权重

## 背景

### 当前实现
- **位置**: `pasco/data/body/params.py` - `compute_class_weights()`
- **公式**: `w_c = 1.0 / (freq_c ^ alpha)`
- **参数**: `--weight_alpha` (默认值 0.5)
- **问题**: 简单的反频率加权对极端类不平衡处理不足（数据集不平衡比达13,297倍）

### 目标：有效样本数方法
- **参考文献**: Cui et al. "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019)
- **公式**: `w_c = (1 - β) / (1 - β^{N_c})`
- **优点**: 对高度不平衡的数据集更稳定，有基于有效样本数的理论基础
- **参考实现**: `pasco/loss/lorentz_loss.py:87-125` (用于ranking loss采样，独立模块)

### 设计决策
1. **独立实现**: 在 `params.py` 中添加新函数，与 `lorentz_loss.py` 分离以避免耦合
2. **权重归一化**: 将权重归一化为 `n_valid` 类的和（与现有 `compute_class_weights` 保持一致）
   - 不影响相对类权重或最终模型性能
   - 保持loss规模一致，切换方法时无需调整学习率
3. **类0处理**: 类0为 `outside_body`，必须完全忽略（权重=0）
4. **Beta参数**: 初始实验使用默认值 `beta=0.999`

---

## 实现计划

### 步骤1：在 `params.py` 中添加有效样本数权重函数

**文件**: `pasco/data/body/params.py`

在 `compute_class_weights()` 函数之后添加新函数：

```python
def compute_effective_number_weights(frequencies, beta=0.999, ignore_index=0):
    """
    Compute class weights using Effective Number of Samples.

    Reference: Cui et al. "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019)

    Formula: w_c = (1 - β) / (1 - β^{N_c})

    Args:
        frequencies: [N_CLASSES] array of class frequencies (sample counts)
        beta: Effective number parameter, typically 0.9-0.9999
              - Higher beta = more weight to rare classes
              - beta -> 1: approaches inverse frequency
              - beta -> 0: uniform weights
        ignore_index: Class index to ignore (weight=0), default 0 for outside_body

    Returns:
        weights: [N_CLASSES] array of class weights (normalized to sum=n_valid, ignore_index has weight 0)
    """
    freq = np.clip(frequencies.astype(np.float64), 1, None)

    # Mask for valid classes (exclude ignore_index)
    valid_mask = np.ones(len(freq), dtype=bool)
    if ignore_index is not None and 0 <= ignore_index < len(freq):
        valid_mask[ignore_index] = False

    # Effective Number: E_n = (1 - β^n) / (1 - β)
    # Weight is inverse: w = (1 - β) / (1 - β^n)
    weights = np.zeros_like(freq)
    valid_freq = freq[valid_mask]
    effective_num = 1.0 - np.power(beta, valid_freq)
    valid_weights = (1.0 - beta) / effective_num

    # Zero frequency classes get zero weight
    valid_weights[valid_freq == 0] = 0.0

    # Normalize to sum to number of valid classes (consistent with compute_class_weights)
    n_valid = valid_mask.sum()
    weight_sum = valid_weights.sum()
    if weight_sum > 0:
        valid_weights = valid_weights / weight_sum * n_valid

    weights[valid_mask] = valid_weights

    return weights.astype(np.float32)
```

### 步骤2：更新训练脚本

**文件**: `scripts/body/train_body.py`

#### 2a. 在 `parse_args()` 中添加新参数（Loss部分）

```python
# Loss
parser.add_argument("--use_class_weights", action="store_true",
                    help="Use class-weighted loss")
parser.add_argument("--weight_method", type=str, default="effective_number",
                    choices=["inverse_freq", "effective_number"],
                    help="Class weight computation method")
parser.add_argument("--weight_alpha", type=float, default=0.5,
                    help="Class weight exponent (for inverse_freq method)")
parser.add_argument("--weight_beta", type=float, default=0.999,
                    help="Effective number beta parameter (for effective_number method)")
```

#### 2b. 更新import语句

```python
from pasco.data.body.params import (
    N_CLASSES, body_class_frequencies,
    compute_class_weights, compute_effective_number_weights
)
```

#### 2c. 在 `main()` 中更新权重计算逻辑

```python
# Compute class weights if requested
class_weights = None
if args.use_class_weights:
    print("\nComputing class weights...")
    if args.weight_method == "effective_number":
        class_weights = compute_effective_number_weights(
            body_class_frequencies, beta=args.weight_beta
        )
        print(f"Using Effective Number weights (beta={args.weight_beta})")
    else:
        class_weights = compute_class_weights(
            body_class_frequencies, alpha=args.weight_alpha
        )
        print(f"Using inverse frequency weights (alpha={args.weight_alpha})")
    print(f"Class weights - min: {class_weights.min():.4f}, max: {class_weights.max():.4f}, "
          f"mean: {class_weights.mean():.4f}, class_0: {class_weights[0]:.4f}")
```

#### 2d. 更新实验名称生成逻辑

```python
if args.use_class_weights:
    if args.weight_method == "effective_number":
        exp_name += f"_en{args.weight_beta}"
    else:
        exp_name += f"_cw{args.weight_alpha}"
```

#### 2e. 更新配置保存（loss部分）

```python
"loss": {
    "use_class_weights": args.use_class_weights,
    "weight_method": args.weight_method if args.use_class_weights else None,
    "weight_alpha": args.weight_alpha if args.use_class_weights and args.weight_method == "inverse_freq" else None,
    "weight_beta": args.weight_beta if args.use_class_weights and args.weight_method == "effective_number" else None,
},
```

---

## 需要修改的文件

| 文件 | 修改内容 |
|------|---------|
| `pasco/data/body/params.py` | 添加 `compute_effective_number_weights()` 函数 |
| `scripts/body/train_body.py` | 添加 `--weight_method`、`--weight_beta` 参数；更新权重计算逻辑 |

---

## 使用示例

```bash
# 使用有效样本数权重（推荐，使用 --use_class_weights 时的默认值）
python scripts/body/train_body.py --use_class_weights --weight_method effective_number --weight_beta 0.999

# 使用有效样本数权重（简化版，使用默认值）
python scripts/body/train_body.py --use_class_weights

# 使用传统的反频率权重
python scripts/body/train_body.py --use_class_weights --weight_method inverse_freq --weight_alpha 0.5
```

---

## 验证

### 1. 烟雾测试
实现后，运行快速验证：

```python
from pasco.data.body.params import body_class_frequencies, compute_class_weights, compute_effective_number_weights

# Compute both weight types
inv_weights = compute_class_weights(body_class_frequencies, alpha=0.5)
en_weights = compute_effective_number_weights(body_class_frequencies, beta=0.999)

# Verify:
# 1. Class 0 (outside_body) weight = 0
assert en_weights[0] == 0.0, "Class 0 should have weight 0"

# 2. Weights sum to n_valid (70 classes)
assert abs(en_weights.sum() - 70.0) < 0.01, "Weights should sum to 70"

# 3. Lower frequency -> higher weight
# Class 24 (freq=109292) should have higher weight than Class 1 (freq=1035425486)
assert en_weights[24] > en_weights[1], "Rare class should have higher weight"

print(f"Inverse freq weights: min={inv_weights.min():.4f}, max={inv_weights.max():.4f}")
print(f"Effective num weights: min={en_weights[en_weights>0].min():.4f}, max={en_weights.max():.4f}")
```

### 2. 集成测试
启动训练并验证：
- 权重计算正确（检查打印的统计信息）
- 类0权重为0
- 训练无错误运行

```bash
python scripts/body/train_body.py --dataset_root /path/to/data --use_class_weights --max_epochs 1
```