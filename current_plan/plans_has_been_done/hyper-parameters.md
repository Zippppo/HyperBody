# Lorentz双曲空间调参优化计划

## 核心问题

**目标**: 最终在完整数据上获得好的结果，小数据调参只是为了节省时间。

### 现状
| 数据规模 | 基线mIoU | Lorentz mIoU | 变化 |
|---------|---------|--------------|------|
| 完整数据 (3,222样本) | 0.2670 | 0.2810 | **+1.40%** ✓ |
| 10%数据 (322样本) | 0.1111 | 0.0989 | **-10.98%** ✗ |

### 核心矛盾
**10%数据无法预测Lorentz在完整数据上的效果** - 小数据调参失去了意义。

### 根本原因分析
1. **数据量不足**: 322样本无法学习32维双曲嵌入的复杂结构
2. **代理失效**: 10%数据上的最优参数可能在完整数据上并非最优
3. **效果反转**: Lorentz在小数据上有害，但在大数据上有益

---

## 新调参策略：25%数据为主代理

### 策略转变
既然10%数据对Lorentz调参无效，改用**25%数据(805样本)**作为调参代理：
- 数据量是10%的2.5倍，更接近完整数据特性
- 训练时间仍远小于完整数据
- 预期能更准确预测Lorentz效果

### 阶段一：25%数据基线建立
**目标**: 在25%数据上验证Lorentz是否恢复有效

| 实验ID | 配置 | 说明 |
|--------|------|------|
| A1 | 无Lorentz基线 | 建立25%数据基线 |
| A2 | Lorentz (完整数据配置) | hyp_weight=0.1, dim=32 |
| A3 | Lorentz + 轻量配置 | hyp_weight=0.05, dim=32 |

**预期结果**:
- 如果A2 > A1: 证明25%数据足够，Lorentz有效
- 如果A2 < A1: 可能需要进一步调整或使用更多数据

### 阶段二：Lorentz超参数微调（如果阶段一Lorentz有效）
**目标**: 在25%数据上优化Lorentz配置

| 参数 | 搜索范围 | 说明 |
|-----|---------|------|
| hyp_weight | 0.05, 0.1, 0.15 | 双曲损失权重 |
| hyp_embed_dim | 16, 32, 64 | 嵌入维度 |
| hyp_margin | 0.05, 0.1, 0.2 | 排名损失边际 |

### 阶段三：最优配置迁移到完整数据
**目标**: 将25%数据上的最优配置应用到完整数据

验证25%数据的调参结果是否能迁移到完整数据

---

## 实验命令

### 阶段一：25%数据基线
```bash
# A1: 25%数据基线（无Lorentz）
python scripts/body/HyperParam_train.py \
    --dataset_root Dataset/voxel_data \
    --split_dir Dataset/voxel_data_25pct \
    --exp_name A1_25pct_baseline \
    --log_dir logs_hp_search/25pct_phase1 \
    --max_epochs 30 \
    --lr 1e-4 \
    --batch_size 2 \
    --precision 16 \
    --gpuids 0

# A2: 25%数据 + Lorentz（使用完整数据的成功配置）
python scripts/body/HyperParam_train.py \
    --dataset_root Dataset/voxel_data \
    --split_dir Dataset/voxel_data_25pct \
    --exp_name A2_25pct_lorentz \
    --log_dir logs_hp_search/25pct_phase1 \
    --max_epochs 30 \
    --lr 1e-4 \
    --batch_size 2 \
    --use_hyperbolic \
    --hyp_weight 0.1 \
    --hyp_embed_dim 32 \
    --hyp_margin 0.1 \
    --precision 16 \
    --gpuids 0

# A3: 25%数据 + 轻量Lorentz
python scripts/body/HyperParam_train.py \
    --dataset_root Dataset/voxel_data \
    --split_dir Dataset/voxel_data_25pct \
    --exp_name A3_25pct_lorentz_light \
    --log_dir logs_hp_search/25pct_phase1 \
    --max_epochs 30 \
    --lr 1e-4 \
    --batch_size 2 \
    --use_hyperbolic \
    --hyp_weight 0.05 \
    --hyp_embed_dim 32 \
    --hyp_margin 0.1 \
    --precision 16 \
    --gpuids 0
```

### 阶段二：Lorentz超参数网格搜索（如果阶段一有效）
```bash
# 网格搜索: hyp_weight × hyp_embed_dim
for weight in 0.05 0.1 0.15; do
  for dim in 16 32 64; do
    python scripts/body/HyperParam_train.py \
      --dataset_root Dataset/voxel_data \
      --split_dir Dataset/voxel_data_25pct \
      --exp_name "25pct_w${weight}_d${dim}" \
      --log_dir logs_hp_search/25pct_phase2 \
      --max_epochs 30 \
      --use_hyperbolic \
      --hyp_weight $weight \
      --hyp_embed_dim $dim \
      --precision 16 \
      --gpuids 0
  done
done
```

---

## 决策树

```
阶段一结果
├── A2(Lorentz) > A1(基线)
│   └── ✓ 25%数据有效，进入阶段二微调
├── A2 ≈ A1 (差距<0.5%)
│   └── 尝试A3轻量配置，或增加正则化
└── A2 < A1
    └── 可能需要更多数据，考虑直接使用完整数据配置
```

---

## 成功标准

| 阶段 | 目标 | 说明 |
|-----|-----|------|
| 阶段一 | A2 > A1 | 验证25%数据上Lorentz有效 |
| 阶段二 | 找到最优配置 | hyp_weight, hyp_embed_dim组合 |
| 阶段三 | 完整数据验证 | 25%最优配置在完整数据上提升 |

---

## 验证方法

1. **TensorBoard对比**: 查看`logs_hp_search/25pct_*/`的训练曲线
2. **性能对比表**: 比较验证集mIoU
3. **过拟合检查**: 监控train-val gap，如果gap过大需要增加正则化

---

## 关键文件路径

| 文件 | 用途 |
|-----|-----|
| `scripts/body/HyperParam_train.py` | 超参数搜索脚本 |
| `scripts/body/train_body.py` | 主训练脚本 |
| `pasco/models/body_net_hyperbolic.py` | Lorentz模型定义 |
| `Dataset/voxel_data_25pct/` | 25%数据集划分 |
| `logs_hp_search/` | 超参数搜索日志目录 |

---

## 备选方案

如果25%数据上Lorentz仍然无效：
1. **延迟策略**: 先使用基线完整数据训练，后期加入Lorentz微调
2. **预训练策略**: 用基线模型预训练，再加Lorentz微调
3. **直接完整数据**: 放弃小数据调参，直接在完整数据上使用已验证的配置
