# 多分类Dice Loss实现计划（优化版）

## 概述

**目标**: 为PaSCo人体分割添加多分类Dice Loss，改进71个器官分割的性能，处理13,297倍的类不平衡。

**相比初始计划的关键改进**：
1. 内存高效的逐类迭代计算（每次前向传播节省~2.2GB）
2. 一致的元组返回模式（如BodyNetHyperbolic）
3. 完整的class_weights支持与CE loss对齐
4. 全面的单元测试

---

## 文件修改总结

| 文件 | 修改内容 |
|------|---------|
| `pasco/loss/losses.py` | 添加 `multi_class_dice_loss` 函数 |
| `pasco/models/body_net.py` | 添加参数 + 修改 `compute_loss`、`training_step`、`validation_step` |
| `pasco/models/body_net_hyperbolic.py` | 传递dice参数到父类 + 更新日志记录 |
| `scripts/body/train_body.py` | 添加CLI参数 + 更新模型创建 + 配置保存 |
| `tests/loss/test_dice_loss.py` | 新增测试文件 |

---

## 步骤1：在 `pasco/loss/losses.py` 中添加 `multi_class_dice_loss`

**位置**: 第68行之后（现有 `dice_loss` 函数之后）

```python
def multi_class_dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    ignore_index: int = 0,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Multi-class Dice loss for 3D segmentation.

    Memory-efficient: iterates per class instead of full one-hot tensor.

    Args:
        inputs: [B, C, H, W, D] logits (before softmax)
        targets: [B, H, W, D] ground truth class indices
        class_weights: [C] optional per-class weights
        ignore_index: class index to exclude (default 0)
        smooth: smoothing factor (default 1.0)

    Returns:
        Scalar Dice loss
    """
    B, C, H, W, D = inputs.shape
    device = inputs.device

    # Softmax probabilities
    probs = F.softmax(inputs, dim=1)

    # Flatten spatial dimensions
    probs_flat = probs.view(B, C, -1)  # [B, C, N]
    targets_flat = targets.view(B, -1)  # [B, N]

    # Valid mask (exclude ignore_index)
    valid_mask = (targets_flat != ignore_index).float()  # [B, N]

    # Compute per-class Dice
    dice_scores = []
    weights_used = []

    for cls in range(C):
        if cls == ignore_index:
            continue

        # Binary mask for this class
        target_cls = (targets_flat == cls).float()  # [B, N]
        prob_cls = probs_flat[:, cls, :]  # [B, N]

        # Apply valid mask
        target_cls = target_cls * valid_mask
        prob_cls = prob_cls * valid_mask

        # Dice coefficient: 2*|A∩B| / (|A| + |B|)
        intersection = (prob_cls * target_cls).sum()
        cardinality = prob_cls.sum() + target_cls.sum()

        dice = (2.0 * intersection + smooth) / (cardinality + smooth)
        dice_scores.append(dice)

        if class_weights is not None:
            weights_used.append(class_weights[cls])

    if len(dice_scores) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    dice_scores = torch.stack(dice_scores)

    # Weighted average
    if class_weights is not None and len(weights_used) > 0:
        weights_tensor = torch.stack(weights_used)
        dice_loss = 1.0 - (dice_scores * weights_tensor).sum() / weights_tensor.sum()
    else:
        dice_loss = 1.0 - dice_scores.mean()

    return dice_loss
```

---

## 步骤2：修改 `pasco/models/body_net.py`

### 2.1 更新 `__init__`（约第32行）

添加参数：
```python
def __init__(
    self,
    n_classes=71,
    in_channels=1,
    base_channels=32,
    lr=1e-4,
    weight_decay=0.0,
    class_weights=None,
    ignore_index=0,
    use_light_model=False,
    warmup_epochs=5,
    max_epochs=100,
    use_dice_loss=False,      # 新增
    dice_weight=0.5,          # 新增
):
```

在 `__init__` 方法体中存储：
```python
self.use_dice_loss = use_dice_loss
self.dice_weight = dice_weight
```

### 2.2 修改 `compute_loss`（约第83行）

```python
def compute_loss(self, logits, labels):
    """Compute CE loss with optional Dice loss."""
    from pasco.loss.losses import multi_class_dice_loss

    criterion = nn.CrossEntropyLoss(
        weight=self.class_weights,
        ignore_index=self.ignore_index,
    )
    ce_loss = criterion(logits, labels)

    if not self.use_dice_loss:
        return ce_loss

    dice_loss = multi_class_dice_loss(
        logits, labels,
        class_weights=self.class_weights,
        ignore_index=self.ignore_index,
    )

    total_loss = ce_loss + self.dice_weight * dice_loss
    return total_loss, ce_loss, dice_loss
```

### 2.3 修改 `training_step`（约第127行）

```python
def training_step(self, batch, batch_idx):
    occupancy = batch["occupancy"]
    labels = batch["labels"]

    logits = self(occupancy)
    loss_result = self.compute_loss(logits, labels)

    if self.use_dice_loss:
        loss, ce_loss, dice_loss = loss_result
        self.log("train/ce_loss", ce_loss, sync_dist=True)
        self.log("train/dice_loss", dice_loss, sync_dist=True)
    else:
        loss = loss_result

    # Accuracy computation (existing code)
    pred = logits.argmax(dim=1)
    valid_mask = labels != self.ignore_index
    accuracy = (pred[valid_mask] == labels[valid_mask]).float().mean()

    self.log("train/loss", loss, prog_bar=True, sync_dist=True)
    self.log("train/accuracy", accuracy, prog_bar=True, sync_dist=True)

    return loss
```

### 2.4 类似修改 `validation_step`

采用相同的模式：如果 `use_dice_loss` 为真，则解包元组，记录 `val/ce_loss` 和 `val/dice_loss`。

---

## 步骤3：修改 `pasco/models/body_net_hyperbolic.py`

### 3.1 更新 `__init__`（约第56行）

添加参数并传递给父类：
```python
def __init__(
    self,
    # ... existing params ...
    use_dice_loss=False,      # 新增
    dice_weight=0.5,          # 新增
    **kwargs
):
    super().__init__(
        n_classes=n_classes,
        use_dice_loss=use_dice_loss,
        dice_weight=dice_weight,
        **kwargs
    )
```

### 3.2 更新 `training_step`（约第150行）

```python
def training_step(self, batch, batch_idx):
    occupancy = batch["occupancy"]
    labels = batch["labels"]

    logits, voxel_emb = self.forward_with_hyperbolic(occupancy)

    # Handle CE + optional Dice
    loss_result = self.compute_loss(logits, labels)
    if self.use_dice_loss:
        ce_dice_loss, ce_loss, dice_loss = loss_result
    else:
        ce_loss = loss_result
        dice_loss = None
        ce_dice_loss = ce_loss

    # Hyperbolic ranking loss
    real_label_emb = self.label_emb.get_real_embeddings()
    hyp_loss = self.hyp_loss_fn(voxel_emb, labels, real_label_emb)

    total_loss = ce_dice_loss + self.hyperbolic_weight * hyp_loss

    # Entailment cone loss (existing code)
    # ...

    # Logging
    self.log("train/loss", total_loss, prog_bar=True, sync_dist=True)
    self.log("train/ce_loss", ce_loss, sync_dist=True)
    if dice_loss is not None:
        self.log("train/dice_loss", dice_loss, sync_dist=True)
    self.log("train/hyp_loss", hyp_loss, sync_dist=True)
    # ... rest of logging ...
```

---

## 步骤4：更新 `scripts/body/train_body.py`

### 4.1 添加CLI参数（约第262行）

```python
# Dice loss
parser.add_argument("--use_dice_loss", action="store_true",
                    help="Enable multi-class Dice loss")
parser.add_argument("--dice_weight", type=float, default=0.5,
                    help="Weight for Dice loss (default 0.5)")
```

### 4.2 更新模型创建（约第376行）

在 `BodyNet` 和 `BodyNetHyperbolic` 实例化时添加：
```python
use_dice_loss=args.use_dice_loss,
dice_weight=args.dice_weight,
```

### 4.3 更新配置保存（约第193行）

```python
"loss": {
    "use_class_weights": args.use_class_weights,
    "weight_alpha": args.weight_alpha if args.use_class_weights else None,
    "use_dice_loss": args.use_dice_loss,
    "dice_weight": args.dice_weight if args.use_dice_loss else None,
},
```

---

## 步骤5：添加单元测试

**文件**: `tests/loss/test_dice_loss.py`

关键测试用例：
1. 输出是标量且范围正确 [0, 1]
2. 完美预测得到接近零的loss
3. 梯度正确流向logits
4. ignore_index=0 排除背景
5. class_weights支持正常工作
6. 大/小logits数值稳定
7. 在现实输入大小上无OOM

---

## 验证

### 快速验证
```bash
# 导入测试
python -c "from pasco.loss.losses import multi_class_dice_loss; print('OK')"

# 单元测试
pytest tests/loss/test_dice_loss.py -v
```

### 训练测试
```bash
# 使用Dice loss进行短期训练
python scripts/body/train_body.py \
    --dataset_root Dataset/voxel_data \
    --batch_size 2 \
    --max_epochs 2 \
    --use_dice_loss \
    --dice_weight 0.5 \
    --use_class_weights \
    --gpuids 0

# 验证日志包含train/dice_loss和val/dice_loss
```

### 完整训练（推荐）
```bash
# BodyNet + Dice
python scripts/body/train_body.py \
    --dataset_root Dataset/voxel_data \
    --use_dice_loss --dice_weight 0.5 \
    --use_class_weights --weight_alpha 0.9 \
    --exp_name body_dice

# BodyNetHyperbolic + Dice
python scripts/body/train_body.py \
    --dataset_root Dataset/voxel_data \
    --use_hyperbolic --hyp_weight 0.1 \
    --use_dice_loss --dice_weight 0.3 \
    --use_class_weights \
    --exp_name body_hyp_dice
```

---

## 设计基础

| 决策 | 基础 |
|----------|-----------|
| 逐类迭代 | 避免分配2.2GB的one-hot张量 |
| 启用时返回元组 | 与BodyNetHyperbolic模式一致 |
| 默认dice_weight=0.5 | CE仍然占主导，Dice作为正则化器 |
| smooth=1.0 | 标准Dice平滑以保证稳定性 |
| Dice中的类权重 | 与CE加权策略对齐 |
