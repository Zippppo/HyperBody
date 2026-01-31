# 实施计划：Hyperbolic模块分类输出

## 目标

让Hyperbolic模块能够基于测地距离输出分类结果，与现有CE分类头形成互补。

## 核心思路

对于每个voxel embedding，计算其到70个类别embedding的测地距离，距离最近的类别作为预测结果。

```
voxel_emb [B, D, H, W, Z] × label_emb [70, D] → distances [B, 70, H, W, Z] → hyp_logits (负距离)
```

---

## Phase 1: 添加批量距离计算函数

**文件**: `pasco/models/hyperbolic/lorentz_ops.py`

添加 `pairwise_dist_voxel` 函数，专门处理5D voxel tensor：

```python
def pairwise_dist_voxel(
    x: Tensor,  # [B, D, H, W, Z] voxel embeddings
    y: Tensor,  # [N, D] class embeddings
    curv: float = 1.0,
    eps: float = 1e-7
) -> Tensor:  # [B, N, H, W, Z] distances
    """计算voxel到各类别的测地距离"""
    B, D, H, W, Z = x.shape
    N = y.shape[0]

    # Reshape: [B, D, H, W, Z] → [B*H*W*Z, D]
    x_flat = x.permute(0, 2, 3, 4, 1).reshape(-1, D)

    # 计算时间分量
    x_time = torch.sqrt(1/curv + (x_flat ** 2).sum(-1, keepdim=True))
    y_time = torch.sqrt(1/curv + (y ** 2).sum(-1, keepdim=True))

    # Lorentz内积: [B*H*W*Z, N]
    lorentz_inner = x_flat @ y.T - x_time @ y_time.T

    # 测地距离
    c_xyl = -curv * lorentz_inner
    distances = torch.acosh(torch.clamp(c_xyl, min=1+eps)) / curv**0.5

    # Reshape: [B*H*W*Z, N] → [B, N, H, W, Z]
    return distances.reshape(B, H, W, Z, N).permute(0, 4, 1, 2, 3)
```

---

## Phase 2: 在BodyNetHyperbolic中添加分类方法

**文件**: `pasco/models/body_net_hyperbolic.py`

### 2.1 添加温度参数

在 `__init__` 中添加：
```python
# 新参数
use_hyp_classification: bool = False,
hyp_temperature: float = 0.1,
```

初始化：
```python
self.use_hyp_classification = use_hyp_classification
if use_hyp_classification:
    self.register_buffer('hyp_temperature', torch.tensor(hyp_temperature))
```

### 2.2 添加分类方法

```python
def compute_hyperbolic_logits(self, voxel_embeddings: Tensor) -> Tensor:
    """
    基于测地距离计算分类logits。

    Args:
        voxel_embeddings: [B, D, H, W, Z] 双曲空间中的voxel特征
    Returns:
        hyp_logits: [B, N_classes, H, W, Z] 分类logits
    """
    label_emb = self.label_emb.get_real_embeddings()  # [70, D]
    distances = pairwise_dist_voxel(voxel_embeddings, label_emb, curv=CURV)

    # 距离转logits：负距离/温度
    temp = torch.clamp(self.hyp_temperature, min=0.01)
    hyp_logits = -distances / temp
    return hyp_logits

def forward_with_all_logits(self, x):
    """返回CE logits和hyperbolic logits"""
    ce_logits, voxel_emb = self.forward_with_hyperbolic(x)
    hyp_logits = self.compute_hyperbolic_logits(voxel_emb)
    return ce_logits, hyp_logits, voxel_emb
```

---

## Phase 3: 推理方法

**文件**: `pasco/models/body_net_hyperbolic.py`

**设计决策**：分别输出CE和hyperbolic两个头的预测结果，供用户分析和比较。

```python
def predict(self, x):
    """
    推理时返回两个头的预测结果。

    Returns:
        ce_pred: [B, H, W, Z] CE头预测
        hyp_pred: [B, H, W, Z] hyperbolic头预测
        ce_logits: [B, N, H, W, Z] CE logits
        hyp_logits: [B, N, H, W, Z] hyperbolic logits
    """
    ce_logits, hyp_logits, _ = self.forward_with_all_logits(x)
    ce_pred = ce_logits.argmax(dim=1)
    hyp_pred = hyp_logits.argmax(dim=1)
    return ce_pred, hyp_pred, ce_logits, hyp_logits

def predict_hyperbolic(self, x):
    """仅使用hyperbolic头推理"""
    _, voxel_emb = self.forward_with_hyperbolic(x)
    hyp_logits = self.compute_hyperbolic_logits(voxel_emb)
    return hyp_logits.argmax(dim=1)
```

---

## Phase 4: 验证步骤中记录hyperbolic预测指标

在validation_step中添加hyperbolic头的IoU统计，便于比较两个头的性能：

```python
# validation_step中添加
if self.use_hyp_classification:
    hyp_logits = self.compute_hyperbolic_logits(voxel_emb)
    hyp_pred = hyp_logits.argmax(dim=1)

    # 计算hyperbolic头的IoU
    hyp_iou_per_class, hyp_valid_mask = self.compute_iou(hyp_pred, labels, self.n_classes)

    # 累积
    if self.val_hyp_iou_sum is None:
        self.val_hyp_iou_sum = torch.zeros(self.n_classes, device=self.device)
        self.val_hyp_iou_count = torch.zeros(self.n_classes, device=self.device)

    self.val_hyp_iou_sum += hyp_iou_per_class
    self.val_hyp_iou_count += hyp_valid_mask.float()
```

---

## Phase 5: CLI参数更新

**文件**: `scripts/body/train_body.py`

```python
# Hyperbolic classification arguments
parser.add_argument("--use_hyp_classification", action="store_true",
                    help="Enable hyperbolic distance-based classification output")
parser.add_argument("--hyp_temperature", type=float, default=0.1,
                    help="Temperature for distance-to-logits conversion")
```

---

## 关键文件

| 文件 | 修改内容 |
|------|---------|
| `pasco/models/hyperbolic/lorentz_ops.py` | 添加 `pairwise_dist_voxel` |
| `pasco/models/body_net_hyperbolic.py` | 添加分类方法和推理方法 |
| `scripts/body/train_body.py` | 添加CLI参数 |

---

## 验证方案

1. **单元测试**: 验证 `pairwise_dist_voxel` 输出形状和数值稳定性
2. **集成测试**: 验证 `compute_hyperbolic_logits` 梯度流
3. **功能测试**: 比较CE头和hyperbolic头的预测一致性
4. **性能测试**: 验证内存使用和计算效率

```bash
# 测试命令
python -c "
from pasco.models.body_net_hyperbolic import BodyNetHyperbolic
import torch

model = BodyNetHyperbolic(n_classes=70, use_hyp_classification=True)
x = torch.randn(1, 1, 32, 32, 32)

# 测试forward_with_all_logits
ce_logits, hyp_logits, voxel_emb = model.forward_with_all_logits(x)
print(f'CE logits: {ce_logits.shape}')      # [1, 70, 32, 32, 32]
print(f'Hyp logits: {hyp_logits.shape}')    # [1, 70, 32, 32, 32]

# 测试predict
ce_pred, hyp_pred, _, _ = model.predict(x)
print(f'CE pred: {ce_pred.shape}')          # [1, 32, 32, 32]
print(f'Hyp pred: {hyp_pred.shape}')        # [1, 32, 32, 32]

# 比较两个头的预测一致性
agreement = (ce_pred == hyp_pred).float().mean()
print(f'Prediction agreement: {agreement:.2%}')
"
```

---

## 设计决策（已确认）

| 问题 | 决策 |
|------|------|
| 温度参数 | 固定值 0.1 |
| 训练策略 | 不添加额外CE loss，仅使用现有ranking loss |
| 推理输出 | 分别输出CE和hyperbolic两个头的预测结果 |
