# 3D U-Net Baseline 详细实现计划

## 概述

**目标:** 实现3D U-Net baseline，从部分人体表面点云预测完整人体体素及70个解剖结构类别

**当前状态:** 项目为空白状态，数据集已就绪 (10,779样本)

**硬件:** 2× NVIDIA A100 40GB GPUs

---

## 实现步骤总览

| 步骤 | 文件 | 依赖 | 验证方法 |
|------|------|------|----------|
| 1 | `config.py`, `requirements.txt` | 无 | 导入并打印配置 |
| 2 | `data/voxelizer.py` | 步骤1 | 体素化一个样本，检查形状 |
| 3 | `data/dataset.py` | 步骤2 | 通过Dataset加载样本，检查张量形状 |
| 4a | `models/dense_block.py` | 步骤1 | 随机张量前向传播，验证输出通道数 |
| 4b | `models/unet3d.py` | 步骤4a | 随机张量前向传播 |
| 5 | `models/losses.py` | 步骤4b | 随机数据反向传播 |
| 6 | `utils/metrics.py` | 无 | 已知预测测试 |
| 7 | `utils/checkpoint.py` | 无 | 保存/加载往返测试 |
| 8 | `train.py` | 步骤1-7 | 完整训练1-2个epoch |

**注意:** 步骤4a/4b、5、6、7可并行实现

---

## 步骤1: 配置和依赖

### requirements.txt
```
torch>=2.0.0
numpy>=1.24.0
tensorboard>=2.12.0
tqdm>=4.65.0
```

### config.py
```python
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class Config:
    # 数据
    data_dir: str = "Dataset/voxel_data"
    split_file: str = "Dataset/dataset_split.json"
    num_classes: int = 70
    voxel_size: float = 4.0
    volume_size: Tuple[int, int, int] = (128, 96, 256)  # X, Y, Z

    # 模型
    in_channels: int = 1
    base_channels: int = 32
    num_levels: int = 4

    # Dense Bottleneck
    growth_rate: int = 32      # 每层新增 channels
    dense_layers: int = 4      # 密集块层数
    bn_size: int = 4           # 1×1×1 压缩倍数

    # 训练
    batch_size: int = 4  # 每GPU
    num_workers: int = 4
    epochs: int = 120
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # 损失
    ce_weight: float = 0.5
    dice_weight: float = 0.5

    # 学习率调度
    lr_patience: int = 10
    lr_factor: float = 0.5

    # 检查点
    checkpoint_dir: str = "checkpoints"
    save_every: int = 10
    log_dir: str = "runs"

    # GPU
    gpu_ids: List[int] = field(default_factory=lambda: [0, 1])

    # 恢复训练
    resume: str = ""
```

**验证:** 导入config，打印值，确认volume_size覆盖所有样本尺寸 (最大117×92×241 → 填充到128×96×256)

---

## 步骤2: 体素化模块

### data/voxelizer.py

**核心功能:**
1. `voxelize_point_cloud()` - 点云转二值占用网格
2. `pad_labels()` - 标签填充到固定尺寸

```python
def voxelize_point_cloud(
    sensor_pc: np.ndarray,       # (N, 3) float32
    grid_world_min: np.ndarray,  # (3,) float32
    grid_voxel_size: np.ndarray, # (3,) float32
    volume_size: tuple           # (128, 96, 256)
) -> np.ndarray:
    """
    转换步骤:
    1. 计算体素索引: idx = floor((pc - grid_world_min) / voxel_size)
    2. 裁剪索引到 [0, volume_size - 1]
    3. 创建二值体积，占用位置设为1

    返回: shape=volume_size, dtype=float32 (0.0 或 1.0)
    """

def pad_labels(
    voxel_labels: np.ndarray,  # (X, Y, Z) uint8, 可变尺寸
    volume_size: tuple          # (128, 96, 256)
) -> np.ndarray:
    """
    用0填充voxel_labels到固定volume_size
    Class 0 (inside_body_empty) 作为填充值

    返回: shape=volume_size, dtype=int64
    """
```

**验证:** 加载一个样本，体素化，验证占用数量大致匹配点数，验证填充后标签形状

---

## 步骤3: 数据集模块

### data/dataset.py

```python
class HyperBodyDataset(Dataset):
    def __init__(self, data_dir: str, split_file: str, split: str, volume_size: tuple):
        """
        Args:
            data_dir: Dataset/voxel_data/ 路径
            split_file: dataset_split.json 路径
            split: 'train', 'val', 或 'test'
            volume_size: (128, 96, 256)
        """

    def __getitem__(self, idx):
        # 1. 加载 .npz 文件
        # 2. 体素化点云 -> (1, 128, 96, 256) float32
        # 3. 填充标签 -> (128, 96, 256) int64
        # 返回 (input_tensor, label_tensor)
```

### data/transforms.py (占位符)
```python
# 基线版本不使用数据增强
# 后续可添加
```

**验证:** 创建数据集实例，加载样本，验证张量形状 `(1,128,96,256)` 和 `(128,96,256)`，检查batch_size=2的DataLoader

---

## 步骤4a: Dense Bottleneck 模块

### models/dense_block.py

**Dense Block 结构:**
```
输入: 256 channels (16×12×32)

Layer 1: BN→ReLU→Conv1×1×1(256→128) → BN→ReLU→Conv3×3×3(128→32) → 拼接: 288 ch
Layer 2: BN→ReLU→Conv1×1×1(288→128) → BN→ReLU→Conv3×3×3(128→32) → 拼接: 320 ch
Layer 3: BN→ReLU→Conv1×1×1(320→128) → BN→ReLU→Conv3×3×3(128→32) → 拼接: 352 ch
Layer 4: BN→ReLU→Conv1×1×1(352→128) → BN→ReLU→Conv3×3×3(128→32) → 拼接: 384 ch

输出: 384 channels (16×12×32)
```

**关键组件:**
```python
class DenseLayer(nn.Module):
    """单个密集层: BN→ReLU→Conv1×1×1→BN→ReLU→Conv3×3×3"""
    def __init__(self, in_channels: int, growth_rate: int, bn_size: int = 4):
        # 1×1×1: in_channels → bn_size * growth_rate (128)
        # 3×3×3: 128 → growth_rate (32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 返回新特征 (不含输入)

class DenseBlock(nn.Module):
    """密集块: 多个 DenseLayer 密集连接"""
    def __init__(self, in_channels: int, num_layers: int = 4,
                 growth_rate: int = 32, bn_size: int = 4):
        # 创建 num_layers 个 DenseLayer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 密集连接: 每层输出与所有前层拼接
        # 返回: 输入 + 所有层输出拼接
```

**可配置参数:**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_layers` | 4 | 密集块层数 |
| `growth_rate` | 32 | 每层新增 channels |
| `bn_size` | 4 | 1×1×1 压缩倍数 |

**验证:**
```python
block = DenseBlock(in_channels=256, num_layers=4, growth_rate=32)
x = torch.randn(1, 256, 16, 12, 32)
out = block(x)
assert out.shape == (1, 384, 16, 12, 32)
```

---

## 步骤4b: 3D U-Net 模型

### models/unet3d.py

**架构概览:**
```
输入: (B, 1, 128, 96, 256)

编码器:
  enc1: 1 → 32   (128×96×256)
  enc2: 32 → 64  (64×48×128)
  enc3: 64 → 128 (32×24×64)
  enc4: 128 → 256 (16×12×32)

Dense Bottleneck:
  DenseBlock: 256 → 384 (16×12×32)
  - 4层密集连接, growth_rate=32
  - 1×1×1 压缩 (bn_size=4)

解码器:
  dec4: 384+128 → 128 (32×24×64)  [注意: 输入从384+128=512]
  dec3: 128+64 → 64   (64×48×128)
  dec2: 64+32 → 32    (128×96×256)

最终: 32 → 70 (1×1×1 conv)

输出: (B, 70, 128, 96, 256)
```

**关键组件:**
```python
class ConvBlock(nn.Module):
    """两个 3×3×3 卷积 + BatchNorm + ReLU"""

class Encoder(nn.Module):
    """MaxPool3d(2) + ConvBlock"""

class Decoder(nn.Module):
    """Trilinear上采样 + 跳跃连接拼接 + ConvBlock"""

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=70, base_channels=32,
                 growth_rate=32, dense_layers=4, bn_size=4):
        # ...
        self.bottleneck = DenseBlock(256, dense_layers, growth_rate, bn_size)
        bottleneck_out = 256 + dense_layers * growth_rate  # 384
        self.decoder4 = Decoder(bottleneck_out + 128, 128)  # 512 → 128
```

**参数量:** 约6.9M

**验证:** 创建模型，传入随机张量 `(1,1,128,96,256)`，验证输出形状 `(1,70,128,96,256)`，检查DataParallel封装

---

## 步骤5: 损失函数

### models/losses.py

```python
class DiceLoss(nn.Module):
    """逐类别Dice损失，平均所有类别"""
    def forward(self, logits, targets):
        # logits: (B, C, X, Y, Z)
        # targets: (B, X, Y, Z)
        # 返回: 1 - mean_dice

class CombinedLoss(nn.Module):
    def __init__(self, num_classes=70, ce_weight=0.5, dice_weight=0.5, class_weights=None):
        """
        class_weights: (C,) 张量，逐类别CE权重
        """

    def forward(self, logits, targets):
        return ce_weight * CE_loss + dice_weight * Dice_loss
```

**类别权重计算:**
- 从训练集100个随机样本计算逆频率权重
- 使用 `1/sqrt(freq)` 并归一化
- 避免罕见类别权重过大

**验证:** 创建损失函数，随机logits和targets，验证产生标量且梯度可反向传播

---

## 步骤6: 评估指标

### utils/metrics.py

```python
class DiceMetric:
    """跨batch累积逐类别Dice分数"""

    def reset(self):
        # 重置累积器

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        # 累积 intersection, pred_sum, target_sum

    def compute(self):
        """返回 (逐类别Dice, 平均Dice)"""
        # 排除target中不存在的类别
```

**验证:** 创建metric，用已知预测更新，验证Dice值正确 (完美预测Dice≈1.0)

---

## 步骤7: 检查点工具

### utils/checkpoint.py

```python
def save_checkpoint(state, checkpoint_dir, filename):
    """保存检查点到指定目录"""

def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """加载检查点，恢复状态，返回 (start_epoch, best_dice)"""
```

**检查点内容:**
```python
{
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_dice': best_dice,
}
```

**验证:** 保存/加载往返测试

---

## 步骤8: 训练脚本

### train.py

**主要功能:**

```python
def compute_class_weights(dataset, num_classes, num_samples=100):
    """计算逐类别权重 (逆频率)"""

def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip):
    """训练一个epoch"""

@torch.no_grad()
def validate(model, loader, criterion, metric, device):
    """验证，返回 (val_loss, dice_per_class, mean_dice)"""

def main():
    """主入口"""
```

**命令行参数:**
```
--resume      恢复训练的检查点路径
--batch_size  每GPU批次大小 (默认4)
--epochs      训练轮数 (默认120)
--gpuids      GPU ID (默认 "0,1")
```

**训练流程:**
1. 加载配置和数据集
2. 计算类别权重 (100个样本)
3. 创建模型 (DataParallel封装)
4. 创建损失函数、优化器、调度器
5. 恢复检查点 (如指定)
6. 训练循环:
   - 训练一个epoch
   - 验证
   - 更新学习率调度器
   - TensorBoard日志
   - 保存检查点 (best/latest/定期)

**TensorBoard日志:**
- Loss/train, Loss/val
- Dice/mean
- LR
- 关键器官Dice (liver, heart, brain, lung, spine)

**检查点保存:**
- `latest.pth` - 每个epoch
- `best.pth` - 最佳mean_dice
- `epoch_N.pth` - 每10个epoch

---

## 关键挑战与解决方案

| 挑战 | 解决方案 |
|------|----------|
| 可变尺寸体积 | 零填充到128×96×256，class 0填充语义正确 |
| 类别不平衡 (73.8% class 0) | 逆频率CE权重 + Dice损失天然处理不平衡 |
| 内存限制 | batch_size=4/GPU，如OOM则降至2 |
| 多GPU训练 | DataParallel，保存时解包model.module |
| 训练中断恢复 | 完整保存/恢复 model、optimizer、scheduler、epoch、best_dice |

---

## 目录结构

```
HyperBody/
├── config.py
├── requirements.txt
├── train.py
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   ├── voxelizer.py
│   └── transforms.py
├── models/
│   ├── __init__.py
│   ├── unet3d.py
│   ├── dense_block.py     # Dense Bottleneck (DenseLayer, DenseBlock)
│   └── losses.py
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   └── checkpoint.py
├── checkpoints/           (训练时创建)
├── runs/                  (TensorBoard日志)
└── Dataset/               (已存在)
    ├── voxel_data/
    ├── dataset_split.json
    └── dataset_info.json
```

---

## 验证方法

### 单元测试 (每个模块完成后)
1. **config.py:** 导入并打印配置值
2. **voxelizer.py:** 加载样本，体素化，检查形状和值范围
3. **dataset.py:** 创建Dataset和DataLoader，检查张量形状
4a. **dense_block.py:** 验证 DenseBlock 输出形状 (256→384 channels)
4b. **unet3d.py:** 随机输入前向传播，检查输出形状
5. **losses.py:** 随机数据前向+反向传播
6. **metrics.py:** 已知预测计算Dice
7. **checkpoint.py:** 保存/加载往返测试

### 集成测试 (完成train.py后)
```bash
# 小规模测试 (1-2 epochs, 少量数据)
python train.py --epochs 2 --batch_size 2 --gpuids 0

# 完整训练
python train.py --gpuids 0,1
```

### 成功标准
- 训练损失持续下降
- 验证Dice > 0.3 (整体)
- 主要器官 (liver, lungs, heart) Dice > 0.5
- 无OOM错误
- 可成功恢复训练

