# Dense 3D CNN Bottleneck 设计

**日期:** 2026-02-01
**目标:** 在 3D U-Net 的 Bottleneck 层使用 Dense 3D CNN 增强特征复用

## 设计决策摘要

| 决策项 | 选择 |
|--------|------|
| Dense 结构类型 | 密集连接残差块 |
| 层数 | 4 层 |
| Growth Rate | 32 (可配置) |
| 1×1×1 压缩 | 使用 (bn_size=4) |
| 输出处理 | 直接 384 channels 给解码器 |

## 架构变化

### 原 Bottleneck

```
Encoder Level 4 (256 ch) → ConvBlock (256 ch) → Decoder Level 4
```

### 新 Dense Bottleneck

```
Encoder Level 4 (256 ch) → DenseBlock (4层, 输出384 ch) → Decoder Level 4
```

## DenseBlock 内部结构

```
输入: 256 channels

Layer 1: BN→ReLU→Conv1×1×1(256→128) → BN→ReLU→Conv3×3×3(128→32) → 拼接: 288 ch
Layer 2: BN→ReLU→Conv1×1×1(288→128) → BN→ReLU→Conv3×3×3(128→32) → 拼接: 320 ch
Layer 3: BN→ReLU→Conv1×1×1(320→128) → BN→ReLU→Conv3×3×3(128→32) → 拼接: 352 ch
Layer 4: BN→ReLU→Conv1×1×1(352→128) → BN→ReLU→Conv3×3×3(128→32) → 拼接: 384 ch

最终输出: 384 channels
```

### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_layers` | 4 | 密集块层数 |
| `growth_rate` | 32 | 每层新增 channels |
| `bn_size` | 4 | 1×1×1 压缩倍数 (128 = 4 × 32) |

## 解码器适配

只需修改 Decoder Level 4（第一个解码层）：

```
原: 上采样(256 ch) + 跳跃连接(128 ch) → ConvBlock(384→128 ch)
新: 上采样(384 ch) + 跳跃连接(128 ch) → ConvBlock(512→128 ch)
```

## 参数量对比

| 组件 | 原参数量 | 新参数量 | 变化 |
|------|----------|----------|------|
| Bottleneck | ~1.2M | ~0.9M | -25% |
| Decoder Level 4 | ~1.3M | ~1.7M | +30% |
| **总计** | ~19M | ~19.3M | +1.5% |

## 内存影响

- Bottleneck 特征图: 384 × 16 × 12 × 32 ≈ 2.4M (原 1.6M)
- 增加约 50%，但该层尺寸最小，影响可控
- 预计仍可维持 batch_size=4/GPU

## 代码实现

### 新增 `models/dense_block.py`

```python
import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    """单个密集层: BN→ReLU→Conv1×1×1→BN→ReLU→Conv3×3×3"""

    def __init__(self, in_channels: int, growth_rate: int, bn_size: int = 4):
        super().__init__()
        intermediate_channels = bn_size * growth_rate  # 128

        self.block = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, intermediate_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(intermediate_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(intermediate_channels, growth_rate, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DenseBlock(nn.Module):
    """密集块: 多个 DenseLayer 密集连接"""

    def __init__(
        self,
        in_channels: int,
        num_layers: int = 4,
        growth_rate: int = 32,
        bn_size: int = 4,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = DenseLayer(
                in_channels=in_channels + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
            )
            self.layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            new_feat = layer(torch.cat(features, dim=1))
            features.append(new_feat)
        return torch.cat(features, dim=1)
```

### 修改 `models/unet3d.py`

```python
from models.dense_block import DenseBlock

class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 70,
        base_channels: int = 32,
        growth_rate: int = 32,
        dense_layers: int = 4,
        bn_size: int = 4,
    ):
        super().__init__()
        # ... 编码器保持不变 ...

        # 替换 bottleneck
        self.bottleneck = DenseBlock(
            in_channels=256,
            num_layers=dense_layers,
            growth_rate=growth_rate,
            bn_size=bn_size,
        )

        # 计算 bottleneck 输出通道数
        bottleneck_out = 256 + dense_layers * growth_rate  # 384

        # 修改 decoder4 输入通道
        self.decoder4 = Decoder(bottleneck_out + 128, 128)  # 512 → 128

        # ... 其余解码器保持不变 ...
```

### 配置项新增 (`config.py`)

```python
# Dense Bottleneck
growth_rate: int = 32
dense_layers: int = 4
bn_size: int = 4
```

## 验证方法

```python
# 测试 DenseBlock
block = DenseBlock(in_channels=256, num_layers=4, growth_rate=32)
x = torch.randn(1, 256, 16, 12, 32)
out = block(x)
assert out.shape == (1, 384, 16, 12, 32)

# 测试完整 UNet3D
model = UNet3D(growth_rate=32, dense_layers=4)
x = torch.randn(1, 1, 128, 96, 256)
out = model(x)
assert out.shape == (1, 70, 128, 96, 256)
```

## 后续可调参数

- `growth_rate`: 可尝试 48 或 64 以增强表达能力
- `dense_layers`: 可尝试 3 或 5 层
- `bn_size`: 通常保持 4 即可
