# 训练日志文件格式说明

改进后的 `train_body.py` 会自动保存完整的训练信息到以下文件：

## 📁 输出文件结构

```
logs/
└── body_unet_bs2_lr0.0001_ch32_cw0.5_aug/
    ├── training_config.json      # 训练配置（启动时保存）
    ├── training_log.json          # 训练过程记录（每个epoch更新）
    ├── training_summary.json      # 训练总结（完成时保存）
    ├── checkpoints/
    │   └── best_model.ckpt        # 最佳模型权重（仅保存最佳）
    └── version_X/                 # TensorBoard日志
        └── events.out.tfevents.*
```

---

## 1️⃣ `training_config.json` - 训练配置

**保存时机**：训练开始时
**内容**：完整的训练参数配置

```json
{
  "experiment": {
    "name": "body_unet",
    "start_time": "2024-01-15 10:30:00",
    "log_dir": "logs"
  },
  "data": {
    "dataset_root": "voxel-output/merged_data",
    "target_size": [160, 160, 256],
    "batch_size": 2,
    "num_workers": 4,
    "data_aug": true
  },
  "model": {
    "n_classes": 71,
    "base_channels": 32,
    "use_light_model": false,
    "total_parameters": 20458391
  },
  "training": {
    "lr": 0.0001,
    "weight_decay": 0.0,
    "max_epochs": 100,
    "warmup_epochs": 5,
    "n_gpus": 2,
    "precision": "16",
    "seed": 42
  },
  "loss": {
    "use_class_weights": true,
    "weight_alpha": 0.5,
    "class_weights_stats": {
      "min": 0.123,
      "max": 5.678,
      "mean": 1.234,
      "std": 0.987
    }
  },
  "system": {
    "pytorch_version": "1.13.0+cu117",
    "cuda_available": true,
    "cuda_version": "11.7"
  }
}
```

---

## 2️⃣ `training_log.json` - 训练过程记录

**保存时机**：每个epoch后自动更新
**内容**：完整的训练历史曲线

```json
{
  "training_history": {
    "epoch": [0, 1, 2, 3, 4, 5],
    "train_loss": [2.456, 1.987, 1.654, 1.432, 1.298, 1.187],
    "train_accuracy": [0.234, 0.356, 0.445, 0.512, 0.567, 0.612],
    "val_loss": [2.123, 1.876, 1.598, 1.387, 1.245, 1.134],
    "val_mIoU": [0.156, 0.234, 0.298, 0.345, 0.389, 0.421],
    "learning_rate": [0.00002, 0.00004, 0.00006, 0.00008, 0.0001, 0.0001]
  },
  "best_metrics": {
    "best_val_mIoU": 0.421,
    "best_epoch": 5
  },
  "last_update": "2024-01-15 12:45:30"
}
```

**用途**：
- 绘制训练曲线
- 分析训练过程
- 监控过拟合/欠拟合
- 学习率调整参考

---

## 3️⃣ `training_summary.json` - 训练总结

**保存时机**：训练完成时
**内容**：最终训练结果总结

```json
{
  "status": "completed",
  "completion_time": "2024-01-15 14:30:00",
  "best_checkpoint": "logs/body_unet_bs2_lr0.0001_ch32/checkpoints/best_model.ckpt",
  "best_val_mIoU": 0.5234,
  "total_epochs": 100
}
```

---

## 🎯 主要改进点

### 1. **完整的训练信息记录**
- ✅ 每个epoch的loss、accuracy、mIoU
- ✅ 学习率变化曲线
- ✅ 最佳模型的epoch和性能
- ✅ 实时更新，训练中断也不丢失

### 2. **优化的模型保存策略**
- ✅ 只保存最佳模型（节省磁盘空间）
- ✅ 固定文件名 `best_model.ckpt`（易于使用）
- ✅ 不保存中间checkpoint（避免混淆）

### 3. **配置可追溯性**
- ✅ 完整记录所有训练参数
- ✅ 记录系统环境信息
- ✅ 记录类别权重统计
- ✅ 记录模型参数量

---

## 📊 如何使用这些日志

### 1. 绘制训练曲线
```python
import json
import matplotlib.pyplot as plt

# 加载训练日志
with open('logs/body_unet_bs2_lr0.0001_ch32/training_log.json', 'r') as f:
    log = json.load(f)

history = log['training_history']

# 绘制Loss曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(history['epoch'], history['train_loss'], label='Train')
plt.plot(history['epoch'], history['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')

# 绘制mIoU曲线
plt.subplot(1, 3, 2)
plt.plot(history['epoch'], history['val_mIoU'])
plt.xlabel('Epoch')
plt.ylabel('mIoU')
plt.title('Validation mIoU')

# 绘制学习率曲线
plt.subplot(1, 3, 3)
plt.plot(history['epoch'], history['learning_rate'])
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')

plt.tight_layout()
plt.savefig('training_curves.png')
```

### 2. 加载最佳模型进行评估
```python
from pasco.models.body_net import BodyNet

# 最佳模型路径在 training_summary.json 中
checkpoint_path = "logs/body_unet_bs2_lr0.0001_ch32/checkpoints/best_model.ckpt"

model = BodyNet.load_from_checkpoint(checkpoint_path)
model.eval()
```

### 3. 对比不同实验
```python
import json
import pandas as pd

experiments = [
    'body_unet_bs2_lr0.0001_ch16',
    'body_unet_bs2_lr0.0001_ch32',
    'body_unet_bs2_lr0.0001_ch64',
]

results = []
for exp in experiments:
    with open(f'logs/{exp}/training_summary.json', 'r') as f:
        summary = json.load(f)

    with open(f'logs/{exp}/training_config.json', 'r') as f:
        config = json.load(f)

    results.append({
        'experiment': exp,
        'best_mIoU': summary['best_val_mIoU'],
        'total_epochs': summary['total_epochs'],
        'base_channels': config['model']['base_channels'],
        'parameters': config['model']['total_parameters'],
    })

df = pd.DataFrame(results)
print(df)
```

---

## 🔍 训练过程监控

训练期间可以实时查看训练日志：

```bash
# 查看最新的训练进度
tail -f logs/body_unet_bs2_lr0.0001_ch32/training_log.json

# 或使用 watch 实时监控
watch -n 5 'cat logs/body_unet_bs2_lr0.0001_ch32/training_log.json | jq ".best_metrics"'
```

---

## 💾 磁盘空间管理

**改进前**：
- 保存 top-3 checkpoint + last checkpoint
- 每个checkpoint ~80MB (base_channels=32)
- 总计: ~320MB

**改进后**：
- 只保存 best checkpoint
- 每个checkpoint ~80MB
- 总计: ~80MB

**节省空间**: ~75% 💰

---

## 📝 训练日志示例输出

训练开始时：
```
============================================================
Experiment: body_unet_bs2_lr0.0001_ch32_cw0.5_aug
Log directory: logs/body_unet_bs2_lr0.0001_ch32_cw0.5_aug
============================================================

Setting up data...
BodyDataset [train]: 3222 samples
BodyDataset [val]: 403 samples
Train samples: 3222
Val samples: 403

Computing class frequencies...
Class weights - min: 0.123, max: 5.678, mean: 1.234

Creating model...
Total parameters: 20,458,391
Trainable parameters: 20,458,391

Saving training configuration...
Training configuration saved to: logs/body_unet_bs2_lr0.0001_ch32/training_config.json

============================================================
Starting training...
============================================================
```

训练完成时：
```
============================================================
Training complete!
============================================================
Best checkpoint: logs/body_unet_bs2_lr0.0001_ch32/checkpoints/best_model.ckpt
Best val mIoU: 0.5234
Training log: logs/body_unet_bs2_lr0.0001_ch32/training_log.json
Config file: logs/body_unet_bs2_lr0.0001_ch32/training_config.json
============================================================

Training summary saved to: logs/body_unet_bs2_lr0.0001_ch32/training_summary.json
Training log saved to: logs/body_unet_bs2_lr0.0001_ch32/training_log.json
```
