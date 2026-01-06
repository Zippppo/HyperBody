# 训练脚本改进说明

## 🎯 改进内容

对 `scripts/train_body.py` 进行了以下改进：

### ✅ 1. 完整的训练信息记录

新增 `TrainingInfoLogger` callback，自动记录：
- 每个epoch的训练loss和accuracy
- 每个epoch的验证loss和mIoU
- 学习率变化曲线
- 最佳模型的epoch和性能
- 实时保存到JSON文件（训练中断也不丢失）

### ✅ 2. 优化的模型保存策略

**改进前**：
- 保存top-3 checkpoint + last checkpoint
- 文件名包含epoch和mIoU：`epoch=042-val_mIoU=0.4567.ckpt`
- 磁盘占用：~320MB（4个checkpoint × 80MB）

**改进后**：
- 只保存最佳模型
- 固定文件名：`best_model.ckpt`
- 磁盘占用：~80MB（节省75%空间）

### ✅ 3. 详细的配置保存

新增 `save_training_config()` 函数，保存：
- 实验配置（数据、模型、训练参数）
- 类别权重统计信息
- 模型参数量
- 系统环境信息（PyTorch版本、CUDA版本）

### ✅ 4. 训练总结报告

训练完成后自动生成总结：
- 训练状态和完成时间
- 最佳checkpoint路径
- 最佳性能指标
- 总训练轮数

### ✅ 5. 更友好的输出信息

改进的命令行输出：
- 清晰的分隔符和格式化
- 完整的训练前信息（数据集、模型、配置）
- 详细的训练后总结

---

## 📁 生成的文件

训练后会在实验目录下生成以下文件：

```
logs/body_unet_bs2_lr0.0001_ch32_cw0.5_aug/
├── training_config.json      # 训练配置（启动时保存）
├── training_log.json          # 训练过程（每epoch更新）
├── training_summary.json      # 训练总结（完成时保存）
├── checkpoints/
│   └── best_model.ckpt        # 最佳模型（仅此一个）
└── version_X/                 # TensorBoard日志
    └── events.out.tfevents.*
```

---

## 🚀 使用方法

### 1. 训练模型（和之前一样）

```bash
python scripts/train_body.py \
    --dataset_root voxel-output/merged_data \
    --batch_size 2 \
    --lr 1e-4 \
    --max_epochs 100 \
    --base_channels 32 \
    --n_gpus 2 \
    --precision 16 \
    --use_class_weights \
    --data_aug \
    --exp_name body_unet
```

训练完成后会自动生成所有日志文件。

### 2. 查看训练总结

```bash
# 使用新的可视化脚本
python scripts/plot_training.py \
    --log_dir logs/body_unet_bs2_lr0.0001_ch32 \
    --summary
```

输出示例：
```
============================================================
Training Summary: body_unet_bs2_lr0.0001_ch32_cw0.5_aug
============================================================
Total epochs: 100
Best val mIoU: 0.5234
Best epoch: 87
Final val loss: 1.0234
Final val mIoU: 0.5123
Final train loss: 0.8901
Final train acc: 0.7456
Last update: 2024-01-15 14:30:00
============================================================
```

### 3. 绘制训练曲线

```bash
# 单个实验
python scripts/plot_training.py \
    --log_dir logs/body_unet_bs2_lr0.0001_ch32

# 对比多个实验
python scripts/plot_training.py \
    --log_dirs logs/exp1 logs/exp2 logs/exp3 \
    --labels "16 channels" "32 channels" "64 channels" \
    --output_dir results/
```

生成的图表包括：
- Loss curves (train & val)
- Training accuracy
- Validation mIoU (标记最佳点)
- Learning rate schedule

### 4. 加载最佳模型

```python
import json
from pasco.models.body_net import BodyNet

# 从summary读取最佳模型路径
with open('logs/body_unet_bs2_lr0.0001_ch32/training_summary.json', 'r') as f:
    summary = json.load(f)

checkpoint_path = summary['best_checkpoint']
model = BodyNet.load_from_checkpoint(checkpoint_path)
model.eval()

# 或直接使用固定路径
model = BodyNet.load_from_checkpoint(
    'logs/body_unet_bs2_lr0.0001_ch32/checkpoints/best_model.ckpt'
)
```

---

## 📊 JSON文件格式

### `training_config.json` 示例

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

### `training_log.json` 示例

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

### `training_summary.json` 示例

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

## 🔧 自定义分析

### 使用Python加载日志

```python
import json
import matplotlib.pyplot as plt

# 加载训练日志
with open('logs/body_unet_bs2_lr0.0001_ch32/training_log.json', 'r') as f:
    log = json.load(f)

history = log['training_history']

# 自定义绘图
plt.figure(figsize=(10, 6))
plt.plot(history['epoch'], history['val_mIoU'])
plt.xlabel('Epoch')
plt.ylabel('Validation mIoU')
plt.title('My Custom Plot')
plt.grid(True)
plt.savefig('my_plot.png')
```

### 对比不同实验

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
    # 加载配置
    with open(f'logs/{exp}/training_config.json', 'r') as f:
        config = json.load(f)

    # 加载总结
    with open(f'logs/{exp}/training_summary.json', 'r') as f:
        summary = json.load(f)

    results.append({
        'experiment': exp,
        'base_channels': config['model']['base_channels'],
        'parameters': config['model']['total_parameters'],
        'best_mIoU': summary['best_val_mIoU'],
        'total_epochs': summary['total_epochs'],
    })

df = pd.DataFrame(results)
print(df.to_markdown(index=False))
```

输出：
```
| experiment                      | base_channels | parameters | best_mIoU | total_epochs |
|:--------------------------------|--------------:|-----------:|----------:|-------------:|
| body_unet_bs2_lr0.0001_ch16     | 16            | 5114695    | 0.4567    | 100          |
| body_unet_bs2_lr0.0001_ch32     | 32            | 20458391   | 0.5234    | 100          |
| body_unet_bs2_lr0.0001_ch64     | 64            | 81831367   | 0.5456    | 100          |
```

---

## 💡 实时监控训练进度

### 方法1：查看JSON文件

```bash
# 实时查看最新指标
watch -n 5 'cat logs/body_unet_bs2_lr0.0001_ch32/training_log.json | jq ".best_metrics"'

# 查看最后一个epoch的结果
cat logs/body_unet_bs2_lr0.0001_ch32/training_log.json | jq '.training_history | {
  epoch: .epoch[-1],
  train_loss: .train_loss[-1],
  val_mIoU: .val_mIoU[-1]
}'
```

### 方法2：使用TensorBoard

```bash
tensorboard --logdir logs/body_unet_bs2_lr0.0001_ch32
```

---

## 🎨 优势总结

| 特性 | 改进前 | 改进后 |
|------|--------|--------|
| **训练历史** | 仅TensorBoard | JSON + TensorBoard |
| **配置记录** | 无 | 完整JSON配置 |
| **最佳模型** | 需手动查找 | 固定路径 best_model.ckpt |
| **磁盘占用** | ~320MB | ~80MB (-75%) |
| **可追溯性** | 低 | 高（完整参数记录） |
| **易用性** | 中 | 高（直接加载JSON） |
| **对比分析** | 困难 | 简单（脚本支持） |

---

## 📌 注意事项

1. **JSON文件实时更新**：`training_log.json` 每个epoch后自动更新，训练中断也不会丢失数据

2. **只保存最佳模型**：如果需要保存多个checkpoint，可修改 `ModelCheckpoint` 的 `save_top_k` 参数

3. **可视化脚本依赖**：需要安装 matplotlib
   ```bash
   pip install matplotlib
   ```

4. **兼容性**：所有改进都向后兼容，不影响原有功能

---

## 📚 相关文档

- [TRAINING_LOG_FORMAT.md](./TRAINING_LOG_FORMAT.md) - 详细的日志格式说明
- [scripts/plot_training.py](./scripts/plot_training.py) - 可视化工具
- [scripts/train_body.py](./scripts/train_body.py) - 改进后的训练脚本

---

## 🤝 反馈与建议

如有问题或建议，请创建issue或联系开发者。
