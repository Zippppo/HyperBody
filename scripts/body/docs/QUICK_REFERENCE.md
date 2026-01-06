# 🚀 训练脚本快速参考

## 训练命令

```bash
# 基础训练
python scripts/train_body.py \
    --dataset_root voxel-output/merged_data \
    --batch_size 2 \
    --lr 1e-4 \
    --max_epochs 100

# 推荐配置（带类别权重和数据增强）
python scripts/train_body.py \
    --dataset_root voxel-output/merged_data \
    --batch_size 2 \
    --lr 1e-4 \
    --max_epochs 100 \
    --base_channels 32 \
    --use_class_weights \
    --data_aug \
    --n_gpus 2 \
    --precision 16
```

## 输出文件（自动生成）

```
logs/experiment_name/
├── training_config.json      ← 训练配置
├── training_log.json          ← 训练历史（实时更新）
├── training_summary.json      ← 训练总结
└── checkpoints/
    └── best_model.ckpt        ← 最佳模型（仅此一个）
```

## 可视化命令

```bash
# 查看训练总结
python scripts/plot_training.py --log_dir logs/exp_name --summary

# 绘制训练曲线
python scripts/plot_training.py --log_dir logs/exp_name

# 对比多个实验
python scripts/plot_training.py \
    --log_dirs logs/exp1 logs/exp2 logs/exp3 \
    --labels "Exp1" "Exp2" "Exp3"
```

## 加载最佳模型

```python
from pasco.models.body_net import BodyNet

model = BodyNet.load_from_checkpoint(
    'logs/exp_name/checkpoints/best_model.ckpt'
)
model.eval()
```

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--base_channels` | 32 | UNet基础通道数（16/32/64） |
| `--batch_size` | 2 | 批次大小 |
| `--lr` | 1e-4 | 学习率 |
| `--max_epochs` | 100 | 最大训练轮数 |
| `--use_class_weights` | False | 使用类别权重 |
| `--data_aug` | False | 数据增强 |
| `--use_light_model` | False | 使用轻量模型 |
| `--precision` | 32 | 训练精度（16/32/bf16） |

## 最佳实践

✅ 使用 `--use_class_weights` 处理类别不平衡
✅ 使用 `--data_aug` 增强泛化能力
✅ 使用 `--precision 16` 节省显存
✅ 使用 `--use_light_model` 在显存受限时
✅ 定期检查 `training_log.json` 监控进度
