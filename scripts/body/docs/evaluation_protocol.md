# LOOC Evaluation Protocol

## Overview

**LOOC** (Learning Organ Occupancy from Calibrated skin surface) is a standardized evaluation framework for fair comparison of organ occupancy prediction models.

This framework evaluates models based on organ-level Axis-Aligned Bounding Boxes (AABBs) rather than voxel-wise metrics, providing more interpretable and clinically relevant measurements.

## Pipeline

```
Input (sensor_pc) --> Model --> Prediction (npz) --> Evaluate --> Metrics
                                     |                  ^
                                     v                  |
                               Ground Truth (npz) ------+
```

## Data Format

All predictions must follow the GT npz format:

| Field             | Shape     | Description                   |
| ----------------- | --------- | ----------------------------- |
| `voxel_labels`    | (H, W, D) | Per-voxel class labels (0-71) |
| `grid_world_min`  | (3,)      | Grid origin in mm             |
| `grid_world_max`  | (3,)      | Grid extent in mm             |
| `grid_voxel_size` | (3,)      | Voxel size (default: 4mm)     |
| `grid_occ_size`   | (3,)      | Grid dimensions (H, W, D)     |

**Important**: Predictions must be on the **same grid** as ground truth (same origin, voxel size, and dimensions).

## Evaluation Process

1. **Input**: Model receives `sensor_pc` (body surface point cloud)
2. **Predict**: Model outputs `voxel_labels` on the **same grid** as GT
3. **Extract AABBs**: Compute per-organ AABBs from both pred and GT
4. **Compute Metrics**: Calculate CD, IoU, ESF per organ

### AABB Extraction

For each organ class:
- Find all voxels with that class label
- Compute minimal axis-aligned bounding box
- Convert to world coordinates (mm)

## Metrics

| Metric  | Description                                                  | Formula                      | Better |
| ------- | ------------------------------------------------------------ | ---------------------------- | ------ |
| **CD**  | Center Distance: Euclidean distance between AABB centers    | \|\|center_pred - center_gt\|\| | Lower  |
| **IoU** | Intersection over Union: Overlap ratio of AABBs              | intersection / union         | Higher |
| **ESF** | Encompassment Scaling Factor: Scaling needed for encompassment | max(scale_x, scale_y, scale_z) | Lower |

### Metric Details

- **CD (Center Distance)**: Measures position accuracy of organ localization in millimeters
- **IoU**: Measures size and position accuracy; 1.0 = perfect overlap
- **ESF**: Measures how much the predicted AABB needs to scale (from its center) to fully encompass the GT AABB; 1.0 = perfect encompassment

## Usage

### Quick Start

```bash
# Run complete evaluation pipeline
bash scripts/body/run_evaluation.sh
```

### Manual Steps

```bash
# Step 1: Generate predictions
python scripts/body/predict_voxel.py \
    --checkpoint logs/body_unet/checkpoints/best_model.ckpt \
    --dataset_root voxel-output/merged_data \
    --split test \
    --output predictions/my_model/

# Step 2: Evaluate predictions
python scripts/body/evaluate_predictions.py \
    --pred_dir predictions/my_model/ \
    --dataset_root voxel-output/merged_data \
    --split test
```

### Output

Evaluation results are saved to `<pred_dir>/evaluation_<split>.json`:

```json
{
  "overall": {
    "mean_cd": 12.34,
    "mean_iou": 0.8521,
    "mean_esf": 1.045,
    "num_organs_evaluated": 45
  },
  "per_organ": {
    "1": {
      "organ_name": "liver",
      "cd_mean": 8.5,
      "iou_mean": 0.92,
      "esf_mean": 1.02,
      "count": 100
    },
    ...
  }
}
```

## Directory Structure

```
predictions/
├── model_A/
│   ├── case_001.npz           # Predictions in LOOC format
│   ├── case_002.npz
│   ├── ...
│   └── evaluation_test.json   # Evaluation results
└── model_B/
    ├── case_001.npz
    ├── ...
    └── evaluation_test.json
```

## Implementation

The evaluation implementation consists of:

1. **predict_voxel.py**: Generates predictions from trained model
2. **evaluate_predictions.py**: Computes LOOC metrics
3. **pasco/utils/looc_metrics.py**: Core metric computation utilities
   - AABB class for bounding box operations
   - `extract_organ_aabb()`: Extract AABB from voxel labels
   - `evaluate_sample()`: Evaluate single sample
   - `aggregate_metrics()`: Aggregate across samples

## Comparing Models

To compare multiple models:

```bash
# Generate predictions for each model
python scripts/body/predict_voxel.py --checkpoint model_A.ckpt --output predictions/model_A/
python scripts/body/predict_voxel.py --checkpoint model_B.ckpt --output predictions/model_B/

# Evaluate each model
python scripts/body/evaluate_predictions.py --pred_dir predictions/model_A/ --split test
python scripts/body/evaluate_predictions.py --pred_dir predictions/model_B/ --split test

# Compare results
cat predictions/model_A/evaluation_test.json
cat predictions/model_B/evaluation_test.json
```

## Best Practices

1. **Always use the same grid**: Predictions must match GT grid exactly
2. **Evaluate on held-out test set**: Don't evaluate on training data
3. **Report all three metrics**: CD, IoU, and ESF provide complementary information
4. **Report per-organ results**: Different organs may have different accuracies
5. **Handle missing predictions**: If organ not predicted, report as inf/0.0
