# Plan: Test Set Evaluation Script with Per-Class Metrics (Revised)

## Objective
Create an evaluation script that loads trained BodyNet checkpoints and evaluates performance on the test set, displaying detailed per-class IoU metrics for each organ.

---

## Key Decisions (Based on Review)

| Item | Decision |
|------|----------|
| Classes | 72 total, ignore class 0 (outside_body), evaluate 71 classes |
| Model Support | BodyNet only (Hyperbolic deferred) |
| Output Format | JSON only (no CSV) |
| Edge Cases | Handle as encountered |
| Checkpoint Path | User provides via `--ckpt_path` |

---

## Implementation Plan

### New Script: `scripts/body/eval_body_detailed.py`

#### Core Features
1. Load actual organ names from `Dataset/dataset_info.json`
2. Compute per-class metrics: IoU, Precision, Recall, Dice, Support
3. Output formatted console table (sorted by IoU)
4. Save results to JSON file

---

### Script Structure

```
eval_body_detailed.py
├── parse_args()                    # CLI arguments
├── load_class_names(dataset_root)  # Load from dataset_info.json
├── load_model(ckpt_path)           # Load BodyNet from checkpoint
├── compute_per_class_metrics()     # TP/FP/FN accumulation
├── format_results_table()          # Console output
├── save_results_json()             # JSON export
└── main()
```

---

### Implementation Details

#### 1. Class Names Loading
```python
def load_class_names(dataset_root):
    """Load class names from dataset_info.json."""
    info_path = os.path.join(dataset_root, "dataset_info.json")
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            info = json.load(f)
        return info.get("class_names", None)
    return None
```

#### 2. Metrics Computation (Micro-Average)
Accumulate TP/FP/FN across all samples, then compute:
- **IoU** = TP / (TP + FP + FN)
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **Dice** = 2*TP / (2*TP + FP + FN)
- **Support** = Total GT voxels

**Note**: Use micro-average (accumulate globally) to match training mIoU calculation.

#### 3. ignore_index Strategy
- Use `ignore_index=0` (outside_body) consistent with BodyNet training
- Class 0 will not be included in mIoU calculation

#### 4. Output Format

**Console Output**:
```
============================================================
EVALUATION RESULTS: test set
Checkpoint: logs/.../best_model.ckpt
============================================================
Overall mIoU: 65.32%
Overall Accuracy: 89.45%
Evaluated Classes: 71 (ignoring class 0: outside_body)

Per-Class Results (sorted by IoU):
------------------------------------------------------------
 ID  Class Name              IoU%   Prec%  Recall%  Dice%    Support
------------------------------------------------------------
  2  liver                  85.42  91.23   92.15   91.69   82560900
 11  heart                  82.31  88.45   90.12   89.27   18207721
...
------------------------------------------------------------
Classes with IoU < 50%: 15
Classes with IoU >= 80%: 8
```

**JSON Output Schema**:
```json
{
  "checkpoint": "/path/to/checkpoint.ckpt",
  "split": "test",
  "n_samples": 100,
  "overall": {
    "mIoU": 0.6532,
    "accuracy": 0.8945,
    "n_evaluated_classes": 71
  },
  "per_class": [
    {
      "id": 2,
      "name": "liver",
      "iou": 0.8542,
      "precision": 0.9123,
      "recall": 0.9215,
      "dice": 0.9169,
      "support": 82560900
    }
  ]
}
```

---

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--ckpt_path` | str | required | Path to model checkpoint |
| `--dataset_root` | str | required | Path to dataset |
| `--split` | str | `test` | Evaluation split (val/test) |
| `--batch_size` | int | `1` | Batch size |
| `--target_size` | int[3] | `[128,128,256]` | Grid size |
| `--output` | str | None | Output JSON file path |
| `--sort_by` | str | `iou` | Sort metric (iou/name/support) |
| `--num_workers` | int | `4` | Data loading workers |

---

### Files to Modify

| File | Action | Description |
|------|--------|-------------|
| `scripts/body/eval_body_detailed.py` | **CREATE** | Main evaluation script |
| `pasco/data/body/params.py` | **UPDATE** | Replace placeholder class names (organ_1, organ_2...) with real organ names (liver, spleen, etc.) |

**params.py Update**:
```python
body_class_names = [
    "outside_body",      # 0 (ignored)
    "inside_body_empty", # 1
    "liver",             # 2
    "spleen",            # 3
    # ... all 72 names from dataset_info.json
]
```

---

## Verification Plan

1. **Run evaluation**:
   ```bash
   python scripts/body/eval_body_detailed.py \
       --ckpt_path logs/body_unet_bs2_lr0.0001_ch16/checkpoints/best_model.ckpt \
       --dataset_root Dataset/voxel_data \
       --split test
   ```

2. **Verify JSON output**:
   ```bash
   python scripts/body/eval_body_detailed.py \
       --ckpt_path <path> \
       --dataset_root Dataset/voxel_data \
       --output results.json
   cat results.json | python -m json.tool
   ```

3. **Compare mIoU with training logs** to ensure consistency

---

## Summary

Create `eval_body_detailed.py` that:
- Loads BodyNet checkpoint (Hyperbolic support deferred)
- Loads 72 class names from dataset_info.json, ignores class 0
- Computes per-class IoU, Precision, Recall, Dice using micro-average
- Outputs formatted console table and optional JSON file
