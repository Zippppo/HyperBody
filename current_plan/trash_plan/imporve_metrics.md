# Plan: Per-Image Evaluation Script

## Problem

Current `evaluate.py` uses dataset-level confusion matrix aggregation, which masks Lorentz's improvement on small classes (ribs). Baseline ribs at 0-0.36% vs Lorentz at 5-12% — a qualitative leap invisible in mIoU (26.76% vs 28.10%).

## Solution

Create `scripts/body/eval/evaluate_per_image.py` — a new evaluation script computing **per-image, per-class metrics** then averaging, so each (image, class) pair has equal weight.

### Why a new file (not modify existing)?

- `evaluate.py` serves its purpose well for dataset-level metrics
- `analyze_per_image_dice.py` exists but has performance issues (class-by-class Python loop) and limited structure
- A clean new script avoids regression risk and serves as the "fair comparison" evaluation tool

## Key Design Decisions

### 1. Metric Averaging: Two-stage averaging (recommended)

```
Per-class Dice = mean over images where class is present
Mean Dice = mean over all classes with valid data
```

**Rationale:** This gives each class equal weight in the final metric, regardless of how many images contain it. A rare rib class appearing in 50 images gets the same weight as liver appearing in 403 images. This maximally reflects small-class performance differences.

### 2. Missing Class Handling

- Class **not present in GT AND not predicted** → `NaN` (excluded from that image's computation)
- Class **present in GT but not predicted** → Dice = 0 (penalize misses)
- Class **not in GT but predicted** (false positive) → Dice = 0 (penalize false alarms)

**Rationale:** Only exclude when the class genuinely doesn't exist in the image. Failed predictions must count as 0.

### 3. Performance: Vectorized via confusion matrix

Use the same `bincount` confusion matrix approach from `evaluate.py`, then extract per-class Dice from it. This avoids the slow per-class loop in `analyze_per_image_dice.py`.

## Implementation Plan

### File: `scripts/body/eval/evaluate_per_image.py`

#### Functions to implement:

1. **`compute_per_image_metrics(pred, gt, n_classes, ignore_index)`**
   - Build confusion matrix using `np.bincount` (vectorized)
   - Extract per-class: intersection, pred_sum, gt_sum
   - Compute per-class Dice: `2*TP / (pred_sum + gt_sum)`, NaN when both are 0
   - Compute per-class IoU: `TP / (TP + FP + FN)`, NaN when union is 0
   - Return: `dice_per_class [N]`, `iou_per_class [N]`, `gt_present [N]` (bool)

2. **`evaluate_single_model(pred_dir, dataset_root, split, ignore_index, verbose)`**
   - Load meta.json for target_size and sample_ids
   - For each image: load pred + GT, call `compute_per_image_metrics`
   - Accumulate into `[N_images, N_classes]` arrays
   - Compute:
     - **Per-class mean Dice/IoU** (nanmean over images where class present)
     - **Overall mean Dice/IoU** (mean of per-class means, excluding ignored class)
     - **Category breakdown**: organs (2-22), ribs (24-47), bones (23-60), muscles (61-70)
     - **Zero-Dice ratio** per class (fraction of images where Dice=0 despite class being present)
   - Print detailed results and save to `results_per_image.json`

3. **`evaluate_all_models(base_dir, dataset_root, ...)`**
   - Find all model dirs, evaluate each
   - Print comparison table with:
     - Overall mDice, mIoU
     - Category-level mDice (organs, ribs, bones, muscles)
     - Rib zero-Dice ratio
     - Delta vs baseline
   - Save comparison to `summary_per_image.json`

4. **`main()`** with same CLI interface as evaluate.py:
   ```
   python scripts/body/eval/evaluate_per_image.py \
       --eval_all model_eval_res \
       --dataset_root Dataset/voxel_data
   ```

### Output Format

```
======================================================================
PER-IMAGE EVALUATION RESULTS
======================================================================
Model: hyperbolic_lorentz
Split: test | Samples: 403

Overall (per-image averaged):
  Mean Dice: XX.XX%    Mean IoU: XX.XX%

Category Breakdown:
  Organs (20 classes):  Dice XX.XX%  IoU XX.XX%
  Ribs (24 classes):    Dice XX.XX%  IoU XX.XX%
  Other Bones (14):     Dice XX.XX%  IoU XX.XX%
  Muscles (10):         Dice XX.XX%  IoU XX.XX%

Per-class Metrics (sorted by Dice):
 Idx  Class Name                   Dice      IoU   Present  ZeroDice%
----------------------------------------------------------------------
  5   liver                       85.20%   74.20%     403      0.0%
 ...
 35   rib_left_12                  8.50%    4.45%      52     45.2%

======================================================================
COMPARISON (delta vs baseline)
======================================================================
                        mDice     mIoU   RibDice  RibZeroDice%
baseline               XX.XX%   XX.XX%   XX.XX%      XX.X%
hyperbolic_lorentz     XX.XX%   XX.XX%   XX.XX%      XX.X%   (+X.XX)
hyperbolic_poincare    XX.XX%   XX.XX%   XX.XX%      XX.X%   (+X.XX)
```

## Verification

1. Run the script: `python scripts/body/eval/evaluate_per_image.py --eval_all model_eval_res --dataset_root Dataset/voxel_data`
2. Confirm Lorentz shows meaningfully higher rib Dice than baseline
3. Confirm per-image mDice differs from dataset-level mDice (validates the averaging difference)
4. Check `summary_per_image.json` is saved correctly
