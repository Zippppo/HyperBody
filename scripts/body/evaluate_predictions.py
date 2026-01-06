"""Standalone evaluation script for LOOC predictions.

Usage:
python scripts/body/evaluate_predictions.py --pred_dir predictions/pasco-20260102 --gt_dir voxel-output/merged_data --split test --split_file dataset_split.json --output pasco.json
"""
import os
import json
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# ============== Constants ==============
NUM_CLASSES = 72


# ============== Split Loading ==============
def load_split_ids(split_file, split):
    """Load sample IDs from unified split file."""
    with open(split_file, 'r') as f:
        split_info = json.load(f)
    return split_info['splits'][split]


# ============== Metrics ==============
def aabb_center(min_xyz, max_xyz):
    return (min_xyz + max_xyz) / 2


def aabb_volume(min_xyz, max_xyz):
    dims = np.maximum(max_xyz - min_xyz, 0)
    return float(np.prod(dims))


def compute_cd(pred_min, pred_max, gt_min, gt_max):
    """Center Distance between AABBs (mm)."""
    pred_center = aabb_center(pred_min, pred_max)
    gt_center = aabb_center(gt_min, gt_max)
    return float(np.linalg.norm(pred_center - gt_center))


def compute_iou(pred_min, pred_max, gt_min, gt_max):
    """Intersection over Union of AABBs."""
    inter_min = np.maximum(pred_min, gt_min)
    inter_max = np.minimum(pred_max, gt_max)
    inter_vol = aabb_volume(inter_min, inter_max)
    pred_vol = aabb_volume(pred_min, pred_max)
    gt_vol = aabb_volume(gt_min, gt_max)
    union_vol = pred_vol + gt_vol - inter_vol
    return inter_vol / union_vol if union_vol > 0 else 0.0


def compute_esf(pred_min, pred_max, gt_min, gt_max):
    """Encompassment Scaling Factor."""
    pred_center = aabb_center(pred_min, pred_max)
    pred_half_size = (pred_max - pred_min) / 2
    gt_corners = np.array([
        [gt_min[0], gt_min[1], gt_min[2]], [gt_min[0], gt_min[1], gt_max[2]],
        [gt_min[0], gt_max[1], gt_min[2]], [gt_min[0], gt_max[1], gt_max[2]],
        [gt_max[0], gt_min[1], gt_min[2]], [gt_max[0], gt_min[1], gt_max[2]],
        [gt_max[0], gt_max[1], gt_min[2]], [gt_max[0], gt_max[1], gt_max[2]],
    ])
    esf_per_axis = np.ones(3)
    for axis in range(3):
        if pred_half_size[axis] <= 0:
            gt_extent = abs(gt_corners[:, axis] - pred_center[axis]).max()
            esf_per_axis[axis] = np.inf if gt_extent > 1e-6 else 1.0
            continue
        max_dist = abs(gt_corners[:, axis] - pred_center[axis]).max()
        esf_per_axis[axis] = max_dist / pred_half_size[axis]
    return max(1.0, float(np.max(esf_per_axis)))


def compute_voxel_iou(pred_mask, gt_mask):
    """3D voxel-level IoU."""
    inter = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()
    return float(inter / union) if union > 0 else 0.0


def compute_dice(pred_mask, gt_mask):
    """Dice coefficient."""
    inter = (pred_mask & gt_mask).sum()
    total = pred_mask.sum() + gt_mask.sum()
    return float(2 * inter / total) if total > 0 else 0.0


def compute_precision_recall_f1(pred_mask, gt_mask):
    """Compute Precision, Recall, F1."""
    tp = (pred_mask & gt_mask).sum()
    fp = (pred_mask & ~gt_mask).sum()
    fn = (~pred_mask & gt_mask).sum()
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ============== AABB Extraction ==============
def extract_aabb_from_voxel(voxel_labels, class_id, grid_world_min, grid_voxel_size):
    """Extract AABB for a class from voxel grid."""
    mask = voxel_labels == class_id
    if not mask.any():
        return None
    indices = np.argwhere(mask)
    min_idx = indices.min(axis=0)
    max_idx = indices.max(axis=0)
    aabb_min = grid_world_min + min_idx * grid_voxel_size
    aabb_max = grid_world_min + (max_idx + 1) * grid_voxel_size
    return aabb_min, aabb_max


# ============== Main Evaluation ==============
def evaluate_predictions(pred_dir, gt_dir, split='test', output_path=None, split_file=None):
    if split_file:
        case_ids = load_split_ids(split_file, split)
    else:
        # If no split file, use all npz files in pred_dir
        case_ids = [f.replace('.npz', '') for f in os.listdir(pred_dir) if f.endswith('.npz')]
    print(f"Evaluating {len(case_ids)} samples from {split} split")

    # Metrics storage
    cd_per_organ = defaultdict(list)
    iou_per_organ = defaultdict(list)
    esf_per_organ = defaultdict(list)
    voxel_iou_per_organ = defaultdict(list)
    dice_per_organ = defaultdict(list)
    precision_per_organ = defaultdict(list)
    recall_per_organ = defaultdict(list)
    f1_per_organ = defaultdict(list)
    fp_counts = []

    for case_id in tqdm(case_ids, desc="Evaluating"):
        pred_path = os.path.join(pred_dir, f"{case_id}.npz")
        gt_path = os.path.join(gt_dir, f"{case_id}.npz")

        if not os.path.exists(pred_path):
            print(f"Warning: Missing prediction for {case_id}")
            continue
        if not os.path.exists(gt_path):
            print(f"Warning: Missing GT for {case_id}")
            continue

        pred_data = np.load(pred_path)
        gt_data = np.load(gt_path)

        pred_voxels = pred_data['voxel_labels']
        gt_voxels = gt_data['voxel_labels']
        grid_world_min = gt_data['grid_world_min']
        grid_voxel_size = gt_data['grid_voxel_size']
        grid_world_max = gt_data['grid_world_max']
        cavity_diagonal = np.linalg.norm(grid_world_max - grid_world_min)

        gt_organs = set(np.unique(gt_voxels)) - {0}
        pred_organs = set(np.unique(pred_voxels)) - {0}

        for organ_id in gt_organs:
            pred_mask = pred_voxels == organ_id
            gt_mask = gt_voxels == organ_id

            voxel_iou_per_organ[organ_id].append(compute_voxel_iou(pred_mask, gt_mask))
            dice_per_organ[organ_id].append(compute_dice(pred_mask, gt_mask))
            p, r, f1 = compute_precision_recall_f1(pred_mask, gt_mask)
            precision_per_organ[organ_id].append(p)
            recall_per_organ[organ_id].append(r)
            f1_per_organ[organ_id].append(f1)

            gt_aabb = extract_aabb_from_voxel(gt_voxels, organ_id, grid_world_min, grid_voxel_size)
            pred_aabb = extract_aabb_from_voxel(pred_voxels, organ_id, grid_world_min, grid_voxel_size)

            if gt_aabb is None:
                continue

            gt_min, gt_max = gt_aabb
            if pred_aabb is None:
                cd_per_organ[organ_id].append(cavity_diagonal)
                iou_per_organ[organ_id].append(0.0)
                esf_per_organ[organ_id].append(np.inf)
            else:
                pred_min, pred_max = pred_aabb
                cd_per_organ[organ_id].append(compute_cd(pred_min, pred_max, gt_min, gt_max))
                iou_per_organ[organ_id].append(compute_iou(pred_min, pred_max, gt_min, gt_max))
                esf_per_organ[organ_id].append(compute_esf(pred_min, pred_max, gt_min, gt_max))

        fp_counts.append(len(pred_organs - gt_organs))

    # Aggregate results
    results = {'per_organ': {}, 'aggregate': {}}
    all_cd, all_iou, all_esf = [], [], []
    all_voxel_iou, all_dice = [], []
    all_precision, all_recall, all_f1 = [], [], []

    for organ_id in range(NUM_CLASSES):
        if organ_id not in cd_per_organ:
            continue
        cds = cd_per_organ[organ_id]
        ious = iou_per_organ[organ_id]
        esfs = [e for e in esf_per_organ[organ_id] if e != np.inf]

        results['per_organ'][organ_id] = {
            'cd_mean': float(np.mean(cds)),
            'iou_mean': float(np.mean(ious)),
            'esf_mean': float(np.mean(esfs)) if esfs else float('inf'),
            'voxel_iou_mean': float(np.mean(voxel_iou_per_organ[organ_id])),
            'dice_mean': float(np.mean(dice_per_organ[organ_id])),
            'precision_mean': float(np.mean(precision_per_organ[organ_id])),
            'recall_mean': float(np.mean(recall_per_organ[organ_id])),
            'f1_mean': float(np.mean(f1_per_organ[organ_id])),
            'num_samples': len(cds),
        }
        all_cd.extend(cds)
        all_iou.extend(ious)
        all_esf.extend(esfs)
        all_voxel_iou.extend(voxel_iou_per_organ[organ_id])
        all_dice.extend(dice_per_organ[organ_id])
        all_precision.extend(precision_per_organ[organ_id])
        all_recall.extend(recall_per_organ[organ_id])
        all_f1.extend(f1_per_organ[organ_id])

    results['aggregate'] = {
        'cd_mean': float(np.mean(all_cd)) if all_cd else 0.0,
        'iou_mean': float(np.mean(all_iou)) if all_iou else 0.0,
        'esf_mean': float(np.mean(all_esf)) if all_esf else float('inf'),
        'voxel_iou_mean': float(np.mean(all_voxel_iou)) if all_voxel_iou else 0.0,
        'dice_mean': float(np.mean(all_dice)) if all_dice else 0.0,
        'precision_mean': float(np.mean(all_precision)) if all_precision else 0.0,
        'recall_mean': float(np.mean(all_recall)) if all_recall else 0.0,
        'f1_mean': float(np.mean(all_f1)) if all_f1 else 0.0,
        'fp_count_mean': float(np.mean(fp_counts)) if fp_counts else 0.0,
        'num_organs': len(results['per_organ']),
        'num_samples': len(case_ids),
    }

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"CD:        {results['aggregate']['cd_mean']:.2f} mm")
    print(f"AABB IoU:  {results['aggregate']['iou_mean']:.4f}")
    esf_str = f"{results['aggregate']['esf_mean']:.2f}" if results['aggregate']['esf_mean'] != float('inf') else "inf"
    print(f"ESF:       {esf_str}")
    print(f"Voxel IoU: {results['aggregate']['voxel_iou_mean']:.4f}")
    print(f"Dice:      {results['aggregate']['dice_mean']:.4f}")
    print(f"Precision: {results['aggregate']['precision_mean']:.4f}")
    print(f"Recall:    {results['aggregate']['recall_mean']:.4f}")
    print(f"F1:        {results['aggregate']['f1_mean']:.4f}")
    print(f"FP Count:  {results['aggregate']['fp_count_mean']:.2f}")
    print(f"Organs: {results['aggregate']['num_organs']}, Samples: {results['aggregate']['num_samples']}")

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate LOOC predictions (standalone)')
    parser.add_argument('--pred_dir', type=str, required=True, help='Prediction directory')
    parser.add_argument('--gt_dir', type=str, required=True, help='Ground truth directory')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    parser.add_argument('--split_file', type=str, default=None, help='Path to dataset_split.json (optional)')
    args = parser.parse_args()

    evaluate_predictions(args.pred_dir, args.gt_dir, args.split, args.output, args.split_file)
