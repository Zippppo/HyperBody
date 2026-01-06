"""Visualize saved voxel predictions vs GT using Plotly.

Usage:
    python scripts/body/vis/vis_prediction.py  --pred_dir predictions/pasco-20260102 --all
"""
import os
import sys
import argparse
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
GT_DATA_DIR = os.path.join(PROJECT_ROOT, "voxel-output", "merged_data")
DATASET_INFO_PATH = os.path.join(PROJECT_ROOT, "voxel-output", "dataset_info.json")


def load_class_names():
    if os.path.exists(DATASET_INFO_PATH):
        with open(DATASET_INFO_PATH, 'r', encoding='utf-8') as f:
            info = json.load(f)
            return info.get('class_names', None)
    return None


CLASS_NAMES = load_class_names()


def generate_colors(num_classes):
    colors = []
    for i in range(num_classes):
        if i == 0:
            colors.append('rgba(200,200,200,0.1)')
        else:
            h = (i * 137.508) % 360
            colors.append(f'hsl({h:.0f},70%,50%)')
    return colors


def verify_format(pred_path, gt_path):
    """Verify prediction format matches GT format."""
    pred = np.load(pred_path)
    gt = np.load(gt_path)

    print(f"\n=== Format Verification ===")
    print(f"GT keys: {list(gt.keys())}")
    print(f"Pred keys: {list(pred.keys())}")

    # Check required keys
    required_keys = ['voxel_labels', 'grid_world_min', 'grid_world_max', 'grid_voxel_size', 'grid_occ_size']
    missing = [k for k in required_keys if k not in pred.keys()]
    if missing:
        print(f"ERROR: Missing keys in prediction: {missing}")
        return False

    # Check shapes
    print(f"\nGT voxel_labels shape: {gt['voxel_labels'].shape}")
    print(f"Pred voxel_labels shape: {pred['voxel_labels'].shape}")

    if gt['voxel_labels'].shape != pred['voxel_labels'].shape:
        print("ERROR: Shape mismatch!")
        return False

    # Check grid params
    for key in ['grid_world_min', 'grid_world_max', 'grid_voxel_size', 'grid_occ_size']:
        gt_val = gt[key]
        pred_val = pred[key]
        match = np.allclose(gt_val, pred_val)
        print(f"{key}: GT={gt_val}, Pred={pred_val}, Match={match}")
        if not match:
            print(f"WARNING: {key} mismatch!")

    print("Format verification: PASSED")
    return True


def voxel_to_points(voxel_labels, grid_world_min, grid_voxel_size):
    D, H, W = voxel_labels.shape
    d_idx, h_idx, w_idx = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
    voxel_indices = np.stack([d_idx, h_idx, w_idx], axis=-1).reshape(-1, 3)
    world_coords = grid_world_min + (voxel_indices + 0.5) * grid_voxel_size
    labels = voxel_labels.flatten()
    return world_coords.astype(np.float32), labels.astype(np.int64)


def create_scatter3d(points, labels, colors, name, max_points=5000, class_names=None):
    traces = []
    unique_labels = np.unique(labels)

    for label in unique_labels:
        if label == 0:
            continue

        mask = labels == label
        pts = points[mask]

        if len(pts) > max_points:
            idx = np.random.choice(len(pts), max_points, replace=False)
            pts = pts[idx]

        if class_names is not None and label < len(class_names):
            label_name = class_names[label]
        else:
            label_name = f'Class {label}'

        traces.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='markers',
            marker=dict(size=2, color=colors[label], opacity=0.7),
            name=f'{name} {label_name}',
            legendgroup=f'class_{label}',
            showlegend=True
        ))

    return traces


def visualize_case(case_id, pred_dir, gt_dir, output_dir, verify=True):
    pred_path = os.path.join(pred_dir, f"{case_id}.npz")
    gt_path = os.path.join(gt_dir, f"{case_id}.npz")

    if not os.path.exists(pred_path):
        print(f"ERROR: Prediction not found: {pred_path}")
        return None
    if not os.path.exists(gt_path):
        print(f"ERROR: GT not found: {gt_path}")
        return None

    if verify:
        if not verify_format(pred_path, gt_path):
            return None

    pred = np.load(pred_path)
    gt = np.load(gt_path)

    pred_labels = pred['voxel_labels']
    gt_labels = gt['voxel_labels']
    grid_world_min = gt['grid_world_min']
    grid_voxel_size = gt['grid_voxel_size']

    pred_points, pred_flat = voxel_to_points(pred_labels, grid_world_min, grid_voxel_size)
    gt_points, gt_flat = voxel_to_points(gt_labels, grid_world_min, grid_voxel_size)

    num_classes = max(pred_labels.max(), gt_labels.max()) + 1
    colors = generate_colors(num_classes)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=['Ground Truth', 'Prediction'],
        horizontal_spacing=0.02
    )

    gt_traces = create_scatter3d(gt_points, gt_flat, colors, 'GT', class_names=CLASS_NAMES)
    for trace in gt_traces:
        fig.add_trace(trace, row=1, col=1)

    pred_traces = create_scatter3d(pred_points, pred_flat, colors, 'Pred', class_names=CLASS_NAMES)
    for trace in pred_traces:
        trace.showlegend = False
        fig.add_trace(trace, row=1, col=2)

    # Compute metrics
    acc = (pred_flat == gt_flat).mean()
    gt_unique = len(np.unique(gt_flat[gt_flat > 0]))
    pred_unique = len(np.unique(pred_flat[pred_flat > 0]))

    fig.update_layout(
        title=f'Case: {case_id} | Acc: {acc:.4f} | GT Classes: {gt_unique} | Pred Classes: {pred_unique}',
        scene=dict(aspectmode='data'),
        scene2=dict(aspectmode='data'),
        height=800,
        legend=dict(x=1.02, y=0.5),
        updatemenus=[
            dict(
                type="buttons", direction="left", x=0.0, y=1.15,
                buttons=[
                    dict(label="Show All", method="restyle", args=[{"visible": True}]),
                    dict(label="Hide All", method="restyle", args=[{"visible": "legendonly"}])
                ]
            )
        ]
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"pred_vis_{case_id}.html")
    fig.write_html(output_path)
    print(f"Saved: {output_path}")
    return output_path


def visualize_all(pred_dir, gt_dir, output_dir):
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.npz')]
    print(f"Found {len(pred_files)} prediction files")

    first = True
    for pf in pred_files:
        case_id = pf.replace('.npz', '')
        visualize_case(case_id, pred_dir, gt_dir, output_dir, verify=first)
        first = False

    print(f"\nDone! All {len(pred_files)} visualizations saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize saved voxel predictions')
    parser.add_argument('--pred_dir', type=str, required=True, help='Prediction directory')
    parser.add_argument('--gt_dir', type=str, default=GT_DATA_DIR, help='GT data directory')
    parser.add_argument('--case_id', type=str, default=None, help='Case ID to visualize')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--all', action='store_true', help='Visualize all predictions')
    args = parser.parse_args()

    output_dir = args.output or os.path.join(PROJECT_ROOT, "vis_results", os.path.basename(args.pred_dir))

    if args.all:
        visualize_all(args.pred_dir, args.gt_dir, output_dir)
    elif args.case_id:
        visualize_case(args.case_id, args.pred_dir, args.gt_dir, output_dir)
    else:
        # Visualize first available
        pred_files = [f for f in os.listdir(args.pred_dir) if f.endswith('.npz')]
        if pred_files:
            case_id = pred_files[0].replace('.npz', '')
            visualize_case(case_id, args.pred_dir, args.gt_dir, output_dir)
        else:
            print("No prediction files found!")
