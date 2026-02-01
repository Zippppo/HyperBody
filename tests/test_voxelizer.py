"""TDD tests for data/voxelizer.py - Step 2"""
import numpy as np
import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SAMPLE_PATH = "Dataset/voxel_data/BDMAP_00000001.npz"
VOLUME_SIZE = (128, 96, 256)


# --- Unit tests ---

class TestVoxelizePointCloud:
    def test_output_shape(self):
        """Output shape matches volume_size."""
        from data.voxelizer import voxelize_point_cloud
        data = np.load(SAMPLE_PATH)
        result = voxelize_point_cloud(
            data['sensor_pc'], data['grid_world_min'],
            data['grid_voxel_size'], VOLUME_SIZE
        )
        assert result.shape == VOLUME_SIZE

    def test_output_dtype(self):
        """Output is float32."""
        from data.voxelizer import voxelize_point_cloud
        data = np.load(SAMPLE_PATH)
        result = voxelize_point_cloud(
            data['sensor_pc'], data['grid_world_min'],
            data['grid_voxel_size'], VOLUME_SIZE
        )
        assert result.dtype == np.float32

    def test_binary_values(self):
        """Output contains only 0.0 and 1.0."""
        from data.voxelizer import voxelize_point_cloud
        data = np.load(SAMPLE_PATH)
        result = voxelize_point_cloud(
            data['sensor_pc'], data['grid_world_min'],
            data['grid_voxel_size'], VOLUME_SIZE
        )
        unique = np.unique(result)
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_occupied_voxels_reasonable(self):
        """Number of occupied voxels is reasonable (less than num points, more than 0)."""
        from data.voxelizer import voxelize_point_cloud
        data = np.load(SAMPLE_PATH)
        result = voxelize_point_cloud(
            data['sensor_pc'], data['grid_world_min'],
            data['grid_voxel_size'], VOLUME_SIZE
        )
        num_occupied = int(result.sum())
        num_points = len(data['sensor_pc'])
        assert num_occupied > 0
        # Multiple points can map to the same voxel
        assert num_occupied <= num_points

    def test_synthetic_data(self):
        """Voxelize a known set of points and verify placement."""
        from data.voxelizer import voxelize_point_cloud
        volume_size = (10, 10, 10)
        grid_world_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        voxel_size = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        # Points at (0.5, 0.5, 0.5) and (3.5, 3.5, 3.5)
        pc = np.array([[0.5, 0.5, 0.5], [3.5, 3.5, 3.5]], dtype=np.float32)
        result = voxelize_point_cloud(pc, grid_world_min, voxel_size, volume_size)
        assert result[0, 0, 0] == 1.0
        assert result[3, 3, 3] == 1.0
        assert result.sum() == 2.0


class TestPadLabels:
    def test_output_shape(self):
        """Padded labels match volume_size."""
        from data.voxelizer import pad_labels
        data = np.load(SAMPLE_PATH)
        result = pad_labels(data['voxel_labels'], VOLUME_SIZE)
        assert result.shape == VOLUME_SIZE

    def test_output_dtype(self):
        """Padded labels are int64."""
        from data.voxelizer import pad_labels
        data = np.load(SAMPLE_PATH)
        result = pad_labels(data['voxel_labels'], VOLUME_SIZE)
        assert result.dtype == np.int64

    def test_original_data_preserved(self):
        """Original label data is preserved in the padded volume."""
        from data.voxelizer import pad_labels
        data = np.load(SAMPLE_PATH)
        labels = data['voxel_labels']
        result = pad_labels(labels, VOLUME_SIZE)
        x, y, z = labels.shape
        np.testing.assert_array_equal(result[:x, :y, :z], labels.astype(np.int64))

    def test_padding_is_zero(self):
        """Padded region is filled with 0 (class 0 = inside_body_empty)."""
        from data.voxelizer import pad_labels
        data = np.load(SAMPLE_PATH)
        labels = data['voxel_labels']
        result = pad_labels(labels, VOLUME_SIZE)
        x, y, z = labels.shape
        # Check padding region along x
        if x < VOLUME_SIZE[0]:
            assert result[x:, :, :].sum() == 0


# --- Visualization test ---

@pytest.mark.parametrize("sample_name", ["BDMAP_00000001"])
def test_visualize_voxelization(sample_name, tmp_path):
    """Generate interactive HTML visualization comparing point cloud and voxelized result."""
    from data.voxelizer import voxelize_point_cloud, pad_labels
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    data = np.load(f"Dataset/voxel_data/{sample_name}.npz")
    pc = data['sensor_pc']
    labels = data['voxel_labels']
    grid_world_min = data['grid_world_min']
    voxel_size = data['grid_voxel_size']

    occ = voxelize_point_cloud(pc, grid_world_min, voxel_size, VOLUME_SIZE)
    padded_labels = pad_labels(labels, VOLUME_SIZE)

    # Subsample point cloud for rendering
    step = max(1, len(pc) // 10000)
    pc_sub = pc[::step]

    # Get occupied voxel centers in world coordinates
    occ_idx = np.argwhere(occ > 0)  # (N, 3)
    vox_centers = grid_world_min + (occ_idx + 0.5) * voxel_size

    # Get label voxel centers (non-zero classes only) for a slice
    label_idx = np.argwhere(padded_labels > 0)
    step_l = max(1, len(label_idx) // 15000)
    label_idx_sub = label_idx[::step_l]
    label_centers = grid_world_min + (label_idx_sub + 0.5) * voxel_size
    label_classes = padded_labels[label_idx_sub[:, 0], label_idx_sub[:, 1], label_idx_sub[:, 2]]

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=[
            f"Input Point Cloud ({len(pc)} pts)",
            f"Voxelized Occupancy ({int(occ.sum())} voxels)",
            f"Ground Truth Labels ({len(label_idx)} voxels)"
        ],
    )

    # 1. Point cloud
    fig.add_trace(go.Scatter3d(
        x=pc_sub[:, 0], y=pc_sub[:, 1], z=pc_sub[:, 2],
        mode='markers', marker=dict(size=1, color='steelblue', opacity=0.6),
        name='Point Cloud',
    ), row=1, col=1)

    # 2. Voxelized occupancy
    fig.add_trace(go.Scatter3d(
        x=vox_centers[:, 0], y=vox_centers[:, 1], z=vox_centers[:, 2],
        mode='markers', marker=dict(size=1.5, color='tomato', opacity=0.5),
        name='Voxelized',
    ), row=1, col=2)

    # 3. Ground truth labels
    fig.add_trace(go.Scatter3d(
        x=label_centers[:, 0], y=label_centers[:, 1], z=label_centers[:, 2],
        mode='markers', marker=dict(
            size=1.5, color=label_classes, colorscale='Rainbow',
            opacity=0.5, colorbar=dict(title="Class", x=1.02),
        ),
        name='Labels',
    ), row=1, col=3)

    fig.update_layout(
        title=f"Voxelization Pipeline: {sample_name}",
        width=1800, height=600,
        showlegend=False,
    )

    out_dir = "docs/visualizations"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"voxelization_{sample_name}.html")
    fig.write_html(out_path)
    print(f"\nVisualization saved to: {out_path}")
    assert os.path.exists(out_path)
