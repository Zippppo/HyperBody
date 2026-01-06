"""
Quick test script to verify Plotly visualization works.

Generates synthetic uncertainty data and creates a test visualization.
"""

import numpy as np
import plotly.graph_objects as go
from pathlib import Path


def create_test_data():
    """Create synthetic uncertainty data for testing."""
    # Create a simple 64x64x64 volume
    H, W, D = 64, 64, 64

    # Create a sphere of occupied voxels
    center = np.array([H//2, W//2, D//2])
    radius = 20

    x, y, z = np.meshgrid(np.arange(H), np.arange(W), np.arange(D), indexing='ij')
    distance = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)

    # Occupancy: sphere
    occupancy = (distance < radius).astype(float)

    # Uncertainty: higher at the surface, lower at the center
    uncertainty = np.zeros_like(occupancy)
    uncertainty[occupancy > 0] = np.abs(distance[occupancy > 0] - radius/2) / radius * 2
    uncertainty = np.clip(uncertainty, 0, 2)

    # Predicted classes: random
    pred_classes = np.random.randint(0, 10, size=(H, W, D))

    return {
        'occupancy': occupancy,
        'uncertainty': uncertainty,
        'pred_classes': pred_classes,
    }


def create_test_visualization(data, output_path="test_uncertainty.html"):
    """Create a simple test visualization."""
    occupancy = data['occupancy']
    uncertainty = data['uncertainty']
    pred_classes = data['pred_classes']

    # Get occupied voxel coordinates
    mask = occupancy > 0.5
    x, y, z = np.where(mask)

    # Subsample for performance
    subsample = 2
    x, y, z = x[::subsample], y[::subsample], z[::subsample]

    # Get uncertainty values
    uncertainty_vals = uncertainty[x, y, z]

    print(f"Creating test visualization with {len(x)} voxels...")
    print(f"Uncertainty range: [{uncertainty_vals.min():.3f}, {uncertainty_vals.max():.3f}]")

    # Create 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=3,
            color=uncertainty_vals,
            colorscale='Hot',  # Black -> Red -> Yellow -> White
            showscale=True,
            colorbar=dict(
                title="Uncertainty",
                thickness=20,
                len=0.7,
            ),
        ),
        text=[f"Uncertainty: {u:.3f}" for u in uncertainty_vals],
        hovertemplate="<b>Position</b>: (%{x}, %{y}, %{z})<br>" +
                      "%{text}<br>" +
                      "<extra></extra>",
    )])

    fig.update_layout(
        title=dict(
            text="Test 3D Uncertainty Visualization (Red = High Uncertainty)",
            x=0.5,
            xanchor='center',
        ),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
        ),
        width=1200,
        height=900,
    )

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))

    print(f"\n✅ Test visualization saved to: {output_path}")
    print("\nOpen this file in a web browser to verify the visualization works!")
    print("\nExpected result:")
    print("  - A sphere made of points")
    print("  - Center points are black/dark (low uncertainty)")
    print("  - Surface points are red/yellow (high uncertainty)")
    print("  - You can rotate, zoom, and pan the view")

    return fig


if __name__ == "__main__":
    print("=" * 60)
    print("Plotly Uncertainty Visualization - Test Script")
    print("=" * 60)

    # Check if plotly is installed
    try:
        import plotly
        print(f"\n✅ Plotly version: {plotly.__version__}")
    except ImportError:
        print("\n❌ Plotly is not installed!")
        print("Install it with: pip install plotly")
        exit(1)

    # Create test data
    print("\n1. Creating synthetic test data...")
    data = create_test_data()
    print(f"   Data shape: {data['occupancy'].shape}")
    print(f"   Number of occupied voxels: {(data['occupancy'] > 0).sum()}")

    # Create visualization
    print("\n2. Creating test visualization...")
    fig = create_test_visualization(data, "test_uncertainty_vis.html")

    print("\n" + "=" * 60)
    print("Test complete! 🎉")
    print("=" * 60)
