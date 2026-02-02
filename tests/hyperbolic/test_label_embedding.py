import torch
import pytest
import json


class TestLorentzLabelEmbedding:
    """Test LorentzLabelEmbedding module."""

    @pytest.fixture
    def class_depths(self):
        """Load real class depths from dataset."""
        from data.organ_hierarchy import load_organ_hierarchy
        with open("Dataset/dataset_info.json") as f:
            class_names = json.load(f)["class_names"]
        return load_organ_hierarchy("Dataset/tree.json", class_names)

    def test_output_shape(self, class_depths):
        """Output should be [num_classes, embed_dim]."""
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding

        emb = LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=class_depths
        )
        out = emb()
        assert out.shape == (70, 32), f"Expected (70, 32), got {out.shape}"

    def test_output_is_on_manifold(self, class_depths):
        """Output should be valid Lorentz points (finite values)."""
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding

        emb = LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=class_depths
        )
        out = emb()
        assert torch.isfinite(out).all(), "Output contains inf or nan"

    def test_deeper_organs_farther_from_origin(self, class_depths):
        """Deeper organs should be initialized farther from origin."""
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding
        from models.hyperbolic.lorentz_ops import distance_to_origin

        torch.manual_seed(42)
        emb = LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=class_depths,
            min_radius=0.1,
            max_radius=2.0
        )
        out = emb()
        distances = distance_to_origin(out)

        # Find a shallow and deep class
        min_depth = min(class_depths.values())
        max_depth = max(class_depths.values())

        shallow_idx = [i for i, d in class_depths.items() if d == min_depth][0]
        deep_idx = [i for i, d in class_depths.items() if d == max_depth][0]

        assert distances[deep_idx] > distances[shallow_idx], \
            f"Deep class dist {distances[deep_idx]:.4f} should be > shallow {distances[shallow_idx]:.4f}"

    def test_gradient_flow(self, class_depths):
        """Gradients should flow through the embedding."""
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding

        emb = LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=class_depths
        )
        out = emb()
        loss = out.sum()
        loss.backward()

        # Check tangent_embeddings has gradients
        assert emb.tangent_embeddings.grad is not None
        assert (emb.tangent_embeddings.grad != 0).any()

    def test_different_seeds_different_directions(self, class_depths):
        """Different random seeds should give different initial directions."""
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding

        torch.manual_seed(42)
        emb1 = LorentzLabelEmbedding(num_classes=70, embed_dim=32, class_depths=class_depths)

        torch.manual_seed(123)
        emb2 = LorentzLabelEmbedding(num_classes=70, embed_dim=32, class_depths=class_depths)

        # Directions should differ
        assert not torch.allclose(emb1.tangent_embeddings, emb2.tangent_embeddings)
