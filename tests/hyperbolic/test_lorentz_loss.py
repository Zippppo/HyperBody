import torch
import pytest


class TestLorentzRankingLoss:
    """Test LorentzRankingLoss module."""

    def test_output_is_scalar(self):
        """Loss should return a scalar."""
        from models.hyperbolic.lorentz_loss import LorentzRankingLoss
        from models.hyperbolic.lorentz_ops import exp_map0

        loss_fn = LorentzRankingLoss(margin=0.1, num_samples_per_class=8, num_negatives=4)

        # Create fake data
        voxel_emb = exp_map0(torch.randn(2, 32, 4, 4, 4) * 0.3)
        labels = torch.randint(0, 70, (2, 4, 4, 4))
        label_emb = exp_map0(torch.randn(70, 32) * 0.5)

        loss = loss_fn(voxel_emb, labels, label_emb)
        assert loss.dim() == 0, f"Expected scalar, got shape {loss.shape}"

    def test_loss_is_non_negative(self):
        """Loss should be non-negative (triplet margin loss)."""
        from models.hyperbolic.lorentz_loss import LorentzRankingLoss
        from models.hyperbolic.lorentz_ops import exp_map0

        loss_fn = LorentzRankingLoss(margin=0.1, num_samples_per_class=8, num_negatives=4)

        voxel_emb = exp_map0(torch.randn(2, 32, 4, 4, 4) * 0.3)
        labels = torch.randint(0, 70, (2, 4, 4, 4))
        label_emb = exp_map0(torch.randn(70, 32) * 0.5)

        loss = loss_fn(voxel_emb, labels, label_emb)
        assert loss >= 0, f"Loss should be >= 0, got {loss}"

    def test_gradient_flow_to_voxel_emb(self):
        """Gradients should flow to voxel embeddings."""
        from models.hyperbolic.lorentz_loss import LorentzRankingLoss
        from models.hyperbolic.lorentz_ops import exp_map0

        loss_fn = LorentzRankingLoss(margin=0.1, num_samples_per_class=8, num_negatives=4)

        tangent = torch.randn(2, 32, 4, 4, 4) * 0.3
        tangent.requires_grad = True
        voxel_emb = exp_map0(tangent)

        labels = torch.randint(0, 70, (2, 4, 4, 4))
        label_emb = exp_map0(torch.randn(70, 32) * 0.5)

        loss = loss_fn(voxel_emb, labels, label_emb)
        loss.backward()

        assert tangent.grad is not None, "No gradient for voxel tangent vectors"

    def test_gradient_flow_to_label_emb(self):
        """Gradients should flow to label embeddings."""
        from models.hyperbolic.lorentz_loss import LorentzRankingLoss
        from models.hyperbolic.lorentz_ops import exp_map0

        loss_fn = LorentzRankingLoss(margin=0.1, num_samples_per_class=8, num_negatives=4)

        voxel_emb = exp_map0(torch.randn(2, 32, 4, 4, 4) * 0.3)
        labels = torch.randint(0, 70, (2, 4, 4, 4))

        label_tangent = torch.randn(70, 32) * 0.5
        label_tangent.requires_grad = True
        label_emb = exp_map0(label_tangent)

        loss = loss_fn(voxel_emb, labels, label_emb)
        loss.backward()

        assert label_tangent.grad is not None, "No gradient for label tangent vectors"

    def test_handles_single_class_batch(self):
        """Should handle batches with only one class present."""
        from models.hyperbolic.lorentz_loss import LorentzRankingLoss
        from models.hyperbolic.lorentz_ops import exp_map0

        loss_fn = LorentzRankingLoss(margin=0.1, num_samples_per_class=8, num_negatives=4)

        voxel_emb = exp_map0(torch.randn(1, 32, 4, 4, 4) * 0.3)
        labels = torch.zeros(1, 4, 4, 4, dtype=torch.long)  # All class 0
        label_emb = exp_map0(torch.randn(70, 32) * 0.5)

        # Should not crash, may return 0 if no negatives available
        loss = loss_fn(voxel_emb, labels, label_emb)
        assert torch.isfinite(loss), "Loss is not finite"
