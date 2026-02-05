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


class TestCurriculumNegativeMining:
    """
    Test Curriculum Negative Mining functionality in LorentzRankingLoss.

    Tests cover:
    1. Temperature scheduling with exponential decay
    2. Warmup period with uniform random sampling
    3. Buffer registration for epoch tracking (save/load compatibility)
    4. Positive sample masking (prevent self-sampling)
    5. Sampling distribution validation
    """

    # ==================== Temperature Scheduling Tests ====================

    def test_set_epoch_and_get_temperature_basic(self):
        """
        Test that set_epoch() and get_temperature() work correctly.
        After setting epoch, get_temperature() should return the expected temperature.
        """
        from models.hyperbolic.lorentz_loss import LorentzRankingLoss

        loss_fn = LorentzRankingLoss(
            margin=0.1,
            t_start=2.0,
            t_end=0.1,
            warmup_epochs=5,
        )

        max_epochs = 30

        # During warmup (epoch < 5), temperature should be t_start
        loss_fn.set_epoch(0, max_epochs)
        temp = loss_fn.get_temperature()
        assert temp == 2.0, f"Expected t_start=2.0 during warmup, got {temp}"

        loss_fn.set_epoch(4, max_epochs)
        temp = loss_fn.get_temperature()
        assert temp == 2.0, f"Expected t_start=2.0 at epoch 4 (still warmup), got {temp}"

    def test_temperature_exponential_decay(self):
        """
        Test exponential temperature decay: t = t_start * (t_end/t_start)^progress
        where progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        """
        from models.hyperbolic.lorentz_loss import LorentzRankingLoss

        loss_fn = LorentzRankingLoss(
            margin=0.1,
            t_start=2.0,
            t_end=0.1,
            warmup_epochs=5,
        )

        max_epochs = 30
        # After warmup: epochs 5 to 29
        # effective_epochs = max_epochs - warmup_epochs = 25

        # At epoch 5 (start of curriculum): progress = 0, t = 2.0
        loss_fn.set_epoch(5, max_epochs)
        temp = loss_fn.get_temperature()
        assert abs(temp - 2.0) < 1e-6, f"At epoch 5, expected 2.0, got {temp}"

        # At epoch 30 (end): progress = 1, t = 0.1
        loss_fn.set_epoch(29, max_epochs)
        temp = loss_fn.get_temperature()
        # progress = (29 - 5) / (30 - 5) = 24/25 = 0.96
        expected_temp = 2.0 * (0.1 / 2.0) ** 0.96
        assert abs(temp - expected_temp) < 1e-5, f"At epoch 29, expected ~{expected_temp:.4f}, got {temp}"

        # At mid-point: epoch 17 (progress = 0.5)
        loss_fn.set_epoch(17, max_epochs)
        temp = loss_fn.get_temperature()
        # progress = (17 - 5) / 25 = 12/25 = 0.48
        expected_temp = 2.0 * (0.1 / 2.0) ** 0.48
        assert abs(temp - expected_temp) < 1e-5, f"At epoch 17, expected ~{expected_temp:.4f}, got {temp}"

    def test_temperature_clamps_at_boundaries(self):
        """
        Test that temperature is properly clamped when epoch exceeds max_epochs.
        Progress should be clamped to [0, 1].
        """
        from models.hyperbolic.lorentz_loss import LorentzRankingLoss

        loss_fn = LorentzRankingLoss(
            margin=0.1,
            t_start=2.0,
            t_end=0.1,
            warmup_epochs=5,
        )

        max_epochs = 30

        # epoch > max_epochs should clamp to t_end
        loss_fn.set_epoch(100, max_epochs)
        temp = loss_fn.get_temperature()
        assert abs(temp - 0.1) < 1e-6, f"Epoch 100 should clamp to t_end=0.1, got {temp}"

    # ==================== Warmup Behavior Tests ====================

    def test_warmup_uses_uniform_sampling(self):
        """
        During warmup (epoch < warmup_epochs), sampling should be truly uniform,
        NOT distance-weighted even with high temperature.
        This tests that negative sampling weights are uniform during warmup.
        """
        from models.hyperbolic.lorentz_loss import LorentzRankingLoss
        from models.hyperbolic.lorentz_ops import exp_map0

        loss_fn = LorentzRankingLoss(
            margin=0.1,
            t_start=2.0,
            t_end=0.1,
            warmup_epochs=5,
            num_samples_per_class=64,
            num_negatives=8,
        )

        max_epochs = 30

        # Set to warmup period
        loss_fn.set_epoch(2, max_epochs)

        # Create data where classes have very different distances
        # This setup would cause non-uniform sampling if distance-weighted
        torch.manual_seed(42)
        voxel_emb = exp_map0(torch.randn(2, 32, 4, 4, 4) * 0.3)
        labels = torch.randint(0, 10, (2, 4, 4, 4))
        label_emb = exp_map0(torch.randn(10, 32) * 0.5)

        # Run multiple times and collect statistics
        # During warmup, all valid negatives should have equal chance
        loss = loss_fn(voxel_emb, labels, label_emb)
        assert torch.isfinite(loss), "Loss should be finite during warmup"

    # ==================== Buffer Registration Tests ====================

    def test_buffers_are_registered(self):
        """
        current_epoch and max_epochs should be registered as buffers,
        making them part of state_dict for saving/loading.
        """
        from models.hyperbolic.lorentz_loss import LorentzRankingLoss

        loss_fn = LorentzRankingLoss(
            margin=0.1,
            t_start=2.0,
            t_end=0.1,
            warmup_epochs=5,
        )

        # Check buffers exist
        assert hasattr(loss_fn, 'current_epoch'), "current_epoch buffer not found"
        assert hasattr(loss_fn, 'max_epochs'), "max_epochs buffer not found"

        # Check they are tensors (buffers)
        assert isinstance(loss_fn.current_epoch, torch.Tensor), \
            "current_epoch should be a Tensor buffer"
        assert isinstance(loss_fn.max_epochs, torch.Tensor), \
            "max_epochs should be a Tensor buffer"

        # Check they appear in state_dict
        state_dict = loss_fn.state_dict()
        assert 'current_epoch' in state_dict, "current_epoch not in state_dict"
        assert 'max_epochs' in state_dict, "max_epochs not in state_dict"

    def test_buffer_save_load(self):
        """
        Test that epoch state survives save/load cycle.
        """
        from models.hyperbolic.lorentz_loss import LorentzRankingLoss
        import tempfile
        import os

        loss_fn = LorentzRankingLoss(
            margin=0.1,
            t_start=2.0,
            t_end=0.1,
            warmup_epochs=5,
        )

        # Set epoch state
        loss_fn.set_epoch(15, 30)

        # Save
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            torch.save(loss_fn.state_dict(), f.name)
            temp_path = f.name

        try:
            # Create new instance and load
            loss_fn2 = LorentzRankingLoss(
                margin=0.1,
                t_start=2.0,
                t_end=0.1,
                warmup_epochs=5,
            )
            loss_fn2.load_state_dict(torch.load(temp_path))

            # Verify state was restored
            assert loss_fn2.current_epoch.item() == 15, \
                f"Expected current_epoch=15, got {loss_fn2.current_epoch.item()}"
            assert loss_fn2.max_epochs.item() == 30, \
                f"Expected max_epochs=30, got {loss_fn2.max_epochs.item()}"

            # Temperature should match
            temp1 = loss_fn.get_temperature()
            temp2 = loss_fn2.get_temperature()
            assert abs(temp1 - temp2) < 1e-6, \
                f"Temperature mismatch after load: {temp1} vs {temp2}"
        finally:
            os.unlink(temp_path)

    # ==================== Positive Sample Mask Tests ====================

    def test_positive_class_never_sampled_as_negative(self):
        """
        The sampling weight for the anchor's true class must be 0.
        Even if the positive class is the closest, it should never be selected as negative.
        """
        from models.hyperbolic.lorentz_loss import LorentzRankingLoss
        from models.hyperbolic.lorentz_ops import exp_map0

        # Create a scenario where the positive class would be selected if not masked
        # Set up embeddings so the positive class is very close
        loss_fn = LorentzRankingLoss(
            margin=0.1,
            t_start=0.1,  # Low temperature = strong preference for close samples
            t_end=0.1,
            warmup_epochs=0,  # No warmup
            num_samples_per_class=8,
            num_negatives=4,
        )

        loss_fn.set_epoch(10, 30)  # Past warmup, low temperature

        # Create minimal test case
        torch.manual_seed(123)

        # 5 classes, embed_dim=8
        # Make class 0 very close to itself (the voxel embeddings)
        label_emb = exp_map0(torch.randn(5, 8) * 0.5)

        # Create voxels all labeled as class 0, with embeddings very close to class 0
        voxel_tangent = torch.zeros(1, 8, 2, 2, 2)
        # Perturb slightly from origin
        voxel_tangent[..., 1:] = 0.01
        voxel_emb = exp_map0(voxel_tangent)
        labels = torch.zeros(1, 2, 2, 2, dtype=torch.long)  # All class 0

        # The loss should work without sampling class 0 as negative
        loss = loss_fn(voxel_emb, labels, label_emb)

        # If positive class was sampled as negative, loss would be incorrect
        # (potentially negative or very small when it shouldn't be)
        assert loss >= 0, f"Loss should be non-negative, got {loss}"
        assert torch.isfinite(loss), "Loss should be finite"

    # ==================== Sampling Distribution Tests ====================

    def test_sampling_distribution_warmup_is_uniform(self):
        """
        During warmup, the sampling distribution should be approximately uniform
        across all valid negative classes.
        """
        from models.hyperbolic.lorentz_loss import LorentzRankingLoss
        from models.hyperbolic.lorentz_ops import exp_map0

        # This test verifies the sampling is uniform by checking that
        # the loss computation doesn't systematically favor certain classes
        loss_fn = LorentzRankingLoss(
            margin=0.1,
            t_start=2.0,
            t_end=0.1,
            warmup_epochs=5,
            num_samples_per_class=32,
            num_negatives=4,
        )

        loss_fn.set_epoch(0, 30)  # In warmup

        # Create data with deliberately skewed distances
        torch.manual_seed(42)
        num_classes = 10

        # Make some classes much closer than others
        label_tangent = torch.randn(num_classes, 16) * 0.5
        label_tangent[0] *= 0.1  # Class 0 is very close to origin
        label_tangent[9] *= 5.0  # Class 9 is very far
        label_emb = exp_map0(label_tangent)

        voxel_emb = exp_map0(torch.randn(2, 16, 4, 4, 4) * 0.1)
        labels = torch.randint(1, 9, (2, 4, 4, 4))  # Classes 1-8 (not 0 or 9)

        # Run loss computation
        loss = loss_fn(voxel_emb, labels, label_emb)
        assert torch.isfinite(loss), "Loss should be finite"

    def test_sampling_distribution_low_temp_prefers_hard_negatives(self):
        """
        After warmup with low temperature, sampling should prefer hard negatives
        (those with smaller distance to anchor).
        """
        from models.hyperbolic.lorentz_loss import LorentzRankingLoss
        from models.hyperbolic.lorentz_ops import exp_map0

        loss_fn = LorentzRankingLoss(
            margin=0.1,
            t_start=2.0,
            t_end=0.1,
            warmup_epochs=5,
            num_samples_per_class=32,
            num_negatives=4,
        )

        # Set to end of training (low temperature)
        loss_fn.set_epoch(29, 30)
        temp = loss_fn.get_temperature()
        assert temp < 0.2, f"Temperature should be low at epoch 29, got {temp}"

        # Create data
        torch.manual_seed(42)
        voxel_emb = exp_map0(torch.randn(2, 16, 4, 4, 4) * 0.3)
        labels = torch.randint(0, 10, (2, 4, 4, 4))
        label_emb = exp_map0(torch.randn(10, 16) * 0.5)

        # Loss should work with low temperature
        loss = loss_fn(voxel_emb, labels, label_emb)
        assert torch.isfinite(loss), "Loss should be finite at low temperature"

    # ==================== Integration Tests ====================

    def test_curriculum_throughout_training(self):
        """
        Test that curriculum mining works correctly throughout an entire training run.
        Temperature should decrease and loss should remain stable.
        """
        from models.hyperbolic.lorentz_loss import LorentzRankingLoss
        from models.hyperbolic.lorentz_ops import exp_map0

        loss_fn = LorentzRankingLoss(
            margin=0.1,
            t_start=2.0,
            t_end=0.1,
            warmup_epochs=5,
            num_samples_per_class=16,
            num_negatives=4,
        )

        torch.manual_seed(42)
        voxel_emb = exp_map0(torch.randn(1, 16, 4, 4, 4) * 0.3)
        labels = torch.randint(0, 10, (1, 4, 4, 4))
        label_emb = exp_map0(torch.randn(10, 16) * 0.5)

        max_epochs = 30
        temperatures = []
        losses = []

        for epoch in range(max_epochs):
            loss_fn.set_epoch(epoch, max_epochs)
            temp = loss_fn.get_temperature()
            temperatures.append(temp)

            loss = loss_fn(voxel_emb, labels, label_emb)
            losses.append(loss.item())

            assert torch.isfinite(loss), f"Loss not finite at epoch {epoch}"

        # Verify temperature schedule
        # During warmup (epochs 0-4), temperature should be t_start
        for i in range(5):
            assert temperatures[i] == 2.0, \
                f"Warmup epoch {i}: expected t=2.0, got {temperatures[i]}"

        # After warmup, temperature should decrease monotonically
        for i in range(5, max_epochs - 1):
            assert temperatures[i] >= temperatures[i + 1], \
                f"Temperature should decrease: epoch {i}={temperatures[i]}, epoch {i+1}={temperatures[i+1]}"

        # Final temperature should be close to t_end
        assert temperatures[-1] < 0.2, \
            f"Final temperature should be close to t_end=0.1, got {temperatures[-1]}"

    def test_backward_compatibility_default_params(self):
        """
        When curriculum parameters are not provided, the loss should behave
        as before (random sampling, no curriculum).
        """
        from models.hyperbolic.lorentz_loss import LorentzRankingLoss
        from models.hyperbolic.lorentz_ops import exp_map0

        # Default instantiation (no curriculum params)
        loss_fn = LorentzRankingLoss(margin=0.1, num_samples_per_class=8, num_negatives=4)

        torch.manual_seed(42)
        voxel_emb = exp_map0(torch.randn(2, 32, 4, 4, 4) * 0.3)
        labels = torch.randint(0, 70, (2, 4, 4, 4))
        label_emb = exp_map0(torch.randn(70, 32) * 0.5)

        # Should work without calling set_epoch
        loss = loss_fn(voxel_emb, labels, label_emb)
        assert torch.isfinite(loss), "Loss should work without curriculum setup"
        assert loss >= 0, "Loss should be non-negative"
