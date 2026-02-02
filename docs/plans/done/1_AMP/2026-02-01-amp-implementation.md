# AMP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Automatic Mixed Precision (AMP) to reduce memory usage and improve training speed.

**Architecture:** Global AMP with conditional GradScaler instantiation. Scaler state saved in checkpoints. Loss scale logged to TensorBoard.

**Tech Stack:** PyTorch 2.0+ `torch.amp` API (autocast, GradScaler)

---

## Task 1: Add use_amp config option

**Files:**
- Modify: `config.py:24` (Training section)

**Step 1: Add use_amp field to Config**

Add after `grad_clip` line (line 30):

```python
    # AMP
    use_amp: bool = True
```

**Step 2: Verify config loads correctly**

Run:
```bash
cd /home/comp/25481568/code/HyperBody/.worktrees/feature-amp && conda activate pasco && python -c "from config import Config; cfg = Config(); print(f'use_amp: {cfg.use_amp}')"
```
Expected: `use_amp: True`

**Step 3: Commit**

```bash
git add config.py
git commit -m "feat: add use_amp config option for AMP training

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Add AMP imports and GradScaler creation

**Files:**
- Modify: `train.py:1-15` (imports)
- Modify: `train.py:305-308` (after optimizer/scheduler creation)

**Step 1: Add torch.amp import**

Add after line 11 (`from torch.nn.parallel import ...`):

```python
from torch.amp import autocast, GradScaler
```

**Step 2: Create GradScaler after scheduler**

Add after line 308 (after scheduler creation), before `# Metrics`:

```python
    # AMP GradScaler (only if use_amp is enabled)
    scaler = GradScaler(device='cuda') if cfg.use_amp else None
    logger.info(f"AMP enabled: {cfg.use_amp}")
```

**Step 3: Verify imports work**

Run:
```bash
cd /home/comp/25481568/code/HyperBody/.worktrees/feature-amp && conda activate pasco && python -c "from torch.amp import autocast, GradScaler; print('OK')"
```
Expected: `OK`

**Step 4: Commit**

```bash
git add train.py
git commit -m "feat: add AMP imports and GradScaler initialization

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Modify train_one_epoch for AMP

**Files:**
- Modify: `train.py:116-162` (train_one_epoch function)

**Step 1: Update function signature**

Change line 116 from:
```python
def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip, epoch=0):
```
To:
```python
def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip, epoch=0, scaler=None):
```

**Step 2: Update docstring**

Add to docstring after `epoch:` parameter (around line 127):
```python
        scaler: Optional GradScaler for AMP training (None to disable AMP)
```

**Step 3: Replace training loop body**

Replace lines 146-156 (from `optimizer.zero_grad()` to `optimizer.step()`) with:

```python
        optimizer.zero_grad()

        if scaler is not None:
            with autocast(device_type='cuda'):
                logits = model(inputs)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
```

**Step 4: Verify syntax**

Run:
```bash
cd /home/comp/25481568/code/HyperBody/.worktrees/feature-amp && conda activate pasco && python -c "from train import train_one_epoch; print('OK')"
```
Expected: `OK`

**Step 5: Commit**

```bash
git add train.py
git commit -m "feat: add AMP support to train_one_epoch

- Add scaler parameter to function signature
- Use autocast context for forward pass when AMP enabled
- Use scaler.scale/unscale_/step/update for gradient handling

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Update train_one_epoch call site

**Files:**
- Modify: `train.py:335-337` (train_one_epoch call in main)

**Step 1: Pass scaler to train_one_epoch**

Change lines 335-337 from:
```python
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, cfg.grad_clip, epoch=epoch
        )
```
To:
```python
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, cfg.grad_clip, epoch=epoch, scaler=scaler
        )
```

**Step 2: Commit**

```bash
git add train.py
git commit -m "feat: pass scaler to train_one_epoch call

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Save scaler state in checkpoint

**Files:**
- Modify: `train.py:387-396` (checkpoint_state dict in main)

**Step 1: Add scaler_state to checkpoint**

Change lines 387-396 from:
```python
            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_dice": best_dice,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "mean_dice": mean_dice,
            }
```
To:
```python
            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_dice": best_dice,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "mean_dice": mean_dice,
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            }
```

**Step 2: Commit**

```bash
git add train.py
git commit -m "feat: save GradScaler state in checkpoint

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Load scaler state from checkpoint

**Files:**
- Modify: `train.py:316-321` (resume section in main)

**Step 1: Add scaler state loading**

Change lines 316-321 from:
```python
    if cfg.resume:
        logger.info(f"Resuming from checkpoint: {cfg.resume}")
        start_epoch, best_dice = load_checkpoint(
            cfg.resume, model, optimizer, scheduler, device=device
        )
        logger.info(f"Resumed at epoch {start_epoch}, best_dice={best_dice:.4f}")
```
To:
```python
    if cfg.resume:
        logger.info(f"Resuming from checkpoint: {cfg.resume}")
        start_epoch, best_dice = load_checkpoint(
            cfg.resume, model, optimizer, scheduler, device=device
        )
        # Load scaler state if available
        if scaler is not None:
            checkpoint = torch.load(cfg.resume, map_location=device)
            if checkpoint.get("scaler_state_dict") is not None:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
                logger.info("Loaded GradScaler state from checkpoint")
        logger.info(f"Resumed at epoch {start_epoch}, best_dice={best_dice:.4f}")
```

**Step 2: Commit**

```bash
git add train.py
git commit -m "feat: load GradScaler state from checkpoint on resume

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Add loss_scale TensorBoard logging

**Files:**
- Modify: `train.py:347-364` (TensorBoard logging section in main)

**Step 1: Add AMP loss_scale logging**

After line 352 (`writer.add_scalar("LR", current_lr, epoch)`), add:

```python
            # Log AMP loss scale
            if scaler is not None:
                writer.add_scalar("AMP/loss_scale", scaler.get_scale(), epoch)
```

**Step 2: Commit**

```bash
git add train.py
git commit -m "feat: log AMP loss_scale to TensorBoard

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Add float32 safety to DiceLoss

**Files:**
- Modify: `models/losses.py:17-29` (DiceLoss forward method)

**Step 1: Force float32 in DiceLoss forward**

Change lines 26-29 from:
```python
        num_classes = logits.shape[1]

        # Softmax to get probabilities
        probs = F.softmax(logits, dim=1)  # (B, C, D, H, W)
```
To:
```python
        num_classes = logits.shape[1]

        # Force float32 for numerical stability in AMP
        logits = logits.float()

        # Softmax to get probabilities
        probs = F.softmax(logits, dim=1)  # (B, C, D, H, W)
```

**Step 2: Run existing loss tests**

Run:
```bash
cd /home/comp/25481568/code/HyperBody/.worktrees/feature-amp && conda activate pasco && python -m pytest tests/test_model.py -k "loss" -v
```
Expected: All loss tests pass

**Step 3: Commit**

```bash
git add models/losses.py
git commit -m "feat: force float32 in DiceLoss for AMP numerical stability

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 9: Write AMP unit test

**Files:**
- Create: `tests/test_amp.py`

**Step 1: Create test file**

```python
"""Tests for AMP (Automatic Mixed Precision) training support."""

import pytest
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from config import Config
from models.unet3d import UNet3D
from models.losses import CombinedLoss, DiceLoss


class TestAMPConfig:
    """Test AMP configuration."""

    def test_config_has_use_amp(self):
        """Config should have use_amp field."""
        cfg = Config()
        assert hasattr(cfg, 'use_amp')
        assert isinstance(cfg.use_amp, bool)

    def test_config_use_amp_default_true(self):
        """use_amp should default to True."""
        cfg = Config()
        assert cfg.use_amp is True


class TestAMPForwardPass:
    """Test model forward pass with AMP."""

    @pytest.fixture
    def model(self):
        """Create a small model for testing."""
        return UNet3D(
            in_channels=1,
            num_classes=10,
            base_channels=8,
            growth_rate=8,
            dense_layers=2,
            bn_size=2,
        ).cuda()

    @pytest.fixture
    def inputs(self):
        """Create test inputs."""
        return torch.randn(1, 1, 32, 32, 32).cuda()

    @pytest.fixture
    def targets(self):
        """Create test targets."""
        return torch.randint(0, 10, (1, 32, 32, 32)).cuda()

    def test_forward_with_autocast(self, model, inputs):
        """Model forward pass should work with autocast."""
        with autocast(device_type='cuda'):
            outputs = model(inputs)
        assert outputs.shape == (1, 10, 32, 32, 32)
        # Output should be float16 inside autocast
        assert outputs.dtype == torch.float16

    def test_loss_with_autocast(self, model, inputs, targets):
        """Loss computation should work with autocast."""
        criterion = CombinedLoss(num_classes=10)
        with autocast(device_type='cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        assert loss.dtype == torch.float32  # Loss should be float32

    def test_backward_with_scaler(self, model, inputs, targets):
        """Backward pass should work with GradScaler."""
        criterion = CombinedLoss(num_classes=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = GradScaler(device='cuda')

        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Should complete without error
        assert scaler.get_scale() > 0


class TestDiceLossFloat32:
    """Test DiceLoss float32 safety."""

    def test_dice_loss_outputs_float32(self):
        """DiceLoss should output float32 even with float16 input."""
        loss_fn = DiceLoss()
        # Simulate float16 input (as would happen inside autocast)
        logits = torch.randn(1, 10, 8, 8, 8, dtype=torch.float16).cuda()
        targets = torch.randint(0, 10, (1, 8, 8, 8)).cuda()

        loss = loss_fn(logits, targets)
        assert loss.dtype == torch.float32


class TestGradScalerState:
    """Test GradScaler state save/load."""

    def test_scaler_state_dict(self):
        """GradScaler should have saveable state."""
        scaler = GradScaler(device='cuda')
        state = scaler.state_dict()
        assert 'scale' in state
        assert '_growth_tracker' in state

    def test_scaler_load_state_dict(self):
        """GradScaler should restore from state dict."""
        scaler1 = GradScaler(device='cuda')
        # Simulate some training that changes scale
        scaler1._scale = torch.tensor(1024.0)

        state = scaler1.state_dict()

        scaler2 = GradScaler(device='cuda')
        scaler2.load_state_dict(state)

        assert scaler2.get_scale() == 1024.0
```

**Step 2: Run the test**

Run:
```bash
cd /home/comp/25481568/code/HyperBody/.worktrees/feature-amp && conda activate pasco && python -m pytest tests/test_amp.py -v
```
Expected: All tests pass

**Step 3: Commit**

```bash
git add tests/test_amp.py
git commit -m "test: add AMP unit tests

- Test config has use_amp field
- Test forward pass with autocast
- Test loss computation with autocast
- Test backward pass with GradScaler
- Test DiceLoss float32 safety
- Test GradScaler state save/load

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 10: Run all tests and verify

**Files:** None (verification only)

**Step 1: Run all tests**

Run:
```bash
cd /home/comp/25481568/code/HyperBody/.worktrees/feature-amp && conda activate pasco && python -m pytest tests/test_amp.py tests/test_model.py tests/test_ddp.py tests/test_checkpoint.py -v
```
Expected: All core tests pass

**Step 2: Verify train.py syntax**

Run:
```bash
cd /home/comp/25481568/code/HyperBody/.worktrees/feature-amp && conda activate pasco && python -c "import train; print('train.py OK')"
```
Expected: `train.py OK`

**Step 3: Final commit (if any fixes needed)**

If all tests pass, no commit needed. Otherwise fix and commit.

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Add use_amp config | config.py |
| 2 | Add imports + GradScaler | train.py |
| 3 | Modify train_one_epoch | train.py |
| 4 | Update call site | train.py |
| 5 | Save scaler in checkpoint | train.py |
| 6 | Load scaler from checkpoint | train.py |
| 7 | TensorBoard logging | train.py |
| 8 | DiceLoss float32 safety | models/losses.py |
| 9 | Write AMP tests | tests/test_amp.py |
| 10 | Verify all tests pass | (verification) |
