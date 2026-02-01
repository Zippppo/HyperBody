# AMP (Automatic Mixed Precision) Design

Date: 2026-02-01

## Overview

Add Automatic Mixed Precision (AMP) to the HyperBody training pipeline to reduce memory usage and improve training speed.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Granularity | Simple global mode | Single `use_amp` flag, autocast wraps entire forward + loss |
| GradScaler handling | Conditional instantiation | `if cfg.use_amp` guards, clearer logic |
| Checkpoint | Save scaler state | Resume training without warmup |
| Logging | Record loss_scale | Monitor gradient overflow/underflow |

## Implementation

### 1. Configuration (`config.py`)

```python
use_amp: bool = True  # Enable automatic mixed precision
```

### 2. Imports (`train.py`)

```python
from torch.amp import autocast, GradScaler  # PyTorch 2.0+ API
```

### 3. GradScaler Initialization (`train.py` main)

```python
scaler = GradScaler(device='cuda') if cfg.use_amp else None
logger.info(f"AMP enabled: {cfg.use_amp}")
```

### 4. Training Loop (`train_one_epoch`)

**Function signature:**
```python
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, cfg, scaler=None):
```

**Core logic:**
```python
optimizer.zero_grad()

if scaler is not None:
    with autocast(device_type='cuda'):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    scaler.step(optimizer)
    scaler.update()
else:
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
```

### 5. Checkpoint Save/Load

**Save:**
```python
checkpoint = {
    "epoch": epoch,
    "model_state": model_state,
    "optimizer_state": optimizer.state_dict(),
    "scheduler_state": scheduler.state_dict(),
    "best_dice": best_dice,
    "train_losses": train_losses,
    "val_losses": val_losses,
    "scaler_state": scaler.state_dict() if scaler is not None else None,
}
```

**Load:**
```python
if scaler is not None and checkpoint.get("scaler_state") is not None:
    scaler.load_state_dict(checkpoint["scaler_state"])
```

### 6. TensorBoard Logging

```python
if is_main_process() and scaler is not None:
    writer.add_scalar("AMP/loss_scale", scaler.get_scale(), epoch)
```

## Numerical Precision Protection

### Loss Function Safety

DiceLoss uses `F.softmax()` which autocast handles correctly. For extra safety, force float32 in custom loss:

```python
def forward(self, pred, target):
    pred = pred.float()  # Force float32
    pred = F.softmax(pred, dim=1)
    # ... rest of computation ...
```

### Metric Calculation

Metrics compute outside autocast context. For probability-based metrics, explicitly convert:

```python
probs = outputs.float().softmax(dim=1)  # Explicit float32
```

## Files to Modify

| File | Changes |
|------|---------|
| `config.py` | Add `use_amp: bool = True` |
| `train.py` | Import torch.amp, create GradScaler, modify train_one_epoch, checkpoint save/load, TensorBoard logging |
| `models/losses.py` | (Optional) Force float32 in DiceLoss forward |

## Files Unchanged

- `models/unet3d.py` — autocast handles automatically
- `models/dense_block.py` — autocast handles automatically
- `utils/metrics.py` — already operates in float32

## Expected Results

- Memory reduction: ~30-50%
- Speed improvement: significant on Ampere+ GPUs
- Numerical precision: maintained
