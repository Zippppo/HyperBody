# Plan: Pre-trained Text Embeddings for Hyperbolic Label Embedding

## Goal
Replace random label embedding initialization with pre-trained text embeddings:
```
text_embedding (768/512) -> MLP -> 32dim -> project to Hyperbolic (exp_map0)
```

## Current Problem
- `LorentzLabelEmbedding._init_from_hierarchy()` uses `torch.randn(embed_dim)` with random direction
- Only hierarchy depth is used for scaling, no semantic information

## Available Pre-trained Embeddings
| Model | Shape | dtype | File |
|-------|-------|-------|------|
| SAT | [72, 768] | float32 | `Dataset/text_embeddings/sat_labeled_embeddings.pt` |
| CLIP | [72, 768] | float16 | `Dataset/text_embeddings/clip_labeled_embeddings.pt` |
| BioMedCLIP | [72, 512] | float32 | `Dataset/text_embeddings/biomedclip_labeled_embeddings.pt` |

**File Structure**:
```python
{
    "label_names": ["outside_body", "inside_body_empty", "liver", ...],  # 72 items
    "label_ids": tensor([0, 1, 2, 7, 8, 6, ...]),  # NOT consecutive!
    "embeddings": tensor([72, 768/512])  # embeddings[i] -> label_ids[i]
}
```

**Critical**: `label_ids` are NOT consecutive. Must build mapping: `reordered[label_id] = embeddings[idx]`

---

## Implementation Steps

### Step 1: Create TextEmbeddingProjector Module
**File**: `pasco/models/hyperbolic/text_projector.py` (NEW)

```python
import torch
import torch.nn as nn


class TextEmbeddingProjector(nn.Module):
    """
    Project pre-trained text embeddings to hyperbolic tangent space.

    Architecture: text_dim -> hidden_dim (LayerNorm + GELU) -> embed_dim

    The output layer uses small initialization (std=0.02) to keep
    initial tangent vectors in a reasonable range for exp_map0.
    """

    def __init__(self, text_dim: int = 768, embed_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.text_dim = text_dim
        self.embed_dim = embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Small initialization for output layer to keep tangent vectors reasonable
        self._init_weights()

    def _init_weights(self):
        """Initialize output layer with small weights."""
        # Output layer is the last Linear in mlp (index 3)
        output_layer = self.mlp[3]
        nn.init.normal_(output_layer.weight, std=0.02)
        nn.init.zeros_(output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Text embeddings [N, text_dim]
        Returns:
            Tangent vectors [N, embed_dim]
        """
        return self.mlp(x)
```

### Step 2: Modify LorentzLabelEmbedding
**File**: `pasco/models/hyperbolic/label_embedding.py`

#### 2.1 Add constants and default paths
```python
# Default paths for text embeddings
TEXT_EMBEDDING_PATHS = {
    "sat": "Dataset/text_embeddings/sat_labeled_embeddings.pt",
    "clip": "Dataset/text_embeddings/clip_labeled_embeddings.pt",
    "biomedclip": "Dataset/text_embeddings/biomedclip_labeled_embeddings.pt",
}

TEXT_EMBEDDING_DIMS = {
    "sat": 768,
    "clip": 768,
    "biomedclip": 512,
}
```

#### 2.2 Add new parameters to `__init__`
```python
def __init__(
    self,
    n_classes=71,
    embed_dim=32,
    curv=1.0,
    ignore_class=0,
    min_radius=0.1,
    max_radius=2.0,
    include_virtual=False,
    # NEW: Text embedding parameters
    use_text_embeddings: bool = False,
    text_embedding_type: str = None,  # "sat", "clip", "biomedclip"
):
```

#### 2.3 Add new initialization method
```python
def _init_from_text_embeddings(self, embedding_type: str):
    """
    Initialize from pre-trained text embeddings.

    Steps:
    1. Load embeddings from file
    2. Convert to float16 (unified dtype)
    3. Build label_id -> array_index mapping
    4. Reorder to tensor[label_id] = embedding
    5. Create TextEmbeddingProjector
    6. Store text_embeddings as buffer (non-trainable)

    Args:
        embedding_type: One of "sat", "clip", "biomedclip"
    """
    from .text_projector import TextEmbeddingProjector

    # 1. Resolve path and load
    if embedding_type not in TEXT_EMBEDDING_PATHS:
        raise ValueError(f"Unknown embedding type: {embedding_type}. "
                        f"Choose from {list(TEXT_EMBEDDING_PATHS.keys())}")

    path = TEXT_EMBEDDING_PATHS[embedding_type]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Text embedding file not found: {path}")

    data = torch.load(path, map_location="cpu", weights_only=False)

    # 2. Extract and validate
    label_ids = data["label_ids"]  # [72]
    embeddings = data["embeddings"]  # [72, text_dim]

    assert len(label_ids) == embeddings.shape[0], \
        f"label_ids ({len(label_ids)}) != embeddings ({embeddings.shape[0]})"
    assert label_ids.max().item() < self.n_classes, \
        f"max label_id ({label_ids.max()}) >= n_classes ({self.n_classes})"

    text_dim = embeddings.shape[1]
    expected_dim = TEXT_EMBEDDING_DIMS[embedding_type]
    assert text_dim == expected_dim, \
        f"Expected dim {expected_dim}, got {text_dim}"

    # 3. Convert to float16 (unified dtype for all embedding types)
    embeddings = embeddings.to(torch.float16)

    # 4. Reorder: tensor[label_id] = embedding
    reordered = torch.zeros(self.n_classes, text_dim, dtype=torch.float16)
    for idx, label_id in enumerate(label_ids):
        lid = label_id.item()
        if lid < self.n_classes:
            reordered[lid] = embeddings[idx]

    # 5. Register as buffer (non-trainable)
    self.register_buffer("text_embeddings", reordered)

    # 6. Create projector (trainable)
    self.projector = TextEmbeddingProjector(
        text_dim=text_dim,
        embed_dim=self.embed_dim,
        hidden_dim=256,
    )

    # Note: tangent_vectors parameter is NOT used when use_text_embeddings=True
    # We keep it for backward compatibility but set requires_grad=False
    self.tangent_vectors.requires_grad_(False)
```

#### 2.4 Modify `forward()`
```python
def forward(self, class_indices=None):
    """
    Get embeddings for given class indices.

    When use_text_embeddings=True:
    - Projects text_embeddings through trainable MLP
    - Recomputes every forward (intentional: projector params update during training)
    - Class 0 (ignore_class) is explicitly zeroed

    Args:
        class_indices: Optional tensor of class indices. If None, return all.

    Returns:
        Embeddings tensor (space components on hyperboloid)
    """
    if self.use_text_embeddings:
        # Project text embeddings to tangent space
        # NOTE: Recomputed every forward because projector is trainable
        tangent_vectors = self.projector(self.text_embeddings.float())  # float16 -> float32 for MLP

        # CRITICAL: Force ignore_class to zero tangent vector
        tangent_vectors[self.ignore_class] = 0.0
    else:
        tangent_vectors = self.tangent_vectors

    # Map tangent vectors to hyperboloid
    embeddings = exp_map0(tangent_vectors, curv=self.curv)

    if class_indices is None:
        return embeddings
    return embeddings[class_indices]
```

### Step 3: Update BodyNetHyperbolic
**File**: `pasco/models/body_net_hyperbolic.py`

```python
def __init__(
    self,
    n_classes=71,
    embed_dim=32,
    hyperbolic_weight=0.1,
    margin=0.1,
    use_entailment_cone=True,
    entailment_weight=0.1,
    entail_loss_weight=0.1,
    contra_loss_weight=1.0,
    pos_loss_weight=0.1,
    include_virtual_nodes=True,
    max_voxels_ranking: Optional[int] = None,
    ranking_beta: float = 0.999,
    class_frequencies: Optional[np.ndarray] = None,
    # NEW: Text embedding parameters
    use_text_embeddings: bool = False,
    text_embedding_type: str = None,
    **kwargs
):
    super().__init__(n_classes=n_classes, **kwargs)

    # Store text embedding config
    self.use_text_embeddings = use_text_embeddings
    self.text_embedding_type = text_embedding_type

    # IMPORTANT: When using text embeddings, disable virtual nodes
    # (text embeddings only cover real classes, not virtual hierarchy nodes)
    if use_text_embeddings and include_virtual_nodes:
        print("Warning: Disabling virtual_nodes when use_text_embeddings=True")
        include_virtual_nodes = False

    # ... existing code ...

    # Lorentz label embeddings
    self.label_emb = LorentzLabelEmbedding(
        n_classes=n_classes,
        embed_dim=embed_dim,
        curv=CURV,
        ignore_class=0,
        include_virtual=include_virtual_nodes,
        use_text_embeddings=use_text_embeddings,
        text_embedding_type=text_embedding_type,
    )
```

### Step 4: Add CLI Arguments
**File**: `scripts/body/train_body.py`

```python
# Text embedding arguments (add after hyperbolic args)
parser.add_argument("--use_text_embeddings", action="store_true",
                    help="Use pre-trained text embeddings for label initialization")
parser.add_argument("--text_embedding_type", type=str, default="sat",
                    choices=["sat", "clip", "biomedclip"],
                    help="Type of text embeddings to use")
```

Update model creation:
```python
if args.use_hyperbolic:
    from pasco.models.body_net_hyperbolic import BodyNetHyperbolic
    model = BodyNetHyperbolic(
        # ... existing args ...
        use_text_embeddings=args.use_text_embeddings,
        text_embedding_type=args.text_embedding_type if args.use_text_embeddings else None,
    )
```

Update config saving (in `save_training_config`):
```python
"hyperbolic": {
    # ... existing fields ...
    "use_text_embeddings": args.use_text_embeddings if args.use_hyperbolic else None,
    "text_embedding_type": args.text_embedding_type if args.use_hyperbolic and args.use_text_embeddings else None,
},
```

---

## Files to Modify

| File | Action |
|------|--------|
| `pasco/models/hyperbolic/text_projector.py` | CREATE - MLP module with small output init |
| `pasco/models/hyperbolic/label_embedding.py` | MODIFY - Add text embedding support |
| `pasco/models/hyperbolic/__init__.py` | MODIFY - Export TextEmbeddingProjector |
| `pasco/models/body_net_hyperbolic.py` | MODIFY - Pass through params, disable virtual nodes |
| `scripts/body/train_body.py` | MODIFY - Add CLI args and config |

---

## Key Design Decisions

### 1. MLP Architecture
- `text_dim -> 256 (LayerNorm + GELU) -> embed_dim`
- Output layer uses **small initialization** (std=0.02) to keep initial tangent vectors reasonable
- Prevents large initial values that could cause numerical issues in exp_map0

### 2. Label ID Mapping
```python
# File contains: embeddings[i] corresponds to label_ids[i]
# We need: reordered[label_id] = embedding
for idx, label_id in enumerate(label_ids):
    reordered[label_id] = embeddings[idx]
```

### 3. dtype Unification
- **All embeddings converted to float16** on load (CLIP is already float16, others converted)
- MLP computation uses float32 (automatic upcast in `projector(text_embeddings.float())`)
- This balances memory efficiency with numerical stability

### 4. Forward Pass Recomputation (Intentional)
- `tangent_vectors = self.projector(self.text_embeddings)` runs every forward
- **This is intentional**: projector parameters update during training
- Cannot cache because gradients need to flow through projector

### 5. Class 0 (outside_body) Handling
- **Always forced to zero tangent vector** after MLP projection
- Explicit: `tangent_vectors[self.ignore_class] = 0.0`
- Ensures ignored class stays at hyperboloid origin regardless of text embedding

### 6. Virtual Nodes Compatibility
- **Disabled when using text embeddings** (text embeddings only have 72 real classes)
- Warning printed if user tries to enable both
- Virtual nodes are for Entailment Cone loss (future work)

### 7. Backward Compatibility
- When `use_text_embeddings=False`, behavior is **completely unchanged**
- Original `tangent_vectors` parameter still works as before

---

## Verification

### 1. Unit Tests (add to `label_embedding.py`)
```python
# Test text embedding mode
print("\nTesting LorentzLabelEmbedding with text embeddings...")

emb_text = LorentzLabelEmbedding(
    n_classes=N_CLASSES,
    embed_dim=32,
    curv=1.0,
    use_text_embeddings=True,
    text_embedding_type="sat",
)
e_text = emb_text()

# Shape check
assert e_text.shape == (N_CLASSES, 32), f"Expected ({N_CLASSES}, 32), got {e_text.shape}"

# Class 0 must be at origin
dist_0 = hyperbolic_distance_to_origin(e_text[0:1], curv=1.0)
assert dist_0.item() < 0.01, f"Class 0 should be at origin, got distance {dist_0.item()}"

# Text embeddings buffer check
assert hasattr(emb_text, "text_embeddings"), "Missing text_embeddings buffer"
assert emb_text.text_embeddings.dtype == torch.float16, "text_embeddings should be float16"

# Projector exists and is trainable
assert hasattr(emb_text, "projector"), "Missing projector"
assert any(p.requires_grad for p in emb_text.projector.parameters()), "Projector should be trainable"

# Backward pass
e_text.sum().backward()
print("  Text embedding mode: PASS")
```

### 2. Run Tests
```bash
conda activate pasco

# Unit test
python -m pasco.models.hyperbolic.label_embedding

# Integration test
python -c "
from pasco.models.body_net_hyperbolic import BodyNetHyperbolic
from pasco.data.body.params import N_CLASSES
import torch

model = BodyNetHyperbolic(
    n_classes=N_CLASSES,
    use_text_embeddings=True,
    text_embedding_type='sat',
    use_light_model=True,
)
print(f'Label embedding text_embeddings shape: {model.label_emb.text_embeddings.shape}')
print(f'Label embedding projector: {model.label_emb.projector}')

# Test forward
x = torch.randn(1, 1, 32, 32, 32)
logits, emb = model.forward_with_hyperbolic(x)
print(f'Output shapes: logits={logits.shape}, emb={emb.shape}')
print('Integration test: PASS')
"
```

### 3. Training Test
```bash
python scripts/body/train_body.py \
    --dataset_root Dataset/voxel_data \
    --use_hyperbolic \
    --use_text_embeddings \
    --text_embedding_type sat \
    --max_epochs 1 \
    --exp_name test_text_emb
```

### 4. Shape Verification Checklist
- [ ] `label_emb.text_embeddings`: [72, 768] for SAT/CLIP, [72, 512] for BioMedCLIP
- [ ] `label_emb.text_embeddings.dtype`: torch.float16
- [ ] `label_emb.projector` output: [72, 32]
- [ ] After `exp_map0`: [72, 32] on hyperboloid
- [ ] Class 0 distance to origin: < 0.01

---

## Usage Example

```bash
# Train with SAT text embeddings
python scripts/body/train_body.py \
    --dataset_root Dataset/voxel_data \
    --use_hyperbolic \
    --use_text_embeddings \
    --text_embedding_type sat \
    --hyp_weight 0.1

# Train with BioMedCLIP text embeddings
python scripts/body/train_body.py \
    --dataset_root Dataset/voxel_data \
    --use_hyperbolic \
    --use_text_embeddings \
    --text_embedding_type biomedclip \
    --hyp_weight 0.1

# Train with CLIP text embeddings
python scripts/body/train_body.py \
    --dataset_root Dataset/voxel_data \
    --use_hyperbolic \
    --use_text_embeddings \
    --text_embedding_type clip \
    --hyp_weight 0.1
```

---

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| `Unknown embedding type: xxx` | Invalid `text_embedding_type` | Use "sat", "clip", or "biomedclip" |
| `Text embedding file not found` | Missing .pt file | Check Dataset/text_embeddings/ exists |
| `max label_id >= n_classes` | Mismatch between file and model | Verify N_CLASSES matches file |
| `Warning: Disabling virtual_nodes` | Both flags enabled | Expected behavior, virtual nodes not supported with text embeddings |
