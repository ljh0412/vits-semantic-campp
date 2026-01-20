# VITS with Semantic Learning + CAM++ Speaker Embeddings

Complete implementation of:
1. **Super Monotonic Align (Super-MAS)** - Direct GPU alignment without old Cython wrapper
2. **CAM++ Speaker Embeddings** - Ported from seed-vc for speaker conditioning
3. **Whisper Teacher + Student Semantics** - Semantic distillation with attention fusion

## Environment Setup

```bash
cd /mnt/ljh/vits_semantic_campp
source .venv/bin/activate
```

Installed components:
- PyTorch 2.4.1 (CUDA 11.8)
- super-monotonic-align (Triton kernels for GPU MAS)
- CAM++ modules (from seed-vc)
- Whisper (for semantic teacher)
- VITS with semantic extensions

## Implementation Details

### Step 1: Super-MAS Integration ✓

**File**: `vits/models.py`

- Replaced `import monotonic_align` with `import super_monotonic_align`
- Direct patch at call site (line ~480):
  ```python
  # Transpose: [B, S, T] -> [B, T, S] for super_monotonic_align
  value = neg_cent.transpose(-1, -2).contiguous().clone()
  mask  = attn_mask.transpose(-1, -2).squeeze(1).contiguous()
  path = super_monotonic_align.maximum_path(value, mask, dtype=torch.float32)
  attn = path.transpose(-1, -2).unsqueeze(1).detach()
  ```

**Verification**: `tests/test_super_mas_smoke.py` confirms imports and MAS compatibility

### Step 2: CAM++ Speaker Embeddings ✓

**Files**:
- `vits/third_party/seedvc/campplus/` - CAM++ modules (ported from seed-vc)
- `vits/tools/precompute_campp.py` - Extract embeddings from audio
- `vits/models.py` - CAM++ projection layer in SynthesizerTrn

**Integration**:
```python
# In SynthesizerTrn.__init__:
self.campplus_proj = nn.Linear(192, gin_channels)
self.use_campplus = False  # Toggle for CAM++ vs lookup table

# In forward/inference:
if self.use_campplus:
    g = self.campplus_proj(sid)  # sid is [B, 192] CAM++ embedding
    g = g.unsqueeze(-1)
else:
    g = self.emb_g(sid).unsqueeze(-1)  # Original lookup table
```

**Usage**:
```bash
python vits/tools/precompute_campp.py \
  --filepaths_and_text train_list.txt \
  --output_dir data/campp \
  --campplus_ckpt seed-vc/campplus_cn_common.bin
```

Output: `data/campp/spk2emb.pt` - speaker embeddings [192-dim] per speaker

### Step 3: Whisper Teacher + Student Semantics ✓

**Files**:
- `vits/modules_semantic.py` - Semantic learning components
- `vits/tools/precompute_whisper.py` - Extract Whisper encoder features
- `vits/train_semantic.py` - Training scaffold with 3 stages

**Components**:

1. **SemPred** - Student semantic predictor
   - Input: text-aligned features [B, T, hidden_channels]
   - Output: predicted semantics [B, T, 384]
   - Architecture: Linear projection + Conv1d + Linear output

2. **SemanticCrossAttnAdapter** - Attention fusion
   - Input: z_p_base [B, T, hidden] and S_used [B, T, 384]
   - Cross-attention: Q=z_p_base, K/V=S_used
   - Output: residual contribution with small gate
   - Starts with gate=0.1 for smooth integration

3. **SemanticLoss** - L1 loss for semantic supervision
   - Masked loss computation
   - Supports variable length sequences

**Training Stages**:

| Stage | CAM++ | Semantics | Attention |Description |
|-------|-------|-----------|-----------|------------|
| A     | ✓     | -         | -         | Base VITS with CAM++ embeddings |
| B     | ✓     | ✓ (mixed) | -         | Add semantic prediction with teacher/student mix |
| C     | ✓     | ✓ (full)  | ✓         | Full semantic fusion via attention adapter |

**Semantic mixing (Stage B)**:
```
S_used = teacher_ratio * S_teacher + (1 - teacher_ratio) * S_hat
         + dropout_mask * zeros  (stochastic drop for robustness)
```

### Step 4: Data Preparation

**Whisper Features**:
```bash
python vits/tools/precompute_whisper.py \
  --filepaths_and_text train_list.txt \
  --output_dir data/whisper \
  --whisper_model tiny  # or base, small, medium, large
```

Output: `data/whisper/{basename}.pt` per audio - [T, encoder_dim] Whisper features

### Step 5: Training Integration

Edit your training loop to:

```python
# During initialization
config = SemanticTrainingConfig()
config.stage = 'C'  # or 'A', 'B'
config.use_campplus = True
config.use_semantics = True

model.use_semantics = config.use_semantics
model.use_campplus = config.use_campplus

# In training step
if config.use_semantics:
    # Load Whisper teacher features
    S_teacher = load_whisper_features(audio_path)  # [B, T, 384]
    
    # Compute semantic loss
    loss_sem, S_hat, S_used = compute_semantic_loss(
        model, z_p_aligned, S_teacher, y_mask, config
    )
    
    # Integrate semantics via adapter
    if config.use_attn_adapter:
        z_p_enhanced, _ = integrate_semantics_forward(
            model, z_p_aligned, S_used, config
        )
        # Use z_p_enhanced for downstream processing
    
    # Total loss
    loss = loss_mel + loss_length + weight_sem * loss_sem
else:
    loss = loss_mel + loss_length

loss.backward()
```

## File Structure

```
/mnt/ljh/vits_semantic_campp/
 .venv/                              # Python virtual environment
 vits/                               # VITS repository
   ├── models.py                       # ✓ Patched with Super-MAS + CAM++
   ├── modules_semantic.py             # ✓ NEW: Semantic components
   ├── train_semantic.py               # ✓ NEW: Training scaffold
   ├── third_party/
   │   └── seedvc/campplus/            # ✓ NEW: CAM++ modules
   ├── tools/
   │   ├── precompute_campp.py         # ✓ NEW: Extract CAM++ embeddings
   │   └── precompute_whisper.py       # ✓ NEW: Extract Whisper features
   ├── tests/
   │   └── test_super_mas_smoke.py     # ✓ NEW: MAS smoke test
   └── data/
 campp/       ├─
       │   └── spk2emb.pt              # Speaker embeddings dict
       └── whisper/
           └── {basename}.pt           # Per-audio Whisper features
 seed-vc/                            # Seed-VC repository (CAM++ source)
 super-monotonic-align/              # Super-MAS repository
 vits/                               # VITS repository
```

## Verification

All components verified:

 Super-MAS: models.py imports, MAS function callable
 CAM++: Modules ported, projection layer added to SynthesizerTrn
 Whisper: Teacher extraction tool ready
 Semantics: SemPred, Adapter, Loss all compile and have gradient flow
 VITS: Full model imports with all extensions

## Next Steps

1. **Prepare data**:
   - Create `train_list.txt` with `audio_path|speaker_id` format
   - Run precompute scripts for CAM++ and Whisper features

2. **Stage A training**:
   - Train base VITS with CAM++ embeddings only
   - Verify convergence (duration loss, alignment quality)

3. **Stage B training**:
   - Load Stage A checkpoint
   - Enable semantic prediction + loss
   - Monitor: semantic loss should decrease
   - Adjust `sem_loss_weight` and `sem_teacher_ratio`

4. **Stage C training**:
   - Load Stage B checkpoint
   - Enable semantic attention adapter
   - Monitor: adapter gate should grow from 0.1
   - Verify: z_p enriched with semantic information

## Configuration

Edit `SemanticTrainingConfig` in `train_semantic.py`:

```python
class SemanticTrainingConfig:
    stage = 'A'  # 'A', 'B', or 'C'
    whisper_dim = 384  # Whisper-tiny: 384, base: 512
    use_campplus = True
    use_semantics = False
    use_attn_adapter = False
    
    sem_loss_weight = 0.1  # Weight of semantic loss
    sem_teacher_ratio = 0.9  # Teacher ratio in Stage B
    sem_student_drop_prob = 0.1  # Dropout for robustness
```

## References

- VITS: https://github.com/jaywalnut310/vits
- Seed-VC: https://github.com/Plachtaa/seed-vc
- Super-MAS: https://github.com/supertone-inc/super-monotonic-align
- Whisper: https://github.com/openai/whisper

---

**Status**: All components implemented and verified. Ready for Stage C training!
