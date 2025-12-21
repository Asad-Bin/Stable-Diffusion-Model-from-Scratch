# Model Module

UNet architecture implementation for the diffusion model with cross-attention text conditioning.

## Files

| File | Description |
|------|-------------|
| `unet.py` | Main UNet class with encoder-decoder structure |
| `blocks.py` | Building blocks: ConvBlock, DownBlock, UpBlock, CrossAttention |

## Architecture

### UNet Overview

```
Input Latent (4×64×64)
        ↓
    ┌─────────────────────────────────────┐
    │            ENCODER                   │
    │  enc1: 4 → 64 (DownBlock)           │
    │  enc2: 64 → 128 (DownBlock)         │
    │  enc3: 128 → 256 (DownBlock + Attn) │
    └─────────────────────────────────────┘
        ↓
    ┌─────────────────────────────────────┐
    │          BOTTLENECK                  │
    │  256 → 512 (ConvBlock + CrossAttn)  │
    └─────────────────────────────────────┘
        ↓
    ┌─────────────────────────────────────┐
    │            DECODER                   │
    │  dec3: 512 → 256 (UpBlock + Attn)   │
    │  dec2: 256 → 128 (UpBlock)          │
    │  dec1: 128 → 64 (UpBlock)           │
    └─────────────────────────────────────┘
        ↓
Output Predicted Noise (4×64×64)
```

## UNet Class

### Initialization

```python
from model.unet import Unet

model = Unet(
    input_ch=4,              # Input channels (VAE latent)
    base_ch=64,              # Base channel multiplier
    output_ch=4,             # Output channels
    time_emb_dim=128,        # Time embedding dimension
    text_emb_dim=512,        # CLIP embedding dimension
    time_steps=1000,         # Total diffusion timesteps
    use_attn_bottleneck=True,# Enable bottleneck attention
    num_groups=8             # GroupNorm groups
)
```

### Forward Pass

```python
# Inputs
x = latent_tensor        # (B, 4, 64, 64)
t = timestep_tensor      # (B,) integer timesteps
text_emb = clip_embedding # (B, 512) or (B, L, 512)

# Forward
noise_pred = model(x, t, text_emb=text_emb)
# Output shape: (B, 4, 64, 64)
```

## Building Blocks

### `ConvBlock`

Basic convolutional block with time and text conditioning:

```python
ConvBlock(
    in_channels=128,
    out_channels=256,
    time_emb_dim=128,
    text_emb_dim=128,
    num_groups=8
)
```

### `DownBlock`

Encoder block with convolution and optional cross-attention:

```python
DownBlock(
    in_channels=64,
    out_channels=128,
    time_emb_dim=128,
    text_emb_dim=128,
    cross_attn=False  # Enable for text conditioning
)
```

### `UpBlock`

Decoder block with transposed convolution, skip connections, and optional attention:

```python
UpBlock(
    in_channels=256,
    skip_channels=128,
    out_channels=128,
    time_emb_dim=128,
    text_emb_dim=128,
    cross_attn=True
)
```

### `CrossAttentionBlock`

Multi-head cross-attention for text conditioning:

```python
CrossAttentionBlock(
    channels=256,
    cross_attention=True
)
```

## Conditioning Mechanism

### Time Embedding
- Sinusoidal positional encoding for timestep `t`
- Projected through MLP: `Linear → SiLU`

### Text Embedding
- CLIP text features projected to match time embedding dim
- Mean-pooled if sequence length > 1
- Concatenated with time embedding for context projection

### Context Fusion
```python
# Combined embedding for cross-attention context
comb = torch.cat([t_emb, text_emb_proj], dim=-1)  # (B, 2×time_emb_dim)
context = context_proj(comb).unsqueeze(1)          # (B, 1, C)
```
