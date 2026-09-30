# model

The text-conditioned UNet that predicts noise in latent space.

## Files

- `blocks.py`: `get_sinusoidal_embedding`, `CrossAttentionBlock`, `ConvBlock`, `DownBlock`, `UpBlock`
- `unet.py`: `Unet`

## Building blocks (`blocks.py`)

| Block | Behavior |
|---|---|
| `get_sinusoidal_embedding(t, dim)` | Standard sin/cos embedding, `(B,) → (B, dim)`, base 10000 |
| `CrossAttentionBlock(channels, num_heads=4, cross_attention=False)` | `GroupNorm(16)` → flatten to `(B, HW, C)` → `nn.MultiheadAttention` (query = image features; key/value = `context` if `cross_attention` and a context is given, else self-attention) → 1×1 conv → **residual add** |
| `ConvBlock(in, out, time_emb_dim, text_emb_dim, num_groups=8)` | conv3×3 → GroupNorm → SiLU → **add `Linear(cat[t_emb, text_emb])` as a per-channel bias** → conv3×3 → GroupNorm → SiLU |
| `DownBlock(in, out, ..., cross_attn)` | `ConvBlock` → optional `CrossAttentionBlock` → `MaxPool2d(2)`; returns `(x, x_pooled)` where `x` (pre-pool) is the skip |
| `UpBlock(in, skip, out, ..., cross_attn)` | `ConvTranspose2d(k=2, s=2)` → concat skip → `ConvBlock` → optional `CrossAttentionBlock` |

## `Unet` (`unet.py`)

Defaults: `input_ch=4, base_ch=64, output_ch=4, time_emb_dim=128, text_emb_dim=512, time_steps=1000, num_groups=8`.

Conditioning:
- `t_emb = SiLU(Linear(sinusoidal(t)))`, dim 128.
- `text_emb_proj = Linear(512→128)(text_emb)`; mean-pooled if the input has a sequence axis. If `text_emb is None`, zeros are used (created on the module-level `config.device`).
- `comb = cat[t_emb, text_emb_proj]` (256-d). It is injected into every `ConvBlock` (via `t_emb` and `text_emb_proj` separately) and used as the single-token cross-attention context.

Shapes for a `(B, 4, 64, 64)` input:

| Stage | In → out channels | Spatial | Cross-attn |
|---|---|---|---|
| `enc1` | 4 → 64 | 64 (skip) → 32 | – |
| `enc2` | 64 → 128 | 32 (skip) → 16 | – |
| `enc3` | 128 → 256 | 16 (skip) → 8 | ✔ context = `enc3_ctx_proj(comb)` |
| `base` | 256 → 512 | 8 | ✔ (`base_att`, context = `base_ctx_proj(comb)`) |
| `dec3` | 512 (+256 skip) → 256 | 8 → 16 | ✔ context = `dec3_ctx_proj(comb)` |
| `dec2` | 256 (+128 skip) → 128 | 16 → 32 | – |
| `dec1` | 128 (+64 skip) → 64 | 32 → 64 | – |
| `out` | 64 → 4 (1×1 conv) | 64 | – |

`forward(x, t, text_emb=None) → (B, 4, 64, 64)`: predicted noise.

## Things to know

- The cross-attention context is **one token** (`(B, 1, C)`) derived from the timestep and the pooled prompt. Softmax over one key is always 1, so these blocks reduce to a learned, spatially-constant projection of that vector, not per-word attention. The prompt is reduced to a mean-pooled 512-d vector before it enters the model.
- `use_attn_enc` and `use_attn_dec` are accepted but ignored. Only `use_attn_bottleneck` has an effect.
- `text_emb_dim` sets the input width of `text_proj`; everything downstream uses `time_emb_dim`.
- The old self-attention block and older forward variants remain as comments in `blocks.py`.
- No dropout, no classifier-free-guidance dropout, and no time-conditioned residual scaling.

## Usage

```python
from model.unet import Unet
model = Unet(input_ch=4, output_ch=4, base_ch=64, time_emb_dim=128, text_emb_dim=512, num_groups=8)
eps_hat = model(z_t, t, text_emb=clip_vec)   # z_t: (B,4,64,64), t: (B,) long, clip_vec: (B,512)
```

`unet.py` imports `config.config`, which loads the VAE on import.
