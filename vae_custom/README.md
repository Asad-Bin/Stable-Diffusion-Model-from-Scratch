# vae_custom

A small VAE written from scratch (encoder, decoder, blocks) and a wrapper to load its checkpoints. It is trained by [`vae_train`](../vae_train/). **The diffusion pipeline currently uses the pretrained TAESD VAE instead** (see [pre_vae](../pre_vae/)); the custom-VAE lines in `config/config.py` are commented out.

There is no `__init__.py` in this directory (it works as a namespace package).

## Files

| File | Contents |
|---|---|
| `blocks.py` | `SelfAttention`, `AttentionBlock`, `ResidualBlock`, `ResAttnBlock` |
| `encoder.py` | `Encoder(latent_channels=4)` |
| `decoder.py` | `Decoder()` |
| `custom_vae_wrap.py` | `CustomVAEWrapper`, `load_custom_vae` |
| `checkpoints/encoder.pt`, `decoder.pt` | Tracked weights (~14.7 MB and ~9.0 MB), copied from epoch 1000 by `vae_train/checkpoint_updater.py` |

## Blocks

- `SelfAttention(n_heads, embd_dim)`: manual multi-head attention (`in_proj` → q, k, v; scaled dot-product; optional causal mask; `out_proj`).
- `AttentionBlock(channels)`: `GroupNorm(32)` → single-head `SelfAttention` over `H·W` tokens → residual add.
- `ResidualBlock(in, out)`: `GroupNorm(8)` → SiLU → conv3×3 → `GroupNorm(8)` → conv3×3 (no SiLU before the second conv), plus a 1×1 conv (or identity) skip.
- `ResAttnBlock(in, out, use_attn=True)`: `ResidualBlock` + `AttentionBlock`. **Not used** by the encoder or decoder (the attention calls are commented out).

## Encoder: `(B, 3, 512, 512) → z, mu, logvar`, each `(B, 4, 64, 64)`

```
Conv4×4 s2 (3→64) + SiLU                      256×256
ResidualBlock 64→128
Conv4×4 s2 (128→128) + SiLU                   128×128
ResidualBlock 128→256
Conv4×4 s2 (256→256) + SiLU                   64×64
ResidualBlock 256→256
conv_mu, conv_logvar: Conv3×3 256→4
z = mu + ε·exp(0.5·logvar)                    (always sampled, even in eval())
```

## Decoder: `(B, 4, 64, 64) → (B, 3, 512, 512)` in [-1, 1]

```
Conv3×3 4→256 + SiLU
ResidualBlock 256→256
ConvTranspose4×4 s2 256→128 + SiLU            128×128
ResidualBlock 128→128
ConvTranspose4×4 s2 128→64  + SiLU            256×256
ResidualBlock 64→64
ConvTranspose4×4 s2 64→32   + SiLU            512×512
Conv3×3 32→3, tanh
```

## Wrapper

```python
from vae_custom.custom_vae_wrap import load_custom_vae
vae = load_custom_vae("vae_custom/checkpoints/encoder.pt", "vae_custom/checkpoints/decoder.pt", device)
z = vae.encode(images)     # returns the sampled z only (mu, logvar are dropped)
x = vae.decode(z)
```

`CustomVAEWrapper` puts both parts in `eval()` and sets `requires_grad=False`.

## Using it in the diffusion pipeline (not done by default)

The rest of the code expects a `diffusers.AutoencoderTiny`-like object, and the wrapper differs:

- `sample_ddpm` calls `vae.decoder(x)` and reads `vae.dtype` (the wrapper has `.decoder` but no `.dtype`).
- `cache_latents_to_disk` calls `vae.eval().to(device).half()` and expects `vae.encode(...)` to return an object with `latent_dist`/`latents`/`sample`, or a tensor. The wrapper returns a tensor, which is accepted.
- `latent_shift` and `latent_magnitude` must be provided separately (`config.py` has a commented-out block reading them from `sensitive_config`).
- Cached latents must be regenerated after switching VAEs.
