# Pre-trained VAE Module

Wrapper for loading pretrained Stable Diffusion VAE from Hugging Face.

## Files

| File | Description |
|------|-------------|
| `pre_vae.py` | VAE initialization and configuration |
| `__init__.py` | Module exports for easy importing |

## Usage

### Importing the VAE

```python
from pre_vae.pre_vae import vae

# Or via package import
import pre_vae
vae = pre_vae.vae
```

### Model Details

- **Model**: `stabilityai/sd-vae-ft-mse`
- **Source**: Hugging Face Diffusers
- **Latent Channels**: 4
- **Downsampling Factor**: 8× (512×512 → 64×64)

## VAE Operations

### Encoding (Image → Latent)

```python
# Input: RGB image tensor normalized to [-1, 1]
# Shape: (B, 3, 512, 512)
image = image_tensor.to(vae.dtype)

# Encode to latent space
latent = vae.encode(image).latent_dist.sample()
# Output shape: (B, 4, 64, 64)

# Apply scaling (optional)
latent = (latent - vae.latent_shift) / vae.latent_magnitude
```

### Decoding (Latent → Image)

```python
# Input: Latent tensor
# Shape: (B, 4, 64, 64)
latent = latent_tensor.to(vae.dtype)

# Undo scaling
latent = (latent * vae.latent_magnitude) + vae.latent_shift

# Decode to image
image = vae.decoder(latent).clamp(-1, 1)
# Output shape: (B, 3, 512, 512)
```

## Latent Space Parameters

The VAE exposes normalization parameters:

```python
vae.latent_shift      # Mean shift for latent normalization
vae.latent_magnitude  # Scale factor for latent normalization
vae.dtype             # Model precision (float16/float32)
```

## Automatic Download

The pretrained VAE is automatically downloaded on first use:
- **Cache Location**: `~/.cache/huggingface/hub/`
- **Approximate Size**: ~160MB

## Switching to Custom VAE

To use a custom VAE instead, modify `config/config.py`:

```python
# Comment out pretrained VAE
# import pre_vae
# vae = pre_vae.vae

# Use custom VAE
from vae_custom.custom_vae_wrap import load_custom_vae
vae = load_custom_vae(
    encoder_ckpt_path="vae_custom/checkpoints/encoder.pt",
    decoder_ckpt_path="vae_custom/checkpoints/decoder.pt",
    device=device
)
```
