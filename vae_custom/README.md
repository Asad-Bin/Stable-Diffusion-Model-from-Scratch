# Custom VAE Module

Custom Variational Autoencoder implementation optimized for the target image domain.

## Files

| File | Description |
|------|-------------|
| `encoder.py` | VAE Encoder network |
| `decoder.py` | VAE Decoder network |
| `blocks.py` | Building blocks: ResidualBlock, AttentionBlock |
| `custom_vae_wrap.py` | Wrapper for loading and using the custom VAE |
| `checkpoints/` | Directory for encoder and decoder weights |

## Architecture

### Encoder

Compresses 512×512 RGB images to 64×64 latent representations:

```
Input (3, 512, 512)
    ↓ Conv2d (stride=2)
(64, 256, 256)
    ↓ ResidualBlock + Conv2d (stride=2)
(128, 128, 128)
    ↓ ResidualBlock + Conv2d (stride=2)
(256, 64, 64)
    ↓ ResidualBlock
    ↓ Conv2d (mu) + Conv2d (logvar)
Latent (4, 64, 64) + KL terms
```

### Decoder

Reconstructs 512×512 images from latent codes:

```
Latent (4, 64, 64)
    ↓ Conv2d + ResidualBlock
(256, 64, 64)
    ↓ ConvTranspose2d (stride=2) + ResidualBlock
(128, 128, 128)
    ↓ ConvTranspose2d (stride=2) + ResidualBlock
(64, 256, 256)
    ↓ ConvTranspose2d (stride=2)
(32, 512, 512)
    ↓ Conv2d + Tanh
Output (3, 512, 512)
```

## Usage

### Loading the Custom VAE

```python
from vae_custom.custom_vae_wrap import load_custom_vae

vae = load_custom_vae(
    encoder_ckpt_path="vae_custom/checkpoints/encoder.pt",
    decoder_ckpt_path="vae_custom/checkpoints/decoder.pt",
    device="cuda"
)
```

### Encoding Images

```python
from vae_custom.encoder import Encoder

encoder = Encoder(latent_channels=4)
encoder.load_state_dict(torch.load("checkpoints/encoder.pt"))

# Encode
z, mu, logvar = encoder(image)  # image: (B, 3, 512, 512)
# z: (B, 4, 64, 64) - reparameterized latent
```

### Decoding Latents

```python
from vae_custom.decoder import Decoder

decoder = Decoder()
decoder.load_state_dict(torch.load("checkpoints/decoder.pt"))

# Decode
reconstructed = decoder(z)  # z: (B, 4, 64, 64)
# reconstructed: (B, 3, 512, 512)
```

## Building Blocks

### `ResidualBlock`

```python
from vae_custom.blocks import ResidualBlock

block = ResidualBlock(in_channels=128, out_channels=256)
```

Features:
- Two convolutions with SiLU activation
- Skip connection with projection if channels differ
- GroupNorm for normalization

### `AttentionBlock`

```python
from vae_custom.blocks import AttentionBlock

attn = AttentionBlock(channels=256)
```

Features:
- Self-attention mechanism
- Spatial attention for capturing long-range dependencies

## Checkpoints

Store trained weights in `checkpoints/`:

```
vae_custom/
├── checkpoints/
│   ├── encoder.pt    # Encoder weights
│   └── decoder.pt    # Decoder weights
├── encoder.py
├── decoder.py
└── ...
```

## Training

See `vae_train/` module for VAE training scripts. The custom VAE is trained separately before diffusion model training.

## Switching Between VAEs

In `config/config.py`:

```python
# Use custom VAE
from vae_custom.custom_vae_wrap import load_custom_vae
vae = load_custom_vae(encoder_path, decoder_path, device)

# OR use pretrained VAE
# import pre_vae
# vae = pre_vae.vae
```
