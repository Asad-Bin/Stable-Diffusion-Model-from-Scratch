# Sampling Module

DDPM sampling algorithm for generating images from the trained diffusion model.

## Files

| File | Description |
|------|-------------|
| `sample_ddpm.py` | DDPM reverse diffusion sampling implementation |

## Usage

### Basic Sampling

```python
from sampling.sample_ddpm import sample_ddpm

# Generate images
generated = sample_ddpm(
    model=unet_model,                    # Trained UNet
    betas=betas,                         # Noise schedule
    shape=(16, 4, 64, 64),               # (batch, channels, height, width)
    device=device,
    vae=vae,                             # VAE for decoding (optional)
    timesteps=1000,                      # Diffusion steps
    text_embeddings=clip_embeddings      # Text conditioning (optional)
)
# Output: (16, 3, 512, 512) decoded images
```

## Function Signature

```python
@torch.no_grad()
def sample_ddpm(
    model,              # UNet model
    betas,              # Beta schedule tensor
    shape,              # Output shape (B, C, H, W)
    device,             # Torch device
    vae=None,           # Optional VAE for decoding
    timesteps=1000,     # Number of sampling steps
    text_embeddings=None # Optional text conditioning
)
```

## Sampling Algorithm

### DDPM Reverse Process

```
1. Start with random noise: x_T ~ N(0, I)

2. For t = T, T-1, ..., 1:
   
   a. Predict noise: ε_θ(x_t, t, text)
   
   b. Compute x_{t-1}:
      x_{t-1} = (1/√αₜ) * (x_t - βₜ/√(1-ᾱₜ) * ε_θ)
   
   c. Add noise (if t > 0):
      x_{t-1} += √βₜ * z, where z ~ N(0, I)

3. Return x_0 (denoised latent)

4. If VAE provided: decode x_0 to image
```

### Mathematical Formulation

```
p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t²I)

where:
  μ_θ = (1/√αₜ) * (x_t - βₜ/√(1-ᾱₜ) * ε_θ(x_t, t))
  σ_t² = βₜ
```

## Output Processing

When a VAE is provided:

```python
# Undo latent normalization
x = (x * latent_magnitude) + latent_shift

# Decode with VAE
x = vae.decoder(x).clamp(-1, 1)

# Result: RGB images in [-1, 1] range
```

## Example: Text-Conditioned Generation

```python
from sampling.sample_ddpm import sample_ddpm
from main.main import get_clip_text_embedding_batch
from dataset.clip import clip_tokenizer, clip_text_model
from noise.noise_generation import linear_beta_schedule

# Prepare text embeddings
prompts = ["a monarch butterfly", "a blue morpho butterfly"]
text_emb = get_clip_text_embedding_batch(
    prompts, clip_tokenizer, clip_text_model, device
)

# Generate
betas = linear_beta_schedule(timesteps=1000).to(device)
images = sample_ddpm(
    model=model,
    betas=betas,
    shape=(2, 4, 64, 64),
    device=device,
    vae=vae,
    text_embeddings=text_emb
)

# images: (2, 3, 512, 512) RGB tensors
```
