# Noise Module

Noise scheduling and forward diffusion process implementation for DDPM.

## Files

| File | Description |
|------|-------------|
| `noise_generation.py` | Beta schedule and forward diffusion sampling |

## Functions

### `linear_beta_schedule()`

Creates a linear noise schedule for the diffusion process:

```python
from noise.noise_generation import linear_beta_schedule

betas = linear_beta_schedule(
    timesteps=1000,    # Number of diffusion steps
    start=1e-4,        # Starting beta value
    end=0.02           # Ending beta value
)
# Returns: Tensor of shape (1000,)
```

### `prepare_alphas()`

Computes alpha values from betas for diffusion:

```python
from noise.noise_generation import prepare_alphas

alphas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = prepare_alphas(betas)
```

**Returns:**
- `alphas` — `1 - betas`
- `alphas_cumprod` — Cumulative product of alphas
- `sqrt_alphas_cumprod` — √(ᾱₜ) for forward diffusion
- `sqrt_one_minus_alphas_cumprod` — √(1 - ᾱₜ) for noise scaling

### `forward_diffusion_sample()`

Adds noise to clean samples according to the diffusion schedule:

```python
from noise.noise_generation import forward_diffusion_sample

x_t, noise = forward_diffusion_sample(
    x0=clean_latent,           # Clean latent (B, C, H, W)
    t=timesteps,               # Timestep tensor (B,)
    betas=betas,
    sqrt_ac=sqrt_alphas_cumprod,
    sqrt_1mac=sqrt_one_minus_alphas_cumprod
)
```

**Returns:**
- `x_t` — Noised latent at timestep t
- `eps` — The noise that was added (used as target)

## Diffusion Mathematics

### Forward Process

The forward diffusion adds Gaussian noise:

```
q(xₜ|x₀) = N(xₜ; √ᾱₜ·x₀, (1-ᾱₜ)·I)
```

Reparameterized as:
```
xₜ = √ᾱₜ · x₀ + √(1-ᾱₜ) · ε
```

where `ε ~ N(0, I)`

### Schedule Visualization

```
β (beta)
  ↑
  │          ╱
  │        ╱
  │      ╱
  │    ╱
  │  ╱
  │╱
  └──────────────→ t (timestep)
  0        500      1000

  Linear: β increases from 1e-4 to 0.02
```

## Usage in Training

```python
# During training loop
betas = linear_beta_schedule(timesteps=1000).to(device)
alphas, ac, sqrt_ac, sqrt_1mac = prepare_alphas(betas)

# Sample random timesteps
t = torch.randint(0, timesteps, (batch_size,), device=device)

# Add noise
x_t, noise = forward_diffusion_sample(x0, t, betas, sqrt_ac, sqrt_1mac)

# UNet predicts the noise
noise_pred = model(x_t, t, text_emb=text_embeddings)

# MSE loss
loss = F.mse_loss(noise_pred, noise)
```
