# noise

Forward (noising) side of DDPM. Single file: `noise_generation.py`.

## Functions

| Function | Description |
|---|---|
| `linear_beta_schedule(timesteps, start=1e-4, end=0.02)` | `torch.linspace(start, end, timesteps)`, CPU tensor |
| `prepare_alphas(betas)` | Returns `(alphas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)` with `alphas = 1 - betas` |
| `forward_diffusion_sample(x0, t, betas, sqrt_ac, sqrt_1mac)` | `@torch.no_grad()`. Draws `eps ~ N(0, I)` like `x0` and returns `(x_t, eps)` with `x_t = sqrt_ac[t]·x0 + sqrt_1mac[t]·eps`. |

`betas` is accepted by `forward_diffusion_sample` but not used in the computation. All operands are moved to `config.device`, so the return value is on that device regardless of where the inputs were.

## Math

```
x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε ,   ε ~ N(0, I),   ᾱ_t = ∏_{s≤t} (1 − β_s)
```

## Usage (as in `main/main.py`)

```python
betas = linear_beta_schedule(timesteps=1000, start=1e-4, end=0.02).to(device)
_, _, sqrt_ac, sqrt_1mac = prepare_alphas(betas)
t = torch.randint(0, 1000, (B,), device=device)
z_t, noise = forward_diffusion_sample(z0, t, betas=betas, sqrt_ac=sqrt_ac, sqrt_1mac=sqrt_1mac)
```

`z0` are the cached, normalized VAE latents `(B, 4, 64, 64)`. Only the linear schedule exists (no cosine schedule).

The module imports `device` from `config.config`.
