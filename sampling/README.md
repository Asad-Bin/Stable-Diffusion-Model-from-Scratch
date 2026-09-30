# sampling

Reverse diffusion. Single file: `sample_ddpm.py`.

## `sample_ddpm(model, betas, shape, device, vae=None, timesteps=timesteps, text_embeddings=None)`

Decorated with `@torch.no_grad()`. Puts `model` in `eval()`.

Algorithm (ancestral DDPM):

```
x_T ~ N(0, I) with the given shape
for i = T-1 … 0:
    ε̂ = model(x, t=i, text_emb=text_embeddings)     # model(x, t) if text_embeddings is None
    x  = 1/√α_i · ( x − β_i / √(1 − ᾱ_i) · ε̂ )
    if i > 0: x += √β_i · N(0, I)                    # σ_t² = β_t
```

Output:
- With `vae`: `x·latent_magnitude + latent_shift` → cast to `vae.dtype` (fp16 for TAESD) → `vae.decoder(x).clamp(-1, 1)`. Returns images in [-1, 1], shape `(B, 3, 512, 512)`.
- Without `vae`: the latent, clamped to [-1, 1].

| Arg | Meaning |
|---|---|
| `betas` | 1-D schedule tensor. `alphas[i]` and `betas[i]` are indexed directly, so it must be on `device`. |
| `shape` | e.g. `(3, 4, 64, 64)` |
| `text_embeddings` | `(B, 512)` CLIP vectors, matched to the batch size |
| `timesteps` | defaults to `config.timesteps` (1000) |

## Notes

- `latent_shift` and `latent_magnitude` come from `from config.config import *`, i.e. the TAESD attributes, regardless of the `vae` argument.
- It calls `vae.decoder(...)` directly (the `AutoencoderTiny` API), so a `vae` without a compatible `.decoder` attribute will not work. See [vae_custom](../vae_custom/).
- No classifier-free guidance, no DDIM or step skipping. Every call runs all `timesteps` network evaluations.
- The final latent is not clamped before decoding.
- Called from `main/main.py::print_image` (3 samples) and `img_generation/img_generation.py` (16 samples).
