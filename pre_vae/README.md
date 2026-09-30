# pre_vae

Pretrained VAE used by the diffusion pipeline.

## Files

- `pre_vae.py`: loads the model and exposes `vae`.
- `__init__.py`: `from pre_vae.pre_vae import vae`, so `import pre_vae; pre_vae.vae` works. `config/config.py` uses exactly that.

## What it loads

```python
vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16)
vae = vae.to(device).eval()
```

That is **TAESD** (Tiny AutoEncoder for Stable Diffusion) through `diffusers.AutoencoderTiny`. It is not the Stable Diffusion KL-VAE. It uses fp16 weights and downsamples 8× (512×512 image ↔ 4×64×64 latent).

## How the rest of the project uses it

| Where | Use |
|---|---|
| `config/config.py` | `vae = pre_vae.vae`; `latent_shift = vae.latent_shift`; `latent_magnitude = vae.latent_magnitude` (attributes of `AutoencoderTiny`) |
| `dataset/load_dataset.py` | `vae.encode(image_fp16)` for caching, then `(z - latent_shift) / latent_magnitude` |
| `sampling/sample_ddpm.py` | `x * latent_magnitude + latent_shift`, cast to `vae.dtype`, `vae.decoder(x).clamp(-1, 1)` |

## Notes

- Requires the `diffusers` package, which is **not** in `requirements.txt`. Run `pip install diffusers`.
- The `for params in vae.parameters(): params.required_grad = False` loop has a typo (`required_grad`, not `requires_grad`), so it freezes nothing. This is harmless because the VAE is only used under `no_grad`.
- The module does `from config.config import *` to get `device`, while `config.config` imports this package back. It works because `device` is defined before `import pre_vae` in `config.py`. Import `config` first, or expect this ordering to matter.
- The dtype is float16 on both CPU and GPU. On CPU some ops may be slow or unsupported.
