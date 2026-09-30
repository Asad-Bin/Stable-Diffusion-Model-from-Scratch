# img_generation

Command-line script that generates images from the latest checkpoint of a training run. Single file: `img_generation.py`. It has no `main()`; everything runs at import, so run it as a module.

## Run

```bash
python -m img_generation.img_generation --run_id <mlflow_run_id> [--output_dir <path>]
```

| Arg | Meaning |
|---|---|
| `--run_id` (required) | MLflow run whose `output_dir` param locates the checkpoints |
| `--output_dir` | Fallback used only if the run has no `output_dir` param (the training script does not log one unless `python -m config.config` was run for that run) |

## Steps

1. `MlflowClient(tracking_uri="http://127.0.0.1:5000")` (hardcoded) → `get_run(run_id)` → `params["output_dir"]`, or `--output_dir`.
2. Finds `checkpoint_epoch_<N>.pth` files in `<output_dir>/checkpoints/` and picks the largest `N`.
3. `Unet()` with **default** arguments (must match the trained architecture), `load_state_dict(..., strict=False)`, `eval()`. `strict=False` silently ignores missing or unexpected keys.
4. Picks **one** random prompt from 8 hardcoded butterfly prompts and embeds it via `get_clip_text_embedding_batch` (CLIP ViT-B/16, from `main/main.py`). The embedding is repeated ×16.
5. `linear_beta_schedule(1000)` and `sample_ddpm(model, betas, shape=(16, 4, 64, 64), vae=vae, text_embeddings=...)`, where `vae` is the TAESD VAE from `config.config`.
6. `make_grid(nrow=4, normalize=True, value_range=(-1, 1))` → PIL, with a white strip containing the prompt (PIL default font) pasted above the grid.
7. Saves `<output_dir>/generated_output/generated_imgs_checkpoint_<epoch>_<timestamp>.png`.

## Caveats

- `from main.main import get_clip_text_embedding_batch` imports the training script, which runs `LoadData(...)` at import: it downloads the dataset, loads CLIP ViT-B/32 and may run Qwen2-VL and rebuild the latent cache if those artifacts are missing.
- Uses `cuda:0` (its own `device`), while `config.config` uses `cuda:1`. The model and the VAE are on different devices if you have several GPUs. Edit one of them.
- Only the prompt list at the top is configurable; there is no CLI option for the prompt, seed or batch size.
- All 16 samples share one prompt, start from different noise, and there is no guidance scale.
- Generation runs all 1000 sampling steps.
