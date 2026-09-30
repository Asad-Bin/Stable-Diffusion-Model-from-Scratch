# main

Training entry point.

## Files

| File | Purpose |
|---|---|
| `main.py` | Data loading, `train_model`, `print_image`, and the `__main__` launcher |
| `old_run.py` | `old_run_data()`: load a checkpoint for resuming |

## Run

```bash
python -m main.main            # foreground
bash ../run.sh                 # background: exports MY_TIMESTAMP, logs to output/0_output_logs/log_<ts>.out
```

`python main/main.py` won't work (it needs package imports); use `-m`.

## Import-time behavior

Importing `main.main` executes `dataloader = LoadData(prompt_style="queenvl", regenerate_prompts=False)` at module level (see [dataset](../dataset/)). This is also why `img_generation` triggers data loading when it imports `get_clip_text_embedding_batch`.

## `main.py`

### `get_clip_text_embedding_batch(prompts, tokenizer, text_model, device)`
Tokenizes (`padding=True, truncation=True, max_length=77`), runs CLIP under `no_grad`, returns the mean over tokens of `last_hidden_state`: shape `(B, 512)`. The mean includes padding tokens, and there is no normalization.

### `train_model(dataloader, model, betas, epochs, lr, save_interval)`

Setup:
- `AdamW(lr=1e-4)`, `CosineAnnealingLR(T_max=epochs, eta_min=1e-6)`, stepped once per epoch.
- `prepare_alphas(betas)` → `sqrt_ac`, `sqrt_1mac`.
- A `Linear` text projector is created only if CLIP's dim differs from `model.text_proj.in_features`. With `text_emb_dim=512` and CLIP ViT-B/16 (512), none is created.

Per batch:
1. `imgs` (cached latents) = `batch[0]`, `prompts` = `batch[2]` (`batch[1]`, the cached embedding, is unused).
2. Text embedding from CLIP ViT-B/16 (recomputed each step).
3. `t ~ randint(0, timesteps)`; `z_t, noise = forward_diffusion_sample(...)`.
4. `pred = model(z_t, t, text_emb=...)`; `loss = F.l1_loss(pred, noise)`; backward; `optimizer.step()`. No gradient clipping and no mixed precision.
5. If MLflow is on: log `train_loss` per step and accumulate L2 grad norms of parameters whose names contain both `weight` and `conv1`.

Per epoch (only if `is_ml_flow_off` is False; otherwise the loop `continue`s right after `scheduler.step()`):
- Log `grad_norm_<param>` averages, generate gradient plots (`utils.gradient_descent`), log `avg_loss`.
- Every `checkpoint_interval` (50) epochs: save `checkpoint_epoch_<N>.pth` (`epoch`, model/optimizer/scheduler state, `loss`) to `output/output_<ts>/checkpoints/`. Only the newest 5 (by mtime) are kept. It is uploaded to MLflow only when `num_epochs - epoch <= 200`.
- If `epoch <= 200` or `(epoch+1) % save_interval == 0` (5): `print_image(epoch)`.

Resume: `if old_run == True:` load the checkpoint from `old_run_data()` (model, optimizer, scheduler, start epoch). **This never fires**: `from main import old_run` binds the *module* `main.old_run`, shadowing the boolean of the same name imported from `config`, and a module is never `== True`. If it did fire, training would continue from the saved epoch up to `num_epochs`.

### `print_image(epoch)`
Uses the module-level `model`, `betas` and `vae`. Runs `sample_ddpm` for 3 fixed prompts (`(3, 4, 64, 64)` latents), maps `[-1,1]→[0,1]`, builds a 3-wide grid, logs it to MLflow (`generated_image_epoch_NNNN.png`), calls `plt.show()`, and saves `output/output_<ts>/training_outputs/epoch_NNNN_samples.png`. On a headless matplotlib backend `plt.show()` just emits a warning.

### `__main__`

```python
Unet(input_ch=4, output_ch=4, base_ch=64, time_emb_dim=128, time_steps=timesteps,
     num_groups=8, text_emb_dim=512)
betas = linear_beta_schedule(timesteps=1000, start=1e-4, end=0.02)
train_model(dataloader, model, betas, epochs=3000, lr=1e-4, save_interval=5)
```

## `old_run.py`

`old_run_data()` loads `<old_checkpoint_dir>/checkpoint_epoch_<old_run_checkpoint_no>.pth` (both from `config`, the directory being a hardcoded absolute path), prints the loss, and returns `(checkpoint, checkpoint['epoch'])`.

## MLflow

`main.main` never calls `mlflow.set_tracking_uri` or `set_experiment`. Logging goes to the default store (`./mlruns`) unless `MLFLOW_TRACKING_URI` is set in the environment. Run params are only logged if you run `python -m config.config`.
