# config

Single module, `config.py`. It is imported (`from config.config import *`) by almost every other module, and **runs side effects at import time**.

## Import-time behavior (in order)

1. Tries `from config.sensitive_config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, OLD_RUN_ID, VAE_ENCODER_PATH, VAE_DECODER_PATH, LATENT_SHIFT, LATENT_MAGNITUDE`. `config/sensitive_config.py` is git-ignored. If it is missing, only `"Sensitive configuration not found. Using default values."` is printed (there are no actual defaults).
2. Creates `./output/output_<timestamp>/` (`get_unique_output_dir`). The timestamp is the `MY_TIMESTAMP` env var (set by `run.sh`) or `YYYYmmdd_HHMMSS`.
3. Creates `<output_dir>/checkpoints/` and `./dataset/huggingface_butterflies/`.
4. Defines hyperparameters, `device`, and the `CONFIG` dict, then prints it.
5. `import pre_vae; vae = pre_vae.vae`, then `latent_shift = vae.latent_shift` and `latent_magnitude = vae.latent_magnitude`. This works even though `pre_vae` imports `config` back, because `device` is defined earlier in this file.

## Values

| Name | Value | Used by |
|---|---|---|
| `image_size` | 512 | dataset transforms, `print_image` (latent = `image_size // 8`) |
| `batch_size` | 32 | `DataLoader` |
| `num_epochs` | 3000 | `train_model` |
| `timesteps` | 1000 | noise schedule, sampling |
| `learning_rate` | 1e-4 | AdamW |
| `save_image_every` | 5 | sample-image cadence |
| `checkpoint_interval` | 50 | checkpoint cadence |
| `input_channels` / `output_channels` | 4 / 4 | listed in `CONFIG` |
| `base_channels`, `time_embedding_dim` | 64, 128 | listed in `CONFIG` |
| `device` | `cuda:1` if CUDA is available, else CPU | everything |
| `is_ml_flow_off` | `False` | `main.main` |
| `old_run` | `False` | intended resume switch |
| `old_run_checkpoint_no` | 3000 | `main/old_run.py` |
| `old_checkpoint_dir` | hardcoded absolute path | `main/old_run.py` |
| `note` | free text about the VAE and attention setup | stored in `CONFIG` |

`main/main.py` builds the `Unet` from literals (`input_ch=4, output_ch=4, base_ch=64, time_emb_dim=128, num_groups=8, text_emb_dim=512`), so `input_channels`, `output_channels`, `base_channels` and `time_embedding_dim` are informational only.

## Functions

| Function | Purpose |
|---|---|
| `get_unique_output_dir(base_dir)` | Create and return `base_dir/output_<timestamp>` |
| `print_config(cfg)` | Pretty-print a config dict (called at import) |
| `log_config_mlflow(cfg, is_ml_flow_off=False, old_run=False)` | `set_tracking_uri(MLFLOW_TRACKING_URI)`, `set_experiment(MLFLOW_EXPERIMENT_NAME)`; if `old_run`, `start_run(run_id=OLD_RUN_ID)`, otherwise log every config entry as a param |
| `log_config_local(cfg, output_dir)` | Write `config.json` |
| `write_config_readme(cfg, output_dir)` | Write a `README.md` listing the config into the run's output dir |

The three logging functions run **only** when the file is executed directly (`python -m config.config`). `main.main` never calls them, so a normal training run does not set the MLflow URI or log these params. This is the only place the `sensitive_config` names are needed.

## VAE selection

The active VAE is the pretrained TAESD from `pre_vae`. The custom-VAE alternative (`load_custom_vae(VAE_ENCODER_PATH, VAE_DECODER_PATH, device)` with `LATENT_SHIFT`/`LATENT_MAGNITUDE` from the sensitive config) is commented out. The other modules read `vae`, `latent_shift` and `latent_magnitude` from here.

## `sensitive_config.py` template

```python
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "my_experiment"
OLD_RUN_ID = ""
VAE_ENCODER_PATH = "vae_custom/checkpoints/encoder.pt"
VAE_DECODER_PATH = "vae_custom/checkpoints/decoder.pt"
LATENT_SHIFT = 0.0
LATENT_MAGNITUDE = 1.0
```

The values above are placeholders; the code does not prescribe them.
