# Config Module

Configuration management for the diffusion model training and inference pipeline.

## Files

| File | Description |
|------|-------------|
| `config.py` | Main configuration file containing all hyperparameters and settings |
| `sensitive_config.py` | User-specific sensitive configuration (not tracked in git) |

## Configuration Parameters

### Training Parameters

```python
image_size = 512          # Output image resolution
batch_size = 32           # Training batch size
num_epochs = 3000         # Total training epochs
timesteps = 1000          # DDPM diffusion timesteps
learning_rate = 0.0001    # AdamW optimizer learning rate
```

### Model Architecture

```python
input_channels = 4        # VAE latent channels
output_channels = 4       # UNet output channels
base_channels = 64        # UNet base channel multiplier
time_embedding_dim = 128  # Sinusoidal time embedding dimension
```

### Logging & Checkpoints

```python
save_image_every = 5      # Generate samples every N epochs
checkpoint_interval = 50  # Save checkpoint every N epochs
```

## Sensitive Configuration

Create `sensitive_config.py` with the following template:

```python
# MLflow experiment tracking
MLFLOW_TRACKING_URI = "sqlite:///mlruns.db"
MLFLOW_EXPERIMENT_NAME = "custom-diffusion"

# Custom VAE checkpoint paths
VAE_ENCODER_PATH = "vae_custom/checkpoints/encoder.pt"
VAE_DECODER_PATH = "vae_custom/checkpoints/decoder.pt"

# Latent space normalization
LATENT_SHIFT = 0.0
LATENT_MAGNITUDE = 1.0

# For resuming training runs
OLD_RUN_ID = None  # Set to MLflow run ID when resuming
```

## VAE Selection

The configuration controls which VAE is used. Modify `config.py`:

```python
# Pretrained VAE (default)
import pre_vae
vae = pre_vae.vae

# OR Custom VAE
# from vae_custom.custom_vae_wrap import load_custom_vae
# vae = load_custom_vae(VAE_ENCODER_PATH, VAE_DECODER_PATH, device)
```

## Utility Functions

- `get_unique_output_dir(base_dir)` — Creates timestamped output directory
- `print_config(cfg)` — Pretty-prints configuration to console
- `log_config_mlflow(cfg)` — Logs parameters to MLflow
- `log_config_local(cfg, output_dir)` — Saves config as JSON locally
- `write_config_readme(cfg, output_dir)` — Generates README with config summary
