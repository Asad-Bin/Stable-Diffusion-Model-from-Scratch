# VAE Training Module

Scripts and utilities for training the custom Variational Autoencoder.

## Files

| File | Description |
|------|-------------|
| `train.py` | Main VAE training script |
| `test.py` | VAE evaluation and reconstruction testing |
| `dataloader.py` | Dataset loading for VAE training |
| `config.py` | VAE-specific training configuration |
| `plot.py` | Visualization utilities for training progress |
| `checkpoint_updater.py` | Checkpoint management utilities |

## Training the VAE

### Quick Start

```bash
python -m vae_train.train
```

### Training Configuration

Edit `vae_train/config.py`:

```python
# Training parameters
batch_size = 32
num_epochs = 1000
learning_rate = 1e-4

# Loss weights
kl_weight = 0.0001      # KL divergence weight
recon_weight = 1.0      # Reconstruction loss weight
```

## Training Pipeline

### `train.py`

Main training loop with:
- Reconstruction loss (MSE)
- KL divergence regularization
- Learning rate scheduling
- Checkpoint saving

```python
# VAE loss function
loss = recon_loss + kl_weight * kl_loss

where:
  recon_loss = MSE(reconstructed, original)
  kl_loss = KL(q(z|x) || p(z))
```

## Data Loading

### `dataloader.py`

Prepares image datasets for VAE training:

```python
from vae_train.dataloader import get_vae_dataloader

dataloader = get_vae_dataloader(
    dataset_path="dataset/huggingface_butterflies",
    batch_size=32,
    image_size=512
)
```

## Testing & Evaluation

### `test.py`

Evaluate reconstruction quality:

```bash
python -m vae_train.test
```

Features:
- Computes reconstruction MSE
- Visualizes original vs. reconstructed images
- Analyzes latent space distribution

## Visualization

### `plot.py`

Training progress visualization:

```python
from vae_train.plot import plot_training_curves

plot_training_curves(
    losses=train_losses,
    save_path="vae_training_curve.png"
)
```

## Checkpoint Management

### `checkpoint_updater.py`

Utilities for managing VAE checkpoints:

```python
from vae_train.checkpoint_updater import save_checkpoint, load_checkpoint

# Save
save_checkpoint(
    encoder=encoder,
    decoder=decoder,
    epoch=100,
    path="checkpoints/"
)

# Load
encoder, decoder = load_checkpoint("checkpoints/", device="cuda")
```

## Output Structure

```
vae_train/
├── outputs/
│   ├── reconstructions/    # Sample reconstructions during training
│   └── training_curves/    # Loss plots
└── checkpoints saved to vae_custom/checkpoints/
```

## Training Tips

1. **Start with low KL weight** — Prevents posterior collapse
2. **Monitor reconstruction quality** — Visual inspection is important
3. **Train for sufficient epochs** — VAE needs 500+ epochs typically
4. **Check latent distributions** — Should approximate N(0, 1)
