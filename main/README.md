# Main Module

Training entry point and main execution loop for the diffusion model.

## Files

| File | Description |
|------|-------------|
| `main.py` | Main training script with complete training loop |
| `old_run.py` | Utilities for resuming previous training runs |

## Usage

### Start Training

```bash
python -m main.main
```

This will:
1. Load the dataset with cached latents
2. Initialize the UNet model
3. Run the training loop for configured epochs
4. Save checkpoints and sample images periodically
5. Log metrics to MLflow

## Training Loop

### `train_model()`

Main training function with comprehensive logging:

```python
def train_model(
    dataloader,           # DataLoader with cached latents
    model,                # UNet model instance
    betas,                # Noise schedule
    epochs=3000,          # Total epochs
    lr=0.0001,            # Learning rate
    save_interval=5       # Sample generation interval
)
```

### Training Steps

```
For each epoch:
    For each batch:
        1. Sample random timesteps
        2. Add noise to latents (forward diffusion)
        3. Predict noise with UNet (conditioned on text)
        4. Compute MSE loss
        5. Backpropagate and update weights
    
    Every save_interval epochs:
        - Generate sample images
        - Log metrics to MLflow
    
    Every checkpoint_interval epochs:
        - Save model checkpoint
```

## Helper Functions

### `get_clip_text_embedding_batch()`

Batch encode text prompts with CLIP:

```python
embeddings = get_clip_text_embedding_batch(
    prompts=["prompt1", "prompt2"],
    tokenizer=clip_tokenizer,
    text_model=clip_text_model,
    device=device
)
```

### `print_image()`

Generate and save sample images during training:

```python
print_image(epoch=100)  # Saves grid to output_dir
```

## Model Initialization

The script initializes the UNet with:

```python
model = Unet(
    input_ch=4,           # VAE latent channels
    output_ch=4,          # Output channels
    base_ch=64,           # Base channel multiplier
    time_emb_dim=128,     # Time embedding dimension
    time_steps=1000,      # Total timesteps
    num_groups=8,         # GroupNorm groups
    use_attn_bottleneck=True
).to(device)
```

## Checkpoints

Checkpoints are saved to `<output_dir>/checkpoints/` containing:
- `model_state_dict` — UNet weights
- `optimizer_state_dict` — Optimizer state
- `epoch` — Current epoch number
- `loss` — Training loss

## Resuming Training

Set in `config/config.py`:

```python
old_run = True
old_checkpoint_dir = "/path/to/previous/run/checkpoints"
old_run_checkpoint_no = 1000  # Epoch to resume from
```
