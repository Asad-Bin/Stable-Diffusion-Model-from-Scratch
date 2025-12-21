# Image Generation Module

Inference scripts for generating images using trained diffusion model checkpoints.

## Files

| File | Description |
|------|-------------|
| `img_generation.py` | Main inference script with MLflow integration |

## Usage

### Command Line

Generate images from a trained model checkpoint:

```bash
python -m img_generation.img_generation --run_id <MLFLOW_RUN_ID>
```

With custom output directory:

```bash
python -m img_generation.img_generation \
    --run_id <MLFLOW_RUN_ID> \
    --output_dir /path/to/output
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--run_id` | Yes | MLflow run ID to load checkpoint from |
| `--output_dir` | No | Custom output directory (optional) |

## Generation Pipeline

```
1. Load checkpoint from MLflow run
        ↓
2. Initialize UNet model with saved weights
        ↓
3. Select random prompt from predefined set
        ↓
4. Encode prompt with CLIP
        ↓
5. Run DDPM sampling (1000 timesteps)
        ↓
6. Decode latents with VAE
        ↓
7. Save 4×4 image grid with prompt overlay
```

## Output

Generated images are saved to:
```
<output_dir>/generated_output/generated_imgs_checkpoint_<epoch>_<timestamp>.png
```

Each output image includes:
- 4×4 grid of generated samples (16 images)
- Prompt text overlaid at the top
- Normalized to [0, 1] range

## Built-in Prompts

The script includes sample butterfly prompts:

```python
butterfly_prompts = [
    "a beautiful butterfly with colorful wings",
    "a blue morpho butterfly on a leaf",
    "a monarch butterfly in flight",
    "a realistic butterfly with detailed patterns",
    ...
]
```

## Example Output Structure

```
output/output_20250101_120000/
├── checkpoints/
│   └── checkpoint_epoch_3000.pth
└── generated_output/
    ├── generated_imgs_checkpoint_3000_20250102_140000.png
    └── generated_imgs_checkpoint_3000_20250102_141500.png
```
