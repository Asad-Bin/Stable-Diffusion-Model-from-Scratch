# Custom Stable Diffusion Model from Scratch

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow" alt="MLflow">
</p>

A custom implementation of a latent diffusion model for text-to-image generation, featuring a UNet-based denoiser with CLIP text conditioning and support for both custom and pretrained VAE architectures.

## ✨ Key Features

- **UNet Diffusion Architecture** — Custom UNet with cross-attention blocks for text-conditioned image generation
- **CLIP Text Conditioning** — Leverages OpenAI's CLIP model for semantic text-to-image alignment
- **Dual VAE Support** — Choose between a custom-trained VAE or pretrained Stable Diffusion VAE
- **Qwen2-VL Integration** — Automatic caption generation for training images using Qwen2-VL vision-language model
- **Latent Space Training** — Efficient training in compressed latent space (64×64 latents for 512×512 images)
- **Experiment Tracking** — Full MLflow integration for logging metrics, parameters, and artifacts
- **Caching System** — Disk caching for VAE latents and text embeddings to accelerate training

## 📁 Project Structure

```
Custom-Stable-Diffusion-Model/
├── config/              # Configuration files and hyperparameters
├── dataset/             # Dataset loading, CLIP encoding, and caching utilities
├── img_generation/      # Inference scripts for generating images from checkpoints
├── main/                # Training entry point and main execution loop
├── model/               # UNet architecture with attention blocks
├── noise/               # Noise scheduling and forward diffusion process
├── pre_vae/             # Pretrained VAE wrapper (Stable Diffusion VAE)
├── sampling/            # DDPM sampling algorithm for image generation
├── utils/               # Utility functions for debugging and training
├── vae_custom/          # Custom VAE encoder/decoder implementation
├── vae_train/           # VAE training scripts and utilities
├── requirements.txt     # Python dependencies
└── run.sh               # Shell script for quick execution
```

### 📚 Module Documentation

Each module contains its own README with detailed documentation:

| Module | Description | Key Components |
|--------|-------------|----------------|
| [config](./config/) | Configuration management | `config.py`, `sensitive_config.py` |
| [dataset](./dataset/) | Data pipeline & caching | `ButterflyDatasetWithPrompts`, `CachedLatentDataset`, CLIP encoding |
| [img_generation](./img_generation/) | Inference & generation | Checkpoint loading, batch generation, MLflow integration |
| [main](./main/) | Training entry point | Training loop, loss computation, checkpoint saving |
| [model](./model/) | UNet architecture | `Unet`, `DownBlock`, `UpBlock`, `CrossAttentionBlock` |
| [noise](./noise/) | Diffusion process | `linear_beta_schedule`, `forward_diffusion_sample` |
| [pre_vae](./pre_vae/) | Pretrained VAE | Stable Diffusion VAE wrapper (`sd-vae-ft-mse`) |
| [sampling](./sampling/) | Image generation | `sample_ddpm` - DDPM reverse diffusion |
| [utils](./utils/) | Utilities | Debugging, gradient analysis, progress tracking |
| [vae_custom](./vae_custom/) | Custom VAE | `Encoder`, `Decoder`, `ResidualBlock` |
| [vae_train](./vae_train/) | VAE training | Training scripts, dataloader, visualization |

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended: 24GB+ VRAM)
- PyTorch 2.0+

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Custom-Stable-Diffusion-Model.git
   cd Custom-Stable-Diffusion-Model
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure sensitive settings**
   
   Create `config/sensitive_config.py` with your configuration:
   ```python
   # MLflow settings
   MLFLOW_TRACKING_URI = "sqlite:///mlruns.db"
   MLFLOW_EXPERIMENT_NAME = "custom-diffusion"
   
   # VAE checkpoint paths (for custom VAE)
   VAE_ENCODER_PATH = "vae_custom/checkpoints/encoder.pt"
   VAE_DECODER_PATH = "vae_custom/checkpoints/decoder.pt"
   
   # Latent space normalization
   LATENT_SHIFT = 0.0
   LATENT_MAGNITUDE = 1.0
   
   # For resuming training
   OLD_RUN_ID = None
   ```

### Dataset Setup

Download and prepare the Smithsonian Butterflies dataset:

```bash
cd dataset
mkdir -p dataset/huggingface_butterflies
python -c "from datasets import load_dataset; \
    dataset = load_dataset('huggan/smithsonian_butterflies_subset', split='train'); \
    dataset.save_to_disk('dataset/huggingface_butterflies')"
cd ..
```

### Training

Start training the diffusion model:

```bash
python -m main.main
```

Training progress is automatically logged to MLflow and checkpoints are saved to the `output/` directory.

### Inference

Generate images using a trained checkpoint:

```bash
python -m img_generation.img_generation --run_id <MLFLOW_RUN_ID>
```

## ⚙️ Configuration

### Model Configuration (`config/config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_size` | 512 | Output image resolution |
| `batch_size` | 32 | Training batch size |
| `num_epochs` | 3000 | Total training epochs |
| `timesteps` | 1000 | Diffusion timesteps |
| `learning_rate` | 0.0001 | Optimizer learning rate |
| `save_image_every` | 5 | Sample generation interval (epochs) |
| `checkpoint_interval` | 50 | Checkpoint saving interval (epochs) |

### VAE Options

**Option 1: Pretrained VAE** (Default)
- Uses `stabilityai/sd-vae-ft-mse` from Hugging Face
- Automatically downloaded on first use
- Recommended for general-purpose generation

**Option 2: Custom VAE**
- Trained specifically on your dataset
- Modify `config/config.py` to use custom VAE:
  ```python
  from vae_custom.custom_vae_wrap import vae  # Custom VAE
  # from pre_vae.pre_vae import vae           # Pretrained VAE
  ```

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│  Image → VAE Encoder → Latent (4×64×64) → Add Noise → UNet     │
│                                                  ↑               │
│  Text Prompt → CLIP Encoder → Text Embedding ────┘               │
│                                                                  │
│  UNet predicts noise → MSE Loss → Gradient Update                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        Inference Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│  Text Prompt → CLIP Encoder → Text Embedding                     │
│                                       ↓                          │
│  Random Noise → DDPM Sampling (1000 steps) → Clean Latent        │
│                                       ↓                          │
│                              VAE Decoder → Generated Image        │
└─────────────────────────────────────────────────────────────────┘
```

### UNet Architecture

The UNet consists of:
- **Encoder**: 3 downsampling blocks with cross-attention at the deepest level
- **Bottleneck**: ConvBlock with cross-attention for text conditioning
- **Decoder**: 3 upsampling blocks with skip connections and cross-attention
- **Conditioning**: Time embeddings + CLIP text embeddings projected and fused

## 📊 Experiment Tracking

MLflow tracks all experiments with:
- Training metrics (loss, gradients)
- Hyperparameters and configuration
- Model checkpoints
- Generated sample images

Launch the MLflow UI:
```bash
mlflow ui --port 5000
```

## 🔧 Cache Management

The project uses caching to optimize training:

### Prompt Cache
- Location: `dataset/dataset/huggingface_butterflies/prompts/`
- Contains Qwen2-VL generated captions

### Latent Cache
- Location: `dataset/dataset/huggingface_butterflies/latents/`
- Contains pre-computed VAE latents and CLIP embeddings

**Clear caches** (if changing models or parameters):
```bash
rm -rf dataset/dataset/huggingface_butterflies/prompts/*
rm -rf dataset/dataset/huggingface_butterflies/latents/*
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run from project root using `python -m main.main` |
| CUDA OOM | Reduce `batch_size` in `config/config.py` |
| VAE loading fails | Verify checkpoint paths in `sensitive_config.py` |
| Slow training | Enable latent caching; first epoch generates cache |

## 📝 Requirements

```
torch>=1.10.0
torchvision>=0.11.0
numpy>=1.20.0
matplotlib>=3.4.0
mlflow>=1.20.0
datasets>=2.0.0
Pillow>=8.0.0
transformers>=4.15.0
scikit-learn>=1.0.0
tqdm>=4.60.0
tensorboard>=2.5.0
huggingface_hub>=0.4.0
accelerate>=0.5.0
```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) for pretrained VAE models
- [OpenAI CLIP](https://github.com/openai/CLIP) for text encoding
- [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) for vision-language captioning
- [Smithsonian Butterflies Dataset](https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset) for training data