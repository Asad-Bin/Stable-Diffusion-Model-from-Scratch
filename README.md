# Butterfly Image Generation Project

## Project Description

This project implements a diffusion-based image generation model specialized for generating butterfly images. It uses a combination of:

- A UNet-based diffusion model for the core generation process
- CLIP text embeddings for text-to-image conditioning
- A VAE (Variational Autoencoder) for encoding/decoding images (with options for custom or pretrained)
- Qwen2-VL for generating descriptive prompts

The system allows for generating high-quality butterfly images based on text descriptions, with the ability to fine-tune various parameters through the configuration system.

## Quick Start Guide for First-Time Users

1. **Setup Environment**:
   ```bash
   git clone <repository-url>
   cd butterfly-image-generation
   pip install -r requirements.txt
   ```

2. **Configure the System**:
   - Create `config/sensitive_config.py` (see template in Installation section)
   - Download the dataset (instructions below)

3. **Choose VAE Option**:
   - Custom VAE: Specialized for butterflies (default)
   - Pretrained VAE: General purpose from Hugging Face

4. **Run Training**:
   ```bash
   python -m main.main
   ```

5. **Generate Images**:
   ```bash
   python -m main.main --inference
   ```

## Project Structure

- `config/`: Configuration files and settings
- `dataset/`: Dataset loading and processing utilities
- `img_generation/`: Image generation components
- `main/`: Main execution scripts
- `model/`: Core model architecture (UNet)
- `noise/`: Noise generation and scheduling
- `sampling/`: Sampling algorithms for the diffusion process
- `utils/`: Utility functions
- `vae_custom/`: Custom VAE implementation
- `vae_train/`: VAE training utilities

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd butterfly-image-generation
   ```

2. Install dependencies using the requirements.txt file:
   ```bash
   pip install -r requirements.txt
   ```
   
   This will install the following key dependencies:
   - PyTorch (with CUDA support if available)
   - Transformers (for CLIP and Qwen2-VL models)
   - Diffusers (for pretrained VAE option)
   - MLflow (for experiment tracking)
   - Matplotlib (for visualization)
   - NumPy and other utilities

3. Create a `config/sensitive_config.py` file with the following template:
   ```python
   # MLflow settings
   MLFLOW_TRACKING_URI = "sqlite:///mlruns.db"  # Local SQLite database
   MLFLOW_EXPERIMENT_NAME = "butterfly-diffusion"
   
   # VAE paths (for custom VAE)
   VAE_ENCODER_PATH = "vae_custom/checkpoints/encoder.pt"
   VAE_DECODER_PATH = "vae_custom/checkpoints/decoder.pt"
   
   # Latent space parameters
   LATENT_SHIFT = 0.0
   LATENT_MAGNITUDE = 1.0
   
   # For resuming training
   OLD_RUN_ID = None  # Set to MLflow run ID if resuming
   ```

4. Download the dataset:
   ```bash
   # Navigate to the dataset directory
   cd dataset
   
   # Create directory for the dataset
   mkdir -p dataset/huggingface_butterflies
   
   # Download the dataset using the Hugging Face datasets library
   python -c "from datasets import load_dataset; dataset = load_dataset('huggan/smithsonian_butterflies_subset', split='train'); dataset.save_to_disk('dataset/huggingface_butterflies')"
   
   # Return to the project root
   cd ..
   ```

## Weight Files Management

The project uses several types of weight files that need to be properly managed:

### 1. VAE Weights

#### Custom VAE Weights
- **Location**: `vae_custom/checkpoints/`
- **Files**: 
  - `encoder.pt`: The encoder part of the custom VAE
  - `decoder.pt`: The decoder part of the custom VAE
- **How to Get**: These should be included in the repository or can be trained using the VAE training scripts in `vae_train/`
- **Configuration**: Paths are set in `sensitive_config.py` as `VAE_ENCODER_PATH` and `VAE_DECODER_PATH`

#### Pretrained VAE Weights
- **Location**: Downloaded automatically to `.cache/huggingface/`
- **Model ID**: `stabilityai/sd-vae-ft-mse`
- **First Use**: Will download automatically when first importing `pre_vae.pre_vae`

### 2. UNet Model Weights

- **Location**: Generated during training in timestamped directories under `output/`
- **Files**: `model_<epoch>.pt` files containing UNet weights
- **Resuming Training**: Set `old_run = True` and specify `old_checkpoint_dir` in config

### 3. CLIP Model Weights

- **Location**: Downloaded automatically to `.cache/huggingface/`
- **Model ID**: `openai/clip-vit-large-patch14`
- **First Use**: Will download automatically when first importing CLIP modules

### 4. Qwen2-VL Weights (for prompt generation)

- **Location**: Downloaded automatically to `.cache/huggingface/`
- **Model ID**: `Qwen/Qwen2-VL-7B`
- **First Use**: Will download automatically when generating prompts

## Choosing Between Custom VAE and Pretrained VAE

This project supports two VAE options:

### Option 1: Using the Custom VAE (Default)

The custom VAE is implemented in the `vae_custom` directory and provides specialized encoding/decoding for butterfly images.

To use the custom VAE:
1. Ensure the VAE checkpoint files are in the correct location:
   ```bash
   # Check if the checkpoint files exist
   ls -la vae_custom/checkpoints/encoder.pt vae_custom/checkpoints/decoder.pt
   ```

2. In `config/config.py`, ensure the VAE configuration is set to use the custom VAE:
   ```python
   # Uncomment this line to use the custom VAE
   from vae_custom.custom_vae_wrap import vae
   
   # Comment out this line if it exists
   # from pre_vae.pre_vae import vae
   ```

### Option 2: Using a Pretrained VAE

The project also supports using a pretrained VAE from the Hugging Face Diffusers library.

To use the pretrained VAE:
1. In `config/config.py`, modify the VAE configuration:
   ```python
   # Comment out the custom VAE import
   # from vae_custom.custom_vae_wrap import vae
   
   # Uncomment this line to use the pretrained VAE
   from pre_vae.pre_vae import vae
   ```

2. The pretrained VAE will be automatically downloaded when first used.

## Cache Management System

The project uses a sophisticated caching system to optimize training and inference performance. Understanding how these caches work is essential for efficient usage.

### 1. Prompt Caches

#### What Are Prompt Caches?
- Text descriptions for each butterfly image generated using Qwen2-VL
- Stored as JSON files to avoid regenerating prompts for each training run

#### Location and Format
- **Directory**: `dataset/dataset/huggingface_butterflies/prompts/`
- **Files**: JSON files containing image ID to prompt mappings
- **Format**: `{"image_id": "detailed butterfly description", ...}`

#### How to Use Prompt Caches
- **First Run**: Set `regenerate_prompts=True` in `ButterflyDatasetWithPrompts` initialization
- **Subsequent Runs**: Set `regenerate_prompts=False` to use cached prompts
- **Customization**: Edit the JSON files directly to modify specific prompts

#### Benefits
- Significantly speeds up dataset loading
- Ensures consistent prompts across training runs
- Reduces API calls to Qwen2-VL model

### 2. Latent Caches

#### What Are Latent Caches?
- VAE-encoded representations of images, pre-computed to speed up training
- CLIP text embeddings for prompts, pre-computed for consistent conditioning

#### Location and Format
- **Directory**: `dataset/dataset/huggingface_butterflies/latents/`
- **Files**: `.pt` (PyTorch) files containing tensor data
- **Naming**: Based on image IDs and encoding parameters

#### How to Use Latent Caches
- **Generation**: Automatically created during first training run
- **Usage**: Automatically loaded in subsequent runs
- **Regeneration**: Delete cache files if you change the VAE or encoding parameters

#### Benefits
- Eliminates VAE encoding overhead during training
- Ensures consistent latent representations
- Reduces VRAM usage during training

### 3. Cache Management Commands

```bash
# Clear all caches (use if changing models or parameters)
rm -rf dataset/dataset/huggingface_butterflies/prompts/*
rm -rf dataset/dataset/huggingface_butterflies/latents/*

# Regenerate only prompt caches
python -c "from dataset.load_dataset import ButterflyDatasetWithPrompts; ButterflyDatasetWithPrompts(regenerate_prompts=True)"

# Regenerate only latent caches
python -c "from dataset.load_dataset import ButterflyDatasetWithPrompts; dataset = ButterflyDatasetWithPrompts(regenerate_prompts=False); dataset.cache_latents_to_disk()"
```

## Data Flow and Processing Pipeline

Understanding the data flow in this project helps you optimize and customize the system:

### 1. Dataset Loading and Preparation
```
Raw Images → ButterflyDatasetWithPrompts → Prompt Generation/Loading → VAE Encoding → Training Dataset
```

- **Raw Images**: Loaded from `dataset/huggingface_butterflies/`
- **Prompt Generation**: Using Qwen2-VL or loading from cache
- **VAE Encoding**: Converting images to latent space using selected VAE

### 2. Training Pipeline
```
Dataset → Batch Sampling → Text Encoding → Noise Addition → UNet Prediction → Loss Calculation → Optimization
```

- **Batch Sampling**: Images and prompts loaded from dataset
- **Text Encoding**: CLIP embeddings for text conditioning
- **Noise Addition**: Random noise added according to diffusion schedule
- **UNet Prediction**: Model predicts noise to remove
- **Loss Calculation**: MSE between predicted and actual noise
- **Optimization**: Gradient descent to update UNet weights

### 3. Inference Pipeline
```
Text Prompt → CLIP Encoding → Random Noise → Iterative Denoising → VAE Decoding → Generated Image
```

- **Text Prompt**: User-provided or randomly selected description
- **CLIP Encoding**: Converting text to embeddings for conditioning
- **Random Noise**: Starting point for the diffusion process
- **Iterative Denoising**: Gradually removing noise using trained UNet
- **VAE Decoding**: Converting latent representation to pixel space

## Training Procedure

### Prerequisites

1. Configure your settings in `config/config.py` and `config/sensitive_config.py`
2. Prepare your dataset in the specified directory
3. Choose your VAE option as described above

### Training Steps

1. **Start Training**:
   ```bash
   python -m main.main
   ```

2. **Training Process**:
   - The model will train for the number of epochs specified in the config
   - Sample images will be generated at intervals defined by `save_image_every`
   - Checkpoints will be saved according to `checkpoint_interval`
   - Training progress is logged to MLflow if enabled

3. **Monitoring**:
   - Training metrics are logged to MLflow
   - Generated samples are saved to the output directory
   - Checkpoints are stored for later use or resuming training

4. **Resuming Training**:
   - Set `old_run = True` in the config to resume from a previous checkpoint
   - Specify the checkpoint directory in `old_checkpoint_dir`
   - Set `OLD_RUN_ID` in `sensitive_config.py` to the MLflow run ID if applicable

## Inference Procedure

To generate butterfly images using a trained model:

1. **Load a Trained Model**:
   - Ensure the checkpoint path is correctly set in the configuration

2. **Generate Images**:
   ```bash
   # Run the inference script
   python -m main.main --inference
   ```

3. **Programmatic Generation**:
   ```python
   from sampling.sample_ddpm import sample_ddpm
   from dataset.clip import clip_tokenizer, clip_text_model
   
   # Prepare text embeddings
   def get_clip_text_embedding_batch(prompts, tokenizer, text_model, device):
       inputs = tokenizer(prompts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
       with torch.no_grad():
           out = text_model(**inputs).last_hidden_state
           text_features = out.mean(dim=1)
       return text_features
   
   # Define prompts
   prompts = ["a black and yellow butterfly with blue markings", 
              "a beautiful butterfly with green wings"]
   
   # Get text embeddings
   text_embedding = get_clip_text_embedding_batch(prompts, clip_tokenizer, clip_text_model, device)
   
   # Generate images
   sampled = sample_ddpm(
       model, 
       betas, 
       shape=(len(prompts), 4, image_size//8, image_size//8), 
       device=device, 
       timesteps=timesteps, 
       vae=vae,
       text_embeddings=text_embedding
   )
   
   # Convert to images and save
   with torch.no_grad():
       images = vae.decoder(sampled.to(vae.dtype)).clamp(-1, 1)
       images = (images + 1) / 2  # Normalize to [0, 1]
       # Save or display images
   ```

4. **Batch Generation**:
   - For generating multiple images, prepare a batch of prompts and process them together
   - The system supports efficient batch processing for faster generation

## Modifying the Configuration

### Main Configuration (`config/config.py`)

The `config.py` file contains the primary settings for the model. Key parameters to modify include:

- **Model Parameters**:
  - `image_size`: Size of the generated images (default: 512)
  - `batch_size`: Batch size for training
  - `num_epochs`: Total training epochs
  - `timesteps`: Number of diffusion steps
  - `learning_rate`: Learning rate for optimization

- **Training Settings**:
  - `save_image_every`: Interval for generating sample images
  - `checkpoint_interval`: Interval for saving model checkpoints
  - `prompt_style`: Choose between "queenvl" (Qwen2-VL generated) or other options

- **Paths and Directories**:
  - `dataset_dir`: Path to the dataset
  - `output_dir`: Directory for saving outputs (auto-generated with timestamp)

### Sensitive Configuration (`config/sensitive_config.py`)

For sensitive data and credentials, use the `sensitive_config.py` file (not tracked by git):

- **MLflow Settings**:
  - `MLFLOW_TRACKING_URI`: URI for MLflow tracking server
  - `MLFLOW_EXPERIMENT_NAME`: Name of the MLflow experiment

- **Model Paths**:
  - `VAE_ENCODER_PATH`: Path to the VAE encoder checkpoint (for custom VAE)
  - `VAE_DECODER_PATH`: Path to the VAE decoder checkpoint (for custom VAE)

- **Other Sensitive Values**:
  - `LATENT_SHIFT` and `LATENT_MAGNITUDE`: Parameters for latent space adjustment
  - `OLD_RUN_ID`: MLflow run ID for resuming previous runs

## Troubleshooting

### Common Issues

1. **Module Import Errors**:
   - If you encounter `ModuleNotFoundError`, ensure you're running the script from the project root directory
   - Use `python -m main.main` instead of directly running the script

2. **CUDA Out of Memory**:
   - Reduce `batch_size` in the configuration
   - Reduce `image_size` if necessary

3. **VAE Loading Issues**:
   - For custom VAE: Verify checkpoint paths in `sensitive_config.py`
   - For pretrained VAE: Ensure internet connection for first-time download

4. **Prompt Generation Errors**:
   - Ensure Qwen2-VL model can be downloaded (requires internet connection)
   - If issues persist, use cached prompts by setting `regenerate_prompts=False`

## Notes

- The model uses either a custom VAE or pretrained VAE for encoding/decoding images
- Text conditioning is handled through CLIP embeddings
- The diffusion process follows a linear beta schedule
- MLflow is used for experiment tracking and visualization
- Cached prompts and latents are used to speed up training when possible