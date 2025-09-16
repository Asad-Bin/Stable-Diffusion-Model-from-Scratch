# Butterfly Image Generation Project

## Project Description

This project implements a diffusion-based image generation model specialized for generating butterfly images. It uses a combination of:

- A UNet-based diffusion model for the core generation process
- CLIP text embeddings for text-to-image conditioning
- A custom VAE (Variational Autoencoder) for encoding/decoding images
- Qwen2-VL for generating descriptive prompts

The system allows for generating high-quality butterfly images based on text descriptions, with the ability to fine-tune various parameters through the configuration system.

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
   cd task1
   ```

2. Install dependencies using the requirements.txt file:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `config/sensitive_config.py` file based on the template in the README

## Training Procedure

### Prerequisites

1. Configure your settings in `config/config.py` and `config/sensitive_config.py`
2. Prepare your dataset in the specified directory

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

## Inference Procedure

To generate butterfly images using a trained model:

1. **Load a Trained Model**:
   - Ensure the checkpoint path is correctly set in the configuration

2. **Generate Images**:
   - Use the `print_image` function to generate samples
   - Alternatively, use the sampling module directly:
   ```python
   from sampling.sample_ddpm import sample_ddpm
   
   # Prepare text embeddings
   text_embedding = get_clip_text_embedding_batch(prompts, clip_tokenizer, clip_text_model, device)
   
   # Generate images
   sampled = sample_ddpm(
       model, 
       betas, 
       shape=sample_shape, 
       device=device, 
       timesteps=timesteps, 
       vae=vae,
       text_embeddings=text_embedding
   )
   ```

3. **Batch Generation**:
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

- **Paths and Directories**:
  - `dataset_dir`: Path to the dataset
  - `output_dir`: Directory for saving outputs (auto-generated with timestamp)

### Sensitive Configuration (`config/sensitive_config.py`)

For sensitive data and credentials, use the `sensitive_config.py` file (not tracked by git):

- **MLflow Settings**:
  - `MLFLOW_TRACKING_URI`: URI for MLflow tracking server
  - `MLFLOW_EXPERIMENT_NAME`: Name of the MLflow experiment

- **Model Paths**:
  - `VAE_ENCODER_PATH`: Path to the VAE encoder checkpoint
  - `VAE_DECODER_PATH`: Path to the VAE decoder checkpoint

- **Other Sensitive Values**:
  - `LATENT_SHIFT` and `LATENT_MAGNITUDE`: Parameters for latent space adjustment
  - `OLD_RUN_ID`: MLflow run ID for resuming previous runs

### Creating a New Configuration

To create a custom configuration:

1. Copy the existing `config.py` as a template
2. Modify the parameters according to your requirements
3. Create a corresponding `sensitive_config.py` with your sensitive data
4. Ensure `sensitive_config.py` is added to `.gitignore` to prevent accidental commits

## Notes

- The model uses a custom VAE for encoding/decoding images
- Text conditioning is handled through CLIP embeddings
- The diffusion process follows a linear beta schedule
- MLflow is used for experiment tracking and visualization