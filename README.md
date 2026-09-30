# Custom Stable Diffusion Model from Scratch

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow" alt="MLflow">
</p>

A from-scratch **latent diffusion** (DDPM) pipeline for text-to-image generation on the Smithsonian butterflies dataset. The denoiser is a hand-written, text-conditioned UNet that operates on 4×64×64 latents of 512×512 images. Captions are produced automatically by Qwen2-VL and embedded with CLIP. A custom VAE is also included, but the pipeline currently uses the pretrained TAESD VAE.

## ✨ What the code does

- **Custom UNet** (`model/`): 3 down blocks, a bottleneck, 3 up blocks, conditioned on a sinusoidal timestep embedding and a CLIP text embedding. Cross-attention is used in `enc3`, the bottleneck and `dec3`.
- **Automatic captions** (`dataset/load_dataset.py`): `Qwen/Qwen2-VL-2B-Instruct` writes one short caption per image.
- **CLIP text conditioning** (`dataset/clip.py`): `openai/clip-vit-base-patch16`, mean-pooled over tokens (512-d).
- **Latent caching**: images are encoded once by the VAE and stored as `.pt` files, so training never touches pixels.
- **DDPM** (`noise/`, `sampling/`): linear β schedule, 1000 steps, ε-prediction with L1 loss, ancestral sampling.
- **Pretrained VAE** (`pre_vae/`): `madebyollin/taesd` (`diffusers.AutoencoderTiny`), fp16.
- **Custom VAE** (`vae_custom/`, `vae_train/`): a small conv/residual VAE trained with MSE + β·KL. Checkpoints are in `vae_custom/checkpoints/`, but it is **not wired into the diffusion pipeline** (see below).
- **MLflow** logging for loss, gradient norms, sample images and checkpoints.

## 🧭 Pipeline

```
HF dataset (huggan/smithsonian_butterflies_subset)
   │  resize 512×512, normalize to [-1, 1]
   ├─► Qwen2-VL-2B  ──► caption per image  ──► generated_prompts_queenvl.txt
   └─► TAESD VAE encode (fp16) ──► (z - latent_shift) / latent_magnitude
                                        │
                        cached_latents_queenvl/latent_{i}.pt  (4×64×64)
                                        ▼
   DataLoader(latent, text_emb, prompt)   batch 32
                                        ▼
   prompt ─► CLIP ViT-B/16 (recomputed every batch, 512-d) ─┐
   t ~ U{0..999}; z_t = √ᾱ·z0 + √(1-ᾱ)·ε                    ▼
                                            Unet(z_t, t, text_emb) ─► ε̂
                                            loss = L1(ε̂, ε)   AdamW + cosine LR
Sampling: N(0, I) ─► 1000 DDPM steps ─► ×latent_magnitude + latent_shift ─► vae.decoder ─► image
```

## 📁 Project structure

```
.
├── config/          # hyperparameters, output dirs, device, VAE selection, MLflow helpers
├── dataset/         # HF dataset, Qwen2-VL captions, latent cache, CLIP model
├── img_generation/  # CLI: generate a 4×4 grid from a trained run's latest checkpoint
├── main/            # training script (main.py) and resume helper (old_run.py)
├── model/           # Unet + building blocks
├── noise/           # beta schedule and forward diffusion
├── pre_vae/         # pretrained TAESD VAE (loaded on import)
├── sampling/        # DDPM reverse process + VAE decode
├── utils/           # GPU memory helpers, gradient-norm plots, progress bar
├── vae_custom/      # custom VAE encoder/decoder, wrapper, tracked .pt checkpoints
├── vae_train/       # custom VAE training / evaluation scripts
├── requirements.txt
├── run.sh           # background launcher for main.main
└── LICENSE          # MIT
```

| Module | Contents |
|---|---|
| [config](./config/) | `config.py`: globals, `CONFIG` dict, MLflow/config logging helpers |
| [dataset](./dataset/) | `load_dataset.py` (`LoadData`, `ButterflyDatasetWithPrompts`, `CachedLatentDataset`), `clip.py` |
| [img_generation](./img_generation/) | `img_generation.py` |
| [main](./main/) | `main.py` (`train_model`, `print_image`), `old_run.py` |
| [model](./model/) | `unet.py` (`Unet`), `blocks.py` |
| [noise](./noise/) | `linear_beta_schedule`, `prepare_alphas`, `forward_diffusion_sample` |
| [pre_vae](./pre_vae/) | `vae` (TAESD) |
| [sampling](./sampling/) | `sample_ddpm` |
| [utils](./utils/) | `debug.py`, `gradient_descent.py`, `epoch_progress.py` |
| [vae_custom](./vae_custom/) | `Encoder`, `Decoder`, `CustomVAEWrapper`, `load_custom_vae` |
| [vae_train](./vae_train/) | `train.py`, `test.py`, `dataloader.py`, `checkpoint_updater.py`, `plot.py` |

## 🚀 Setup

```bash
git clone https://github.com/Asad-Bin/Stable-Diffusion-Model-from-Scratch.git
cd Stable-Diffusion-Model-from-Scratch
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install diffusers        # imported by pre_vae/pre_vae.py but NOT listed in requirements.txt
```

Notes:

- `config/sensitive_config.py` is git-ignored and **optional for training**. `config/config.py` tries to import `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_NAME`, `OLD_RUN_ID`, `VAE_ENCODER_PATH`, `VAE_DECODER_PATH`, `LATENT_SHIFT`, `LATENT_MAGNITUDE` from it and only prints a warning if the file is missing. Those names are used only by `python -m config.config` (MLflow param logging) and by the commented-out custom-VAE block.
- Model weights (Qwen2-VL, CLIP, TAESD) and the dataset are downloaded from the Hugging Face Hub on first use.
- `config.py` selects `cuda:1` if CUDA is available, otherwise CPU. With one GPU, edit the device in `config/config.py`.
- Qwen2-VL captioning is only run if the prompt file is missing (or `regenerate_prompts=True`), so a GPU is effectively required for the first run.

## 🏋️ Train

```bash
# foreground
python -m main.main

# background (sets MY_TIMESTAMP, logs to output/0_output_logs/log_<timestamp>.out)
bash run.sh
```

Start an MLflow server first if you want to use one (`mlflow server --port 5000`). `main.main` does **not** call `mlflow.set_tracking_uri`. Unless `MLFLOW_TRACKING_URI` is set in the environment, MLflow logs to a local `./mlruns` directory.

Each run writes to `output/output_<timestamp>/` (`MY_TIMESTAMP` from `run.sh`, otherwise the current time):

```
output/output_<ts>/
├── checkpoints/checkpoint_epoch_<N>.pth   # every 50 epochs, last 5 kept
└── training_outputs/epoch_<NNNN>_samples.png
```

Key hyperparameters (`config/config.py`, `main/main.py`): 3000 epochs, batch 32, AdamW lr 1e-4, cosine LR to 1e-6, T=1000, β 1e-4→0.02, `Unet(base_ch=64, time_emb_dim=128, text_emb_dim=512, num_groups=8)`.

## 🎨 Generate

```bash
python -m img_generation.img_generation --run_id <mlflow_run_id> [--output_dir <path>]
```

Reads `output_dir` from the run's params (needs an MLflow server at `http://127.0.0.1:5000`, or pass `--output_dir`), loads the highest-numbered checkpoint, and writes a 4×4 grid, with the prompt stamped on top, to `<output_dir>/generated_output/`.

## 🧪 Custom VAE (optional)

```bash
python -m vae_train.train                 # trains; needs MLflow at 127.0.0.1:5000
python -m vae_train.checkpoint_updater    # copies epoch-1000 weights to vae_custom/checkpoints/
```

See [vae_train](./vae_train/) and [vae_custom](./vae_custom/). To use it for diffusion you would have to enable the commented block in `config/config.py` **and** provide `latent_shift`/`latent_magnitude`, `.dtype` and a `.decoder` compatible with `sample_ddpm`.

## ⚠️ Known issues (verified from the code)

1. **Caption/latent misalignment.** `CachedLatentDataset` sorts filenames lexicographically (`latent_1`, `latent_10`, `latent_100`, …, `latent_2`), but reads `prompts.txt` by numeric index. For most items the caption returned does not belong to the returned latent. The caching step itself is correct; the pairing at load time is not.
2. **Resume is a no-op.** In `main/main.py`, `from main import old_run` shadows the `old_run` boolean from `config`, so `if old_run == True` is always false.
3. **`is_ml_flow_off = True` skips checkpoints and samples.** The `continue` after `scheduler.step()` skips the checkpoint and sampling code.
4. **Import-time side effects.** Importing `main.main` runs `LoadData()` (dataset download, CLIP + Qwen setup, cache build if missing). `img_generation` imports `main.main`, so it triggers all of this. `vae_train.train`, `vae_train.test`, and `vae_train.dataloader` also run work on import.
5. **Cache uses a different CLIP than training.** The cached `text_emb_*.pt` come from `clip-vit-base-patch32` (L2-normalized) and are not used. Training re-embeds prompts with `clip-vit-base-patch16`.
6. **`requirements.txt` misses `diffusers`.**
7. **Device mismatch.** `config.py` uses `cuda:1`, `img_generation` and `vae_train` use `cuda:0`.

## 📄 License

MIT. See [LICENSE](./LICENSE).
