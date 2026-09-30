# vae_train

Training, evaluation and export scripts for the custom VAE in [`vae_custom`](../vae_custom/). No `__init__.py`; scripts use relative imports, so run them with `python -m vae_train.<name>` from the repo root.

> `train.py`, `test.py` and `dataloader.py` have no `__main__` guard. Importing them starts work (downloading data, training, or evaluating).

## Files

| File | Role |
|---|---|
| `config.py` | `device = cuda:0` (else CPU), `image_size=512`, `num_epochs=1000`, `learning_rate=1e-4`, `batch_size=8` |
| `dataloader.py` | `HuggingFaceImageDataset`, `LoadData()`, `load_test_images()`; builds `dataloader` at import |
| `train.py` | Training loop |
| `test.py` | Reconstruct test images with a saved checkpoint |
| `checkpoint_updater.py` | Copy a checkpoint pair into `vae_custom/checkpoints/` |
| `plot.py` | `local(outputs)`: matplotlib preview of the first reconstruction (the call is commented out) |

## Data (`dataloader.py`)

- Training: `huggan/smithsonian_butterflies_subset` (train split), `Resize(512, 512)` → `ToTensor` → `Normalize(0.5, 0.5)`, `DataLoader(batch_size=8, shuffle=True, num_workers=2)`. Items are `(image, 0)`.
- Test images: `load_test_images()` reads every `vae_train/test_images/*.png` (resize 512, normalize to [-1, 1]) and stacks them. The directory is git-ignored, so **you must create it and add PNGs**. The `path_pattern` argument of the function is ignored.

## Training (`train.py`)

```bash
mlflow server --port 5000        # required: URI is hardcoded
python -m vae_train.train
```

- MLflow: `http://127.0.0.1:5000`, experiment `Custom_VAE_Training`, run name `vae_run`.
- Model: `Encoder` + `Decoder` from `vae_custom`, one `AdamW(lr=1e-4)` over both, `CosineAnnealingLR(T_max=1000, eta_min=1e-6)` stepped per epoch.
- Loss: `MSE(recon, image) + β(epoch) · KL`, with `KL = -0.5 · mean(1 + logvar − mu² − exp(logvar))` and `β(epoch) = 1e-4 · min(1, epoch / 10)`. (A `beta = 0.1` variable is defined but unused.)
- Resume: if `vae_train/checkpoints/encoder_epoch_500.pt` and `decoder_epoch_500.pt` exist, it loads them and continues from epoch 500. Otherwise it starts from 0. To resume from another epoch, edit `resume_epoch`. Optimizer and scheduler state are **not** saved or restored.
- Each epoch: saves the first 2 originals interleaved with their reconstructions to `vae_train/saved_images/train_recon_<epoch>.png` (single overwriting folder, epoch in the filename) and uploads it to MLflow.
- Logged: `grad_norm` (first batch of each epoch), `recon_loss`, `kl_loss`, `total_loss`, `lr`.
- Every 100 epochs: `vae_train/checkpoints/{encoder,decoder}_epoch_<N>.pt` (also logged as MLflow artifacts).

## Evaluation (`test.py`)

```bash
python -m vae_train.test
```

At import it runs `evaluate_vae("vae_train/checkpoints")`. It loads `encoder_epoch_1000.pt` / `decoder_epoch_1000.pt` (`chkpnt_epoch`), then for each image in `vae_train/test_images/`:
- Reconstructs it (`z` is still sampled from the encoder).
- Prints `MSE + β(0)·KL`, where `β(0)` is 0, so this equals the MSE.
- Saves `[original, reconstruction]` to `vae_train/testing_outputs/test_eval_<i>.png` (written twice; the argparse parser at the top is unused).

## Exporting to `vae_custom` (`checkpoint_updater.py`)

```bash
python -m vae_train.checkpoint_updater
```

Copies `vae_train/checkpoints/{encoder,decoder}_epoch_<copy_target>.pt` (`copy_target = 1000`) to `vae_custom/checkpoints/encoder.pt` and `decoder.pt`.

## Git

`vae_train/checkpoints/`, `saved_images/`, `test_images/` and `testing_outputs/` are git-ignored.
