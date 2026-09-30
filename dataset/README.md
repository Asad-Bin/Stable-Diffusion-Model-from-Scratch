# dataset

Data pipeline: HF images → Qwen2-VL captions → VAE latents (cached) → `DataLoader`. Also holds the CLIP text model used during training.

> `load_dataset.py` and `clip.py` load models **at import time** (see below).

## `load_dataset.py`

Import-time work: `from config.config import *`, `from pre_vae.pre_vae import vae`, then `setup_text_encoder()` loads `openai/clip-vit-base-patch32` (`CLIPTokenizer` + `CLIPTextModel`, eval, on `device`). It raises `ImportError` if that fails. The Qwen2-VL model is loaded lazily on first caption.

| Symbol | Description |
|---|---|
| `setup_text_encoder()` | Sets globals `text_encoder`, `tokenizer`, `encode_method='transformers'` |
| `setup_queenvl()` | Loads `Qwen/Qwen2-VL-2B-Instruct` processor and `Qwen2VLForConditionalGeneration` (fp16) onto `device`. If loading fails the error is printed and the model stays `None`. |
| `generate_prompt_with_queenvl(pil)` | Asks for a short butterfly description (`max_new_tokens=100`), takes the text after `"assistant"`, removes `:`, truncates to CLIP's 77 tokens |
| `ButterflyDatasetWithPrompts(hf_dataset, transform, save_prompts_file, regenerate_prompts)` | Generates captions (stored as `prompt[202:]`) or reads them from a file. Writes lines `[i] caption`. `__getitem__` → `(image_tensor, prompt)` |
| `encode_text_with_clip(text)` | `last_hidden_state.mean(dim=1)`, L2-normalized, on CPU (512-d for ViT-B/32). Prints the text. |
| `cache_latents_to_disk(dataset, save_dir)` | For each item: encodes the image in fp16 (handles `latent_dist`, `.latents`, `.sample()` or raw tensor outputs), normalizes `(z - latent_shift) / latent_magnitude`, embeds the prompt (after the marker `"within 2 0 words assistant "` if present), saves the files below. Logs to `dataset/huggingface_butterflies/cache_log.txt` every 100 items. |
| `CachedLatentDataset(latent_dir)` | Lists `latent_*.pt`, `text_emb_*.pt` (sorted) and `prompts.txt`. `__getitem__` → `(latent, text_embedding, prompt)` |
| `LoadData(prompt_style="queenvl", regenerate_prompts=False)` | Full pipeline, returns a `DataLoader` (`batch_size=32`, `shuffle=True`, `num_workers=2`). `prompt_style` is ignored. |

### What `LoadData` does

1. `load_dataset("huggan/smithsonian_butterflies_subset", split="train")`.
2. Transform: `Resize((512, 512))` → `ToTensor` → `Normalize(0.5, 0.5)` → range [-1, 1].
3. `ButterflyDatasetWithPrompts(...)` is **always constructed**. With `regenerate_prompts=False` it reuses `dataset/huggingface_butterflies/generated_prompts_queenvl.txt` if it exists, otherwise it captions every image with Qwen2-VL.
4. If `dataset/huggingface_butterflies/cached_latents_queenvl/` does not exist (or `regenerate_prompts=True`), calls `cache_latents_to_disk`.
5. Builds the `DataLoader` over `CachedLatentDataset`, prints the first batch's shapes, returns it.

### Cache layout (`dataset/huggingface_butterflies/cached_latents_queenvl/`)

```
latent_<i>.pt      # tensor (4, 64, 64), normalized latent
text_emb_<i>.pt    # tensor (512,), CLIP ViT-B/32, L2-normalized (not used by training)
prompts.txt        # one caption per line
```

Training uses only `latent` and `prompt` from each batch. The text embedding is recomputed with `dataset/clip.py`.

### Known caveat: latent ↔ prompt pairing

`CachedLatentDataset` sorts filenames as strings (`latent_0, latent_1, latent_10, latent_100, …`), but indexes `prompts.txt` by integer position. Latent and cached embedding stay aligned with each other, but the prompt from `prompts.txt` generally does not belong to that latent. Naturally-sorted filenames (or storing the prompt with the latent) would fix it.

## `clip.py`

Loads at import: `CLIPTokenizer` and `CLIPTextModel` from `openai/clip-vit-base-patch16`, moved to `config.device`, set to `eval()`. Exposes `clip_tokenizer` and `clip_text_model`. These are the models `main.main` and `img_generation` use (through `get_clip_text_embedding_batch` in `main/main.py`).

## Run

```bash
python -m dataset.load_dataset      # __main__: LoadData(regenerate_prompts=True), prints the first batch
```

Note `regenerate_prompts=True` recaptions every image with Qwen2-VL and rebuilds the cache.

## Git

`dataset/huggingface_butterflies/`, `cached_latents*/`, `cache_log.txt` and `generated_prompts_*.txt` are git-ignored and regenerated.
