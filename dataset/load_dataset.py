import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset

from config.config import *
from utils.debug import *
# from pre_vae.pre_vae import vae

# --------------------------------------------------
# Dataset that loads HuggingFace images
# --------------------------------------------------
class HuggingFaceImageDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image'].convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0

# --------------------------------------------------
# Save VAE-encoded latents to disk
# --------------------------------------------------
def cache_latents_to_disk(dataset, save_dir="cached_latents"):
    os.makedirs(save_dir, exist_ok=True)
    vae.eval().to(device).half()

    all_latents = []
    for idx in range(len(dataset)):
        image, _ = dataset[idx]
        image = image.to(device).half().unsqueeze(0)

        with torch.no_grad():
            latents = vae.encode(image)
            latents = (latents - latent_shift) / latent_magnitude
            # latents = latents * scaling_factor
            all_latents.append(latents)
        # print(latents.mean(), latents.var())
        # latents = (latents - vae.config.latent_shift) / vae.config.latent_magnitude
        # all_latents.append(latents)
        save_path = os.path.join(save_dir, f"{idx}.pt")
        torch.save(latents.squeeze(0).cpu(), save_path)
        if idx % 100 == 0:
            print(f"Saved latent {idx} to {save_path}")
    
    all_latents = torch.concat(all_latents, dim=0)
    print(f"Latents shape: {all_latents.shape}, mean: {all_latents.mean()}, var: {all_latents.var()}, std: {all_latents.std()}")
    print(all_latents.min(), all_latents.max())

    print(f"✅ Cached {len(dataset)} latent files to {save_dir}")
    # vae.to("cpu")  # free GPU

# --------------------------------------------------
# Dataset for cached latents
# --------------------------------------------------
class CachedLatentDataset(Dataset):
    def __init__(self, latent_dir):
        self.latent_dir = latent_dir
        self.latent_files = sorted([
            os.path.join(latent_dir, f) for f in os.listdir(latent_dir) if f.endswith('.pt')
        ])

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, idx):
        latent = torch.load(self.latent_files[idx])  # stays on CPU
        return latent


# --------------------------------------------------
# Data loader with latent pre-caching
# --------------------------------------------------
def LoadData():
    hf_data = load_dataset("huggan/smithsonian_butterflies_subset", split="train")
    print("Loaded Hugging Face dataset with length:", len(hf_data))

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = HuggingFaceImageDataset(hf_data, transform=transform)

    latent_cache_dir = os.path.join(dataset_dir, "cached_latents")
    if not os.path.exists(latent_cache_dir) or len(os.listdir(latent_cache_dir)) < len(dataset):
        print("🔄 Caching latents...")
        cache_latents_to_disk(dataset, save_dir=latent_cache_dir)

    cached_dataset = CachedLatentDataset(latent_cache_dir)
    dataloader = DataLoader(cached_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    # for i, imgs in enumerate(dataloader):
    #     print(imgs.shape, imgs.min(), imgs.max())

    print(f"✅ Loaded {len(cached_dataset)} cached latent files.")
    return dataloader


# Final call
dataloader = LoadData()
