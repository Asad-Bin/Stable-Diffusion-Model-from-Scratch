import torch

from config.config import *
# from pre_vae.pre_vae import vae


@torch.no_grad()
def sample_ddpm(model, betas, shape, device, vae=None, timesteps=timesteps, text_embeddings=None):
    model.eval()
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).to(device)

    x = torch.randn(shape, device=device)

    for i in reversed(range(timesteps)):
        t = torch.tensor([i] * shape[0], device=device).long()
        pred_noise = model(x, t, text_emb=text_embeddings) if text_embeddings is not None else model(x, t)
        
        x = (1.0 / torch.sqrt(alphas[i])) * (x - betas[i] / sqrt_one_minus_alphas_cumprod[i] * pred_noise)
        if i > 0:
            x += torch.sqrt(betas[i]) * torch.randn_like(x)

    if vae is not None:
        vae.eval()
        x = (x * latent_magnitude) + latent_shift
        # x = x.half()
        sampled = vae.decoder(x.float()).clamp(-1, 1)
        return sampled
    else:
        return x.clamp(-1, 1)
