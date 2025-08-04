import torch

from config.config import *
# from pre_vae.pre_vae import vae


@torch.no_grad()
def sample_ddpm(model, betas, shape, device, vae=vae, timesteps=timesteps):
    model.eval()
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas).to(device)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).to(device)

    x = torch.randn(shape, device=device)
    # print(f"dimension at start {x.shape}")

    for i in reversed(range(timesteps)):
        t = torch.tensor([i] * shape[0], device=device).long()
        pred_noise = model(x, t)

        if i == 0:
            x = (1.0 / torch.sqrt(alphas[i])) * (x - betas[i] / sqrt_one_minus_alphas_cumprod[i] * pred_noise)
        else:
            x = (1.0 / torch.sqrt(alphas[i])) * (x - betas[i] / sqrt_one_minus_alphas_cumprod[i] * pred_noise)
            noise = torch.randn_like(x)
            sigma = torch.sqrt(betas[i])
            x += sigma * noise
            
    if vae is not None:
        vae = vae.to(device).half()
        
        # latent_unscaled = (x * vae.config.latent_magnitude) + vae.config.latent_shift
        x = (x * latent_magnitude) + latent_shift
        x = x.half()

        decoded = vae.decode(x)  # extract tensor
        sampled = decoded.clamp(-1, 1)                # clamp the tensor


    return sampled
