# Noised image generation
import torch


from config.config import device

def linear_beta_schedule(timesteps, start=1e-4, end=0.02):
    return torch.linspace(start, end, timesteps)

def prepare_alphas(betas):
    # device = betas.device
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    return alphas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod

@torch.no_grad()
def forward_diffusion_sample(x0, t, betas, sqrt_ac, sqrt_1mac):
    b_size = x0.size(0)
    sqrt_ac_t = sqrt_ac[t].reshape(b_size, 1, 1, 1)
    sqrt_1mac_t = sqrt_1mac[t].reshape(b_size, 1, 1, 1)
    eps = torch.randn_like(x0)
    x_t = sqrt_ac_t.to(device) * x0.to(device) + sqrt_1mac_t.to(device) * eps.to(device)
    return x_t, eps
