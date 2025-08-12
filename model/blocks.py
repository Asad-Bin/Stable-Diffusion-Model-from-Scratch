import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_sinusoidal_embedding(timesteps, time_emb_dim):
    device = timesteps.device
    half_dim = time_emb_dim // 2
    emb = torch.exp(-np.log(10000) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim).to(device)
    emb = timesteps[:, None].float() * emb[None, :]  # (B, half_dim)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # (B, time_emb_dim)
    return emb

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(16, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.view(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        attn_out, _ = self.attn(h, h, h)
        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)
        return x + self.proj(attn_out)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None, num_groups=8):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels) if time_emb_dim else None

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x, t_emb=None):
        x = self.conv1(x)
        x = self.norm1(x)
        # x = F.silu(x)

        if t_emb is not None and self.time_emb_proj is not None:
            t_proj = self.time_emb_proj(t_emb).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
            x = x + t_proj

        x = self.conv2(x)
        x = self.norm2(x)
        x = F.silu(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None, use_attn=False, num_groups=8):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, time_emb_dim, num_groups)
        self.attn = AttentionBlock(out_channels) if use_attn else nn.Identity()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t_emb):
        x = self.conv(x, t_emb)
        x = self.attn(x)
        x_pooled = self.pool(x)
        return x, x_pooled

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, time_emb_dim=None, use_attn=False, num_groups=8):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(skip_channels + out_channels, out_channels, time_emb_dim=time_emb_dim, num_groups=num_groups)
        self.attn = AttentionBlock(out_channels) if use_attn else nn.Identity()

    def forward(self, x, skip, t_emb=None):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x, t_emb)
        x = self.attn(x)
        return x
