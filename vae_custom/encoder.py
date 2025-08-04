import torch
import torch.nn as nn
import torch.nn.functional as F

from vae_custom.blocks import ResAttnBlock, ResidualBlock, AttentionBlock

class Encoder(nn.Module):
    def __init__(self, latent_channels=4):
        super().__init__()
        # Input: (3, 512, 512)
        self.down1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)  # (64, 256, 256)

        # self.block1 = ResAttnBlock(64, 128, use_attn=False)                 # (128, 256, 256)
        self.res1 = ResidualBlock(64, 128)
        self.down2 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)  # (128, 128, 128)

        # self.block2 = ResAttnBlock(128, 256, use_attn=False)                # (256, 128, 128)
        self.res2 = ResidualBlock(128, 256)
        self.down3 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)  # (256, 64, 64)

        # self.block3 = ResAttnBlock(256, 256, use_attn=True)
        self.res3 = ResidualBlock(256, 256)
        # self.attn3 = AttentionBlock(256)

        self.conv_mu     = nn.Conv2d(256, latent_channels, kernel_size=3, stride=1, padding=1)
        self.conv_logvar = nn.Conv2d(256, latent_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.silu(self.down1(x))
        x = self.res1(x)
        # x = self.block1(x)
        x = F.silu(self.down2(x))
        x = self.res2(x)
        # x = self.block2(x)
        x = F.silu(self.down3(x))
        x = self.res3(x)
        # x = self.block3(x)

        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar
