import torch
import torch.nn as nn
import torch.nn.functional as F

from vae_custom.blocks import ResAttnBlock, ResidualBlock, AttentionBlock

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 256, kernel_size=3, stride=1, padding=1)

        # self.block1 = ResAttnBlock(256, 256, use_attn=True)
        self.res1 = ResidualBlock(256, 256)
        # self.attn1 = AttentionBlock(256)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # (128, 128, 128)

        # self.block2 = ResAttnBlock(128, 128, use_attn=True)
        self.res2 = ResidualBlock(128, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # (64, 256, 256)

        # self.block3 = ResAttnBlock(64, 64, use_attn=True)
        self.res3 = ResidualBlock(64, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)    # (32, 512, 512)

        self.out = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        z = F.silu(self.conv1(z))
        z = self.res1(z)
        # z = self.block1(z)
        z = F.silu(self.up1(z))
        z = self.res2(z)
        # z = self.block2(z)
        z = F.silu(self.up2(z))
        z = self.res3(z)
        # z = self.block3(z)
        z = F.silu(self.up3(z))
        return torch.tanh(self.out(z))
