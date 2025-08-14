import torch
import torch.nn as nn
from .blocks import get_sinusoidal_embedding, DownBlock, UpBlock, AttentionBlock, ConvBlock

class Unet(nn.Module):
    def __init__(self, input_ch=4, base_ch=64, output_ch=4, time_emb_dim=128, text_emb_dim=512, time_steps=1000,
                 use_attn_enc=True, use_attn_bottleneck=True, use_attn_dec=True, num_groups=8):
        super().__init__()
        self.time_steps = time_steps
        self.time_emb_dim = time_emb_dim
        self.text_emb_dim = text_emb_dim

        # Shared time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )

        # Project text embedding to same dim as time embedding
        self.text_proj = nn.Linear(text_emb_dim, time_emb_dim)

        # Encoder
        self.enc1 = DownBlock(input_ch, base_ch, time_emb_dim + time_emb_dim, use_attn=False, num_groups=num_groups)
        self.enc2 = DownBlock(base_ch, base_ch * 2, time_emb_dim + time_emb_dim, use_attn=False, num_groups=num_groups)
        self.enc3 = DownBlock(base_ch * 2, base_ch * 4, time_emb_dim + time_emb_dim, use_attn=use_attn_enc, num_groups=num_groups)

        # Bottleneck
        self.base = ConvBlock(base_ch * 4, base_ch * 8, time_emb_dim + time_emb_dim, num_groups=num_groups)
        self.base_att = AttentionBlock(base_ch * 8) if use_attn_bottleneck else nn.Identity()

        # Decoder
        self.dec3 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4, time_emb_dim + time_emb_dim, use_attn=use_attn_dec, num_groups=num_groups)
        self.dec2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2, time_emb_dim + time_emb_dim, use_attn=False, num_groups=num_groups)
        self.dec1 = UpBlock(base_ch * 2, base_ch, base_ch, time_emb_dim + time_emb_dim, use_attn=False, num_groups=num_groups)

        # Output
        self.out = nn.Conv2d(base_ch, output_ch, kernel_size=1)

    def forward(self, x, t, text_emb=None):
        # Sinusoidal time embedding
        t_emb = get_sinusoidal_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        # Project text embedding
        if text_emb is not None:
            text_emb_proj = self.text_proj(text_emb)
            combined_emb = torch.cat([t_emb, text_emb_proj], dim=-1)
        else:
            combined_emb = t_emb

        # Encoder
        e1, e1p = self.enc1(x, combined_emb)
        e2, e2p = self.enc2(e1p, combined_emb)
        e3, e3p = self.enc3(e2p, combined_emb)

        # Bottleneck
        b = self.base(e3p, combined_emb)
        b = self.base_att(b)

        # Decoder
        d3 = self.dec3(b, e3, combined_emb)
        d2 = self.dec2(d3, e2, combined_emb)
        d1 = self.dec1(d2, e1, combined_emb)

        return self.out(d1)
