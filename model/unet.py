import torch
import torch.nn as nn
from .blocks import get_sinusoidal_embedding, DownBlock, UpBlock, CrossAttentionBlock, ConvBlock
from config.config import device


class Unet(nn.Module):
    def __init__(
        self,
        input_ch=4,
        base_ch=64,
        output_ch=4,
        time_emb_dim=128,
        text_emb_dim=512,
        time_steps=1000,
        use_attn_enc=True,          # kept for API compatibility; not strictly needed
        use_attn_bottleneck=True,   # bottleneck cross-attn enabled when True
        use_attn_dec=True,          # kept for API compatibility; not strictly needed
        num_groups=8
    ):
        super().__init__()
        self.time_steps = time_steps
        self.time_emb_dim = time_emb_dim
        self.text_emb_dim = text_emb_dim

        # Shared time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )

        # Project raw text embedding to same dim as time embedding
        self.text_proj = nn.Linear(text_emb_dim, time_emb_dim)

        # -------------------------
        # Encoder (cross-attn at the lower encoder: enc3)
        # Pass time_emb_dim and text_emb_dim separately so ConvBlock can sum them internally.
        self.enc1 = DownBlock(
            in_channels=input_ch, out_channels=base_ch,
            time_emb_dim=time_emb_dim, text_emb_dim=time_emb_dim,
            cross_attn=False, num_groups=num_groups
        )
        self.enc2 = DownBlock(
            in_channels=base_ch, out_channels=base_ch * 2,
            time_emb_dim=time_emb_dim, text_emb_dim=time_emb_dim,
            cross_attn=False, num_groups=num_groups
        )
        self.enc3 = DownBlock(
            in_channels=base_ch * 2, out_channels=base_ch * 4,
            time_emb_dim=time_emb_dim, text_emb_dim=time_emb_dim,
            cross_attn=True, num_groups=num_groups   # <-- lower encoder cross-attn
        )

        # Context projection for enc3 cross-attn: (B, 2*time_emb_dim) -> (B, C)
        self.enc3_ctx_proj = nn.Linear(2 * time_emb_dim, base_ch * 4)

        # -------------------------
        # Bottleneck (+ optional cross-attn)
        self.base = ConvBlock(
            in_channels=base_ch * 4, out_channels=base_ch * 8,
            time_emb_dim=time_emb_dim, text_emb_dim=time_emb_dim,
            num_groups=num_groups
        )
        self.base_att = (
            CrossAttentionBlock(base_ch * 8, cross_attention=True)
            if use_attn_bottleneck else nn.Identity()
        )
        self.base_ctx_proj = nn.Linear(2 * time_emb_dim, base_ch * 8)

        # -------------------------
        # Decoder (cross-attn at lower decoder: dec3)
        self.dec3 = UpBlock(
            in_channels=base_ch * 8, skip_channels=base_ch * 4, out_channels=base_ch * 4,
            time_emb_dim=time_emb_dim, text_emb_dim=time_emb_dim,
            cross_attn=True, num_groups=num_groups   # <-- lower decoder cross-attn
        )
        self.dec2 = UpBlock(
            in_channels=base_ch * 4, skip_channels=base_ch * 2, out_channels=base_ch * 2,
            time_emb_dim=time_emb_dim, text_emb_dim=time_emb_dim,
            cross_attn=False, num_groups=num_groups
        )
        self.dec1 = UpBlock(
            in_channels=base_ch * 2, skip_channels=base_ch, out_channels=base_ch,
            time_emb_dim=time_emb_dim, text_emb_dim=time_emb_dim,
            cross_attn=False, num_groups=num_groups
        )

        # Context projection for dec3 cross-attn
        self.dec3_ctx_proj = nn.Linear(2 * time_emb_dim, base_ch * 4)

        # -------------------------
        # Output head
        self.out = nn.Conv2d(base_ch, output_ch, kernel_size=1)

    def forward(self, x, t, text_emb=None):
        # Sinusoidal time embedding -> MLP
        t_emb = get_sinusoidal_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)  # (B, time_emb_dim)

        # Text projection to time_emb_dim
        if text_emb is not None:
            text_emb_proj = self.text_proj(text_emb)  # (B, time_emb_dim) or (B, L, time_emb_dim) if you pass seq
            if text_emb_proj.dim() == 3:
                # Pool sequence to a single vector (CLS-like); mean-pool by default
                text_emb_proj = text_emb_proj.mean(dim=1)
        else:
            batch = t_emb.shape[0]
            text_emb_proj = torch.zeros(batch, self.time_emb_dim, device=device, dtype=t_emb.dtype)

        # Combined embedding token (used for context projections)
        comb = torch.cat([t_emb, text_emb_proj], dim=-1)  # (B, 2*time_emb_dim)

        # -------------------------
        # Encoder
        e1, e1p = self.enc1(x, t_emb=t_emb, text_emb=text_emb_proj, context=None)
        e2, e2p = self.enc2(e1p, t_emb=t_emb, text_emb=text_emb_proj, context=None)

        # Prepare context for enc3 cross-attn: (B, 1, C)
        enc3_ctx = self.enc3_ctx_proj(comb).unsqueeze(1)
        e3, e3p = self.enc3(e2p, t_emb=t_emb, text_emb=text_emb_proj, context=enc3_ctx)

        # -------------------------
        # Bottleneck
        b = self.base(e3p, t_emb=t_emb, text_emb=text_emb_proj)

        if isinstance(self.base_att, CrossAttentionBlock):
            base_ctx = self.base_ctx_proj(comb).unsqueeze(1)  # (B, 1, C_bottleneck)
            b = self.base_att(b, context=base_ctx)

        # -------------------------
        # Decoder
        dec3_ctx = self.dec3_ctx_proj(comb).unsqueeze(1)  # (B, 1, C_dec3)
        d3 = self.dec3(b, e3, t_emb=t_emb, text_emb=text_emb_proj, context=dec3_ctx)
        d2 = self.dec2(d3, e2, t_emb=t_emb, text_emb=text_emb_proj, context=None)
        d1 = self.dec1(d2, e1, t_emb=t_emb, text_emb=text_emb_proj, context=None)

        return self.out(d1)
