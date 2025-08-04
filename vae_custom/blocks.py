import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
  def __init__(self, n_heads, embd_dim, in_proj_bias=True, out_proj_bias=True):
    super().__init__()
    self.n_heads = n_heads
    self.in_proj = nn.Linear(embd_dim, 3 * embd_dim, bias=in_proj_bias)
    self.out_proj = nn.Linear(embd_dim, embd_dim, bias=out_proj_bias)

    self.d_heads = embd_dim // n_heads

  def forward(self, x, casual_mask=False):
    # x: (batch_size, seq_len, dim)

    batch_size, seq_len, d_emed = x.shape

    interim_shape = (batch_size, seq_len, self.n_heads, self.d_heads)

    # (batch_size, seq_len, dim) -> 3 * (batch_size, seq_len, d_embed)
    q, k, v = self.in_proj(x).chunk(3, dim=-1)

    # change the shape of q, k and v to match the interim shape
    q = q.view(interim_shape)
    k = k.view(interim_shape)
    v = v.view(interim_shape)

    # swap the elements within matrix using transpose
    # take n_heads before seq_len, like that: (batch_size, n_heads, seq_len, d_embed)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # calculate the attention
    weight = q @ k.transpose(-1, -2)

    if casual_mask:
        # mask where the upper traingle (above the prinicpal dagonal) is 1
        mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
        # fill the upper traingle with -inf
        weight.masked_fill_(mask, -torch.inf)

    weight /= math.sqrt(self.d_heads)

    weight = F.softmax(weight, dim=-1)

    # (batch_size, h_heads, seq_len, dim / h)
    output = weight @ v

    # (batch_size, h_heads, seq_len, dim / h) -> (batch_size, seq_len, n_heads, dim / h)
    output = output.transpose(1, 2)

    # change the shape to the shape of out_proj
    output = output.reshape((batch_size, seq_len, d_emed))

    output = self.out_proj(output)

    return output


class AttentionBlock(nn.Module):
  def __init__(self, channels):
      super().__init__()
      self.groupnorm = nn.GroupNorm(32, channels)
      self.attention = SelfAttention(1, channels)

  def forward(self, x):
      # x: (batch_size, channels, h, w)
      residual = x.clone()

      # (batch_size, channels, h, w) -> (batch_size, channels, h, w)
      x = self.groupnorm(x)

      n, c, h, w = x.shape

      # (batch_size, channels, h, w) -> (batch_size, channels, h * w)
      x = x.view((n, c, h * w))

      # (batch_size, channels, h * w) -> (batch_size, h * w, channels)
      x = x.transpose(-1, -2)

      # perform self-attention without mask
      # (batch_size, h * w, channels) -> (batch_size, h * w, channels)
      x = self.attention(x)

      # (batch_size, h * w, channels) -> (batch_size, channels, h * w)
      x = x.transpose(-1, -2)

      # (batch_size, channels, h * w) -> (batch_size, channels, h, w)
      x = x.view((n, c, h, w))

      # (batch_size, channels, h, w) -> (batch_size, channels, h, w)
      x += residual

      return x

class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.groupnorm1 = nn.GroupNorm(8, in_channels)
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    self.groupnorm2 = nn.GroupNorm(8, out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    if in_channels == out_channels:
      self.residual_layer = nn.Identity()
    else:
      self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

  def forward(self, x):
    # x: (batch_size, in_channels, h, w)
    residue = x.clone()

    x = self.groupnorm1(x)
    x = F.silu(x)
    x = self.conv1(x)
    x = self.groupnorm2(x)
    x = self.conv2(x)

    return x + self.residual_layer(residue)

class ResAttnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attn=True):
        super().__init__()
        self.resblock = ResidualBlock(in_channels, out_channels)
        self.attn = AttentionBlock(out_channels) if use_attn else nn.Identity()

    def forward(self, x):
        x = self.resblock(x)
        x = self.attn(x)
        return x
