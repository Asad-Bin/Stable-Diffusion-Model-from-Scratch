import torch
from diffusers import AutoencoderTiny

from config.config import *

vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16)
vae = vae.to(device)
vae = vae.eval()

for params in vae.parameters():
    params.required_grad = False
