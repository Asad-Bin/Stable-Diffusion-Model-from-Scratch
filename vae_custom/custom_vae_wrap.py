# vae_custom_wrapper.py
import torch
from vae_custom.encoder import Encoder
from vae_custom.decoder import Decoder

class CustomVAEWrapper(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder.eval()
        self.decoder = decoder.eval()

        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

    def encode(self, x):
        z, mu, logvar = self.encoder(x)
        return z  # assuming encoder returns sampled z

    def decode(self, z):
        return self.decoder(z)

def load_custom_vae(encoder_ckpt_path, decoder_ckpt_path, device):
    encoder = Encoder()
    decoder = Decoder()
    encoder.load_state_dict(torch.load(encoder_ckpt_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_ckpt_path, map_location=device))

    vae = CustomVAEWrapper(encoder, decoder).to(device)
    return vae
