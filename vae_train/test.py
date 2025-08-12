import torch
from vae_custom.encoder import Encoder
from vae_custom.decoder import Decoder
from vae_train.dataloader import load_test_images
from vae_train.config import device
import torchvision.utils as vutils
import os
import argparse
import torch.nn as nn

# from vae_train.train import kl

parser = argparse.ArgumentParser(description="")
# parser.add_argument('--idx', type=int, required=True, help='image index to test')
# idx = parser.parse_args().idx

def kl_divergence(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
mseloss = nn.MSELoss()
# def criterion(outputs, images, mu, logvar, epoch=None):
#     loss = l1loss(outputs, images)
#     kl = kl_divergence(mu, logvar)
#     print(f"[Epoch {epoch+1}] KL: {kl:.4f}, Recon: {loss:.4f}, Total: {loss.item():.4f}")
#     return loss + beta * kl

def get_beta(epoch, max_beta=1e-4, warmup_epochs=10):
    return max_beta * min(1.0, epoch / warmup_epochs)


chkpnt_epoch = 1000  # Change this to the epoch you want to evaluate
def evaluate_vae(model_ckpt_dir):
    # Load model
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    encoder.load_state_dict(torch.load(os.path.join(model_ckpt_dir, f"encoder_epoch_{chkpnt_epoch}.pt")))
    decoder.load_state_dict(torch.load(os.path.join(model_ckpt_dir, f"decoder_epoch_{chkpnt_epoch}.pt")))

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        test_imgs = load_test_images().to(device)

        for idx in range(0, len(test_imgs), 1):
            img = test_imgs[idx].unsqueeze(0)  # Add batch dimension

            z, a, b = encoder(img)
            recon = decoder(z)

            loss = mseloss(recon, img) + get_beta(epoch=0) * kl_divergence(a, b)

            # Stack original and recon side-by-side
            interleaved = torch.stack([img, recon], dim=1).reshape(-1, 3, 512, 512)
            interleaved = interleaved * 0.5 + 0.5  # de-normalize if needed

            save_path=f"vae_train/testing_outputs/test_eval_{idx}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            if interleaved.size(0) > 0:
                vutils.save_image(interleaved, save_path, nrow=2)
            else:
                print(f"Skipping save_image — tensor is empty at")


            vutils.save_image(interleaved, save_path, nrow=2)
            print(f"test loss = {loss}")
            print(f"Test reconstructions saved to: {save_path}")
        # test_imgs = test_imgs[idx:idx]

        # z, _, _ = encoder(test_imgs)
        # recon = decoder(z)

        # Stack original and recon side-by-side
        # interleaved = torch.stack([test_imgs, recon], dim=1).reshape(-1, 3, 512, 512)
        # interleaved = interleaved * 0.5 + 0.5  # de-normalize if needed

        # vutils.save_image(interleaved, save_path, nrow=2)
        # print(f"Test reconstructions saved to: {save_path}")

evaluate_vae(model_ckpt_dir="vae_train/checkpoints")
