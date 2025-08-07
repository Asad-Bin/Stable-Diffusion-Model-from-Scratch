import torch.nn as nn
import torch
from vae_custom.encoder import Encoder
from vae_custom.decoder import Decoder

from .config import num_epochs, learning_rate
from .dataloader import dataloader, load_test_images
from .config import device
from .plot import local
from utils.epoch_progress import print_epoch_progress

import torchvision.utils as vutils
import os
import mlflow
import mlflow.pytorch

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Custom_VAE_Training")
mlflow.start_run(run_name="vae_run")
# mlflow.start_run(run_id="")
mlflow.log_params({
    "num_epochs": num_epochs,
    "learning_rate": learning_rate,
    "beta_schedule": "linear_warmup",
    "optimizer": "AdamW",
})



checkpoint_dir = "vae_train/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

def save_images(epoch, originals, reconstructions, save_dir_base="vae_train/saved_images"):
    # Folder for this epoch
    # save_dir = os.path.join(save_dir_base, f"epoch_{epoch:02d}")
    save_dir = save_dir_base
    os.makedirs(save_dir, exist_ok=True)

    # Save training images
    originals = originals[:2]
    reconstructions = reconstructions[:2]
    interleaved = torch.stack([originals, reconstructions], dim=1).reshape(-1, *originals.shape[1:])
    interleaved = interleaved * 0.5 + 0.5
    vutils.save_image(interleaved, os.path.join(save_dir, f"train_recon_{epoch}.png"), nrow=2)

    # Save test reconstructions (if given)
    # if test_originals is not None and test_reconstructions is not None:
    #     test_originals = test_originals[1:5]
    #     test_reconstructions = test_reconstructions[1:5]
    #     test_interleaved = torch.stack([test_originals, test_reconstructions], dim=1).reshape(-1, *test_originals.shape[1:])
    #     test_interleaved = test_interleaved * 0.5 + 0.5
    #     vutils.save_image(test_interleaved, os.path.join(save_dir, "test_recon.png"), nrow=2)
    print(f"Saved images for epoch {epoch} to {save_dir}")



# Assuming you instantiate encoder and decoder objects somewhere, e.g.:
encoder = Encoder().to(device)
decoder = Decoder().to(device)

resume_epoch = 500  # set this to the epoch number of the checkpoint you want to resume from
enc_path = f"{checkpoint_dir}/encoder_epoch_{resume_epoch}.pt"
dec_path = f"{checkpoint_dir}/decoder_epoch_{resume_epoch}.pt"

if os.path.exists(enc_path) and os.path.exists(dec_path):
    encoder.load_state_dict(torch.load(enc_path, map_location=device))
    decoder.load_state_dict(torch.load(dec_path, map_location=device))
    print(f"Resumed encoder and decoder from epoch {resume_epoch}")
else:
    print("Checkpoint not found. Starting from scratch.")
    resume_epoch = 0


print(sum(p.numel() for p in encoder.parameters() if p.requires_grad) + sum(p.numel() for p in decoder.parameters() if p.requires_grad))

# Combine parameters
model_params = list(encoder.parameters()) + list(decoder.parameters())

optimizer = torch.optim.AdamW(model_params, lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

# optimizer.load_state_dict(torch.load("optimizer.pt"))
# scheduler.load_state_dict(torch.load("scheduler.pt"))

beta = 0.1
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

for epoch in range(resume_epoch, num_epochs):
    total_kl = 0.0
    total_recon = 0.0
    total_loss = 0.0
    for idx, (images, _) in enumerate(dataloader):
        images = images.to(device)

        optimizer.zero_grad()
        latent, mu, logvar = encoder(images)
        outputs = decoder(latent)

        # print(images.shape)

        # print("latent min/max:", latent.min().item(), latent.max().item())

        # loss = criterion(outputs, images, mu, lagvar)
        # logvar = torch.clamp(logvar, min=-10.0, max=10.0)

        recon = mseloss(outputs, images)
        kl = kl_divergence(mu, logvar)

        # if idx == 0:
        #     print(f"[Epoch {epoch+1}, Mean {mu.mean().item():.4f}, LogVar {logvar.mean().item():.4f}]")

        loss = recon + get_beta(epoch=epoch) * kl

        loss.backward()

        if idx == 0:
            total_norm = 0
            for p in model_params:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            mlflow.log_metric("grad_norm", total_norm, step=epoch)


        optimizer.step()

        total_kl += kl.item()
        total_recon += recon.item() 
        total_loss += loss.item()


        # call your plotting function (if you want here)
        # local(outputs=outputs)
        if idx == 0:
            # with torch.no_grad():
            #     test_imgs = load_test_images().to(device=device)
            #     test_latent, _, _ = encoder(test_imgs)
            #     test_outputs = decoder(test_latent)

            save_images(
                epoch + 1,
                originals=images,
                reconstructions=outputs
            )

    avg_kl = total_kl / len(dataloader)
    avg_recon = total_recon / len(dataloader)
    avg_total = total_loss / len(dataloader)
    current_lr = scheduler.get_last_lr()[0]

    mlflow.log_metrics({
        "recon_loss": avg_recon,
        "kl_loss": avg_kl,
        "total_loss": avg_total,
        "lr": current_lr,
    }, step=epoch)

    image_path = f"vae_train/saved_images/train_recon_{epoch+1}.png"
    mlflow.log_artifact(image_path, artifact_path="images")

    if (epoch+1) % 100 == 0:
        enc_path = f"{checkpoint_dir}/encoder_epoch_{epoch+1}.pt"
        dec_path = f"{checkpoint_dir}/decoder_epoch_{epoch+1}.pt"
        torch.save(encoder.state_dict(), enc_path)
        torch.save(decoder.state_dict(), dec_path)
        print(f"Saved checkpoint at epoch {epoch+1}")

        mlflow.log_artifact(enc_path, artifact_path="checkpoints")
        mlflow.log_artifact(dec_path, artifact_path="checkpoints")
    
    print(f"[Epoch {epoch+1}, Batch {idx+1}] Recon: {total_recon/len(dataloader):.4f}, KL: {total_kl/len(dataloader):.4f}, Total Loss: {total_loss/len(dataloader):.4f}")

    # Step scheduler after each epoch
    scheduler.step()

    print(f"Epoch {epoch+1}/{num_epochs} completed. LR: {scheduler.get_last_lr()[0]:.6f}")
    # print_epoch_progress(epoch + 1, num_epochs)

mlflow.end_run()
