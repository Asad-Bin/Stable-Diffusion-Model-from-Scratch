# Updated training loop that (1) ensures CLIP embeddings are on-device,
# (2) auto-creates a small projector if CLIP embedding dim != model expected text_emb_dim,
# (3) passes projected text embeddings into the UNet (which handles cross-attention),
# (4) keeps your existing logging/saving behavior.
#
# Drop this into the same file in place of your old train_model / __main__ sections.

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from torchvision.utils import make_grid
import glob
import mlflow
from mlflow.tracking import MlflowClient

# keep your existing imports (config, dataloader, Unet, sample_ddpm, etc.)
from config.config import *
from model.unet import Unet
from dataset.load_dataset import LoadData
from noise.noise_generation import linear_beta_schedule, prepare_alphas, forward_diffusion_sample
from sampling.sample_ddpm import sample_ddpm
from utils.gradient_descent import plot_and_log_grad_norms
from main import old_run
from main.old_run import old_run_data
from dataset.clip import clip_tokenizer, clip_text_model
from utils.debug import *

dataloader = LoadData(prompt_style="queenvl", regenerate_prompts=False)

# -------------------------
# helper: get CLIP text embeddings (already returning on device)
def get_clip_text_embedding_batch(prompts, tokenizer, text_model, device):
    # Set max_length to 77 (CLIP's maximum) and truncate longer sequences
    inputs = tokenizer(prompts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
    with torch.no_grad():
        out = text_model(**inputs).last_hidden_state  # (B, L, D_clip)
        text_features = out.mean(dim=1)               # (B, D_clip)
    return text_features  # shape: (batch_size, embedding_dim)

def print_image(epoch):
    with torch.no_grad():
        model.eval()
        num_samples = 3
        sample_shape = (num_samples, 4, image_size//8, image_size//8)
        
        #prompt 
        random_prompts = ["a black and yellow butterfly with blue and white markings on its wings , featuring prominent black spots and a yellow border on its wings.", "a beautiful butterfly with green wings", "a butterfly with blue and black wings"]
        text_embedding = get_clip_text_embedding_batch(random_prompts, clip_tokenizer, clip_text_model, device)

        sampled = sample_ddpm(
            model, 
            betas, 
            shape=sample_shape, 
            device=device, 
            timesteps=timesteps, 
            vae=vae,
            text_embeddings=text_embedding
        )

        sampled = (sampled + 1) / 2.0
        sampled = torch.clamp(sampled, 0.0, 1.0)

        image_tensor = make_grid(sampled, nrow=3)
        np_image = image_tensor.permute(1, 2, 0).cpu().numpy()

        mlflow.log_image(image=np_image, artifact_file=f"generated_image_epoch_{epoch+1:04d}.png")

        plt.figure(figsize=(12, 6))
        plt.imshow(np_image)
        plt.title(f"Epoch {epoch+1} Sample")
        plt.axis('off')
        plt.show()


        save_img_path = os.path.join(output_dir, "training_outputs", f"epoch_{epoch+1:04d}_samples.png")
        os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
        plt.figure(figsize=(12, 6))
        plt.imsave(save_img_path, np_image)
        plt.close()
        print(f"Saved sample to: {save_img_path}")
    model.train()


# -------------------------
# Updated train loop (keeps almost all of your original logic)
def train_model(dataloader, model, betas, epochs=num_epochs, lr=learning_rate, save_interval=save_image_every):
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    start_epoch = 0
    if old_run == True:
        checkpoint, start_epoch = old_run_data()
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Prepare alphas/schedule helpers (you already had this)
    alphas, alphas_cumprod, sqrt_ac, sqrt_1mac = prepare_alphas(betas)
    best_loss = float('inf')

    # --- Prepare a projector if CLIP dim != model expected text_emb_dim ---
    # We'll lazily create projector after the first batch (so we know CLIP dim).
    projector = None
    projector_created = False

    for epoch in range(start_epoch, epochs):
        total_loss = 0.0
        grad_sums = {}
        grad_counts = {}

        for i, value in enumerate(dataloader):
            imgs = value[0]
            prompts = value[2]
            # print(prompts)

            optimizer.zero_grad(set_to_none=True)
            imgs = imgs.to(device)

            # Get CLIP embeddings (on device)
            text_embedding = get_clip_text_embedding_batch(prompts, clip_tokenizer, clip_text_model, device)  # (B, D_clip)

            # On first iteration create projector if needed
            if not projector_created:
                clip_dim = text_embedding.shape[-1]
                # attempt to inspect model.text_proj if it exists to see expected dim
                model_text_proj_in = None
                if hasattr(model, "text_proj") and isinstance(model.text_proj, torch.nn.Linear):
                    model_text_proj_in = model.text_proj.in_features

                if model_text_proj_in is None:
                    # fallback: if UNet expects text_emb_dim attr (as you set in constructor)
                    if hasattr(model, "text_emb_dim"):
                        model_text_proj_in = model.text_emb_dim
                # If still None, assume no projection needed
                if model_text_proj_in is None or model_text_proj_in == clip_dim:
                    projector = None
                else:
                    # create a lightweight linear projector to map CLIP -> model expected text dim
                    projector = torch.nn.Linear(clip_dim, model_text_proj_in).to(device)
                    # initialize projector (xavier)
                    torch.nn.init.xavier_uniform_(projector.weight)
                    if projector.bias is not None:
                        torch.nn.init.zeros_(projector.bias)
                    print(f"[INFO] Created CLIP->model text projector: {clip_dim} -> {model_text_proj_in}")

                projector_created = True

            # apply projector if present
            if projector is not None:
                text_embedding = projector(text_embedding)

            b_size = imgs.size(0)
            t = torch.randint(0, timesteps, (b_size,), device=device).long()

            print_gpu_memory("after calc t")

            z_t, noise = forward_diffusion_sample(imgs, t, betas=betas, sqrt_ac=sqrt_ac, sqrt_1mac=sqrt_1mac)
            z_t = z_t.to(device)
            noise = noise.to(device)

            print_gpu_memory("before training starts.")
            # Pass text embedding into model. Model is expected to handle text_emb shape.
            pred_noise = model(z_t, t, text_emb=text_embedding)

            print_gpu_memory("After getback from trinaing loop")

            # You used L1 previously; keep consistent
            loss = F.l1_loss(pred_noise, noise)

            print_gpu_memory("After calculating loss")
            global_step = epoch * len(dataloader) + i
            print_gpu_memory("After calculating global_step")

            if not is_ml_flow_off:
                mlflow.log_metric("train_loss", loss.item(), step=global_step)

            loss.backward()
            print_gpu_memory("After loss backward")

            # Grad norms logging (keeps your logic)
            if not is_ml_flow_off:
                for name, param in model.named_parameters():
                    if param.grad is not None and ("weight" in name) and ("conv1" in name):
                        grad_norm = param.grad.data.norm(2).item()
                        grad_sums[name] = grad_sums.get(name, 0.0) + grad_norm
                        grad_counts[name] = grad_counts.get(name, 0) + 1

            total_loss += loss.item()
            optimizer.step()

        scheduler.step()
        print_gpu_memory("After scheduler step")

        if is_ml_flow_off:
            continue

        for name in grad_sums:
            avg_norm = grad_sums[name] / grad_counts[name]
            mlflow.log_metric(f"grad_norm_{name}", avg_norm, step=epoch)

        def log_grad_plots():
            client = MlflowClient()
            run_id = mlflow.active_run().info.run_id
            all_metrics = client.get_run(run_id).data.metrics.keys()
            weight_metrics = [m for m in all_metrics if m.startswith("grad_norm_weight_")]
            bias_metrics = [m for m in all_metrics if m.startswith("grad_norm_bias_")]
            plot_and_log_grad_norms(run_id, weight_metrics, bias_metrics, client)

        log_grad_plots()

        avg_loss = total_loss / len(dataloader)
        mlflow.log_metric("avg_loss", avg_loss, step=epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            print("✅ Current best average loss: ", best_loss)

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.4f}")

        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss
            }, checkpoint_path)

            if num_epochs - epoch <= 200:
                mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
                print(f"✅ Logged checkpoint for epoch {epoch+1} to MLflow.")

            # Keep only last 5 checkpoints
            all_checkpoints = sorted(
                glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth")),
                key=os.path.getmtime
            )
            if len(all_checkpoints) > 5:
                for ckpt_to_delete in all_checkpoints[:-5]:
                    os.remove(ckpt_to_delete)
                    print(f"🗑️ Deleted old checkpoint: {ckpt_to_delete}")

        if epoch <= 200 or (epoch + 1) % save_interval == 0:
            # reuse your existing sampling + save function (print_image)
            print_image(epoch)

    print("Diffusion training complete")


# -------------------------
# main / run
if __name__ == '__main__':
    model = Unet(
        input_ch=4,
        output_ch=4,
        base_ch=64,
        time_emb_dim=128,
        time_steps=timesteps,
        num_groups=8,
        text_emb_dim=512   # this should match what you want inside UNet; projection will adapt if CLIP != 128
    ).to(device)

    print("Num params: ", sum(p.numel() for p in model.parameters()))

    betas = linear_beta_schedule(timesteps=timesteps, start=1e-4, end=0.02).to(device)

    # finally run training
    train_model(dataloader, model, betas, epochs=num_epochs, lr=learning_rate, save_interval=save_image_every)
