import torch
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from torchvision.utils import make_grid
import glob
import mlflow
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt


# ...existing code...
from config.config import *
from model.unet import Unet
from dataset.load_dataset import dataloader
from noise.noise_generation import linear_beta_schedule, prepare_alphas, forward_diffusion_sample
from sampling.sample_ddpm import sample_ddpm
from utils.gradient_descent import plot_and_log_grad_norms
from main import old_run
from main.old_run import old_run_data
from dataset.clip import clip_tokenizer, clip_text_model

# from pre_vae.pre_vae import *
from utils.debug import *
# ...existing code...

def get_clip_text_embedding_batch(prompts, tokenizer, text_model, device):
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = text_model(**inputs).last_hidden_state.mean(dim=1)
    return text_features  # shape: (batch_size, embedding_dim)

# Training the model - train loop
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


    alphas, alphas_cumprod, sqrt_ac, sqrt_1mac = prepare_alphas(betas)
    best_loss = float('inf')  # to store the lowest loss seen so far

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0
        grad_sums = {}
        grad_counts = {}

        for i, (imgs, prompts) in enumerate(dataloader):

            optimizer.zero_grad(set_to_none=True)

            imgs = imgs.to(device)

            text_embedding = []
            for prompt in prompts:
                text_embedding = get_clip_text_embedding_batch(prompts, clip_tokenizer, clip_text_model, device)
                text_embedding.append(text_embedding)
            text_embedding = torch.stack(text_embedding, dim=0).to(device)

            b_size = imgs.size(0)

            t = torch.randint(0, timesteps, (b_size, ), device=device)
            print_gpu_memory("after calc t")


            z_t, noise = forward_diffusion_sample(imgs, t, betas=betas, sqrt_ac=sqrt_ac, sqrt_1mac=sqrt_1mac)
            z_t = z_t.to(device)
            noise = noise.to(device)
            print_gpu_memory("before training starts.")
            pred_noise = model(z_t, t)

            print_gpu_memory("After getback from trinaing loop")


            loss = F.l1_loss(pred_noise, noise)

            print_gpu_memory("After calculating loss")

            global_step = epoch * len(dataloader) + i

            print_gpu_memory("After calculating global_step")
            if is_ml_flow_off == False:
                mlflow.log_metric("train_loss", loss.item(), step=global_step)

            loss.backward()

            print_gpu_memory("After loss backward")


            if is_ml_flow_off == False: 
                for name, param in model.named_parameters():
                    if param.grad is not None and "weight" and "conv1" in name:
                        first_layers = [
                            "enc1", "enc2", "enc3",
                            "base",
                            "dec1", "dec2", "dec3"
                        ]
                        # if any(layer in name for layer in first_layers):
                        #     layer_name = name.split(".")[0]  # 'enc_conv1_1'
                        #     param_type = "weight" if "weight" in name else "bias"
                        #     key = f"{param_type}_{layer_name}"

                        grad_norm = param.grad.data.norm(2).item()

                        grad_sums[name] = grad_sums.get(name, 0.0) + grad_norm
                        grad_counts[name] = grad_counts.get(name, 0) + 1

            total_loss += loss.item()
            optimizer.step()
            
        scheduler.step()

        print_gpu_memory("After scheduler step")

        if is_ml_flow_off == True:
            continue
        for name in grad_sums:
            avg_norm = grad_sums[name] / grad_counts[name]
            mlflow.log_metric(f"grad_norm_{name}", avg_norm, step=epoch)

        def log_grad_plots():
            client = MlflowClient()
            run_id =mlflow.active_run().info.run_id
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
            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss
            }, checkpoint_path)

            mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
            print(f"✅ Logged checkpoint for epoch {epoch+1} to MLflow.")


            # 💡 Always define before using
            all_checkpoints = sorted(
                glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth")),
                key=os.path.getmtime
            )

            # Keep only last 10
            if len(all_checkpoints) > 5:
                for ckpt_to_delete in all_checkpoints[:-5]:
                    os.remove(ckpt_to_delete)
                    print(f"🗑️ Deleted old checkpoint: {ckpt_to_delete}")


        if epoch <= 200 or (epoch+1) % save_interval == 0:
            with torch.no_grad():
                model.eval()
                num_samples = 2
                sample_shape = (num_samples, 4, image_size//8, image_size//8)
                sampled = sample_ddpm(model, betas, shape = sample_shape, device=device, vae=vae, timesteps=timesteps)

                sampled = (sampled + 1) / 2.0
                sampled = torch.clamp(sampled, 0.0, 1.0)

                image_tensor = make_grid(sampled, nrow=2)
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
    print("Diffusion training complete")

if __name__ == '__main__':
    model = Unet(
        input_ch=4,
        output_ch=4,
        base_ch=64,
        time_emb_dim=128,
        time_steps=timesteps,
        num_groups=8
    ).to(device)

    print("Num params: ", sum(p.numel() for p in model.parameters()))

    betas = linear_beta_schedule(timesteps=timesteps, start=1e-4, end=0.02).to(device)

    train_model(dataloader, model, betas, epochs=num_epochs, lr=learning_rate, save_interval=save_image_every)

    # model_save_path = os.path.join(output_dir, 'diffsion_deeper_unet_model.pth')
    # torch.save(model.state_dict(), model_save_path)
    # print("Saved diffusion model to:", model_save_path)

    num_samples = 4
    sample_shape = (num_samples, 4, image_size//8, image_size//8)
    final_samples = sample_ddpm(model, betas, shape=sample_shape, device=device, timesteps=timesteps)
    final_samples = (final_samples+1)/2.0
    final_samples = torch.clamp(final_samples, 0.0, 1.0)

    grid = make_grid(final_samples, nrow=2)
    np_grid = grid.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(np_grid, cmap='gray')
    plt.title(f"final samples (Epoch {num_epochs})")
    plt.axis('off')
    plt.show()

    final_save_path = os.path.join(output_dir, "generated_output", "final_ddpm_samples.png")
    os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
    mlflow.log_image(image=np_grid, artifact_file=f"final_ddpm_samples.png")

    plt.imsave(final_save_path, np_grid, cmap='gray')
    print('saved final sample grid to: ', final_save_path)
