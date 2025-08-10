import mlflow
import torch
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import os


from model.unet import Unet
from sampling.sample_ddpm import sample_ddpm
from noise.noise_generation import linear_beta_schedule, prepare_alphas
from config.config import vae, latent_shift, latent_magnitude
# from pre_vae.pre_vae import vae
# from config.config import device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# === Data parser ===
import argparse

parser = argparse.ArgumentParser(description="Generate images using a trained diffusion model checkpoint.")
# parser.add_argument('--model_id', type=int, required=True, help='Model ID to identify which model to use')
parser.add_argument('--checkpoint', type=int, required=True, help='Checkpoint number to load from MLflow')
parser.add_argument('--run_id', type=str, required=True, help='MLflow run ID to download checkpoint from')
parser.add_argument('--source', type=str, choices=["mlflow", "local"], default="mlflow", help="Source to load checkpoint from: 'mlflow' or 'local'")


args = parser.parse_args()
# model_id = args.model_id
checkpoint_no = args.checkpoint
RUN_ID = args.run_id

client = mlflow.tracking.MlflowClient(tracking_uri="http://127.0.1:5000")
run = client.get_run(RUN_ID)

output_dir_from_run = run.data.params.get("output_dir")
if output_dir_from_run is None:
    raise ValueError("Output directory not found in MLflow run parameters.")


# === CONFIG ===
# mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Replace with your actual MLflow URI                   # <-- Replace with actual run ID
CHECKPOINT_PATH_IN_MLFLOW = f"checkpoints/checkpoint_epoch_{checkpoint_no}.pth"  # Path inside MLflow artifacts

# model_result_path = "pre_vae_final6/"
DEVICE = device
# DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = f"{output_dir_from_run}/generated_output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GRID_IMAGE_PATH = os.path.join(OUTPUT_DIR, f"generated_imgs_for_checkpoint_{checkpoint_no}.png")

import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
GRID_IMAGE_PATH = os.path.join(
    OUTPUT_DIR,
    f"generated_imgs_checkpoint_{checkpoint_no}_{timestamp}.png"
)


# === DOWNLOAD CHECKPOINT FROM MLFLOW ===
if args.source == "mlflow":
    print("Downloading checkpoint from MLflow...")
    local_checkpoint_path = mlflow.artifacts.download_artifacts(
        run_id=RUN_ID,
        artifact_path=CHECKPOINT_PATH_IN_MLFLOW,
        dst_path="/tmp"
    )
    print(f"Downloaded checkpoint to: {local_checkpoint_path}")
else:
    local_checkpoint_path = f"{output_dir_from_run}checkpoints/checkpoint_epoch_{checkpoint_no}.pth"
    if not os.path.exists(local_checkpoint_path):
        raise FileNotFoundError(f"Local checkpoint not found at {local_checkpoint_path}")
    print(f"Loaded local checkpoint from: {local_checkpoint_path}")


# === LOAD MODEL ===
print("Initializing model...")
model = Unet().to(DEVICE)
    
checkpoint = torch.load(local_checkpoint_path, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.eval()
print("Model loaded successfully.")

# === GENERATE IMAGES ===
print("Generating 4 sample images...")
with torch.no_grad():
    # Define parameters
    timesteps = 1000
    betas = linear_beta_schedule(timesteps=timesteps)
    betas = betas.to(DEVICE)
    # Generate samples
    sampled_imgs = sample_ddpm(model, betas=betas, shape=(4, 4, 64, 64), device=DEVICE, vae=vae, timesteps=timesteps)


# === SAVE GRID ===
print(f"Saving image grid to {GRID_IMAGE_PATH}")
grid = make_grid(sampled_imgs, nrow=2, normalize=True, value_range=(-1, 1))
save_image(grid, GRID_IMAGE_PATH)

# Optional: Show image
plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
plt.axis("off")
plt.title("Generated Images (4x4 Grid)")
plt.show()
