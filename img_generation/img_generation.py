import mlflow
import torch
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import os
import argparse
import datetime
import random
import re

from model.unet import Unet
from sampling.sample_ddpm import sample_ddpm
from noise.noise_generation import linear_beta_schedule
from config.config import vae
from dataset.clip import clip_tokenizer, clip_text_model
from main.main import get_clip_text_embedding_batch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# === ARGUMENT PARSER ===
parser = argparse.ArgumentParser(description="Generate images from the latest diffusion model checkpoint in MLflow run output_dir.")
parser.add_argument('--run_id', type=str, required=True, help="MLflow run ID")
parser.add_argument('--output_dir', type=str, default=None, help="Custom output_dir if not found in MLflow run")
args = parser.parse_args()

# === CONNECT TO MLFLOW & GET OUTPUT DIR ===
client = mlflow.tracking.MlflowClient(tracking_uri="http://127.0.0.1:5000")
run = client.get_run(args.run_id)

output_dir_from_run = run.data.params.get("output_dir")
if output_dir_from_run is not None:
    output_dir = output_dir_from_run
    print(f"[INFO] Using output_dir from MLflow run: {output_dir}")
else:
    if args.output_dir is None:
        raise ValueError("MLflow run has no output_dir, and no --output_dir was provided.")
    output_dir = args.output_dir
    print(f"[INFO] Using custom output_dir: {output_dir}")

print("output_dir:", output_dir)

# === FIND LATEST CHECKPOINT ===
checkpoint_dir = os.path.join(output_dir, "checkpoints")
if not os.path.exists(checkpoint_dir):
    raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

checkpoint_files = [
    f for f in os.listdir(checkpoint_dir)
    if re.match(r"checkpoint_epoch_(\d+)\.pth", f)
]

if not checkpoint_files:
    raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

# Get the one with highest epoch number
latest_checkpoint_file = max(
    checkpoint_files,
    key=lambda f: int(re.search(r"checkpoint_epoch_(\d+)\.pth", f).group(1))
)
latest_epoch = int(re.search(r"checkpoint_epoch_(\d+)\.pth", latest_checkpoint_file).group(1))

local_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint_file)
print(f"[INFO] Latest checkpoint found: {latest_checkpoint_file} (epoch {latest_epoch})")

# === OUTPUT FOLDER FOR GENERATED IMAGES ===
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
generated_dir = os.path.join(output_dir, "generated_output")
os.makedirs(generated_dir, exist_ok=True)

GRID_IMAGE_PATH = os.path.join(
    generated_dir,
    f"generated_imgs_checkpoint_{latest_epoch}_{timestamp}.png"
)

# === LOAD MODEL ===
print("[INFO] Initializing model...")
model = Unet().to(device)
checkpoint = torch.load(local_checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.eval()
print("[INFO] Model loaded successfully.")

# === RANDOM PROMPT ===
butterfly_prompts = [
    "a beautiful butterfly with colorful wings",
    "a blue morpho butterfly on a leaf",
    "a monarch butterfly in flight",
    "a realistic butterfly with detailed patterns",
    "a yellow swallowtail butterfly on a flower",
    "a red butterfly with black spots",
    "a red butterfly with green wings",
    "a white butterfly with blue patterns",
]
prompt = random.choice(butterfly_prompts)
print(f"[INFO] Using text prompt: {prompt}")

batch_size = 16  # same as in shape=(16, 4, 64, 64)
text_embedding = get_clip_text_embedding_batch(
    [prompt], clip_tokenizer, clip_text_model, device
)
text_embedding = text_embedding.repeat(batch_size, 1)  # match batch size


# === GENERATE IMAGES ===
with torch.no_grad():
    timesteps = 1000
    betas = linear_beta_schedule(timesteps=timesteps).to(device)
    sampled_imgs = sample_ddpm(
        model,
        betas=betas,
        shape=(16, 4, 64, 64),
        device=device,
        vae=vae,
        timesteps=timesteps,
        text_embeddings=text_embedding
    )

# === SAVE IMAGE GRID ===
print(f"[INFO] Saving image grid to {GRID_IMAGE_PATH}")
grid = make_grid(sampled_imgs, nrow=4, normalize=True, value_range=(-1, 1))
# save_image(grid, GRID_IMAGE_PATH)

# plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
# plt.axis("off")
# plt.title(f"Generated Images (4x4) - Latest Checkpoint (Epoch {latest_epoch})")
# plt.show()

# Plot with prompt as title
# plt.figure(figsize=(12, 12))
# plt.imshow(grid)
# plt.axis("off")
# plt.title(f"Prompt: {prompt}", fontsize=14)  # display the first prompt
# plt.tight_layout()

# # Save the figure
# save_image(grid, GRID_IMAGE_PATH)
# plt.savefig(GRID_IMAGE_PATH, bbox_inches='tight')
# plt.show()

from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T

# Convert PyTorch grid to PIL image
grid_pil = T.ToPILImage()(grid)

# Create a prompt image
font_size = 200
padding = 10
font = ImageFont.load_default()  # Or ImageFont.truetype("arial.ttf", font_size)
dummy_img = Image.new("RGB", (10, 10))
draw = ImageDraw.Draw(dummy_img)

# Get text size using textbbox
bbox = draw.textbbox((0, 0), prompt, font=font)
text_width = bbox[2] - bbox[0]
text_height = bbox[3] - bbox[1]

prompt_img = Image.new("RGB", (grid_pil.width, text_height + 2 * padding), color=(255, 255, 255))
draw = ImageDraw.Draw(prompt_img)
draw.text((padding, padding), prompt, fill=(0, 0, 0), font=font)

# Concatenate prompt image on top of grid
final_img = Image.new("RGB", (grid_pil.width, grid_pil.height + prompt_img.height))
final_img.paste(prompt_img, (0, 0))
final_img.paste(grid_pil, (0, prompt_img.height))

# Save final image
final_img.save(GRID_IMAGE_PATH)
print(f"[INFO] Saved image with prompt: {GRID_IMAGE_PATH}")
