import torch
import mlflow
import os
import json
import datetime


def get_unique_output_dir(base_dir):
    base_dir = os.path.abspath(base_dir)
    os.makedirs(base_dir, exist_ok=True)  # Ensure parent exists
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = os.environ.get("MY_TIMESTAMP") or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_dir = os.path.join(base_dir, f"output_{timestamp}")
    os.makedirs(unique_dir, exist_ok=True)
    return unique_dir

output_dir = get_unique_output_dir("./output/")
os.makedirs(output_dir, exist_ok=True)

image_size = 512
batch_size = 32
num_epochs = 3000
timesteps = 1000
learning_rate = 0.0001
save_image_every = 5
checkpoint_interval = 50

checkpoint_dir = os.path.join(output_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

dataset_dir = "./dataset/huggingface_butterflies"
os.makedirs(dataset_dir, exist_ok=True)

note = (
    "       Used 3 blocks of encoder decoder\n"
    "       Used 3 attention at enc3, bottleneck & dec3,\n"
    "       last conv layer of each block has 'Silu',\n"
    "       no scaling factor, but norm\n"
    "       vae trained at 1000 epoch.\n"
)
input_channels = 4
output_channels = 4
base_channels = 64
time_embedding_dim = 128

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "image_size": image_size,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "timesteps": timesteps,
    "learning_rate": learning_rate,
    "save_image_every": save_image_every,
    "checkpoint_interval": checkpoint_interval,
    "output_dir": output_dir,
    "dataset_dir": dataset_dir,
    "note": note,
    "input_channels": input_channels,
    "output_channels": output_channels,
    "base_channels": base_channels,
    "time_embedding_dim": time_embedding_dim,
    "device": str(device),
}



# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
CONFIG["device"] = str(device)

# ------------------- PRINT CONFIG ------------------- #
def print_config(cfg):
    print("=" * 40)
    print("Training Configuration:")
    for k, v in cfg.items():
        print(f"{k:18}: {v}")
    print("=" * 40)

print_config(CONFIG)

# ------------------- LOGGING ------------------- #
def log_config_mlflow(cfg, is_ml_flow_off=False, old_run=False):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("asad - text_emb + cus_vae + unet")
    if old_run:
        mlflow.start_run(run_id='3bb016f503a94971b24e00910655db52')
    elif not is_ml_flow_off:
        for k, v in cfg.items():
            mlflow.log_param(k, v)

def log_config_local(cfg, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=4)

def write_config_readme(cfg, output_dir):
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write("# Training Configuration\n\n")
        for k, v in cfg.items():
            f.write(f"- **{k}**: `{v}`\n")


# ------------------- EXECUTE LOGGING ------------------- #
is_ml_flow_off = False
old_run = False
old_run_checkpoint_no = 3000
old_checkpoint_dir = "/home/asad/task1/output/output_20250807_185921/checkpoints"

if __name__ == "__main__":
    log_config_mlflow(CONFIG, is_ml_flow_off, old_run)
    log_config_local(CONFIG, CONFIG["output_dir"])
    write_config_readme(CONFIG, CONFIG["output_dir"])


# ------------------- VAE SETTINGS  ------------------- #
# from pre_vae.pre_vae import vae
from vae_custom.custom_vae_wrap import load_custom_vae

vae = load_custom_vae(
    encoder_ckpt_path="./vae_custom/checkpoints/encoder.pt",
    decoder_ckpt_path="./vae_custom/checkpoints/decoder.pt",
    device=device
)

# latent_shift = 0.01165771484375
# latent_magnitude = 1.0478515625
latent_shift = -0.06365966796875
latent_magnitude = 1.0634765625
# scaling_factor = 3.0/8.5
