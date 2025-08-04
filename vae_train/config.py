import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

image_size = 512
num_epochs = 500
learning_rate = 1e-4
batch_size = 8