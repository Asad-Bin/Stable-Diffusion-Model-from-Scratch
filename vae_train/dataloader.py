from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


from .config import image_size, batch_size, device

class HuggingFaceImageDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image'].convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0
    
def LoadData():
    hf_data = load_dataset("huggan/smithsonian_butterflies_subset", split="train")
    print("Loaded Hugging Face dataset with length:", len(hf_data))

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = HuggingFaceImageDataset(hf_data, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    return dataloader

dataloader = LoadData()
# for idx, (images, _) in enumerate(load_dataset):
#     print(images.shape, images.min(), images.max())
    # break  # Just to test the first batch   

import glob
from PIL import Image
import os
import torchvision.transforms as transforms
import torch
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),  # [-1,1] range
])
base_dir = os.path.dirname(os.path.abspath(__file__))
pattern = os.path.join(base_dir, "test_images", "*.png")
image_paths = glob.glob(pattern) 
def load_test_images(path_pattern="/homevae_train/test_images/images*.png"):
    # image_paths = glob.glob(path_pattern)
    if not image_paths:
        raise FileNotFoundError(f"No images found matching: {path_pattern}")
    
    images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        images.append(transform(img))
    
    return torch.stack(images)
