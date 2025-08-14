import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset

from config.config import *
from utils.debug import *
# from pre_vae.pre_vae import vae  # make sure your VAE is available as in your setup

# ==================================================
# CLIP text encoder with multiple fallbacks (unchanged)
# ==================================================
def setup_text_encoder():
    """Setup text encoder with multiple fallback options"""
    global text_encoder, tokenizer, encode_method
    
    # Option 2: Try transformers CLIP
    try:
        from transformers import CLIPTextModel, CLIPTokenizer
        model_name = "openai/clip-vit-base-patch32"
        tokenizer = CLIPTokenizer.from_pretrained(model_name)
        text_encoder = CLIPTextModel.from_pretrained(model_name).to(device)
        text_encoder.eval()
        encode_method = 'transformers'
        print("✅ Using transformers CLIP")
        return
    except Exception as e:
        print(f"Transformers CLIP failed: {e}")
    
    raise ImportError("❌ Could not load any text encoder. Please install one of: clip, transformers, or sentence-transformers")

# Initialize text encoder
setup_text_encoder()

# ==================================================
# Qwen2-VL (QueenVL) setup with robust fallbacks
# ==================================================
qvl_processor = None
qvl_model = None

def setup_queenvl():
    """
    Load Qwen2-VL-2B-Instruct with broad compatibility across transformers versions.
    We try (in order):
      1) Qwen2VLForConditionalGeneration (native class)
      2) AutoModelForVision2Seq with trust_remote_code=True
      3) AutoModelForCausalLM with trust_remote_code=True (older repos sometimes expose this)
    """
    global qvl_processor, qvl_model
    from transformers import AutoProcessor
    qvl_name = "Qwen/Qwen2-VL-2B-Instruct"
    print("🔄 Loading Qwen2-VL processor...")
    qvl_processor = AutoProcessor.from_pretrained(qvl_name, trust_remote_code=True)
    last_err = None

    # Try native class
    try:
        from transformers import Qwen2VLForConditionalGeneration
        print("🔄 Trying Qwen2VLForConditionalGeneration...")
        qvl_model = Qwen2VLForConditionalGeneration.from_pretrained(
            qvl_name,
            torch_dtype=torch.float16,
            device_map=None,
            # trust_remote_code=True,
        ).to(device)
        print("✅ Loaded with Qwen2VLForConditionalGeneration")
        return
    except Exception as e:
        last_err = e
        print(f"Qwen2VLForConditionalGeneration failed: {e}")

def generate_prompt_with_queenvl(image_pil):
    global qvl_processor, qvl_model
    if qvl_processor is None or qvl_model is None:
        setup_queenvl()

    query = (
        "Describe this butterfly briefly but precisely: species traits if evident, "
        "dominant colors, wing pattern (spots/stripes/borders), and notable features."
    )

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": query}
        ]}
    ]

    text_prompt = qvl_processor.apply_chat_template(messages, add_generation_prompt=True)


    # Processor produces either pixel_values or already flattened features
    inputs = qvl_processor(images=image_pil, text=text_prompt, return_tensors="pt")
    for k, v in inputs.items():
        inputs[k] = v.to(device)

    # pixel_values = inputs.pop("pixel_values", None)

    # If pixel_values is missing or flattened, let the model handle it
    # kwargs = {"input_ids": inputs["input_ids"]}
    # if pixel_values is not None:
    #     kwargs["pixel_values"] = pixel_values

        # Attempt to provide image_grid_thw only if pixel_values is 4D
        # if pixel_values.ndim == 4:
        #     _, _, H, W = pixel_values.shape
        #     patch_size = 16
        #     grid_h = H // patch_size
        #     grid_w = W // patch_size
        #     kwargs["image_grid_thw"] = (1, grid_h, grid_w)

    with torch.no_grad():
        output = qvl_model.generate(**inputs, max_new_tokens=80)

    description = qvl_processor.batch_decode(output, skip_special_tokens=True)[0]
    return description.strip()
 


# def generate_prompt_with_queenvl(image_pil):
    # """Generate descriptive prompt from an image using Qwen2-VL."""
    # global qvl_processor, qvl_model
    # if qvl_processor is None or qvl_model is None:
    #     setup_queenvl()

    # # Keep it short, visual-detail-focused for training
    # query = (
    #     "Describe this butterfly briefly but precisely: species traits if evident, "
    #     "dominant colors, wing pattern (spots/stripes/borders), and notable features."
    # )

    # # Most Qwen2-VL versions accept images + text via AutoProcessor
    # inputs = qvl_processor(
    #     images=image_pil,
    #     text=query,
    #     return_tensors="pt"
    # )

    # # inputs = qvl_processor(images=image_pil, text=query, return_tensors="pt")
    # # Move tensors to device individually
    # for k, v in inputs.items():
    #     inputs[k] = v.to(device)

    # with torch.no_grad():
    #     output = qvl_model.generate(**inputs, max_new_tokens=80)

    # # description = qvl_processor.batch_decode(output, skip_special_tokens=True)[0]

    # # Some processors return pairs like "<image>\n..."; decoding strips specials
    # description = qvl_processor.batch_decode(output, skip_special_tokens=True)[0]
    # return description.strip()

# ==================================================
# Dataset that ONLY uses Qwen2-VL to create prompts
# (random prompt generators removed)
# ==================================================
class ButterflyDatasetWithPrompts(Dataset):
    def __init__(self, hf_dataset, transform=None, save_prompts_file=None, regenerate_prompts=True):
        self.dataset = hf_dataset
        self.transform = transform
        self.generated_prompts = []
        self.save_prompts_file = save_prompts_file

        if regenerate_prompts or not (save_prompts_file and os.path.exists(save_prompts_file)):
            print(f"🦋 Generating QueenVL prompts for {len(hf_dataset)} butterfly images...")
            for idx in range(len(hf_dataset)):
                image_pil = hf_dataset[idx]['image'].convert('RGB')
                prompt = generate_prompt_with_queenvl(image_pil)
                self.generated_prompts.append(prompt)

                if((idx+1) % 10 == 0):
                    print(f"✅ Generated {idx + 1}/{len(hf_dataset)} prompts: '{prompt[:50]}...'")

            if save_prompts_file:
                os.makedirs(os.path.dirname(save_prompts_file), exist_ok=True)
                with open(save_prompts_file, 'w', encoding='utf-8') as f:
                    # Maintain your original line-per-prompt format for compatibility
                    for i, p in enumerate(self.generated_prompts):
                        f.write(f"[{i}] {p}\n")
                print(f"✅ Saved prompts to {save_prompts_file}")
        else:
            print(f"📂 Loading cached prompts from {save_prompts_file}")
            with open(save_prompts_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            # Parse the "[i] " prefix to get the prompt text
            self.generated_prompts = [line.split("]", 1)[1].strip() if "]" in line else line.strip()
                                      for line in lines]

        # Show sample prompts
        print("\n📝 Sample generated prompts:")
        for i in range(min(10, len(self.generated_prompts))):
            print(f"  [{i}] {self.generated_prompts[i]}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image'].convert('RGB')
        prompt = self.generated_prompts[idx]
        if self.transform:
            image = self.transform(image)
        return image, prompt

# ==================================================
# Encode text with your CLIP stack (unchanged)
# ==================================================
def encode_text_with_clip(text):
    """Encode text using the available method"""
    with torch.no_grad():
        if encode_method == 'original':
            text_tokens = tokenizer([text]).to(device)
            text_features = text_encoder.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.squeeze(0).cpu()
        
        elif encode_method == 'transformers':
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
            text_features = text_encoder(**inputs).last_hidden_state.mean(dim=1)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.squeeze(0).cpu()
        
        elif encode_method == 'sentence_transformers':
            text_features = text_encoder.encode([text], convert_to_tensor=True)
            return text_features.squeeze(0).cpu()
        
        else:
            raise ValueError(f"Unknown encode method: {encode_method}")

# ==================================================
# Cache VAE latents + CLIP text embeddings (kept your logic)
# ==================================================
def cache_latents_to_disk(dataset, save_dir="cached_latents"):
    os.makedirs(save_dir, exist_ok=True)
    vae.eval().to(device).half()

    all_latents = []
    all_text_embeddings = []
    prompts = []
    
    for idx in range(len(dataset)):
        image, prompt = dataset[idx]
        prompts.append(prompt)
        
        # Process image
        image = image.to(device).half().unsqueeze(0)
        with torch.no_grad():
            latents = vae.encode(image)
            latents = (latents - latent_shift) / latent_magnitude
            all_latents.append(latents)
        
        # Process text with CLIP
        text_embedding = encode_text_with_clip(prompt)
        all_text_embeddings.append(text_embedding)
        
        # Save latents and text embeddings separately
        latent_save_path = os.path.join(save_dir, f"latent_{idx}.pt")
        text_save_path = os.path.join(save_dir, f"text_emb_{idx}.pt")
        
        torch.save(latents.squeeze(0).cpu(), latent_save_path)
        torch.save(text_embedding, text_save_path)
        
        if idx % 100 == 0:
            print(f"✅ Cached {idx}/{len(dataset)}: '{prompt[:50]}...'")

    # Save prompts for reference (line-per-prompt to match your existing reader)
    with open(os.path.join(save_dir, "prompts.txt"), "w", encoding='utf-8') as f:
        for prompt in prompts:
            f.write(prompt + "\n")
    
    # Aggregate for quick sanity stats
    all_latents = torch.concat(all_latents, dim=0)
    all_text_embeddings = torch.stack(all_text_embeddings, dim=0)
    
    print(f"Latents shape: {all_latents.shape}, mean: {all_latents.mean()}, var: {all_latents.var()}")
    print(f"Text embeddings shape: {all_text_embeddings.shape}, mean: {all_text_embeddings.mean()}")
    print(f"✅ Cached {len(dataset)} latent files and text embeddings to {save_dir}")

# ==================================================
# CachedLatentDataset (unchanged interface)
# ==================================================
class CachedLatentDataset(Dataset):
    def __init__(self, latent_dir):
        self.latent_dir = latent_dir
        
        # Get all latent files
        self.latent_files = sorted([
            os.path.join(latent_dir, f) for f in os.listdir(latent_dir) 
            if f.startswith('latent_') and f.endswith('.pt')
        ])
        
        # Get all text embedding files
        self.text_emb_files = sorted([
            os.path.join(latent_dir, f) for f in os.listdir(latent_dir) 
            if f.startswith('text_emb_') and f.endswith('.pt')
        ])

        # Load prompts for reference
        prompts_path = os.path.join(latent_dir, "prompts.txt")
        if os.path.exists(prompts_path):
            with open(prompts_path, "r", encoding='utf-8') as f:
                self.prompts = [line.strip() for line in f.readlines()]
        else:
            self.prompts = [""] * len(self.latent_files)

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, idx):
        # Load image latent and text embedding
        latent = torch.load(self.latent_files[idx])
        text_embedding = torch.load(self.text_emb_files[idx])
        prompt = self.prompts[idx] if idx < len(self.prompts) else ""
        
        return latent, text_embedding, prompt

# ==================================================
# LoadData now uses QueenVL (no random styles)
# ==================================================
def LoadData(prompt_style="queenvl", regenerate_prompts=True):
    """
    Load butterfly dataset with Qwen2-VL prompts (only).
    Args kept for compatibility with your code.
    """
    hf_data = load_dataset("huggan/smithsonian_butterflies_subset", split="train")
    print(f"🦋 Loaded Smithsonian butterflies dataset with {len(hf_data)} images")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Prompts file (keep your original naming but force 'queenvl')
    prompts_file = os.path.join(dataset_dir, f"generated_prompts_queenvl.txt")
    
    # Create dataset with Qwen2-VL prompts
    dataset = ButterflyDatasetWithPrompts(
        hf_data, 
        transform=transform, 
        save_prompts_file=prompts_file,
        regenerate_prompts=regenerate_prompts
    )

    latent_cache_dir = os.path.join(dataset_dir, f"cached_latents_queenvl")
    
    # Cache latents and embeddings
    if regenerate_prompts or not os.path.exists(latent_cache_dir):
        print(f"🔄 Caching latents + text embeddings with Qwen2-VL prompts...")
        os.makedirs(latent_cache_dir, exist_ok=True)
        # Save prompts alongside cache for CachedLatentDataset
        with open(os.path.join(latent_cache_dir, "prompts.txt"), "w", encoding='utf-8') as f:
            for _, prompt in dataset:
                f.write(prompt + "\n")
        cache_latents_to_disk(dataset, save_dir=latent_cache_dir)
    else:
        print(f"✅ Using existing cached latents from {latent_cache_dir}")

    cached_dataset = CachedLatentDataset(latent_cache_dir)
    dataloader = DataLoader(cached_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Test the dataloader
    print("\n🧪 Testing dataloader with Qwen2-VL prompts:")
    for batch_idx, (latents, text_embeddings, prompts) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Latents shape: {latents.shape}")
        print(f"  Text embeddings shape: {text_embeddings.shape}")
        print(f"  Sample prompts:")
        for i, prompt in enumerate(prompts[:3]):  # Show first 3 prompts
            print(f"    [{i}] {prompt}")
        break  # Just show first batch
    
    print(f"✅ Loaded {len(cached_dataset)} butterfly samples with Qwen2-VL prompts")
    return dataloader

# ==================================================
# Main
# ==================================================
dataloader = None
if __name__ == "__main__":
    # Load data using QueenVL prompts
    dataloader = LoadData(prompt_style="queenvl", regenerate_prompts=True)

    for batch in dataloader:
        print(len(batch), type(batch))
        print(type(batch[0]), batch[0].shape)  # Latents
        print(type(batch[1]), batch[1].shape)  # Text embeddings
        # prompts is a tuple/list of strings
        print(type(batch[2]), len(batch[2]))   # Prompts
        break
