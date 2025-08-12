import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
import random
import json

from config.config import *
from utils.debug import *
# from pre_vae.pre_vae import vae

# --------------------------------------------------
# Import CLIP with multiple fallback options
# --------------------------------------------------
def setup_text_encoder():
    """Setup text encoder with multiple fallback options"""
    global text_encoder, tokenizer, encode_method
    
    # Option 1: Try original CLIP
    try:
        import clip
        if hasattr(clip, 'load'):
            model, preprocess = clip.load("ViT-B/32", device=device)
            model.eval()
            text_encoder = model
            tokenizer = clip.tokenize
            encode_method = 'original'
            print("✅ Using original CLIP")
            return
    except Exception as e:
        print(f"Original CLIP failed: {e}")
    
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
    
    # Option 3: Try sentence transformers as last resort
    try:
        from sentence_transformers import SentenceTransformer
        text_encoder = SentenceTransformer('clip-ViT-B-32').to(device)
        tokenizer = None
        encode_method = 'sentence_transformers'
        print("✅ Using sentence-transformers CLIP")
        return
    except Exception as e:
        print(f"Sentence transformers failed: {e}")
    
    raise ImportError("❌ Could not load any text encoder. Please install one of: clip, transformers, or sentence-transformers")

# Initialize text encoder
setup_text_encoder()

# --------------------------------------------------
# Butterfly Prompt Generation Functions
# --------------------------------------------------
def generate_detailed_butterfly_prompt(idx=None):
    """Generate detailed, varied butterfly prompts"""
    
    # Base butterfly types
    butterfly_types = [
        "Monarch butterfly", "Swallowtail butterfly", "Admiral butterfly", 
        "Skipper butterfly", "Fritillary butterfly", "Blue butterfly",
        "Copper butterfly", "Hairstreak butterfly", "White butterfly",
        "Sulphur butterfly", "Longwing butterfly", "Metalmark butterfly"
    ]
    
    # Colors and patterns
    colors = [
        "vibrant orange", "deep blue", "bright yellow", "pristine white",
        "rich black", "emerald green", "royal purple", "copper red",
        "silvery gray", "golden yellow", "coral pink", "turquoise blue"
    ]
    
    patterns = [
        "with intricate spotted wings", "with delicate striped patterns",
        "with symmetrical wing markings", "with iridescent wing scales",
        "with bold geometric patterns", "with subtle mottled designs",
        "with eye-spot patterns", "with metallic wing borders",
        "with transparent wing sections", "with gradient color transitions"
    ]
    
    poses = [
        "perched on a flower", "with wings spread wide", "in a graceful pose",
        "resting on a leaf", "with wings partially folded", "in profile view",
        "showing detailed wing structure", "captured from above", "in natural position",
        "displaying wing patterns clearly"
    ]
    
    settings = [
        "against a neutral background", "in natural lighting",
        "showing fine wing details", "in museum specimen style",
        "with scientific precision", "in high detail photography",
        "showcasing natural colors", "in documentary style"
    ]
    
    descriptors = [
        "beautiful", "delicate", "graceful", "elegant", "stunning",
        "magnificent", "pristine", "detailed", "colorful", "exquisite"
    ]
    
    # Create varied prompt templates
    templates = [
        "{descriptor} {butterfly_type} {pattern} {pose} {setting}",
        "{color} {butterfly_type} {pattern} {pose}",
        "{descriptor} {butterfly_type} {pose} {setting}",
        "{butterfly_type} {pattern} {color} coloring {pose}",
        "{descriptor} {color} {butterfly_type} {setting}",
        "{butterfly_type} specimen {pattern} {pose} {setting}"
    ]
    
    # Select random elements
    template = random.choice(templates)
    
    prompt = template.format(
        descriptor=random.choice(descriptors),
        butterfly_type=random.choice(butterfly_types),
        color=random.choice(colors),
        pattern=random.choice(patterns),
        pose=random.choice(poses),
        setting=random.choice(settings)
    )
    
    # Clean up any double spaces
    prompt = ' '.join(prompt.split())
    
    return prompt

def generate_scientific_butterfly_prompt(idx=None):
    """Generate scientific/specimen style prompts"""
    
    families = [
        "Nymphalidae", "Papilionidae", "Pieridae", "Lycaenidae",
        "Hesperiidae", "Riodinidae", "Danaidae", "Satyridae"
    ]
    
    features = [
        "wingspan measurement visible", "antenna structure detailed",
        "wing venation patterns clear", "body segmentation visible",
        "proboscis coiled", "compound eyes detailed",
        "leg structure visible", "wing scales magnified"
    ]
    
    specimen_terms = [
        "museum specimen", "scientific collection", "taxonomic reference",
        "research specimen", "field guide illustration", "entomological study",
        "species documentation", "morphological study"
    ]
    
    template = random.choice([
        f"Lepidoptera specimen from {random.choice(families)} family with {random.choice(features)}",
        f"Butterfly {random.choice(specimen_terms)} showing {random.choice(features)}",
        f"Scientific illustration of butterfly with {random.choice(features)}",
        f"{random.choice(families)} butterfly specimen for {random.choice(specimen_terms)}"
    ])
    
    return template

def generate_artistic_butterfly_prompt(idx=None):
    """Generate artistic/aesthetic prompts"""
    
    artistic_styles = [
        "watercolor painting style", "botanical illustration",
        "nature photography", "macro photography",
        "vintage naturalist drawing", "field guide illustration",
        "scientific diagram", "detailed sketch"
    ]
    
    aesthetics = [
        "soft natural lighting", "vibrant color palette",
        "high contrast details", "delicate textures",
        "fine art composition", "minimalist background",
        "studio lighting", "dramatic shadows"
    ]
    
    template = f"Butterfly rendered in {random.choice(artistic_styles)} with {random.choice(aesthetics)}"
    return template

# --------------------------------------------------
# Enhanced Dataset with Generated Prompts
# --------------------------------------------------
class ButterflyDatasetWithPrompts(Dataset):
    def __init__(self, hf_dataset, transform=None, prompt_style="detailed", save_prompts_file=None):
        self.dataset = hf_dataset
        self.transform = transform
        self.prompt_style = prompt_style
        self.generated_prompts = []
        
        print(f"🦋 Generating {prompt_style} prompts for {len(hf_dataset)} butterfly images...")
        
        # Pre-generate all prompts for consistency
        for idx in range(len(hf_dataset)):
            if prompt_style == "detailed":
                prompt = generate_detailed_butterfly_prompt(idx)
            elif prompt_style == "scientific":
                prompt = generate_scientific_butterfly_prompt(idx)
            elif prompt_style == "artistic":
                prompt = generate_artistic_butterfly_prompt(idx)
            else:
                prompt = "a beautiful butterfly specimen"
            
            self.generated_prompts.append(prompt)
        
        # Save prompts to file for reference
        if save_prompts_file:
            with open(save_prompts_file, 'w', encoding='utf-8') as f:
                for i, prompt in enumerate(self.generated_prompts):
                    f.write(f"[{i}] {prompt}\n")
            print(f"✅ Saved all generated prompts to {save_prompts_file}")
        
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

# --------------------------------------------------
# Function to encode text with different methods
# --------------------------------------------------
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

# --------------------------------------------------
# Save VAE-encoded latents and CLIP text embeddings to disk
# --------------------------------------------------
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

    # Save prompts for reference
    with open(os.path.join(save_dir, "prompts.txt"), "w", encoding='utf-8') as f:
        for prompt in prompts:
            f.write(prompt + "\n")
    
    all_latents = torch.concat(all_latents, dim=0)
    all_text_embeddings = torch.stack(all_text_embeddings, dim=0)
    
    print(f"Latents shape: {all_latents.shape}, mean: {all_latents.mean()}, var: {all_latents.var()}")
    print(f"Text embeddings shape: {all_text_embeddings.shape}, mean: {all_text_embeddings.mean()}")
    print(f"✅ Cached {len(dataset)} latent files and text embeddings to {save_dir}")

# --------------------------------------------------
# Dataset for cached latents with text embeddings
# --------------------------------------------------
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

# --------------------------------------------------
# Enhanced LoadData with Generated Prompts
# --------------------------------------------------
def LoadData(prompt_style="detailed", regenerate_prompts=True):
    """
    Load butterfly dataset with generated prompts
    
    Args:
        prompt_style: "detailed", "scientific", or "artistic"
        regenerate_prompts: If True, regenerate prompts; if False, try to load existing
    """
    hf_data = load_dataset("huggan/smithsonian_butterflies_subset", split="train")
    print(f"🦋 Loaded Smithsonian butterflies dataset with {len(hf_data)} images")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create prompts file path
    prompts_file = os.path.join(dataset_dir, f"generated_prompts_{prompt_style}.txt")
    
    # Create dataset with generated prompts
    dataset = ButterflyDatasetWithPrompts(
        hf_data, 
        transform=transform, 
        prompt_style=prompt_style,
        save_prompts_file=prompts_file
    )

    latent_cache_dir = os.path.join(dataset_dir, f"cached_latents_{prompt_style}")
    
    # Cache latents and embeddings
    if regenerate_prompts or not os.path.exists(latent_cache_dir):
        print(f"🔄 Caching latents with {prompt_style} prompts...")
        cache_latents_to_disk(dataset, save_dir=latent_cache_dir)
    else:
        print(f"✅ Using existing cached latents from {latent_cache_dir}")

    cached_dataset = CachedLatentDataset(latent_cache_dir)
    dataloader = DataLoader(cached_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Test the dataloader
    print("\n🧪 Testing dataloader with generated prompts:")
    for batch_idx, (latents, text_embeddings, prompts) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Latents shape: {latents.shape}")
        print(f"  Text embeddings shape: {text_embeddings.shape}")
        print(f"  Sample prompts:")
        for i, prompt in enumerate(prompts[:3]):  # Show first 3 prompts
            print(f"    [{i}] {prompt}")
        break  # Just show first batch
    
    print(f"✅ Loaded {len(cached_dataset)} butterfly samples with {prompt_style} prompts")
    return dataloader

# --------------------------------------------------
# Function to test different prompt styles
# --------------------------------------------------
def test_all_prompt_styles():
    """Test all prompt generation styles"""
    print("🧪 Testing all prompt styles:\n")
    
    styles = ["detailed", "scientific", "artistic"]
    for style in styles:
        print(f"=== {style.upper()} PROMPTS ===")
        for i in range(5):
            if style == "detailed":
                prompt = generate_detailed_butterfly_prompt()
            elif style == "scientific":
                prompt = generate_scientific_butterfly_prompt()
            elif style == "artistic":
                prompt = generate_artistic_butterfly_prompt()
            print(f"  {prompt}")
        print()

# Final call - choose your preferred prompt style
if __name__ == "__main__":
    # Test prompt styles first
    test_all_prompt_styles()
    
    # Load data with your preferred prompt style
    dataloader = LoadData(prompt_style="detailed", regenerate_prompts=True)