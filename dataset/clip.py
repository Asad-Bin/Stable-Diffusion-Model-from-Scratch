from transformers import CLIPTokenizer, CLIPTextModel
import torch

clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
clip_text_model.eval()