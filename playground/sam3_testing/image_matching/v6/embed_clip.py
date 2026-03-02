
# embed_clip.py
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--> CLIP using device: {device}")

# Configuration
# MODEL_ID = "openai/clip-vit-base-patch32" # Fast, 512 dim
MODEL_ID = "openai/clip-vit-large-patch14" # Accurate, 768 dim

# Load CLIP (Vision Model only, we don't need text for img-to-img)
model = CLIPVisionModel.from_pretrained(MODEL_ID).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_ID)
model.eval()

def embed_image(img_path_or_pil):
    """Returns normalized embedding from CLIP."""
    if isinstance(img_path_or_pil, str):
        img = Image.open(img_path_or_pil).convert("RGB")
    else:
        img = img_path_or_pil.convert("RGB")

    # Processor handles resizing, normalization, etc.
    inputs = processor(images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        # pooler_output is the global representation (CLS-like)
        emb = outputs.pooler_output # [1, hidden_size]
        
        # Normalize (CLIP embeddings are usually used with cosine sim, so norm is key)
        emb = emb / emb.norm(dim=-1, keepdim=True)

    return emb.cpu().numpy().flatten()
