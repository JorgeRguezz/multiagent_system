# embed.py
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--> DINOv2 using device: {device}")

# Load DINOv2 (once)
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resize HERE!!
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Version 2.0: with patch-token pooling 

def embed_image(img_path_or_pil):
    """Returns normalized 768-dim embedding using patch-token pooling."""
    if isinstance(img_path_or_pil, str):
        img = Image.open(img_path_or_pil).convert("RGB")
    else:
        img = img_path_or_pil.convert("RGB")

    img_tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, H, W]

    with torch.no_grad():
        # Get full token dictionary instead of only CLS head output
        feats = model.forward_features(img_tensor)  # dict with keys incl. "x_norm_patchtokens"[web:125][web:128]

        patch_tokens = feats["x_norm_patchtokens"]      # [1, N_patches, 768]
        emb = patch_tokens.mean(dim=1)                  # [1, 768] average over patches
        emb = F.normalize(emb, dim=-1)                  # L2-normalize

    return emb.cpu().numpy().flatten()                  # [768]

 