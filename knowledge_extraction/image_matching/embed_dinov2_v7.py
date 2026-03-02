import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
import sys
import contextlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--> DINOv2 using device: {device}", file=sys.stderr)

# Load DINOv2 (once)
with contextlib.redirect_stdout(sys.stderr):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Version 2.0: with patch-token pooling 

def embed_image(img_path_or_pil):
    """Returns normalized 1024-dim embedding using patch-token pooling."""
    if isinstance(img_path_or_pil, str):
        img = Image.open(img_path_or_pil).convert("RGB")
    else:
        img = img_path_or_pil.convert("RGB")

    img_tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, H, W]

    with torch.no_grad():
        # Using patch-token pooling (proved robust in V1)
        feats = model.forward_features(img_tensor)
        patch_tokens = feats["x_norm_patchtokens"]      # [1, N_patches, 1024]
        emb = patch_tokens.mean(dim=1)                  # [1, 1024] average over patches
        emb = F.normalize(emb, dim=-1)                  # L2-normalize

    return emb.cpu().numpy().flatten()                  # [1024]

def embed_patch_tokens(img_path_or_pil):
    """Returns normalized patch-token embeddings (N_patches, 1024)."""
    if isinstance(img_path_or_pil, str):
        img = Image.open(img_path_or_pil).convert("RGB")
    else:
        img = img_path_or_pil.convert("RGB")

    img_tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, H, W]

    with torch.no_grad():
        feats = model.forward_features(img_tensor)
        patches = feats["x_norm_patchtokens"][0]      # [N_patches, D]
        patches = F.normalize(patches, dim=-1) # L2-normalize each patch token

    return patches.cpu().numpy()       # [N_patches, D]

 