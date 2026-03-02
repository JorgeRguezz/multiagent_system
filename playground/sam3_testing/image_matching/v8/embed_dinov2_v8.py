
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--> DINOv2 using device: {device}")

# Load DINOv2 (Large)
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_all_embeddings(img_path_or_pil):
    """
    Returns a dictionary containing:
    - 'cls': Normalized CLS token (1024,)
    - 'patch_mean': Normalized Mean of Patch tokens (1024,)
    - 'patches': Normalized Patch tokens (256, 1024)
    """
    if isinstance(img_path_or_pil, str):
        img = Image.open(img_path_or_pil).convert("RGB")
    else:
        img = img_path_or_pil.convert("RGB")

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feats = model.forward_features(img_tensor)
        
        # 1. CLS Token
        cls_token = feats["x_norm_clstoken"][0]
        cls_token = F.normalize(cls_token, dim=-1)
        
        # 2. Patch Tokens
        patch_tokens = feats["x_norm_patchtokens"][0] # [256, 1024]
        
        # 3. Patch Mean
        patch_mean = patch_tokens.mean(dim=0)
        patch_mean = F.normalize(patch_mean, dim=-1)
        
        # Normalize patches individually for MaxSim
        patches_norm = F.normalize(patch_tokens, dim=-1)

    return {
        "cls": cls_token.cpu().numpy(),
        "patch_mean": patch_mean.cpu().numpy(),
        "patches": patches_norm.cpu().numpy()
    }

def embed_image(img_path_or_pil):
    """Legacy wrapper for building DB with CLS token by default."""
    res = get_all_embeddings(img_path_or_pil)
    return res['cls']
