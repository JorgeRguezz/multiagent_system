
# build_db_vlm_context_v3.py
import os
import sys
import json
from nano_vectordb import NanoVectorDB
from tqdm import tqdm
from PIL import Image, ImageOps

# Ensure we can import 'embed' from the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from image_matching.v3.embed_dinov2_v3 import embed_image

# Configuration
ASSETS_ROOT = "/home/gatv-projects/Desktop/project/playground/lol_images_extraction/assets/champions"
DB_FILENAME = os.path.join(current_dir, "lol_champions_square_224_augmented.nvdb")
# EMBEDDING_DIM = 768 # DINOv2 ViT-B/14 (Matches V1)
EMBEDDING_DIM = 1024 # DINOv2 ViT-L/14 (Matches V2)

def preprocess_image(img, target_size=224, fill_color=(0, 0, 0)):
    """
    Resize an image to 224x224 for DINOv2:
    - Keep aspect ratio
    - Use bicubic interpolation
    - Pad to 224x224 with a solid background (default black)
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    img_resized = img.resize((new_w, new_h), resample=Image.LANCZOS)

    new_img = Image.new("RGB", (target_size, target_size), fill_color)
    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2
    new_img.paste(img_resized, (left, top))
    
    return new_img

def create_augmentations(pil_image):
    """
    Generates variations of the input image to simulate HUD conditions.
    """
    augments = []
    
    # 1. Original (Clean)
    augments.append(("original", pil_image))
    
    # 2. Low-Res (Simulate tiny HUD crop)
    # Downscale to 32x32 then upscale back to original size
    # This introduces the blur/pixelation seen in the cutouts
    w, h = pil_image.size
    low_res = pil_image.resize((32, 32), resample=Image.BILINEAR)
    low_res_upscaled = low_res.resize((w, h), resample=Image.NEAREST)
    augments.append(("lowres_32px", low_res_upscaled))
    
    # 3. Grayscale (Simulate 'Dead' state or B&W filter)
    gray = ImageOps.grayscale(pil_image).convert("RGB")
    augments.append(("grayscale", gray))
    
    # 4. Darkened (Simulate 'Cooldown' state or inactive)
    # Simple point operation to darken
    dark = pil_image.point(lambda p: p * 0.5)
    augments.append(("darkened", dark))

    return augments

def build_champion_db():
    # Initialize DB
    print(f"--> Initializing NanoVectorDB with dim={EMBEDDING_DIM}...")
    db = NanoVectorDB(EMBEDDING_DIM, storage_file=DB_FILENAME, metric="cosine")

    if not os.path.exists(ASSETS_ROOT):
        print(f"Error: Assets directory not found at {ASSETS_ROOT}")
        return

    data_list = []
    
    # List all champions
    champions = [d for d in os.listdir(ASSETS_ROOT) if os.path.isdir(os.path.join(ASSETS_ROOT, d))]
    print(f"--> Found {len(champions)} champions.")

    for champion_name in tqdm(champions, desc="Processing Champions"):
        champion_dir = os.path.join(ASSETS_ROOT, champion_name)
        
        # Only process 'square' images (HUD icons)
        sub_categories = ["square"]
        
        for category in sub_categories:
            category_dir = os.path.join(champion_dir, category)
            
            if not os.path.exists(category_dir):
                continue
                
            img_files = [f for f in os.listdir(category_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_file in img_files:
                img_path = os.path.join(category_dir, img_file)
                
                try:
                    # Load base image
                    base_img = Image.open(img_path).convert("RGB")
                    
                    # Generate augmentations
                    # We do augmentation *before* final preprocessing (padding) usually,
                    # but here we want to simulate the whole view. 
                    # Let's preprocess first to get to 224x224 canvas, then augment.
                    processed_base = preprocess_image(base_img)
                    
                    variations = create_augmentations(processed_base)
                    
                    for variant_name, variant_img in variations:
                        # Generate Embedding
                        embedding = embed_image(variant_img)

                        # Construct Metadata
                        # Unique ID includes variation name
                        unique_id = f"{champion_name}_{category}_{img_file}_{variant_name}"
                        
                        data_item = {
                            "__id__": unique_id,
                            "__vector__": embedding,
                            "champion_name": champion_name,
                            "category": category,
                            "filename": img_file,
                            "variation": variant_name,
                            "img_path": img_path
                        }
                        data_list.append(data_item)
                    
                except Exception as e:
                    print(f"Failed to process {img_path}: {e}")

    # Upsert to DB
    print(f"--> Upserting {len(data_list)} vectors to DB...")
    if data_list:
        result = db.upsert(data_list)
        print(f"    Inserted: {len(result['insert'])}, Updated: {len(result['update'])}")
        
        # Save
        db.save()
        print(f"--> DB saved to {DB_FILENAME}. Total vectors: {len(db)}")
    else:
        print("--> No data found to insert.")

if __name__ == "__main__":
    build_champion_db()
