
# build_db_vlm_context_v4.py
import os
import sys
import json
from nano_vectordb import NanoVectorDB
from tqdm import tqdm
from PIL import Image, ImageDraw

# Ensure we can import 'embed' from the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from image_matching.v4.embed_dinov2_v4 import embed_image

# Configuration
ASSETS_ROOT = "/home/gatv-projects/Desktop/project/playground/lol_images_extraction/assets/champions"
DB_FILENAME = os.path.join(current_dir, "lol_champions_masked_518.nvdb")
EMBEDDING_DIM = 1024 # DINOv2 ViT-L/14 (Large)

def preprocess_image_masked(img, target_size=518, fill_color=(0, 0, 0)):
    """
    1. Resizes image to target_size (keeping aspect ratio).
    2. Applies a CIRCULAR MASK (corners become black).
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    # 1. Resize logic
    w, h = img.size
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img_resized = img.resize((new_w, new_h), resample=Image.LANCZOS)

    # 2. Paste onto black square canvas
    new_img = Image.new("RGB", (target_size, target_size), fill_color)
    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2
    new_img.paste(img_resized, (left, top))
    
    # 3. Apply Circular Mask
    # Create an alpha mask: White circle, Black corners
    mask = Image.new("L", (target_size, target_size), 0)
    draw = ImageDraw.Draw(mask)
    # Draw circle touching edges
    draw.ellipse((0, 0, target_size, target_size), fill=255)
    
    # Composite: Apply mask to image, background is black
    # Create a solid black image
    bg = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    # Composite: where mask is white, use new_img; where black, use bg
    final_img = Image.composite(new_img, bg, mask)
    
    return final_img

def build_champion_db():
    print(f"--> Initializing NanoVectorDB with dim={EMBEDDING_DIM}...")
    db = NanoVectorDB(EMBEDDING_DIM, storage_file=DB_FILENAME, metric="cosine")

    if not os.path.exists(ASSETS_ROOT):
        print(f"Error: Assets directory not found at {ASSETS_ROOT}")
        return

    data_list = []
    
    champions = [d for d in os.listdir(ASSETS_ROOT) if os.path.isdir(os.path.join(ASSETS_ROOT, d))]
    print(f"--> Found {len(champions)} champions.")

    for champion_name in tqdm(champions, desc="Processing Champions"):
        champion_dir = os.path.join(ASSETS_ROOT, champion_name)
        
        # Only process 'square' (HUD candidates)
        sub_categories = ["square"]
        
        for category in sub_categories:
            category_dir = os.path.join(champion_dir, category)
            if not os.path.exists(category_dir):
                continue
                
            img_files = [f for f in os.listdir(category_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_file in img_files:
                img_path = os.path.join(category_dir, img_file)
                try:
                    base_img = Image.open(img_path)
                    
                    # Apply Masked Preprocessing
                    processed_img = preprocess_image_masked(base_img)
                    
                    # Embed
                    embedding = embed_image(processed_img)

                    unique_id = f"{champion_name}_{category}_{img_file}_masked"
                    
                    data_item = {
                        "__id__": unique_id,
                        "__vector__": embedding,
                        "champion_name": champion_name,
                        "category": category,
                        "filename": img_file,
                        "img_path": img_path
                    }
                    data_list.append(data_item)
                    
                except Exception as e:
                    print(f"Failed to process {img_path}: {e}")

    print(f"--> Upserting {len(data_list)} vectors to DB...")
    if data_list:
        result = db.upsert(data_list)
        print(f"    Inserted: {len(result['insert'])}, Updated: {len(result['update'])}")
        db.save()
        print(f"--> DB saved to {DB_FILENAME}. Total vectors: {len(db)}")
    else:
        print("--> No data found to insert.")

if __name__ == "__main__":
    build_champion_db()
