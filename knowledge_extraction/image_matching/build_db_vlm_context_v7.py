import os
import sys
from pathlib import Path
from nano_vectordb import NanoVectorDB
from tqdm import tqdm
from PIL import Image

# Ensure we can import 'embed' from the current directory
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[1]
sys.path.append(str(project_root))

from knowledge_extraction.config import CHAMPION_ASSETS_ROOT, DB_PATH
from embed_dinov2_v7 import embed_image

# Configuration
ASSETS_ROOT = Path(CHAMPION_ASSETS_ROOT).resolve()
DB_FILENAME = Path(DB_PATH).resolve()
# EMBEDDING_DIM = 768 # DINOv2 ViT-B/14
EMBEDDING_DIM = 1024 # DINOv2 ViT-L/14


def preprocess_image(input_path, target_size=224, fill_color=(0, 0, 0)):
    """
    Resize an image to 518x518 for DINOv2:
    - Keep aspect ratio
    - Use LANCZOS interpolation
    - Pad to 518x518 with a solid background (default black)
    """
    img = Image.open(input_path).convert("RGB")

    # Original size
    w, h = img.size

    # Compute scale factor so that the longest side == TARGET_SIZE
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize with LANCZOS interpolation
    img_resized = img.resize((new_w, new_h), resample=Image.LANCZOS)

    # Create new 224x224 canvas and paste the resized image centered
    new_img = Image.new("RGB", (target_size, target_size), fill_color)
    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2
    new_img.paste(img_resized, (left, top))
    
    return new_img

def build_champion_db():
    # Initialize DB
    print(f"--> Initializing NanoVectorDB with dim={EMBEDDING_DIM}...")
    db = NanoVectorDB(EMBEDDING_DIM, storage_file=str(DB_FILENAME), metric="cosine")

    if not ASSETS_ROOT.exists():
        print(f"Error: Assets directory not found at {ASSETS_ROOT}")
        return

    data_list = []
    
    # List all champions
    champions = [d.name for d in ASSETS_ROOT.iterdir() if d.is_dir()]
    print(f"--> Found {len(champions)} champions.")

    for champion_name in tqdm(champions, desc="Processing Champions"):
        champion_dir = ASSETS_ROOT / champion_name
        
        # Sub-categories: passives, spells, square
        sub_categories = ["passives", "spells", "square"]
        # sub_categories = ["square"]
        
        for category in sub_categories:
            category_dir = champion_dir / category
            
            if not category_dir.exists():
                continue
                
            # Process images in this category
            img_files = [p.name for p in category_dir.iterdir() if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")]
            
            for img_file in img_files:
                img_path = category_dir / img_file
                
                try:
                    asset_rel_path = img_path.relative_to(ASSETS_ROOT).as_posix()
                    # Preprocess Image (Resize + Pad)
                    pil_image = preprocess_image(img_path)

                    # Generate Embedding
                    embedding = embed_image(pil_image)

                    # Construct Metadata
                    unique_id = f"{champion_name}_{category}_{img_file}"
                    
                    data_item = {
                        "__id__": unique_id,
                        "__vector__": embedding,
                        "champion_name": champion_name,
                        "category": category,
                        "filename": img_file,
                        "asset_rel_path": asset_rel_path,
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
