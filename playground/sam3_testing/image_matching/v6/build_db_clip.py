
# build_db_clip.py
import os
import sys
import json
from nano_vectordb import NanoVectorDB
from tqdm import tqdm
from PIL import Image

# Ensure we can import 'embed' from the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from image_matching.v6.embed_clip import embed_image

# Configuration
ASSETS_ROOT = "/home/gatv-projects/Desktop/project/playground/lol_images_extraction/assets/champions"
DB_FILENAME = os.path.join(current_dir, "lol_champions_clip.nvdb")
EMBEDDING_DIM = 1024 # CLIP ViT-Large hidden size

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
        
        # Only 'square' for consistency
        sub_categories = ["square"]
        
        for category in sub_categories:
            category_dir = os.path.join(champion_dir, category)
            if not os.path.exists(category_dir):
                continue
                
            img_files = [f for f in os.listdir(category_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_file in img_files:
                img_path = os.path.join(category_dir, img_file)
                try:
                    # Embed directly (preprocessing handled inside embed_image via HF Processor)
                    embedding = embed_image(img_path)

                    unique_id = f"{champion_name}_{category}_{img_file}_clip"
                    
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
