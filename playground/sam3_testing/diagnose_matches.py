import os
import sys
import numpy as np
from nano_vectordb import NanoVectorDB
from PIL import Image

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from embed_dinov2 import embed_image

DB_FILENAME = os.path.join(current_dir, "lol_champions.nvdb")
CUTOUTS_DIR = os.path.join(current_dir, "lol_cutouts")

def main():
    print(f"Loading DB from {DB_FILENAME}")
    db = NanoVectorDB(768, storage_file=DB_FILENAME)
    
    # Pick one image
    img_files = [f for f in os.listdir(CUTOUTS_DIR) if f.endswith('.png')]
    if not img_files:
        print("No images found.")
        return
    
    test_img_path = os.path.join(CUTOUTS_DIR, img_files[0])
    print(f"Testing with: {test_img_path}")
    
    # Embed
    img = Image.open(test_img_path).convert("RGB")
    
    emb = embed_image(img)
    print(f"Embedding Shape: {emb.shape}")
    print(f"Embedding Norm: {np.linalg.norm(emb)}")
    print(f"Embedding Min/Max: {emb.min()}, {emb.max()}")
    print(f"Embedding First 10: {emb[:10]}")
    
    if np.allclose(emb, 0):
        print("CRITICAL: Embedding is all zeros!")
    
    # Query
    results = db.query(emb, top_k=1)
    print("\nQuery Results:")
    for i, r in enumerate(results):
        print(f"Rank {i}: {r.get('champion_name', 'Unknown')}")
        print(f"   Dist key present: {'__dist__' in r}")
        print(f"   Dist Value: {r.get('__dist__')}")
        print(f"   All Keys: {list(r.keys())}")

if __name__ == "__main__":
    main()
