
import os
import sys
import glob
from PIL import Image
import numpy as np
from nano_vectordb import NanoVectorDB

# Ensure we can import 'embed' from the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from embed_dinov2_v8 import get_all_embeddings

# Configuration
LOL_CUTOUTS_DIR = "/home/gatv-projects/Desktop/project/playground/sam3_testing"
CUTOUTS_DIR = os.path.join(LOL_CUTOUTS_DIR, "lol_cutouts/champions")
DB_FILENAME = os.path.join(current_dir, "lol_champions_square_224_cls.nvdb")
EMBEDDING_DIM = 1024
# Set high enough to retrieve all candidates for brute-force MaxSim
TOP_K_RETRIEVAL = 10000 

# Cache to avoid re-loading/re-embedding DB images
db_embedding_cache = {}

def preprocess_image(input_path_or_pil, target_size=224, fill_color=(0, 0, 0)):
    if isinstance(input_path_or_pil, str):
        img = Image.open(input_path_or_pil).convert("RGB")
    else:
        img = input_path_or_pil.convert("RGB")

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

def maxsim_score(q_tokens, d_tokens):
    """
    Computes MaxSim: Average of max similarity for each query token against all doc tokens.
    """
    if d_tokens is None: return 0.0
    # q: (Nq, D), d: (Nd, D)
    # sim: (Nq, Nd)
    sim = q_tokens @ d_tokens.T
    return float(sim.max(axis=1).mean())

def select_tokens_by_padding_mask(pil_image, tokens, patch = 14, thr = 10, min_fill = 0.05):
    arr = np.array(pil_image)
    nonblack = (arr.sum(axis=2) > thr).astype(np.float32)
    grid = 224 // patch
    keep = []
    idx = 0
    for gy in range(grid):
        for gx in range(grid):
            y0, y1 = gy*patch, (gy+1)*patch
            x0, x1 = gx*patch, (gx+1)*patch
            fill = nonblack[y0:y1, x0:x1].mean()
            if fill >= min_fill:
                keep.append(idx)
            idx += 1
    if not keep: return tokens
    return tokens[np.array(keep)]

def get_cached_db_tokens(img_path):
    """
    Retrieves ONLY the patch tokens for MaxSim.
    Use cache to speed up repeated lookups (though in this script each image is processed once per query loop,
    so cache helps if multiple queries hit same DB images, or if we parallelize).
    """
    if img_path in db_embedding_cache:
        return db_embedding_cache[img_path]
    
    # We only need patch tokens for MaxSim
    pil = preprocess_image(img_path)
    embs = get_all_embeddings(pil)
    patches = embs['patches'] # (256, 1024)
    
    db_embedding_cache[img_path] = patches
    return patches

def main():
    if not os.path.exists(DB_FILENAME):
        print(f"Error: Database file not found at {DB_FILENAME}")
        return

    print(f"Loading database from {DB_FILENAME}...")
    db = NanoVectorDB(EMBEDDING_DIM, storage_file=DB_FILENAME)
    print(f"Database loaded with {len(db)} vectors.")

    image_files = glob.glob(os.path.join(CUTOUTS_DIR, "*.png")) + \
                  glob.glob(os.path.join(CUTOUTS_DIR, "*.jpg")) + \
                  glob.glob(os.path.join(CUTOUTS_DIR, "*.jpeg"))
    
    image_files = [f for f in image_files if "B&W" not in os.path.basename(f)]
    
    if not image_files:
        print(f"No images found in {CUTOUTS_DIR}")
        return

    print(f"Found {len(image_files)} images to process.")
    print("-" * 140)
    print(f"{ 'Image File':<50} | { 'Matched Champion':<20} | {'MaxSim':<10} | {'Correct':<8}")
    print("-" * 140)

    total_correct = 0
    
    # Pre-fetch all candidate paths from DB to iterate over
    # Use a dummy query with high top_k to get all entries
    dummy_vector = np.zeros(EMBEDDING_DIM)
    all_candidates = db.query(dummy_vector, top_k=10000)
    print(f"Scanning against {len(all_candidates)} candidates using Brute-Force MaxSim...")

    for img_path in image_files:
        try:
            # 0. Prepare Query
            pil_image = preprocess_image(img_path)
            q_embs = get_all_embeddings(pil_image)
            
            # Filter query tokens (ignore black padding)
            q_tokens_filtered = select_tokens_by_padding_mask(pil_image, q_embs['patches'])
            
            best_match = None
            best_score = -1e9
            
            BASE_SKIN_BONUS = 0.00

            # --- BRUTE FORCE MAXSIM LOOP ---
            for cand in all_candidates:
                path = cand['img_path']
                if not os.path.exists(path): continue
                
                # Get DB Tokens (Cached)
                d_tokens = get_cached_db_tokens(path)
                
                # MaxSim Calculation
                score = maxsim_score(q_tokens_filtered, d_tokens)
                
                # Apply Bonus
                filename = os.path.basename(path)
                if filename.lower() == f"{cand['champion_name']}.png".lower():
                    score += BASE_SKIN_BONUS
                
                if score > best_score:
                    best_score = score
                    best_match = cand['champion_name']
            
            is_correct = 1 if best_match.lower() in os.path.basename(img_path).lower() else 0
            total_correct += is_correct
            
            print(f"{os.path.basename(img_path):<50} | {best_match:<20} | {best_score:.4f}     | {is_correct:<8}")

        except Exception as e:
            print(f"Error processing {os.path.basename(img_path)}: {e}")

    print("-" * 140)
    print(f"{ 'Total Correct matches':<103} | {total_correct} / {len(image_files)}")
    print("-" * 140)

if __name__ == "__main__":
    main()
