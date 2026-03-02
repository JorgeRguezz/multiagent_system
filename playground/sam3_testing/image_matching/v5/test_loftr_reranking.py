import os
import sys
import glob
from PIL import Image
import numpy as np
from nano_vectordb import NanoVectorDB
import cv2
import torch
import kornia
from kornia.feature import LoFTR

# Ensure we can import 'embed' from the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from embed_dinov2 import embed_image

# Configuration
LOL_CUTOUTS_DIR = "/home/gatv-projects/Desktop/project/playground/sam3_testing"
CUTOUTS_DIR = os.path.join(LOL_CUTOUTS_DIR, "lol_cutouts/champions")
# Reusing V2 Optimized DB
DB_FILENAME = os.path.join(os.path.dirname(current_dir), "v2", "lol_champions_square_224.nvdb")
EMBEDDING_DIM = 1024
TOP_K_RETRIEVAL = 5 # Fetch top 5 for reranking

# LoFTR Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--> LoFTR using device: {DEVICE}")

# Initialize LoFTR
loftr_matcher = LoFTR(pretrained="outdoor").to(DEVICE) # 'outdoor' is usually better trained than 'indoor' for general objects
loftr_matcher.eval()

def preprocess_image_dino(input_path, target_size=224, fill_color=(0, 0, 0)):
    img = Image.open(input_path).convert("RGB")
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

def preprocess_for_loftr(pil_img, size=(224, 224)):
    """
    Converts PIL RGB image to Grayscale Tensor for LoFTR.
    """
    # LoFTR works on grayscale
    img_gray = pil_img.convert("L")
    img_gray = img_gray.resize(size, Image.BICUBIC)
    img_tensor = torch.from_numpy(np.array(img_gray)).float() / 255.0
    return img_tensor.unsqueeze(0).unsqueeze(0).to(DEVICE) # (1, 1, H, W)

def run_loftr(query_pil, target_pil):
    """
    Runs LoFTR matching between two images.
    Returns the number of high-confidence matches.
    """
    # LoFTR performs best at slightly larger resolutions, 
    # but for icon matching, 224x224 or 320x320 is likely sufficient and faster.
    # Let's try 320x320 to give it some spatial room.
    SIZE = (320, 320)
    
    batch = {
        "image0": preprocess_for_loftr(query_pil, SIZE),
        "image1": preprocess_for_loftr(target_pil, SIZE)
    }
    
    with torch.no_grad():
        out = loftr_matcher(batch)
        # print(f"DEBUG: LoFTR keys: {out.keys()}") # Uncomment if needed
        
    # 'mkpts0' are matches in image0, 'mconf' is confidence
    # We can filter by confidence if needed, but LoFTR usually outputs good matches.
    # Default threshold is implicitly handled by the model logic (usually 0.2)
    
    if "keypoints0" in out:
        # Kornia 0.7+ structure might be different from original LoFTR repo
        # It usually returns 'keypoints0', 'keypoints1', 'confidence', 'batch_indexes'
        # But wait, Kornia LoFTR usually returns a dict with 'keypoints0', etc.
        # Let's count rows in keypoints0
        matches = out["keypoints0"].shape[0]
    elif "mkpts0" in out:
        matches = out["mkpts0"].shape[0]
    else:
        print(f"DEBUG: LoFTR Output Keys: {out.keys()}")
        matches = 0
        
    return matches

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
    
    # Filter out B&W images
    image_files = [f for f in image_files if "B&W" not in os.path.basename(f)]
    
    if not image_files:
        print(f"No images found in {CUTOUTS_DIR}")
        return

    print(f"Found {len(image_files)} images to process.")
    print("-" * 145)
    print(f"{ 'Image File':<50} | { 'Matched Champion':<20} | {'LoFTR Matches':<15} | {'DINO Rank':<10} | {'Correct':<8}")
    print("-" * 145)

    total_correct = 0

    for img_path in image_files:
        try:
            # 1. DINO Retrieval
            pil_image = preprocess_image_dino(img_path)
            query_embedding = embed_image(pil_image)
            results = db.query(query_embedding, top_k=TOP_K_RETRIEVAL)

            if not results:
                continue

            # 2. LoFTR Reranking
            best_match = None
            max_matches = -1
            best_dino_rank = -1
            
            # Keep track of the original query image for LoFTR
            query_pil_original = Image.open(img_path).convert("RGB")

            for rank, candidate in enumerate(results):
                champion_name = candidate.get("champion_name", "Unknown")
                champ_path = candidate.get("img_path")
                
                if champ_path is None or not os.path.exists(champ_path):
                    continue
                
                # Load candidate image
                candidate_pil = Image.open(champ_path).convert("RGB")
                
                # Run LoFTR
                num_matches = run_loftr(query_pil_original, candidate_pil)
                
                # Simple logic: Winner takes all
                if num_matches > max_matches:
                    max_matches = num_matches
                    best_match = champion_name
                    best_dino_rank = rank
            
            if best_match is None:
                # Fallback to DINO top-1
                best_match = results[0]["champion_name"]
                max_matches = 0
                best_dino_rank = 0

            # Correctness Check
            is_correct = 1 if best_match.lower() in os.path.basename(img_path).lower() else 0
            total_correct += is_correct

            print(f"{os.path.basename(img_path):<50} | {best_match:<20} | {max_matches:<15} | {best_dino_rank:<10} | {is_correct:<8}")

        except Exception as e: 
            print(f"Error processing {os.path.basename(img_path)}: {e}")

    print("-" * 145)
    print(f"{ 'Total Correct matches':<108} | {total_correct} / {len(image_files)}")
    print("-" * 145)

if __name__ == "__main__":
    main()
