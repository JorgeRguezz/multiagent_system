import os
import sys
import glob
from PIL import Image
import numpy as np
from nano_vectordb import NanoVectorDB

# Ensure we can import 'embed' from the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


from embed_dinov2_v7 import embed_image, embed_patch_tokens

# Configuration
LOL_CUTOUTS_DIR = "/home/gatv-projects/Desktop/project/playground/sam3_testing"
CUTOUTS_DIR = os.path.join(LOL_CUTOUTS_DIR, "lol_cutouts/champions")
DB_PATH = "/home/gatv-projects/Desktop/project/playground/sam3_testing/image_matching/v7/"
DB_FILENAME = os.path.join(DB_PATH, "lol_champions_224_leagueX.nvdb")
EMBEDDING_DIM = 1024
TOP_K = 30   # number of candidates from the vector DB for pixel-space recheck
token_cache = {} # cache for patch tokens to avoid redundant computation

def preprocess_image(input_path, target_size=224, fill_color=(0, 0, 0)):
    """
    Resize an image to 224x224 for DINOv2:
    - Keep aspect ratio
    - Use LANCZOS interpolation
    - Pad to 224x224 with a solid background (default black)
    """
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

def maxsim_score(q_tokens, d_tokens):
    """
    q_tokens: (Nq, D) numpy array of query tokens (L2-normalized),
    d_tokens: (Nd, D) numpy array of database tokens (L2-normalized).
    Returns: float, higher is better (more similar)
    """
    sim = q_tokens @ d_tokens.T  # (Nq, Nd)
    return float(sim.max(axis=1).mean())  # average of max sim for each query token 

def get_tokens_cached(image_path):
    if image_path not in token_cache:
        pil = preprocess_image(image_path)
        token_cache[image_path] = embed_patch_tokens(pil)
    return token_cache[image_path]

def select_informative_tokens(tokens, keep_ratio=0.5):
    """To reduce influence of black padding tokens. Only use the top 
    keep_ratio% most "informative" tokens based on similarity to the mean token."""

    mean = tokens.mean(axis=0, keepdims=True)  # (1, D)
    mean /= (np.linalg.norm(mean) + 1e-9)  # normalize mean vector
    scores = (tokens @ mean.T).ravel()  # similarity to mean
    k = max(1, int(len(tokens) * keep_ratio))
    idx = np.argsort(scores)[-k:]  # indices of top-k tokens
    return tokens[idx]

def select_tokens_by_padding_mask(pil_image, tokens, patch = 14, thr = 10, min_fill = 0.05):
    """
    pil_img: 224x224 padded image
    tokens: (256, D) patch tokens for 224x224 (16x16 patches)
    thr: pixel intensity threshold to treat as non_padding
    min_fill: minimum fraction of non-black pixels inside patch to keep
    """
    arr = np.array(pil_image)  # (224, 224, 3)
    nonblack = (arr.sum(axis=2) > thr).astype(np.float32)  # (224, 224), 1 for non-black pixels

    grid = 224 // patch
    keep = []
    idx = 0
    for gy in range(grid):
        for gx in range(grid):
            y0, y1 = gy*patch, (gy+1)*patch
            x0, x1 = gx*patch, (gx+1)*patch
            fill = nonblack[y0:y1, x0:x1].mean()  # fraction of non-black pixels
            if fill >= min_fill:
                keep.append(idx)
            idx += 1
    if not keep:  # if all patches are mostly black, keep them all to avoid empty tokens
        return tokens
    return tokens[np.array(keep)]

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
    print("-" * 140)
    print(f"{ 'Image File':<50} | { 'Matched Champion':<20} | {'Combined':<8} | {'Global':<8} | {'Local':<8} | {'Correct':<8}")
    print("-" * 140)

    all_db_scores = []
    all_combined_scores = []
    total_correct = 0

    # Tracking min/max scores for confidence analysis
    min_correct_scores = {'combined': float('inf'), 'global': float('inf'), 'local': float('inf'), 'name': 'N/A'}
    max_incorrect_scores = {'combined': float('-inf'), 'global': float('-inf'), 'local': float('-inf'), 'name': 'N/A'}

    for img_path in image_files:
        try:

            # Preprocess query image
            pil_image = preprocess_image(img_path)
            
            # Embed (returns normalized embedding)
            query_embedding = embed_image(pil_image)

            # 1) Vector DB search: get top-K candidates
            results = db.query(query_embedding, top_k=TOP_K)

            if not results:
                print(f"{os.path.basename(img_path):<50} | {'No Match':<20} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'0':<8}")
                continue
            
            q_tokens = embed_patch_tokens(pil_image)
            # q_tokens = select_informative_tokens(q_tokens, keep_ratio=0.3)
            q_tokens = select_tokens_by_padding_mask(pil_image, q_tokens, min_fill=0.05) # Option B: Alternative more robust method

            # 2) Re-rank using a weighted combination: 

            BASE_SKIN_BONUS = 0.0   # tune this; small positive number
            best_score = -1e9
            best_match = None
            
            # Stats for the best match
            best_global = 0.0
            best_local = 0.0

            for match in results:
                champ_path = match.get("img_path")
                if champ_path is None or not os.path.exists(champ_path):
                    continue
                champion_name = match.get("champion_name", "Unknown")
                filename_db = match.get("filename", os.path.basename(champ_path) if champ_path else "")


                d_tokens = get_tokens_cached(champ_path)
                li = maxsim_score(q_tokens, d_tokens)

                db_metric = float(match.get("__metrics__", 0.0))  # cosine similarity (0-1)
                combined = db_metric 
                # combined = li

                # --- base-skin bonus ---
                expected_base_name = f"{champion_name}.png".lower()
                if filename_db.lower() == expected_base_name:
                    # print(f"Applying base skin bonus for {champion_name}")
                    combined += BASE_SKIN_BONUS

                if combined > best_score:
                    best_score = combined
                    best_match = (champion_name, champ_path)
                    best_global = db_metric
                    best_local = li

            if best_match is None:
                print(f"{os.path.basename(img_path):<50} | {'No Match':<20} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'0':<8}")
                continue

            champion_name, champ_path = best_match
            filename = os.path.basename(img_path)

            # Check if match is correct
            is_correct = 1 if champion_name.lower() in filename.lower() else 0
            total_correct += is_correct

            # Update min/max stats
            if is_correct:
                if best_score < min_correct_scores['combined']:
                    min_correct_scores = {'combined': best_score, 'global': best_global, 'local': best_local, 'name': filename}
            else:
                if best_score > max_incorrect_scores['combined']:
                    max_incorrect_scores = {'combined': best_score, 'global': best_global, 'local': best_local, 'name': filename}

            # Collect stats
            if best_global is not None:
                all_db_scores.append(best_global)
            all_combined_scores.append(best_score)

            print(f"{filename:<50} | {champion_name:<20} | {best_score:.4f}   | {best_global:.4f}   | {best_local:.4f}   | {is_correct:<8}")

        except Exception as e: 
            print(f"Error processing {os.path.basename(img_path)}: {e}")

    # Optional summary
    print("-" * 140)
    if all_db_scores:
        avg_db = sum(all_db_scores) / len(all_db_scores)
        print(f"{'Average DB metric (Global)':<103} | {avg_db:.4f}")
    if all_combined_scores:
        avg_combined = sum(all_combined_scores) / len(all_combined_scores)
        print(f"{'Average Combined metric':<103} | {avg_combined:.4f}")
    
    print(f"{'Total Correct matches':<103} | {total_correct} / {len(image_files)}")
    print("-" * 140)
    
    print("Confidence Boundaries:")
    if min_correct_scores['name'] != 'N/A':
        print(f"  Lowest Correct Score ({min_correct_scores['name']}):")
        print(f"    Combined: {min_correct_scores['combined']:.4f} | Global: {min_correct_scores['global']:.4f} | Local: {min_correct_scores['local']:.4f}")
    
    if max_incorrect_scores['name'] != 'N/A':
        print(f"  Highest Incorrect Score ({max_incorrect_scores['name']}):")
        print(f"    Combined: {max_incorrect_scores['combined']:.4f} | Global: {max_incorrect_scores['global']:.4f} | Local: {max_incorrect_scores['local']:.4f}")
    print("-" * 140)

if __name__ == "__main__":
    main()
 