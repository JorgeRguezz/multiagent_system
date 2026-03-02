import os
import sys
import glob
from PIL import Image
import numpy as np
from nano_vectordb import NanoVectorDB
import cv2

# Ensure we can import 'embed' from the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


from embed_dinov2_v2 import embed_image

# Configuration
LOL_CUTOUTS_DIR = "/home/gatv-projects/Desktop/project/playground/sam3_testing"
CUTOUTS_DIR = os.path.join(LOL_CUTOUTS_DIR, "lol_cutouts/champions")
# DB_FILENAME = os.path.join(current_dir, "lol_champions_square_224.nvdb")
DB_PATH = "/home/gatv-projects/Desktop/project/playground/sam3_testing/image_matching/v2/"
DB_FILENAME = os.path.join(DB_PATH, "lol_champions_square_224.nvdb")
# EMBEDDING_DIM = 768
EMBEDDING_DIM = 1024
# EMBEDDING_DIM = 1536
TOP_K = 10   # number of candidates from the vector DB for pixel-space recheck

def preprocess_image(input_path, target_size=224, fill_color=(0, 0, 0)):
    """
    Resize an image to 224x224 for DINOv2:
    - Keep aspect ratio
    - Use bicubic interpolation
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

# def mse(a, b):
#     """Mean Squared Error between two images (numpy arrays)."""
#     a = a.astype("float32")
#     b = b.astype("float32")
#     return np.mean((a - b) ** 2)

def hsv_histogram_distance(query_img, db_img, bins=(8, 8, 8)):
    # HSV arrays
    q_hsv = cv2.cvtColor(np.array(query_img), cv2.COLOR_RGB2HSV)
    d_hsv = cv2.cvtColor(np.array(db_img), cv2.COLOR_RGB2HSV)

    # 3D histograms
    q_hist = cv2.calcHist([q_hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    d_hist = cv2.calcHist([d_hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    q_hist, d_hist = q_hist.ravel() / q_hist.sum(), d_hist.ravel() / d_hist.sum()

    # Bhattacharyya distance (better than Chi2 colors)
    return cv2.compareHist(q_hist, d_hist, cv2.HISTCMP_BHATTACHARYYA) 

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
    print("-" * 125)
    print(f"{ 'Image File':<50} | { 'Matched Champion':<20} | {'DB_sim':<10} | {'hsv_dist':<10} | {'Combined':<10} | {'Correct':<8}")
    print("-" * 125)

    all_db_scores = []
    all_hsv = []
    total_correct = 0

    for img_path in image_files:
        try:

            # Preprocess query image
            pil_image = preprocess_image(img_path)
            query_arr = np.array(pil_image)

            # Embed (returns normalized embedding)
            query_embedding = embed_image(pil_image)

            # 1) Vector DB search: get top-K candidates
            results = db.query(query_embedding, top_k=TOP_K)

            if not results:
                print(f"{os.path.basename(img_path):<50} | {'No Match':<20} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'0':<8}")
                continue

            # 2) Re-rank using a weighted combination:
            # Primary: use DB cosine similarity (higher is better)
            # Secondary: use pixel MSE (lower is better) as tiebreaker only

            best_match = None
            best_score = None
            best_db_metric = None
            best_hsv = None

            # for match in results:
            #     champion_name = match.get("champion_name", "Unknown")
            #     champ_path = match.get("img_path")
                
            #     if champ_path is None or not os.path.exists(champ_path):
            #         continue

            #     db_metric = float(match.get("__metrics__", 0.0))  # cosine similarity (0-1)
                
            #     # Only compute pixel MSE if within top 2 DB candidates
            #     # (avoid expensive pixel checks if DB already has clear winner)
            #     if best_match is None or db_metric >= best_db_metric - 0.05:  # within 0.05 margin
            #         champ_img = preprocess_image(champ_path)
            #         champ_arr = np.array(champ_img)
            #         px_mse = mse(query_arr, champ_arr)
            #     else:
            #         px_mse = float('inf')  # Don't bother if DB metric is much worse

            #     # Combine: prioritize DB similarity, use MSE to break ties
            #     combined_score = db_metric - (px_mse / 10000.0)  # Weight pixel slightly
                
            #     if best_match is None or combined_score > best_score:
            #         best_match = (champion_name, champ_path)
            #         best_score = combined_score
            #         best_db_metric = db_metric
            #         best_mse = px_mse

            BASE_SKIN_BONUS = 0.2   # tune this; small positive number

            for match in results:
                champion_name = match.get("champion_name", "Unknown")
                champ_path = match.get("img_path")
                filename_db = match.get("filename", os.path.basename(champ_path) if champ_path else "")

                if champ_path is None or not os.path.exists(champ_path):
                    continue

                db_metric = float(match.get("__metrics__", 0.0))  # cosine similarity (0-1)

                # Only compute pixel MSE if within margin of current best db_metric
                if best_match is None or db_metric >= best_db_metric - 0.05:
                    champ_img = preprocess_image(champ_path)
                    champ_arr = np.array(champ_img)
                    # px_mse = mse(query_arr, champ_arr)
                    hsv_dist = hsv_histogram_distance(pil_image, champ_img) # use HSV histogram distance

                else:
                    hsv_dist = float("inf")
                #     px_mse = float("inf")

                # Base combined score: cosine - weighted MSE
                # combined_score = db_metric - (px_mse / 10000.0)
                combined_score = db_metric * 0.8 - hsv_dist * 0.2
                # combined_score = db_metric 


                # --- base-skin bonus ---
                # base skin images are "ChampionName.png" with no extra suffix
                # e.g., "Aatrox.png", not "Aatrox_1.png"
                expected_base_name = f"{champion_name}.png".lower()
                if filename_db.lower() == expected_base_name:
                    # print(f"Applying base skin bonus for {champion_name} (file: {filename_db})")
                    combined_score += BASE_SKIN_BONUS

                if best_match is None or combined_score > best_score:
                    best_match = (champion_name, champ_path)
                    best_score = combined_score
                    best_db_metric = db_metric
                    # best_mse = px_mse
                    best_hsv = hsv_dist

            if best_match is None:
                print(f"{os.path.basename(img_path):<50} | {'No Match':<20} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'0':<8}")
                continue

            champion_name, champ_path = best_match
            filename = os.path.basename(img_path)

            # Check if match is correct
            is_correct = 1 if champion_name.lower() in filename.lower() else 0
            total_correct += is_correct

            # Collect stats
            if best_db_metric is not None:
                all_db_scores.append(best_db_metric)
            all_hsv.append(best_hsv)

            db_metric_str = f"{best_db_metric:.4f}" if isinstance(best_db_metric, (int, float)) else "N/A"
            print(f"{filename:<50} | {champion_name:<20} | {db_metric_str:<10} | {best_hsv:.4f} | {best_score:.4f} | {is_correct:<8}")

        except Exception as e: 
            print(f"Error processing {os.path.basename(img_path)}: {e}")

    # Optional summary
    print("-" * 125)
    if all_db_scores:
        avg_db = sum(all_db_scores) / len(all_db_scores)
        print(f"{'Average DB metric':<88} | {avg_db:.4f}")
    if all_hsv:
        avg_hsv = sum(all_hsv) / len(all_hsv)
        print(f"{'Average HSV distance':<88} | {avg_hsv:.4f}")
    
    print(f"{'Total Correct matches':<88} | {total_correct} / {len(image_files)}")
    print("-" * 125)

if __name__ == "__main__":
    main()
 