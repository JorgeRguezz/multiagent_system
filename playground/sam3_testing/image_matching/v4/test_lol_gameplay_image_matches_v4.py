import os
import sys
import glob
from PIL import Image, ImageDraw
import numpy as np
from nano_vectordb import NanoVectorDB
import cv2 # Added cv2 import

# Ensure we can import 'embed' from the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from embed_dinov2_v4 import embed_image

# Configuration
LOL_CUTOUTS_DIR = "/home/gatv-projects/Desktop/project/playground/sam3_testing"
CUTOUTS_DIR = os.path.join(LOL_CUTOUTS_DIR, "lol_cutouts/champions")
DB_FILENAME = os.path.join(current_dir, "lol_champions_masked_224.nvdb")
EMBEDDING_DIM = 1024
TOP_K = 10

def preprocess_image_masked(img, target_size=224, fill_color=(0, 0, 0)):
    """
    1. Resizes image to target_size (keeping aspect ratio).
    2. Applies a CIRCULAR MASK (corners become black).
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
    
    # Circular Mask
    mask = Image.new("L", (target_size, target_size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, target_size, target_size), fill=255)
    
    bg = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    final_img = Image.composite(new_img, bg, mask)
    
    return final_img

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
    print("-" * 145)
    print(f"{ 'Image File':<50} | { 'Matched Champion':<20} | {'DB_sim':<10} | {'hsv_dist':<10} | {'Combined':<10} | {'Correct':<8}")
    print("-" * 145)

    all_db_scores = []
    total_correct = 0

    for img_path in image_files:
        try:
            base_img = Image.open(img_path)
            pil_image = preprocess_image_masked(base_img) # MASK THE QUERY TOO
            query_arr = np.array(pil_image)
            query_embedding = embed_image(pil_image)

            results = db.query(query_embedding, top_k=TOP_K)

            if not results:
                continue

            best_match = None
            best_score = float('-inf')
            best_db_metric = 0.0
            best_hsv = 0.0

            BASE_SKIN_BONUS = 0.02

            for match in results:
                champion_name = match.get("champion_name", "Unknown")
                champ_path = match.get("img_path")
                
                if champ_path is None or not os.path.exists(champ_path):
                    continue

                db_metric = float(match.get("__metrics__", 0.0))
                
                if best_match is None or db_metric >= best_db_metric - 0.05:
                    original_db_img = Image.open(champ_path)
                    
                    # Ensure we mask the DB image too for fair pixel comparison
                    champ_img_processed = preprocess_image_masked(original_db_img)
                    
                    hsv_dist = hsv_histogram_distance(pil_image, champ_img_processed)
                else:
                    hsv_dist = float("inf")

                combined_score = db_metric * 0.8 - hsv_dist * 0.2

                filename_db = match.get("filename", os.path.basename(champ_path))
                expected_base_name = f"{champion_name}.png".lower()
                if filename_db.lower() == expected_base_name:
                    combined_score += BASE_SKIN_BONUS

                if combined_score > best_score:
                    best_match = champion_name
                    best_score = combined_score
                    best_db_metric = db_metric
                    best_hsv = hsv_dist

            if best_match is None:
                print(f"{os.path.basename(img_path):<50} | {'No Match':<20} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'0':<8}")
                continue

            is_correct = 1 if best_match.lower() in os.path.basename(img_path).lower() else 0
            total_correct += is_correct

            if best_db_metric is not None:
                all_db_scores.append(best_db_metric)

            db_metric_str = f"{best_db_metric:.4f}"
            print(f"{os.path.basename(img_path):<50} | {best_match:<20} | {db_metric_str:<10} | {best_hsv:.4f} | {best_score:.4f} | {is_correct:<8}")

        except Exception as e: 
            print(f"Error processing {os.path.basename(img_path)}: {e}")

    print("-" * 145)
    if all_db_scores:
        avg_db = sum(all_db_scores) / len(all_db_scores)
        print(f"{ 'Average DB metric':<108} | {avg_db:.4f}")
    
    print(f"{ 'Total Correct matches':<108} | {total_correct} / {len(image_files)}")
    print("-" * 145)

if __name__ == "__main__":
    main()
