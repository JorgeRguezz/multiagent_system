"""
MCP server for entity extraction (character detection and identification) 
using SAM3 and DINOv2.
"""
import os
import sys
import contextlib
import numpy as np
import torch
from PIL import Image
from mcp.server.fastmcp import FastMCP

# Add project root to path for local imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "playground"))
sys.path.append(os.path.join(project_root, "sam3"))

from nano_vectordb import NanoVectorDB
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# V7 Embedder imports (assuming the structure from the monolith)
image_matching = os.path.join(project_root, "knowledge_extraction/image_matching/")
sys.path.append(image_matching)
import embed_dinov2_v7
from embed_dinov2_v7 import embed_image, embed_patch_tokens

mcp = FastMCP("entity_extraction_server")

# Global state for models and DB
sam3_model = None
sam3_processor = None
db = None
db_token_cache = {}
EMBEDDING_DIM = 1024
TOP_K = 30

def load_sam3():
    global sam3_model, sam3_processor
    if sam3_model is None:
        print("--> Entity Server: Loading SAM3...", file=sys.stderr)
        with contextlib.redirect_stdout(sys.stderr):
            sam3_model = build_sam3_image_model()
        sam3_processor = Sam3Processor(sam3_model)
        print("--> Entity Server: SAM3 Loaded ✅", file=sys.stderr)

def load_db(db_path):
    global db
    if db is None:
        print(f"--> Entity Server: Loading DB from {db_path}...", file=sys.stderr)
        db = NanoVectorDB(EMBEDDING_DIM, storage_file=db_path)
        print(f"--> Entity Server: DB loaded with {len(db)} vectors ✅", file=sys.stderr)

def preprocess_image(img, target_size=224, fill_color=(0, 0, 0)):
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

def maxsim_score(q_tokens, d_tokens):
    if d_tokens is None: return 0.0
    sim = q_tokens @ d_tokens.T
    return float(sim.max(axis=1).mean())

def select_tokens_by_padding_mask(pil_image, tokens, patch=14, thr=10, min_fill=0.05):
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
    if not keep:
        return tokens
    return tokens[np.array(keep)]

def get_db_tokens(image_path):
    if image_path in db_token_cache:
        return db_token_cache[image_path]
    if not os.path.exists(image_path):
        return None
    try:
        pil = preprocess_image(Image.open(image_path))
        tokens = embed_patch_tokens(pil)
        db_token_cache[image_path] = tokens
        return tokens
    except Exception:
        return None

@mcp.tool()
async def warmup(db_path: str = "") -> str:
    """Pre-loads SAM3, DINOv2 models, and optionally the database."""
    load_sam3()
    if db_path:
        load_db(db_path)
    return "Models (and DB) loaded and ready."

@mcp.tool()
async def detect_and_match_regions(image_path: str, regions_config: list, db_path: str, threshold: float = 0.0) -> dict:
    """
    Detects characters in multiple specific regions of an image.
    regions_config: [{"name": "middle", "region": [x0, y0, x1, y1]}, ...]
    Returns: {"middle": [...matches...], ...}
    """
    load_sam3()
    load_db(db_path)
    
    if not os.path.exists(image_path):
        return {}

    full_image = Image.open(image_path).convert("RGB")
    all_results = {}

    for config in regions_config:
        region_name = config["name"]
        region = config["region"]
        roi = full_image.crop(region)
        # print(
        #     # f"      [DEBUG Entity Server] {os.path.basename(image_path)} "
        #     f"{region_name} ROI size={roi.size} region={region}",
        #     file=sys.stderr,
        # )
        
        matches = []
        try:
            inference_state = sam3_processor.set_image(roi)
            output = sam3_processor.set_text_prompt(state=inference_state, prompt="Character.")
            boxes = output["boxes"]
        except Exception as e:
            print(f"      [ERROR Entity Server] SAM3 Inference failed for {region_name}: {e}", file=sys.stderr)
            all_results[region_name] = []
            continue

        num_boxes = 0 if boxes is None else len(boxes)
        # print(
        #     # f"      [DEBUG Entity Server] {region_name} SAM3 boxes={num_boxes}",
        #     file=sys.stderr,
        # )

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                if hasattr(box, 'cpu'): box = box.cpu().numpy()
                x_min, y_min, x_max, y_max = map(int, box)
                cutout = roi.crop((x_min, y_min, x_max, y_max))
                
                # Aspect Ratio Check
                w, h = cutout.size
                if h == 0: continue
                aspect_ratio = w / h
                if not (0.7 <= aspect_ratio <= 1.4):
                    # print(
                    #     # f"      [DEBUG Entity Server] {region_name} box rejected: "
                    #     f"({x_min},{y_min},{x_max},{y_max}) aspect_ratio={aspect_ratio:.2f}",
                    #     file=sys.stderr,
                    # )
                    continue
                    
                processed_cutout = preprocess_image(cutout)
                query_embedding = embed_image(processed_cutout)
                results = db.query(query_embedding, top_k=TOP_K)
                
                if not results:
                    # print(
                    #     # f"      [DEBUG Entity Server] {region_name} box has no DB results",
                    #     file=sys.stderr,
                    # )
                    continue
                    
                q_tokens = embed_patch_tokens(processed_cutout)
                q_tokens = select_tokens_by_padding_mask(processed_cutout, q_tokens)

                best_match = None
                best_score = float('-inf')

                for match in results:
                    champ_path = match.get("img_path")
                    if not champ_path or not os.path.exists(champ_path):
                        continue

                    d_tokens = get_db_tokens(champ_path)
                    local_score = maxsim_score(q_tokens, d_tokens)
                    
                    if local_score > best_score:
                        best_match = match.get("champion_name", "Unknown")
                        best_score = local_score
                
                # print(
                #     # f"      [DEBUG Entity Server] {region_name} best_match={best_match} "
                #     f"best_score={best_score:.4f} threshold={threshold}",
                #     file=sys.stderr,
                # )
                if best_match and best_score > threshold:
                    matches.append({"name": best_match, "score": best_score})
        
        all_results[region_name] = matches
            
    return all_results

@mcp.tool()
async def detect_and_match(image_path: str, region: list, db_path: str, threshold: float = 0.0) -> list:
    """
    Detects characters in a specific region of an image and matches them against a DINOv2 database.
    region: [x0, y0, x1, y1]
    """
    load_sam3()
    load_db(db_path)
    
    if not os.path.exists(image_path):
        return []

    full_image = Image.open(image_path).convert("RGB")
    roi = full_image.crop(region)
    
    # # Save ROI for debugging
    # debug_dir = os.path.join(os.path.dirname(image_path), "debug_roi")
    # os.makedirs(debug_dir, exist_ok=True)
    # roi_name = f"roi_{os.path.basename(image_path)}_{region[0]}_{region[1]}.png"
    # roi.save(os.path.join(debug_dir, roi_name))
    
    # print(f"      [DEBUG Entity Server] ROI size: {roi.size} saved to {roi_name}", file=sys.stderr)
    
    # SAM3 Inference
    try:
        inference_state = sam3_processor.set_image(roi)
        # Try with a few different prompts if one fails, or just stick to the proven one
        output = sam3_processor.set_text_prompt(state=inference_state, prompt="Character.")
        boxes = output["boxes"]
    except Exception as e:
        print(f"      [ERROR Entity Server] SAM3 Inference failed: {e}", file=sys.stderr)
        return []
    
    # print(f"      [DEBUG Entity Server] SAM3 detected {len(boxes) if boxes is not None else 0} boxes.", file=sys.stderr)
    
    matches = []
    if boxes is None or len(boxes) == 0:
        return matches

    for box_idx, box in enumerate(boxes):
        if hasattr(box, 'cpu'): box = box.cpu().numpy()
        x_min, y_min, x_max, y_max = map(int, box)
        cutout = roi.crop((x_min, y_min, x_max, y_max))
        
        # Aspect Ratio Check
        w, h = cutout.size
        if h == 0: continue
        aspect_ratio = w / h
        if not (0.7 <= aspect_ratio <= 1.4):
            # print(f"      [DEBUG Entity Server] Box {box_idx} rejected: aspect ratio {aspect_ratio:.2f}", file=sys.stderr)
            continue
            
        processed_cutout = preprocess_image(cutout)
        query_embedding = embed_image(processed_cutout)
        results = db.query(query_embedding, top_k=TOP_K)
        
        if not results:
            continue
            
        q_tokens = embed_patch_tokens(processed_cutout)
        q_tokens = select_tokens_by_padding_mask(processed_cutout, q_tokens)

        best_match = None
        best_score = float('-inf')

        for match in results:
            champ_path = match.get("img_path")
            if not champ_path or not os.path.exists(champ_path):
                continue

            d_tokens = get_db_tokens(champ_path)
            local_score = maxsim_score(q_tokens, d_tokens)
            
            if local_score > best_score:
                best_match = match.get("champion_name", "Unknown")
                best_score = local_score
        
        # print(f"      [DEBUG Entity Server] Box {box_idx}: Best Match {best_match} (Score: {best_score:.4f})", file=sys.stderr)
        
        if best_match and best_score > threshold:
            matches.append({"name": best_match, "score": best_score})
            
    return matches

if __name__ == "__main__":
    mcp.run(transport='stdio')
