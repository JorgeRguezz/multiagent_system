import sys
import os
import shutil
import numpy as np
from PIL import Image
from moviepy.video.io.VideoFileClip import VideoFileClip
from nano_vectordb import NanoVectorDB
import torch
# from transformers import AutoProcessor, AutoModelForImageTextToText # Removed SmolVLM imports
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info # Added Qwen utils
from tqdm import tqdm

# Ensure we can import from playground/knowledge_graph_build and sam3
current_dir = os.path.dirname(os.path.abspath(__file__))
playground_dir = os.path.dirname(current_dir) # playground/
project_root = os.path.dirname(playground_dir) # project root
sam3_repo_root = os.path.join(project_root, "sam3")

sys.path.append(project_root)
sys.path.append(playground_dir)
sys.path.append(sam3_repo_root)

# Import dependencies
try:
    from knowledge_graph_build._videoutil.split import split_video
    from knowledge_graph_build._videoutil.caption import encode_video
    # Import embed_image from V7 embedder
    sys.path.append(os.path.join(current_dir, "image_matching/v7"))
    import embed_dinov2_v7 # Import module to access model
    from embed_dinov2_v7 import embed_image, embed_patch_tokens
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# SAM3 Imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Config
VIDEO_PATH = "/home/gatv-projects/Desktop/project/downloads/The_COMPLETE_SMOLDER_STARTER_GUIDE_-_League_of_Legends.mp4"
WORKING_DIR = "./vlm_context_test_cache"
RESULTS_DIR = "./sam3_results"
# Use V7 Database
DB_FILENAME = os.path.join(current_dir, "image_matching/v7/lol_champions_square_224.nvdb")
EMBEDDING_DIM = 1024 # DINOv2 Large
SEGMENT_LENGTH = 30
FRAMES_PER_SEGMENT = 5
TOP_K = 30
MAX_SEGMENTS = 6 # Limit to first 3 minutes (6 * 30s = 180s)

# Regions for contextual champion detection (relative to 1920x1080)
# MIDDLE_HUD = (600, 900, 1235, 1080) # [x0, y0, x1, y1]
MIDDLE_HUD = (500, 900, 1350, 1080) # límites más permisivos
BOTTOM_RIGHT_HUD = (1500, 600, 1923, 750) # [x0, y0, x1, y1]

# Cache for database image tokens to speed up Reranking
db_token_cache = {}

# --- Qwen2.5-VL Setup ---
current_device = "cuda" if torch.cuda.is_available() else "cpu"
# MODEL_PATH = "HuggingFaceTB/SmolVLM2-2.2B-Instruct" 
MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"

asr_model = None   
vlm_model = None 
vlm_processor = None # Ensure processor is global too

def load_models():
    """Loads ASR and VLM models into GPU memory if not already loaded."""
    global asr_model, vlm_processor, vlm_model, current_device
    
    # --- ASR Model ---
    if asr_model is None:
        print("--> Media Server: Loading ASR model (Whisper)...", file=sys.stderr)
        try:
            import whisper
            asr_model = whisper.load_model("base", device=current_device)
            print("--> Media Server: ASR model loaded successfully ✅", file=sys.stderr)
        except ImportError:
            print("--> Media Server: `whisper` not installed, ASR will not be available.", file=sys.stderr)
        except Exception as e:
            print(f"--> Media Server: Error loading ASR model: {e}", file=sys.stderr)

    # --- VLM Model (Qwen2.5-VL) ---
    if vlm_model is None:
        print(f"--> Media Server: Loading VLM model ({MODEL_PATH})...", file=sys.stderr)
        try:
            # Using float16/bfloat16 and device_map="cuda" as per Qwen usage
            vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_PATH, 
                torch_dtype="float16", 
                device_map="cuda"
            )
            vlm_processor = AutoProcessor.from_pretrained(MODEL_PATH)
            
            print("--> Media Server: VLM model loaded successfully ✅", file=sys.stderr)
        except ImportError:
            print("--> Media Server: `transformers` or `qwen_vl_utils` missing.", file=sys.stderr)
        except Exception as e:
            print(f"--> Media Server: Error loading VLM model: {e}", file=sys.stderr)

def run_vlm_inference(vlm_model, vlm_processor, image, main_champ, partner_champs, transcript=""):
    """
    Runs Qwen2.5-VL inference on the given image with champion context and transcript.
    """
    partners_list_str = ", ".join(partner_champs) if partner_champs else "None identified"
    
    # Context dictionary for cleaner prompt injection
    context = {
      "champion": main_champ if main_champ else "Unknown",
      "teammates": partners_list_str,
      "transcript": transcript if transcript else "No speech detected."
    }

    prompt_text = f"""
    **System Role:**  
    You are a vision-language analyst for League of Legends gameplay. Your job is to create a **detailed, comprehensive textual description** of the provided video frame + context + transcript.  

    **Purpose:**  
    Generate a rich natural language summary that captures *every meaningful visual detail*, *contextual relationships*, and *transcript insights* so a downstream knowledge graph system can extract entities, events, and relationships.  

    **Context:**
    - Champion: {context['champion']}
    - Teammates: {context['teammates']}
    - Transcript: {context['transcript']}

    **Output Instructions:**  

    Write a **detailed scene description** covering:  

    ### 1. **Visual Scene Analysis** (most important).
    - **Position**: Where the action is happening relative to important map landmarks (e.g. towers/statues, river, jungle).  
    - **Actions/Animations**: Categorize the intensity of the action (e.g., idle, fighting) and identify any visible spell effects or animations.  
    - **Health/Mana**: Approximate bars visible, who is low HP, who is full. Red health correspond to enemies, blue health correspond to allies and green health bars correspond to the player.
    - **Map Context**: Indicate if there are any allied (blue) or enemy (red) minions and any allied (blue highlight//outline) or enemy (red highlight/outline) towers/statues visible. 
    - **Items/Effects**: Active items, summoner spells, champion abilities on cooldown. Indicate the status of the players abilities (e.g. if they are on cooldown, ready or in use).  

    ### 2. **Transcript Integration**  
    - What the transcript reveals about intentions, strategies, or events not visible.  
    - What the transcript reveals about gameplay information about players abilities, item builds, or team compositions that can be inferred from the transcript but not directly seen in the image.  

    ### 3. **Temporal Context**  
    - What seems to be happening RIGHT NOW   

    ### 4. **CONSTRAINT:**
    - Do NOT guess champion positions or specific interactions if they are unclear.
    - Do NOT invent a narrative. Describe only the visible frame supporting yourself in the provided context.
    - Do NOT mention the knowledge graph or downstream extraction system.
    - Do NOT describe location in too much detail, just locate where is the action happening in relation to important map landmarks (e.g. river, jungle, towers/statues).
    - Do NOT describe animations (spell or abilities effects) in too much detail, just categorize the intensity of the action (e.g. idle, fighting) and identify if there are any visible spell effects or animations.

    **Format your output as continuous descriptive text, in paragraphs (no subheadings).**   

    **Write the detailed scene description below:**
    """

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # Qwen2.5-VL Inference Logic
    text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = vlm_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(vlm_model.device) # Use model device (cuda)

    # Generation
    generated_ids = vlm_model.generate(**inputs, max_new_tokens=512)

    # Trim prefix and decode
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = vlm_processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    
    description = output_text[0]
    
    # Return description and context string for logging
    return description, prompt_text

def get_frames(video, timestamps):
    """
    Extract frames at specific timestamps without resizing.
    """
    frames = []
    for t in timestamps:
        frame_array = video.get_frame(t)
        # Convert numpy array to PIL Image immediately
        frames.append(Image.fromarray(frame_array.astype('uint8')))
    return frames

def preprocess_image(img_or_path, target_size=224, fill_color=(0, 0, 0)):
    """
    Resize an image to 224x224 for DINOv2:
    - Keep aspect ratio
    - Use LANCZOS interpolation
    - Pad to 224x224 with a solid background (default black)
    """
    if isinstance(img_or_path, str):
        img = Image.open(img_or_path)
    else:
        img = img_or_path

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

def mse(a, b):
    """Mean Squared Error between two images (numpy arrays)."""
    a = a.astype("float32")
    b = b.astype("float32")
    return np.mean((a - b) ** 2)

def maxsim_score(q_tokens, d_tokens):
    """
    q_tokens: (Nq, D) numpy array of query tokens (L2-normalized),
    d_tokens: (Nd, D) numpy array of database tokens (L2-normalized).
    Returns: float, higher is better (more similar)
    """
    if d_tokens is None: return 0.0
    sim = q_tokens @ d_tokens.T  # (Nq, Nd)
    return float(sim.max(axis=1).mean())  # average of max sim for each query token 

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

def get_db_tokens(image_path):
    """Retrieves or computes patch tokens for a DB image."""
    if image_path in db_token_cache:
        return db_token_cache[image_path]
    
    if not os.path.exists(image_path):
        return None
        
    try:
        # Preprocess DB image (same as query: resize/pad)
        pil = preprocess_image(image_path)
        tokens = embed_patch_tokens(pil)
        db_token_cache[image_path] = tokens
        return tokens
    except Exception as e:
        print(f"Error loading DB tokens for {image_path}: {e}")
        return None

def extract_cutout(image, box):
    """
    Extracts the image cutout based on bounding box [x0, y0, x1, y1].
    """
    if hasattr(box, 'cpu'):
        box = box.cpu().numpy()
    
    x_min, y_min, x_max, y_max = map(int, box)
    
    # Clip coordinates to image bounds
    w, h = image.size
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)
    
    if x_max <= x_min or y_max <= y_min:
        return None

    cutout = image.crop((x_min, y_min, x_max, y_max))
    return cutout

def process_detections(boxes, source_frame, db, threshold, segment_idx, frame_idx, region_name):
    """
    Processes detected bounding boxes: extracts cutouts, matches against DB, 
    and returns a list of successful matches.
    """
    matches = []
    if boxes is None or len(boxes) == 0:
        return matches

    for box_idx, box in enumerate(boxes):
        # Extract Cutout
        cutout = extract_cutout(source_frame, box)
        if cutout is None:
            continue

        # Aspect Ratio Check (Almost Square)
        w, h = cutout.size
        if h == 0: continue
        aspect_ratio = w / h
        if not (0.7 <= aspect_ratio <= 1.4):
            continue
        
        try:
            # Preprocess & Embed
            processed_cutout = preprocess_image(cutout)
            query_arr = np.array(processed_cutout)
            query_embedding = embed_image(processed_cutout)
            
            # DB Search
            results = db.query(query_embedding, top_k=TOP_K)
            
            if not results:
                continue
            
            # Extract Query Patch Tokens for Reranking
            q_tokens = embed_patch_tokens(processed_cutout)
            q_tokens = select_tokens_by_padding_mask(processed_cutout, q_tokens, min_fill=0.05)

            # Match Logic
            best_match = None
            best_score = float('-inf')
            best_global = 0.0
            best_local = 0.0

            # BASE_SKIN_BONUS = 0.2

            for match in results:
                champion_name = match.get("champion_name", "Unknown")
                champ_path = match.get("img_path")
                filename_db = match.get("filename", os.path.basename(champ_path) if champ_path else "")

                if champ_path is None or not os.path.exists(champ_path):
                    continue

                # 1. Global Score (Cosine Sim)
                db_metric = float(match.get("__metrics__", 0.0))
                
                # 2. Local Score (MaxSim)
                d_tokens = get_db_tokens(champ_path)
                local_score = maxsim_score(q_tokens, d_tokens)
                
                # Combined Score
                combined_score = local_score

                # Base skin bonus
                # expected_base_name = f"{champion_name}.png".lower()
                # if filename_db.lower() == expected_base_name:
                #     combined_score += BASE_SKIN_BONUS

                if combined_score > best_score:
                    best_match = champion_name
                    best_score = combined_score
                    best_global = db_metric
                    best_local = local_score
            
            # Threshold Check (using Combined Score)
            # Note: Combined score range is roughly [0, 1] + Bonus
            if best_match and best_score > threshold:
                matches.append((best_match, best_score))
                # print(f"       + Match: {best_match} (Score: {best_score:.3f} | G: {best_global:.3f} | L: {best_local:.3f})")
                
        except Exception as e:
            # Silence per-object errors for cleaner frame summary
            pass
            
    return matches

def speech_to_text(video_name, segment_index2name):
    """
    Transcribes audio segments using the loaded ASR model.
    """
    if not asr_model:
        raise RuntimeError("ASR model is not loaded.")
    
    # Working dir is global in this script
    cache_path = os.path.join(WORKING_DIR, '_cache', video_name)
    transcripts = {}
    
    # Use 'mp3' as defined in config
    audio_format = 'mp3' 

    for index in tqdm(segment_index2name, desc=f"Speech Recognition for {video_name}"):
        segment_name = segment_index2name[index]
        audio_file = os.path.join(cache_path, f"{segment_name}.{audio_format}")
        
        if not os.path.exists(audio_file):
            transcripts[index] = ""
            continue
            
        result = asr_model.transcribe(audio_file)
        # Format: [Start -> End] Text
        transcripts[index] = "".join([f"[{s['start']:.2f}s -> {s['end']:.2f}s] {s['text']}" for s in result["segments"]])
        
    return transcripts

def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video not found at {VIDEO_PATH}")
        return

    if not os.path.exists(DB_FILENAME):
        print(f"Error: Database file not found at {DB_FILENAME}")
        return

    print(f"--> Loading database from {DB_FILENAME}...")
    db = NanoVectorDB(EMBEDDING_DIM, storage_file=DB_FILENAME)
    print(f"--> Database loaded with {len(db)} vectors.")

    print(f"--> Processing video: {VIDEO_PATH}")
    
    # Clean workspace
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR, exist_ok=True)
    
    # Load SAM3 Model
    print("\n--> Loading SAM3 Model...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    print("--> SAM3 Model Loaded.")

    # 1. Split Video
    print("\n[Step 1] Splitting Video...")
    segment_index2name, segment_times_info = split_video(
        video_path=VIDEO_PATH,
        working_dir=WORKING_DIR,
        segment_length=SEGMENT_LENGTH,
        num_frames_per_segment=FRAMES_PER_SEGMENT,
        audio_output_format='mp3'
    )
    
    print(f"--> Generated {len(segment_index2name)} segments.")

    # 2. Process Segments (Context Building)
    print("\n[Step 2] Building Context with SAM3...")
    
    # Store context for Phase 3
    # List of dicts: {"frame_path": str, "main_champ": str|None, "partners": list}
    context_data = []

    # Sort indices numerically to ensure correct processing order
    sorted_indices = sorted(segment_index2name.keys(), key=lambda x: int(x))

    with VideoFileClip(VIDEO_PATH) as video:
        for index in sorted_indices:
            if int(index) >= MAX_SEGMENTS:
                print(f"--> Reached time limit (3 mins). Stopping.")
                break
                
            segment_name = segment_index2name[index]
            timestamps = segment_times_info[index]["frame_times"]
            
            print(f"\n>> Segment {index} ({segment_name})")
            
            # Use local get_frames (to keep original resolution)
            frames = get_frames(video, timestamps)
            
            print(f"   [Vision] Extracted {len(frames)} frames. Running SAM3 Inference...")
            
            for i, frame in enumerate(frames):
                image_rgb = frame.convert("RGB")
                
                # Save full frame
                frame_filename = f"full_frame_seg{index}_idx{i}.png"
                frame_path = os.path.join(WORKING_DIR, frame_filename)
                frame.save(frame_path)
                
                # --- Region 1: Middle HUD (Main Player) ---
                middle_hud_frame = image_rgb.crop(MIDDLE_HUD)
                middle_hud_inference = processor.set_image(middle_hud_frame)
                middle_hud_output = processor.set_text_prompt(state=middle_hud_inference, prompt="Character.")
                middle_hud_boxes = middle_hud_output["boxes"]

                frame_main_matches = process_detections(
                    boxes=middle_hud_boxes,
                    source_frame=middle_hud_frame,
                    db=db,
                    threshold=0,
                    segment_idx=index,
                    frame_idx=i,
                    region_name="Middle"
                )

                # --- Region 2: Bottom Right HUD (Partner) ---
                bottom_right_hud_frame = image_rgb.crop(BOTTOM_RIGHT_HUD)
                bottom_right_hud_inference = processor.set_image(bottom_right_hud_frame)
                bottom_right_hud_output = processor.set_text_prompt(state=bottom_right_hud_inference, prompt="Character.")
                bottom_right_hud_boxes = bottom_right_hud_output["boxes"]

                frame_partner_matches_raw = process_detections(
                    boxes=bottom_right_hud_boxes,
                    source_frame=bottom_right_hud_frame,
                    db=db,
                    threshold=0,
                    segment_idx=index,
                    frame_idx=i,
                    region_name="BottomRight"
                )

                # Determine final context for this frame
                # Middle HUD: Use the champion with the highest score
                main_champ_name = None
                if frame_main_matches:
                    # print(f"      [DEBUG Middle HUD] Detections for Frame {i}:")
                    # Sort by score descending
                    frame_main_matches.sort(key=lambda x: x[1], reverse=True)
                    # for name, score in frame_main_matches:
                    #     print(f"        - {name}: {score:.4f}")
                    
                    main_champ_name = frame_main_matches[0][0]
                
                # Bottom Right HUD: Use the top 4 champions with highest scores
                partner_names = []
                if frame_partner_matches_raw:
                    # Sort by score descending
                    frame_partner_matches_raw.sort(key=lambda x: x[1], reverse=True)
                    # Get unique names while maintaining score order
                    seen = set()
                    for name, score in frame_partner_matches_raw:
                        if name not in seen:
                            partner_names.append(name)
                            seen.add(name)
                        if len(partner_names) >= 4:
                            break
                
                # Store data
                context_data.append({
                    "frame_path": frame_path,
                    "main_champ": main_champ_name,
                    "partners": partner_names,
                    "segment_idx": index,
                    "frame_idx": i
                })
                
                print(f"      > Frame {i} processed. Main: {main_champ_name}, Partners: {partner_names}")

    # --- Phase 3: Switch Models ---
    print("\n" + "="*50)
    print("PHASE 2 COMPLETE. Unloading SAM3/DINO and Loading VLM...")
    print("="*50)
    
    # Unload SAM3
    del model
    del processor
    
    # Unload DINOv2
    print("--> Unloading DINOv2...")
    del embed_dinov2_v7.model
    embed_dinov2_v7.model = None
    
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    # Load VLM and ASR models
    # vlm_model, vlm_processor = load_vlm_model()
    load_models()
    
    # --- Phase 4.1: Run ASR ---
    print("\n[Step 3a] Running Speech Recognition (ASR)...")
    video_basename = os.path.basename(VIDEO_PATH).split('.')[0]
    try:
        segment_transcripts = speech_to_text(video_basename, segment_index2name)
        print("--> ASR Complete.")
    except Exception as e:
        print(f"--> ASR Failed: {e}")
        segment_transcripts = {}


    # --- Phase 4.2: VLM Inference ---
    print("\n[Step 3b] Running VLM Inference with Context...")
    
    for entry in context_data:
        frame_path = entry["frame_path"]
        main_champ = entry["main_champ"]
        partners = entry["partners"]
        segment_idx = entry["segment_idx"] # Need this to look up transcript
        
        # Look up transcript for this segment
        # Handle key type mismatch (string vs int) if necessary
        # segment_index2name keys are likely strings "0", "1", etc.
        transcript_text = segment_transcripts.get(segment_idx, "")
        if not transcript_text:
             transcript_text = segment_transcripts.get(str(segment_idx), "")

        print(f"\n[Frame {entry['segment_idx']}-{entry['frame_idx']}]: {os.path.join(WORKING_DIR, os.path.basename(frame_path))}")
        
        # Load image for VLM
        try:
            image = Image.open(frame_path).convert("RGB")
            
            vlm_description, vlm_context = run_vlm_inference(vlm_model, vlm_processor, image, main_champ, partners, transcript_text)
            
            print(f"VLM Context:\n{vlm_context}")
            print(f"VLM Output:\n{vlm_description}")
            
        except Exception as e:
            print(f"Error running VLM on {frame_path}: {e}")
            
        print("-" * 30)

    print("\nProcessing complete.")

if __name__ == "__main__":
    main()
