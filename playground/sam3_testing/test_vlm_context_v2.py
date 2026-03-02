import sys
import os
import shutil
import numpy as np
from PIL import Image
from moviepy.video.io.VideoFileClip import VideoFileClip
from nano_vectordb import NanoVectorDB
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
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
    # Import embed_image from embed_dinov2.py
    sys.path.append(current_dir)
    from embed_dinov2 import embed_image
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
DB_FILENAME = os.path.join(current_dir, "lol_champions_224.nvdb")
EMBEDDING_DIM = 768
SEGMENT_LENGTH = 30
FRAMES_PER_SEGMENT = 5
TOP_K = 10
MAX_SEGMENTS = 6 # Limit to first 3 minutes (6 * 30s = 180s)
MIDDLE_HUD = (600, 900, 1235, 1080) # [x0, y0, x1, y1]
BOTTOM_RIGHT_HUD = (1500, 600, 1923, 750) # [x0, y0, x1, y1]

# --- SmolVLM2 Setup ---
current_device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

asr_model = None   
vlm_model = None 

# def load_vlm_model():
#     print(f"--> Initializing SmolVLM2 on {DEVICE}...")
#     vlm_processor = AutoProcessor.from_pretrained(MODEL_PATH)
#     vlm_model = AutoModelForImageTextToText.from_pretrained(
#         MODEL_PATH,
#         torch_dtype=torch.bfloat16, 
#     ).to(DEVICE)
#     print("--> SmolVLM2 Initialized.")
#     return vlm_model, vlm_processor

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

    # --- VLM Model ---
    if vlm_model is None:
        print("--> Media Server: Loading VLM model (SmolVLM2-Video)...", file=sys.stderr)
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
            vlm_model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
            vlm_processor = AutoProcessor.from_pretrained(vlm_model_path)
            vlm_model = AutoModelForImageTextToText.from_pretrained(
                vlm_model_path,
                torch_dtype=torch.bfloat16 if (current_device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32,
            ).to(current_device)
            print("--> Media Server: VLM model loaded successfully ✅", file=sys.stderr)
        except ImportError:
            print("--> Media Server: `transformers` not installed, VLM will not be available.", file=sys.stderr)
        except Exception as e:
            print(f"--> Media Server: Error loading VLM model: {e}", file=sys.stderr)

def run_vlm_inference(vlm_model, vlm_processor, image, main_champ, partner_champs, transcript=""):
    """
    Runs SmolVLM2 inference on the given image with champion context and transcript.
    """
    context_str = "{\n"
    if main_champ:
        context_str += f'  "champion": "{main_champ}",\n'
    if partner_champs:
        partners_list = '", "'.join(partner_champs)
        context_str += f'  "teammates": ["{partners_list}"],\n'
    if transcript:
        # Escape quotes in transcript to avoid JSON breakage
        safe_transcript = transcript.replace('"', "'")
        context_str += f'  "transcript": "{safe_transcript}"\n'
    context_str += "}"

    prompt = f"""# TASK:
    Describe the main visual activity in this League of Legends screenshot.

    # INSTRUCTIONS:
    1. **Environment:** Identify the specific part of the map shown (e.g., Lane, River, Jungle, Base).
    2. **Visual Effects:** Describe the dominant spell effects visible (e.g., "purple laser," "fire particles," "plants"). Connect these visuals to the provided champion names ONLY if the match is obvious.
    3. **Intensity:** State if this appears to be a large team fight (many health bars/effects) or a minor skirmish.
    4. **Speech Context:** Use the provided transcript to infer intent if relevant (e.g., "I'm going in!" suggests aggression) or to provide important information about champions abilities, match situations, etc.

    # CONTEXT (Potential Champions & Transcript):
    {context_str}

    # CONSTRAINT:
    - Do NOT guess champion positions or specific interactions if they are unclear.
    - Do NOT invent a narrative. Describe only the visible frame.
    """

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        },
    ]

    inputs = vlm_processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(vlm_model.device, dtype=torch.bfloat16)

    generated_ids = vlm_model.generate(**inputs, do_sample=False, max_new_tokens=512)
    generated_texts = vlm_processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    
    # Extract only the assistant's response if possible, though batch_decode usually gives full text
    full_text = generated_texts[0]
    description = full_text
    if "Assistant:" in full_text:
        description = full_text.split("Assistant:")[1].strip()
        
    return description, context_str

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

            # Match Logic
            best_match = None
            best_score = float('-inf')
            best_db_metric = float('-inf')
            best_mse = float('inf')

            BASE_SKIN_BONUS = 0.02

            for match in results:
                champion_name = match.get("champion_name", "Unknown")
                champ_path = match.get("img_path")
                filename_db = match.get("filename", os.path.basename(champ_path) if champ_path else "")

                if champ_path is None or not os.path.exists(champ_path):
                    continue

                db_metric = float(match.get("__metrics__", 0.0))
                
                if best_match is None or db_metric >= best_db_metric - 0.05:
                    champ_img = preprocess_image(champ_path)
                    champ_arr = np.array(champ_img)
                    px_mse = mse(query_arr, champ_arr)
                else:
                    px_mse = float("inf")

                combined_score = db_metric - (px_mse / 10000.0)

                # Base skin bonus
                expected_base_name = f"{champion_name}.png".lower()
                if filename_db.lower() == expected_base_name:
                    combined_score += BASE_SKIN_BONUS

                if combined_score > best_score:
                    best_match = champion_name
                    best_score = combined_score
                    best_db_metric = db_metric
                    best_mse = px_mse
            
            if best_match and best_db_metric > threshold:
                matches.append(best_match)
                
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
                    threshold=0.7,
                    segment_idx=index,
                    frame_idx=i,
                    region_name="Middle"
                )

                # --- Region 2: Bottom Right HUD (Partner) ---
                bottom_right_hud_frame = image_rgb.crop(BOTTOM_RIGHT_HUD)
                bottom_right_hud_inference = processor.set_image(bottom_right_hud_frame)
                bottom_right_hud_output = processor.set_text_prompt(state=bottom_right_hud_inference, prompt="Character.")
                bottom_right_hud_boxes = bottom_right_hud_output["boxes"]

                frame_partner_matches = process_detections(
                    boxes=bottom_right_hud_boxes,
                    source_frame=bottom_right_hud_frame,
                    db=db,
                    threshold=0.5,
                    segment_idx=index,
                    frame_idx=i,
                    region_name="BottomRight"
                )

                # Determine final context for this frame
                main_champ_name = None
                if frame_main_matches:
                    from collections import Counter
                    main_champ_name = Counter(frame_main_matches).most_common(1)[0][0]
                
                partner_names = []
                if frame_partner_matches:
                    partner_names = list(set(frame_partner_matches))
                
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
    print("PHASE 2 COMPLETE. Unloading SAM3 and Loading VLM...")
    print("="*50)
    
    del model
    del processor
    torch.cuda.empty_cache()
    
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
