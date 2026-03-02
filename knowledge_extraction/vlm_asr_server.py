"""
MCP server for multi-modal analysis, handling Whisper ASR for transcription 
and Qwen2.5-VL or InternVL3-14B for detailed scene description.
"""
import os
import sys
import contextlib
import torch
import torchvision.transforms as T

# Ensure models are loaded from the correct cache
from knowledge_extraction.config import HF_HOME
# os.environ["HF_HOME"] = HF_HOME
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from transformers import ( 
    AutoProcessor, 
    AutoModel, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
# from transformers import Qwen2_5_VLForConditionalGeneration
# from qwen_vl_utils import process_vision_info
from mcp.server.fastmcp import FastMCP
import whisper

mcp = FastMCP("vlm_asr_server")

# Qwen Config
QWEN_MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
vlm_model = None
vlm_processor = None

# InternVL Config
INTERNVL_MODEL_PATH = "OpenGVLab/InternVL3-14B"
internvl_model = None
internvl_tokenizer = None

# Shared Config
asr_model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------- INTERNVL HELPERS -----------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        processed_images.append(resized_img.crop(box))
    
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images

def load_internvl_image(image_path, input_size=448, max_num=12):
    image = Image.open(image_path).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)

# --------------------- MODEL LOADERS -----------------------

def load_vlm():
    """Loads Qwen2.5-VL and unloads InternVL if necessary."""
    global vlm_model, vlm_processor, internvl_model, internvl_tokenizer
    if internvl_model is not None:
        print("--> VLM Server: Unloading InternVL to free memory...", file=sys.stderr)
        internvl_model = None
        internvl_tokenizer = None
        torch.cuda.empty_cache()
        
    if vlm_model is None:
        print(f"--> VLM Server: Loading {QWEN_MODEL_PATH}...", file=sys.stderr)
        with contextlib.redirect_stdout(sys.stderr):
            vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                QWEN_MODEL_PATH, 
                torch_dtype="float16", 
                device_map="cuda"
            )
            vlm_processor = AutoProcessor.from_pretrained(QWEN_MODEL_PATH)
        print("--> VLM Server: Qwen Loaded ✅", file=sys.stderr)

def load_internvl():
    """Loads InternVL3-14B and unloads Qwen if necessary."""
    global internvl_model, internvl_tokenizer, vlm_model, vlm_processor
    if vlm_model is not None:
        print("--> VLM Server: Unloading Qwen to free memory...", file=sys.stderr)
        vlm_model = None
        vlm_processor = None
        torch.cuda.empty_cache()

    if internvl_model is None:
        print(f"--> VLM Server: Loading {INTERNVL_MODEL_PATH}...", file=sys.stderr)
        with contextlib.redirect_stdout(sys.stderr):
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            internvl_model = AutoModel.from_pretrained(
                INTERNVL_MODEL_PATH,
                quantization_config=quant_config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
                device_map="cuda"
            ).eval()
            internvl_tokenizer = AutoTokenizer.from_pretrained(INTERNVL_MODEL_PATH, trust_remote_code=True)
            internvl_model.img_context_token_id = internvl_tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        print("--> VLM Server: InternVL Loaded ✅", file=sys.stderr)

def load_asr():
    global asr_model
    if asr_model is None:
        print("--> VLM Server: Loading Whisper (base)...", file=sys.stderr)
        with contextlib.redirect_stdout(sys.stderr):
            asr_model = whisper.load_model("base", device=device)
        print("--> VLM Server: ASR Loaded ✅", file=sys.stderr)

def unload_models():
    """Free VLM/ASR models to release GPU memory."""
    global vlm_model, vlm_processor, internvl_model, internvl_tokenizer, asr_model
    if vlm_model is not None or vlm_processor is not None:
        print("--> VLM Server: Unloading Qwen VLM...", file=sys.stderr)
        vlm_model = None
        vlm_processor = None
    if internvl_model is not None or internvl_tokenizer is not None:
        print("--> VLM Server: Unloading InternVL...", file=sys.stderr)
        internvl_model = None
        internvl_tokenizer = None
    if asr_model is not None:
        print("--> VLM Server: Unloading ASR...", file=sys.stderr)
        asr_model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# --------------------- TOOLS -----------------------

@mcp.tool()
async def transcribe_audio(audio_path: str) -> str:
    """Transcribes an audio file using Whisper."""
    load_asr()
    if not os.path.exists(audio_path):
        return ""
    result = asr_model.transcribe(audio_path)
    return "".join([f"[{s['start']:.2f}s -> {s['end']:.2f}s] {s['text']}" for s in result["segments"]])

@mcp.tool()
async def unload_vlm_asr() -> str:
    """Unload VLM and ASR models to free GPU memory."""
    unload_models()
    return "unloaded"

@mcp.tool()
async def run_qwen_description(image_path: str, context: dict, last_description: str) -> str:
    """
    Generates a detailed description of an image using Qwen2.5-VL with provided context.
    context: {"champion": str, "teammates": list, "transcript": str}
    """
    load_vlm()
    if not os.path.exists(image_path):
        return "Error: Image not found."

    image = Image.open(image_path).convert("RGB")
    partners_str = ", ".join(context.get("teammates", []))

    prompt_text = f"""
    **System Role:**  
    You are a vision-language analyst for League of Legends gameplay. Your job is to create a **detailed, comprehensive textual description** of the provided video frame + context + transcript.  

    **Purpose:**  
    Generate a rich natural language summary that captures *every meaningful visual detail*, *contextual relationships*, and *transcript insights* so a downstream knowledge graph system can extract entities, events, and relationships.  

    **Context:**
    - Champion: {context.get('champion', 'None')}
    - Teammates: {partners_str if partners_str else 'None'}
    - Transcript: {context.get('transcript', 'No speech detected.')}

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

    text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = vlm_processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(vlm_model.device)

    with contextlib.redirect_stdout(sys.stderr):
        generated_ids = vlm_model.generate(
            **inputs,
            max_new_tokens=512, 
            do_sample=True,
            temperature=0.4
            )
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = vlm_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
    
    return output_text

@mcp.tool()
async def run_internvl_description(image_path: str, context: dict, last_description: str) -> str:
    """
    Generates a detailed description of an image using InternVL3-14B with provided context.
    context: {"champion": str, "teammates": list, "transcript": str}
    """
    load_internvl()
    if not os.path.exists(image_path):
        return "Error: Image not found."

    if last_description is None or last_description == "":
        last_frame_context = "This is the FIRST frame of the sequence. There is no prior context. Focus on providing a comprehensive initial description."
    else:
        last_frame_context = f"LAST FRAME (for delta comparison only — do not repeat or paraphrase):\n{last_description}"

    partners_str = ", ".join(context.get("teammates", []))

    prompt_text_alt = f"""
    You are a specialized League of Legends gameplay analyst. Your goal is to extract actionable data for a knowledge graph.

    **Context:**
    - Player: {context.get('champion', 'None')} | Teammates: {partners_str if partners_str else 'None'}
    - Transcript Context: {context.get('transcript', 'No transcript available.')}

    ---

    **YOUR TASK:**
    Provide a high-density description of the CURRENT frame. 

    **STRICT PRIORITY: Start with WHAT HAS CHANGED since the last frame.**
    1. **Dynamic Events**: Describe new banners (e.g., "LEVEL UP!"), gold numbers (e.g., "+274"), and kill notifications in the chat.
    2. **Action-State**: Detail the current intensity (e.g., "fighting in the bottom lane"). Identify the specific ability being cast (e.g., "MMOOOMMMM!") and its target.
    3. **Player State**: Report current Level, Health %, and Mana % from the HUD.
    4. **Map Logic**: Describe the movement of minions or champions relative to map landmarks.
    5. **Transcript Tie-In**: Connect the current action to the transcript's tactical advice or facts.

    **Rules:**
    - DO NOT repeat background descriptions (e.g., "in the jungle") if they were mentioned in the last frame.
    - DO NOT paraphrase the transcript; integrate it as strategic context for the VISUAL action.

    ---

    {last_frame_context}

    **Scene description:**
    """

    # prompt_text_alt = f"""
    # You are a specialized League of Legends gameplay analyst. Your goal is to extract visual data from this frame to build a knowledge graph.

    # **CONTEXT:**
    # - Player Champion: {context.get('champion', 'None')} (Green health bar)
    # - Teammate Champions: {partners_str if partners_str else 'None'} (Blue health bars)
    # - Enemies: Red health bars, red minions, red structures.
    # - Transcript Context (~30s window): "{context.get('transcript', 'No transcript available.')}"

    # **PREVIOUS FRAME (5 seconds ago):**
    # {last_frame_context}

    # ---

    # **YOUR TASK:**
    # Provide a high-density, structured description of the CURRENT frame. Focus strictly on what is visually verifiable.

    # **OUTPUT FORMAT:**
    # Use these exact headers for your response:

    # ### 1. WHAT CHANGED (Delta)
    # List new dynamic events since the last frame. Look for kill feed notifications (top right), banners (e.g., "Double Kill", "Level Up"), floating gold numbers, or sudden drops in health bars.

    # ### 2. CURRENT ACTION
    # Describe the immediate gameplay intensity (e.g., "skirmishing in the jungle", "farming minions"). Note specific visual actions like casting projectiles, dashing, retreating, or auto-attacking. 

    # ### 3. PLAYER STATE
    # Read the HUD. Report the Player's current Level, Health value, and Resource/Mana value.

    # ### 4. MAP & POSITIONING
    # Where is the action happening? Describe the movement of the player, visible enemies (red bars), and minions relative to map landmarks (turrets, river, lanes). Do not repeat static background info from the previous frame.

    # ### 5. TRANSCRIPT TIE-IN
    # Briefly connect the visual action to the transcript context. Does the visual confirm the transcript's tactical advice or callouts or valuable champion information?

    # **RULES:**
    # - If you cannot identify an enemy champion's exact name, simply call them "enemy champion". Do not guess.
    # - Do not paraphrase the transcript; use it only to explain the strategic context of the visual action.
    # - Keep sentences concise and highly factual.

    # **Scene description:**
    # """


    # 1. Load and process the image into InternVL's required tensor format
    pixel_values = load_internvl_image(image_path, max_num=12).to(torch.bfloat16).cuda()
    
    # 2. Generation config
    generation_config = dict(max_new_tokens=512, do_sample=False, temperature=1.0)
    
    # 3. Use the built-in model.chat()
    with contextlib.redirect_stdout(sys.stderr):
        response = internvl_model.chat(internvl_tokenizer, pixel_values, prompt_text_alt, generation_config)
    
    return response 

if __name__ == "__main__":
    mcp.run(transport='stdio')
