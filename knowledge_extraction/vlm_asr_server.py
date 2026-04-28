"""
MCP server for multi-modal analysis, handling Whisper ASR for transcription 
and Qwen2.5-VL or InternVL3-14B for detailed scene description.
"""
import os
import sys
import contextlib
import torch
import torchvision.transforms as T


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
from knowledge_pipeline.game_profiles import get_active_game_profile

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


def _get_profile_for_context(context: dict):
    return get_active_game_profile(context.get("game"))


def _build_qwen_description_prompt(context: dict) -> str:
    profile = _get_profile_for_context(context)
    transcript = context.get("transcript", "No speech detected.")

    if profile.id == "league_of_legends":
        partners_str = ", ".join(context.get("teammates", []))
        context_block = (
            f"- Champion: {context.get('champion', 'None')}\n"
            f"- Teammates: {partners_str if partners_str else 'None'}\n"
            f"- Transcript: {transcript}"
        )
    else:
        entities_str = ", ".join(context.get("entities", []))
        context_block = (
            f"- Visible entities: {entities_str if entities_str else 'None'}\n"
            f"- Transcript: {transcript}"
        )

    return (
        f"{profile.vlm_prompt_profile.qwen_description_prompt}\n\n"
        f"Context:\n{context_block}\n\n"
        "Write the detailed scene description below:"
    )


def _build_internvl_description_prompt(context: dict, last_description: str) -> str:
    profile = _get_profile_for_context(context)
    transcript = context.get("transcript", "No transcript available.")

    if last_description:
        last_frame_context = (
            "LAST FRAME (for delta comparison only — do not repeat or paraphrase):\n"
            f"{last_description}"
        )
    else:
        last_frame_context = (
            "This is the FIRST frame of the sequence. There is no prior context. "
            "Focus on providing a comprehensive initial description."
        )

    if profile.id == "league_of_legends":
        partners_str = ", ".join(context.get("teammates", []))
        context_block = (
            f"- Player: {context.get('champion', 'None')} | "
            f"Teammates: {partners_str if partners_str else 'None'}\n"
            f"- Transcript Context: {transcript}"
        )
    else:
        entities_str = ", ".join(context.get("entities", []))
        context_block = (
            f"- Visible entities: {entities_str if entities_str else 'None'}\n"
            f"- Transcript Context: {transcript}"
        )

    return (
        f"{profile.vlm_prompt_profile.internvl_description_prompt}\n\n"
        f"Context:\n{context_block}\n\n"
        "---\n\n"
        f"{last_frame_context}\n\n"
        "Scene description:"
    )


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
    context: profile-aware payload with transcript and optional game-specific entity cues.
    """
    load_vlm()
    if not os.path.exists(image_path):
        return "Error: Image not found."

    image = Image.open(image_path).convert("RGB")
    prompt_text = _build_qwen_description_prompt(context)

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
    context: profile-aware payload with transcript and optional game-specific entity cues.
    """
    load_internvl()
    if not os.path.exists(image_path):
        return "Error: Image not found."

    prompt_text_alt = _build_internvl_description_prompt(context, last_description)

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
