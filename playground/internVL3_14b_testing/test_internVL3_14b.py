import os, time
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# SDPA toggles (won't hurt)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.set_grad_enabled(False)

model_path = "OpenGVLab/InternVL3-14B-hf"
quant_config = BitsAndBytesConfig(load_in_8bit=True)

processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    quantization_config=quant_config,
    device_map="cuda",
    dtype=torch.float16,
    attn_implementation="sdpa",
).eval()

frame_paths = [
    "/home/gatv-projects/Desktop/project/knowledge_extraction/cache/frame_s1_i3.png",
    "/home/gatv-projects/Desktop/project/knowledge_extraction/cache/frame_s1_i4.png",
    "/home/gatv-projects/Desktop/project/knowledge_extraction/cache/frame_s2_i0.png",
    "/home/gatv-projects/Desktop/project/knowledge_extraction/cache/frame_s2_i1.png",
]

images = [Image.open(p).convert("RGB") for p in frame_paths]

context = {
    "champion": "Smolder",
    "teammates": ["Yasuo", "Morgana", "Warwick", "Velkoz"],
    "transcript": "[0.00s -> 3.08s]  You can definitely take chances and look for good fights.[3.08s -> 6.12s]  Not only that, but the stacking process can actually be quite enjoyable.[6.12s -> 10.04s]  You will be deleting waves quickly while powering up in a very satisfying way, so it won't[10.04s -> 12.68s]  feel like a chore as it does on other stacking champions.[12.68s -> 16.28s]  That being said, stacking efficiently every game is without a doubt the most important[16.28s -> 17.60s]  skill to mastering Smolder.[17.60s -> 20.92s]  So towards the end of the guide, we'll give you some quick and easy tips to make stacking[20.92s -> 22.28s]  a breeze every game.[22.28s -> 24.32s]  For now though, let's break down his abilities.[24.32s -> 28.24s]  Smolder has a stacking passive, based on how many stacks he has, his three basic abilities[28.24s -> 29.80s]  will simply deal more damage.",
}

prompt_text = f"""
You are a vision-language analyst for League of Legends gameplay.

**Player champion:** {context.get('champion', 'None')}
**Teammates:** {", ".join(context['teammates'])}
**Transcript:** {context.get('transcript', 'No speech detected.')}

---

**YOUR TASK:**
You will be given multiple frames from the SAME video segment. Provide ONE coherent, segment-level description that integrates visual details across all frames and the transcript.

**Focus on:**
1. Overall location and map context across the segment.
2. Actions and combat intensity across frames.
3. Health/mana trends or major changes across the segment.
4. Abilities, items, or events that persist across multiple frames.
5. Transcript integration for tactical meaning.

**Rules:**
- Do NOT describe each frame separately.
- Do NOT speculate beyond what the frames show.
- If frames conflict, explain the change briefly and summarize the segment.

**Segment description:**
"""

# Multi-image chat-style message (HF InternVL supports multiple images in one turn) :contentReference[oaicite:1]{index=1}
messages = [
    {
        "role": "user",
        "content": (
            [{"type": "image", "image": im} for im in images]
            + [{"type": "text", "text": prompt_text}]
        )
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.float16)

gen_kwargs = dict(max_new_tokens=256, do_sample=False, temperature=1.0, use_cache=False)

start = time.time()
out = model.generate(**inputs, **gen_kwargs)
print(f"Inference time: {time.time()-start:.2f}s")

# Decode only the newly generated tokens
decoded = processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(decoded)