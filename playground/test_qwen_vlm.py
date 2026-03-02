import os
import time

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# 1. Model Setup
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

quant_config = BitsAndBytesConfig(load_in_8bit=True)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=quant_config,
    torch_dtype="float16",
    device_map="cuda",
)

processor = AutoProcessor.from_pretrained(model_id)

# 2. Data Preparation (segment-level)
context = {
    "champion": "Smolder",
    "teammates": ["Yasuo", "Morgana", "Warwick", "Velkoz"],
    "transcript": "[0.00s -> 3.08s]  You can definitely take chances and look for good fights.[3.08s -> 6.12s]  Not only that, but the stacking process can actually be quite enjoyable.[6.12s -> 10.04s]  You will be deleting waves quickly while powering up in a very satisfying way, so it won't[10.04s -> 12.68s]  feel like a chore as it does on other stacking champions.[12.68s -> 16.28s]  That being said, stacking efficiently every game is without a doubt the most important[16.28s -> 17.60s]  skill to mastering Smolder.[17.60s -> 20.92s]  So towards the end of the guide, we'll give you some quick and easy tips to make stacking[20.92s -> 22.28s]  a breeze every game.[22.28s -> 24.32s]  For now though, let's break down his abilities.[24.32s -> 28.24s]  Smolder has a stacking passive, based on how many stacks he has, his three basic abilities[28.24s -> 29.80s]  will simply deal more damage."
}

# Frames representing a single segment
frame_paths = [
    "/home/gatv-projects/Desktop/project/knowledge_extraction/cache/frame_s1_i3.png",
    "/home/gatv-projects/Desktop/project/knowledge_extraction/cache/frame_s1_i4.png",
    "/home/gatv-projects/Desktop/project/knowledge_extraction/cache/frame_s2_i0.png",
    "/home/gatv-projects/Desktop/project/knowledge_extraction/cache/frame_s2_i1.png",
]

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
- Write in continuous descriptive paragraphs, no subheadings or bullet points.

**Segment description:**
"""

# Load images
images = []
for path in frame_paths:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing frame: {path}")
    images.append(Image.open(path).convert("RGB"))

messages = [
    {
        "role": "user",
        "content": ([{"type": "image", "image": img} for img in images] + [{"type": "text", "text": prompt_text}]),
    }
]

# 3. Inference Processing (segment-level)
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to("cuda")

print(f"Running segment inference on {len(frame_paths)} frames...")
start_time = time.time()

generated_ids = model.generate(**inputs, max_new_tokens=512)

end_time = time.time()
print(f"Inference time: {end_time - start_time:.2f} seconds")

# Trim prefix and decode
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

response = output_text[0]
print("\n--- SEGMENT RESPONSE ---")
print(response)
