import re
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import BitsAndBytesConfig  # NEW
from collections import Counter


torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# # 4-bit quantization config (bitsandbytes)
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",          # good default
#     bnb_4bit_use_double_quant=True,     # extra compression, usually fine
# )

# 8-bit quantization config (bitsandbytes)
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,              # default
    llm_int8_has_fp16_weight=False,     # default
)

model_id = "ServiceNow-AI/Apriel-1.6-15b-Thinker"

# Load model in 4-bit
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    quantization_config=bnb_config,     # NEW
    # device_map="auto",                  # let HF place modules on GPU
    # max_memory={0: "22GiB", "cpu": "16GiB"},  
    device_map={"":0},                   # force to single GPU 0
    torch_dtype=torch.bfloat16,         # dtype for non-quant parts
)

processor = AutoProcessor.from_pretrained(model_id)


system_prompt = """
    You are a concise, domain-aware assistant.
    For this task: 
    - Answer directly without showing your reasoning steps.
    - Keep the output under 150 words.
    - Write single cohesive paragraphs without bullet points.
    **IMPORTANT**: Be aware that you only have 200 tokens to answer, this includes any special tags (like [BEGIN FINAL RESPONSE] and <|end|>), though tokens and actual output tokens. If you don't keep the sum of these under 200 tokens (about 900 characters) your answer will get chopped. 
"""

# Example 1: Text-only prompt

video_game_name = "League of Legends"

prompt = f"""### INSTRUCTION
    You are a Context Generator for an Entity Extraction AI.
    Your goal is to provide a **concise, keyword-focused summary** of the video game "{video_game_name}".

    The output will be used to help an AI understand proper nouns and specific terminology in text.
    You must IGNORE release dates, developers, sales history, reviews, or graphical fidelity.

    ### CONTENT REQUIREMENTS
    Focus ONLY on these three categories:
    1.  **Setting & Premise:** The fictional world, time period, and main conflict.
    2.  **Key Factions & Figures:** Major protagonists, antagonists, and groups (e.g., "The Empire", "Mario", "Ganon").
    3.  **Unique Vocabulary:** Specific names for currency, magic, items, or mechanics (e.g., "Rupees", "Mana", "Drifting", "Fatality").

    ### FORMATTING RULES
    - Keep the total output under 150 words.
    - Use dense, informative sentences.
    - Do not use bullet points; write as a cohesive summary paragraph.

    ### EXAMPLE (Pac-Man)
    Pac-Man is an arcade maze game set in a neon labyrinth. The protagonist, Pac-Man, must navigate the maze to eat dots and "Power Pellets" while avoiding four ghosts: Blinky, Pinky, Inky, and Clyde. Consuming a Power Pellet turns the ghosts blue, allowing Pac-Man to eat them for points. Fruit items occasionally appear as bonuses.

    ### INPUT
    Target Game: {video_game_name}

    ### OUTPUT
    """


full_prompt = f"""<s><|begin_system|>
{system_prompt}
<|begin_user|>
{prompt}
<|begin_assistant|>
"""

# task_prompt = f"""You are a context generator for an entity extraction AI.

# Write a single concise paragraph (max 150 words) that helps another AI understand
# the world, key factions/characters, and special vocabulary of the video game "{video_game_name}".

# Follow these rules:
# - Focus ONLY on: (1) setting & premise, (2) key factions & figures, (3) unique vocabulary (currency, magic, items, mechanics).
# - Ignore release date, developer, reviews, graphics, sales.
# - No bullet points. One cohesive paragraph. No explanation of your reasoning.
# """


# chat = [
#     {
#         "role": "system",
#         "content": [
#             {"type": "text", "text": system_prompt},
#         ],
#     },
#     {
#         "role": "user",
#         "content": [
#             {"type": "text", "text": prompt},
#         ],
#     }
# ]

# inputs = processor.apply_chat_template(
#     chat,
#     add_generation_prompt=True,
#     tokenize=True,
#     return_dict=True,
#     return_tensors="pt",
# )

inputs = processor(
    text = full_prompt,
    return_tensors="pt",
).to(model.device)

# inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
inputs.pop("token_type_ids", None)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=600,             # reduce from 1024 for speed
        do_sample=False                # deterministic for testing
    )

generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
output = processor.decode(generated_ids[0], skip_special_tokens=True)
if "[BEGIN FINAL RESPONSE]" in output:
    response = re.findall(r"\[BEGIN FINAL RESPONSE\](.*?)(?:<\|end\|>)", output, re.DOTALL)[0].strip()
else :
    response = "MALFORMED OUTPUT: Missing [BEGIN FINAL RESPONSE] tag."

print("--"*20, "\n", Counter(p.device for _, p in model.named_parameters()), "\n", "--"*20)
print("Full thinking process:\n")
print(output)
print("="*20, "\n")
print("Text-only Response:", response)

