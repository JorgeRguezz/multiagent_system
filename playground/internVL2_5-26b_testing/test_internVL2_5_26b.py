# ── Model response ──
# The current frame shows a dynamic battle scene in the bottom lane of a League of Legends game. Smolder, the champion in focus, is actively engaged in combat with an enemy champion near the jungle entrance. The intensity of the fight is evident from the visible spell effects, with Smolder using abilities that create colorful visual effects around the enemy. Smolder's health bar is partially depleted, indicating they are taking damage, while the enemy champion's health bar is also low, suggesting a back-and-forth exchange.

# The environment is a mix of jungle terrain and the bottom lane path, with a turret visible in the background. Minions from both sides are present, with blue minions (allied) closer to the turret and red minions (enemy) further away. The turret is intact, indicating that the lane is still contested.

# Smolder's abilities are currently in use, as indicated by the highlighted icons and active effects. The minimap in the bottom-right corner shows the positions of all players and key structures, providing strategic context for the ongoing battle. The health and mana bars at the bottom of the screen indicate that Smolder has a relatively healthy amount of both resources, with some mana used up from the recent ability usage.

# In contrast to the last frame, the current scene shows Smolder actively fighting rather than retreating. The enemy champion's position has shifted slightly, indicating movement and engagement. The intensity of the battle is higher, with visible spell effects and a more dynamic combat situation. The health bars at the top of the screen still show Smolder's team ahead, with a score of 7-2, but the focus is now on the immediate combat rather than a strategic retreat.

# The transcript context, which discusses Smolder's stacking passive and abilities, does not directly relate to the current frame, as the focus is on the ongoing battle rather than the mechanics of Smolder's abilities. The overall context remains a competitive match where Smolder's performance is crucial for the team's success.

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from torchvision.transforms.functional import InterpolationMode
import time

model_name = "OpenGVLab/InternVL2_5-26B"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
    llm_int8_skip_modules=["mlp1", "language_model.output"],
)

gpu_llm_layers = 28   # out of 48 total layers.

device_map = {}
device_map["vision_model"] = 0
device_map["mlp1"]         = 0
device_map["language_model.model.tok_embeddings"] = 0

for i in range(gpu_llm_layers):
    device_map[f"language_model.model.layers.{i}"] = 0

for i in range(gpu_llm_layers, 48):
    device_map[f"language_model.model.layers.{i}"] = "cpu"

device_map["language_model.model.norm"] = "cpu"
device_map["language_model.output"]     = "cpu"

start_model_load = time.time()
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print("Loading model (8-bit + CPU offload)...")
model = AutoModel.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    use_flash_attn=False,   # ← explicitly disable to avoid FA2 lookup
).eval()

vram_used  = torch.cuda.memory_allocated(0) / 1024**3
vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"\nVRAM used : {vram_used:.2f} GB / {vram_total:.2f} GB")
end_model_load = time.time()
print(f"Model loaded in {end_model_load - start_model_load:.2f} seconds")

# Image helpers
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def load_frame(path, input_size=448):
    img = Image.open(path).convert("RGB")
    return build_transform(input_size)(img).unsqueeze(0).to(torch.bfloat16).cuda()

# ── Test ──────────────────────────────────────────────────────────────────────
# Replace with your actual frame path
frame_path = "/home/gatv-projects/Desktop/project/knowledge_extraction/cache/frame_s2_i1.png"

print(f"\nLoading frame: {frame_path}")
start_frame_time = time.time()
pixel_values = load_frame(frame_path)
end_frame_time = time.time()
print(f"Frame loaded in {end_frame_time - start_frame_time:.2f} seconds")


# ---------------------PROMPT CONSTRUCTION (for VLM ASR Server)-----------------------

last_description = """
The video frame depicts a tense moment in a League of Legends game, specifically focusing on the bottom lane. In the center of the screen, the champion Smolder is engaged in combat with an enemy champion. Smolder, positioned near the jungle, is taking damage, indicated by the red health bar above their head. The enemy champion, partially obscured by the terrain, appears to be retreating, as suggested by the "Retreat" message in the chat. 
The environment shows a mix of jungle terrain and a path leading towards the turret. Minions from both sides are present, with some blue minions (allied) closer to the turret and some red minions (enemy) further away. The turret itself is visible in the background, indicating that this is the bottom lane. The health bars at the top of the screen show that Smolder's team is currently ahead, with a score of 7-2. 
Smolder’s abilities are not currently being used, as none of the ability icons are highlighted, suggesting they are on cooldown. The minimap in the bottom-right corner shows the positions of all players and key structures, providing strategic context for the ongoing battle. The health and mana bars at the bottom of the screen indicate that Smolder has a relatively healthy amount of both resources.
In the transcript, the speaker provides information about playing Smolder, describing him as an ADC similar to Ezreal, emphasizing his role as a caster rather than a pure auto attacker. The speaker also mentions that unlike Ezreal, Smolder does not require specific Korean challenger mechanics, making the game accessible to a broader audience. The transcript encourages viewers to try out the game without risk, offering a money-back guarantee if they do not rank up, and provides a link for further information.
The current scene captures a critical moment in the game, with Smolder under attack but still holding ground, suggesting a strategic retreat or a defensive stance. The overall context indicates a competitive match where Smolder's performance will be crucial for the team's success.
"""

context = {
  "champion": "Smolder",
  "teammates": ["Yasuo", "Morgana", "Warwick", "Velkoz"],
  "transcript": "[0.00s -> 3.08s]  You can definitely take chances and look for good fights.[3.08s -> 6.12s]  Not only that, but the stacking process can actually be quite enjoyable.[6.12s -> 10.04s]  You will be deleting waves quickly while powering up in a very satisfying way, so it won't[10.04s -> 12.68s]  feel like a chore as it does on other stacking champions.[12.68s -> 16.28s]  That being said, stacking efficiently every game is without a doubt the most important[16.28s -> 17.60s]  skill to mastering Smolder.[17.60s -> 20.92s]  So towards the end of the guide, we'll give you some quick and easy tips to make stacking[20.92s -> 22.28s]  a breeze every game.[22.28s -> 24.32s]  For now though, let's break down his abilities.[24.32s -> 28.24s]  Smolder has a stacking passive, based on how many stacks he has, his three basic abilities[28.24s -> 29.80s]  will simply deal more damage."
}


prompt_text = f"""
You are a vision-language analyst for League of Legends gameplay.

**Champion:** {context.get('champion', 'None')}
**Teammates:** {", ".join(context['teammates'])}
**Transcript:** {context.get('transcript', 'No speech detected.')}

---

**YOUR TASK:**
Describe the CURRENT frame in detail. Then, contrast it with the last frame description to identify what has changed.

**CURRENT FRAME — Describe the following:**

1. **Position**: Location relative to map landmarks (river, jungle, towers).
2. **Actions**: Intensity (idle/fighting/retreating) and any visible spell/ability effects.
3. **Health/Mana**: Approximate bar states. Red = enemy, blue = ally, green = player.
4. **Map Context**: Allied (blue) or enemy (red) minions and towers visible.
5. **Abilities/Items**: Cooldown states, active effects, summoner spell availability.
6. **Transcript Integration**: What audio context reveals that the image doesn't show.

**WHAT HAS CHANGED SINCE LAST FRAME:**
Compare only — do not repeat. Note new positions, health changes, ability usage, or champion entries/exits.
If nothing meaningful has changed, state that explicitly.

---

**LAST FRAME (for delta comparison only — do not repeat or paraphrase):**
{last_description}

---

**Rules:**
- Do NOT repeat or paraphrase content from the last frame description.
- Do NOT invent details not visible in the current frame.
- Do NOT over-describe spell animations; categorize intensity only.
- Write in continuous descriptive paragraphs, no subheadings or bullet points.

**Scene description:**
"""
print("·"*80)
print(f"\n[DEBUG] Prompt for\n{prompt_text}")
print("·"*80)
# ---------------------END PROMPT CONSTRUCTION (for VLM ASR Server)-----------------------

start_inference = time.time()
question = f"<image>\n{prompt_text}"
response, _ = model.chat(
    tokenizer,
    pixel_values,
    question,
    generation_config=dict(max_new_tokens=512, do_sample=False),
    history=None,
    return_history=True,
)
end_inference = time.time()
print(f"\nInference completed in {end_inference - start_inference:.2f} seconds")
print("\n── Model response ──")
print(response)
print("────────────────────")
print("Time breakdown:")
print(f"  Model loading: {end_model_load - start_model_load:.2f} seconds")
print(f"  Frame loading: {end_frame_time - start_frame_time:.2f} seconds")
print(f"  Inference: {end_inference - start_inference:.2f} seconds")    
print(f"  Total: {end_inference - start_model_load:.2f} seconds")
 


