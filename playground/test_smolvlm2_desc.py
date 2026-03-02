from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from PIL import Image

# 1. Setup
current_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"------------> Initializing on device: {current_device}")

model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
).to(current_device)


def run_inference(model, processor, messages, current_device):
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=500)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return generated_texts[0]


# 2. Image Test
print("--- Image Test ---")
image_path = "/home/gatv-projects/Desktop/project/playground/sam3_testing/vlm_context_test_cache/full_frame_seg3_idx3.png"
image = Image.open(image_path).convert("RGB")


context = """
{
  "champion": "Smolder",
  "teammates": ["Yasuo", "Morgana", "Warwick", "Velkoz"],
  "transcript": "[0.00s -> 3.08s]  You can definitely take chances and look for good fights.[3.08s -> 6.12s]  Not only that, but the stacking process can actually be quite enjoyable.[6.12s -> 10.04s]  You will be deleting waves quickly while powering up in a very satisfying way, so it won't[10.04s -> 12.68s]  feel like a chore as it does on other stacking champions.[12.68s -> 16.28s]  That being said, stacking efficiently every game is without a doubt the most important[16.28s -> 17.60s]  skill to mastering Smolder.[17.60s -> 20.92s]  So towards the end of the guide, we'll give you some quick and easy tips to make stacking[20.92s -> 22.28s]  a breeze every game.[22.28s -> 24.32s]  For now though, let's break down his abilities.[24.32s -> 28.24s]  Smolder has a stacking passive, based on how many stacks he has, his three basic abilities[28.24s -> 29.80s]  will simply deal more damage."
}
"""

# prompt = f"""# TASK:
# Describe the main activity in this League of Legends gameplay screenshot using the following provided transcript as context. Extract all valuable information from the transcript since the final user won't be able to see it. 

# # INSTRUCTIONS:
# 1. **Visual Effects:** Describe dominant spell effects visible if they are any. Match them to the champion abilities if mentioned in the transcript context.
# 2. **Intensity:** State if this appears to be a large team fight (many health bars/effects) or a minor skirmish.
# 3. **Speech Context:** Use the provided transcript to help you understand the context of the image.
# 4. **Transcript:** Extract the valuable information from the transcript that provides gameplay/game tactics information that the user wants to know.

# # CONTEXT (Champions & Transcript):
# {context}

# # CONSTRAINT:
# - Do NOT guess champion positions or specific interactions if they are unclear.
# - Do NOT invent a narrative. Describe only the visible frame supporting yourself in the provided context.
# """

prompt = f"""
**System Role:**  
You are a vision-language analyst for League of Legends gameplay. Your job is to create a **detailed, comprehensive textual description** of the provided video frame + context + transcript.  

**Purpose:**  
Generate a rich natural language summary that captures *every meaningful visual detail*, *contextual relationships*, and *transcript insights* so a downstream knowledge graph system can extract entities, events, and relationships.  

**Input Data:**  
- **Visual Frame**  
- **Context:** {context}

**Output Instructions:**  

Write a **detailed scene description** covering:  

### 1. **Visual Scene Analysis** (most important).
- **Position**: Where the action is happening relative to important map landmarks (e.g. towers/statues, river, jungle).  
- **Actions/Animations**: Categorize the intensity of the action (e.g., idle, fighting) and identify any visible spell effects or animations.  
- **Health/Mana**: Approximate bars visible, who is low HP, who is full.
- **Map Context**: Indicate if there are any allied (blue) or enemy (red) minions and any allied (blue highlight) or enemy (red highlight) towers/statues visible. 
- **Items/Effects**: Active items, summoner spells, champion abilities on cooldown.  

### 2. **Transcript Integration**  
- What the transcript reveals about intentions, strategies, or events not visible.  
- What the transcript reveals about gameplay information about players abilities, item builds, or team compositions that can be inferred from the transcript but not directly seen in the image.  

### 3. **Temporal Context**  
- What seems to be happening RIGHT NOW  
- What likely happened 2-3 seconds ago  
- What appears to be the next likely action  

### 4. **CONSTRAINT:**
- Do NOT guess champion positions or specific interactions if they are unclear.
- Do NOT invent a narrative. Describe only the visible frame supporting yourself in the provided context.
- Do NOT mention the knowledge graph or downstream extraction system.

**Format your output as continuous descriptive text.**   

**Write the detailed scene description below:**
"""

messages_with_context = [
    {

        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
    },
]

messages_no_context = [
    {

        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe the image in detail."}
        ]
    },
]

context_answer = run_inference(model, processor, messages_with_context, current_device)
no_context_answer = run_inference(model, processor, messages_no_context, current_device)

if "Assistant:" in context_answer:
    context_answer = context_answer.split("Assistant:")[1].strip()
if "Assistant:" in no_context_answer:
    no_context_answer = no_context_answer.split("Assistant:")[1].strip()

print(">>> Without Context:")
print(no_context_answer)
print("\n>>> With Context:")
print(context_answer)


# # 3. Video Test
# print("\n--- Video Test ---")
# video_path = "/home/gatv-projects/Desktop/project/chatbot_system/downloads/My Nintendo Switch 2 Review.mp4"

# messages_video = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "video", "path": video_path},
#             {"type": "text", "text": "Describe this video in detail"}
#         ]
#     },
# ]

# decoded_text_video = run_inference(model, processor, messages_video, current_device)
# vlm_response_parts = decoded_text_video.split("Assistant:")
# vlm_response_clean = vlm_response_parts[1].strip()
# print(">>> VLM response:")
# print(vlm_response_clean)
