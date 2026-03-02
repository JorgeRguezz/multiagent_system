
import torch
import os
from transformers import (
    Gemma3ForConditionalGeneration, 
    Gemma3Processor,
    BitsAndBytesConfig
    # CHANGE 1: Removed TrainingArguments (unused, adds overhead)
)
from PIL import Image

# CHANGE 2: More aggressive mem config + clear cache multiple times
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# CHANGE 3: Sequential loading - processor first, then model
print("Loading processor...")
model_id = "google/gemma-3-27b-it"
processor = Gemma3Processor.from_pretrained(model_id)

# CHANGE 4: Max aggressive 4bit + strict max_memory
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading model with ultra-low VRAM...")
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=quant_config,
    torch_dtype=torch.float16,
    device_map={"": 0},  # CHANGE 5: Force single GPU, no auto
    low_cpu_mem_usage=True,
    max_memory={0: "21.5GiB", "cpu": "64GiB"},  # Tighter GPU limit
    offload_folder="./offload"
    
)

# CHANGE 6: Skip gradient_checkpointing (causes OOM during load for inference)
# model.gradient_checkpointing_enable()  # COMMENTED OUT

torch.cuda.empty_cache()  # Post-load cleanup
print(f"✅ Model loaded! VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

# YOUR EXACT PROMPT STRUCTURE - UNCHANGED
def generate_vlm(image_path):
    image = Image.open(image_path)
    
    context = {
    "champion": "Smolder",
    "teammates": ["Yasuo", "Morgana", "Warwick", "Velkoz"],
    "transcript": "[0.00s -> 3.08s] You can definitely take chances and look for good fights..."
    }

    # 3. Prompt Construction
    prompt_text = f"""
    **System Role:**  
    You are a vision-language analyst for League of Legends gameplay. Your job is to create a **detailed, comprehensive textual description** of the provided video frame + context + transcript.  

    **Purpose:**  
    Generate a rich natural language summary that captures *every meaningful visual detail*, *contextual relationships*, and *transcript insights* so a downstream knowledge graph system can extract entities, events, and relationships.  

    **Context:**
    - Champion: {context['champion']}
    - Teammates: {", ".join(context['teammates'])}
    - Transcript: {context['transcript']}

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

    # FIX: Use the message list format.
    # This allows the tokenizer to handle the exact location and format of the image token.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    
    # STEP 1: Generate the raw text prompt with the CORRECT image token
    # We set tokenize=False so we get a string we can verify/debug if needed
    prompt = processor.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # STEP 2: Process the text and image together
    # Now the prompt string definitely contains the token the processor looks for
    inputs = processor(
        text=prompt, 
        images=image, 
        return_tensors="pt"
    )
    
    # STEP 3: Move to CUDA cleanly
    # The processor returns a 'BatchFeature' object which has a .to() method
    inputs = inputs.to("cuda")
    
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.1,
            do_sample=True,
            pad_token_id=processor.tokenizer.eos_token_id
        )

    # Decode response
    response = processor.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response

# Test - YOUR EXACT CALL
image_path = "/home/gatv-projects/Desktop/project/playground/sam3_testing/vlm_context_test_cache/full_frame_seg3_idx3.png"
result = generate_vlm(image_path)
print("🎮 Gemma-3-27B Output:")
print(result)
print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
torch.cuda.empty_cache()
 