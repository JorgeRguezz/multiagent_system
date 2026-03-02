# Test script for unsloth/gpt-oss-20b-GGUF using Transformers + llama.cpp backend
# Compatible with Titan RTX (Turing, no Hopper). GGUF loads via llama-cpp-python for quantization support.


import torch
from huggingface_hub import snapshot_download
import os
import re
import json
import time

# Config for Titan RTX 24GB: Use Q4_K_M (~12-14GB VRAM) or Q5_K_M
model_id = "unsloth/gpt-oss-20b-GGUF"
quant_file = "gpt-oss-20b-Q4_K_M.gguf"  # Adjust to available quant in repo

# Download model snapshot
model_path = snapshot_download(
    repo_id=model_id, 
    local_dir="/home/gatv-projects/Desktop/project/gpt-oss-20b",
    allow_patterns=[quant_file]    
)

full_path = os.path.join(model_path, quant_file)

print(f"Model path: {full_path}")
print(f"VRAM check: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB available")

# Load with llama-cpp-python (GGUF native)
from llama_cpp import Llama

llm = Llama(
    model_path=full_path,
    n_gpu_layers=-1,  # Offload all layers to GPU
    n_ctx=20000,       # Context for Titan RTX
    n_batch=512,
    f16_kv=True,      # FP16 KV cache
    verbose=False
)

knowledge_path = "/home/gatv-projects/Desktop/project/knowledge_extraction/cache/kv_store_video_frames.json"

with open(knowledge_path, "r") as f:
    knowledge_json = json.load(f)

# 1. Flatten all segments into a single, easy-to-manage list
all_segments = []
for video in knowledge_json.values():
    for segment in video.values():
        all_segments.append(segment["vlm_output"])

# 2. Define static variables OUTSIDE the loop for better performance
system_prompt = """
    You are an expert on summarizing League of Legends gameplay based on visual descriptions.
    Given the VLM outputs describing the visual content of consecutive video segments, your task is to generate a concise summary of the key events and actions.
    Focus on identifying the main champion(s) involved, their actions, and any significant interactions.
    Use the VLM output as your primary source of information, and infer the most likely gameplay events based on the visual cues provided. Your summary should be clear, informative, and capture the essence of the gameplay.
"""

# Compile the regex pattern once
pattern = re.compile(r"<\|end\|><\|start\|>assistant<\|channel\|>final(?:<\|message\|>|\|assistant_output\|?|>| <\|constrain\|><\|assistant\|>)")

chunk_size = 5
batch_number = 0
total_time = 0

# 3. Iterate through the list in chunks of 4 using Python slicing
for i in range(0, len(all_segments), chunk_size):
    # Grab 4 segments at a time (or however many are left at the end of the list)
    chunk = all_segments[i:i + chunk_size]
    
    # 4. Merge the 4 text descriptions into one big text block
    merged_content = "\n\n--- Next Segment ---\n\n".join(chunk)
    
    print(f"-----> Inferencing batch {batch_number} (contains {len(chunk)} segments) with LLM...")
    
    prompt = f""" 
    Based on the following VLM descriptions of a League of Legends gameplay sequence, describe what is happening overall.
    
    VLM Outputs of the merged segments:
    {merged_content}
    """

    full_prompt = f"""<s><|begin_system|>
    {system_prompt}
    <|begin_user|>
    {prompt}
    <|begin_assistant|>
    """
    llm_start_time = time.time()
    # Generate Output
    output = llm(
        full_prompt,
        max_tokens=10000, 
        temperature=0.7,
        top_p=0.9,
        stop=["User:"]
    )

    clean_output = output['choices'][0]['text'].strip()
    llm_end_time = time.time()
    print(f"Batch {batch_number} inference time: {llm_end_time - llm_start_time:.2f} seconds")

    # 5. Use the pre-compiled regex pattern
    if pattern.search(clean_output):
        response = pattern.split(clean_output, maxsplit=1)
        thought_process = response[0].strip()
        answer = response[1].strip()
        print("="*20, f" Final Response of batch {batch_number} ", "="*20)
        print(">>>>>>>>>> THOUGHT PROCESS <<<<<<<<<<")
        print(thought_process)
        print(">>>>>>>>>> ANSWER <<<<<<<<<<")
        print(answer)
        print("="*60)
    else:
        print(f"============== MALFORMED OUTPUT of batch {batch_number} ==============")
        print("="*10, "RAW MODEL RESPONSE", "="*10)
        print(clean_output) # Printed clean_output instead of the whole dict for readability
        print("="*30)

    batch_number += 1
    total_time += (llm_end_time - llm_start_time)

average_time = total_time / batch_number if batch_number > 0 else 0
print(f"Average inference time for all batches: {average_time:.2f} seconds")

print("Processing complete. Cleaning up...")
llm.close() 
del llm