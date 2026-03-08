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
quant_file = "gpt-oss-20b-F16.gguf"  # Adjust to available quant in repo

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
    n_ctx=16384,       # Context for Titan RTX
    verbose=False,

    #improving speed
    # n_batch=2048,
    # n_ubatch=512,
    # f16_kv=True,
    # n_threads=max(1, os.cpu_count() //2),  # Use half of available CPU cores for inference threads
    # n_threads_batch=max(1, os.cpu_count() //2)  # Use a quarter of available CPU cores for batch processing threads
)

knowledge_path = "/home/gatv-projects/Desktop/project/knowledge_extraction/cache/extracted_data/The_COMPLETE_SMOLDER_STARTER_GUIDE_-_League_of_Legends/kv_store_video_frames.json"

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

    Reasoning: low

    Given the VLM outputs describing the visual content of consecutive video segments, your task is to generate a description of the key events and actions.
    Focus on identifying the main champion(s) involved, their actions, and any significant interactions as well as gameplay events or game situations.
    Use the VLM output as your primary source of information, and infer the most likely gameplay events based on the visual cues provided. Your summary should be clear, informative, and capture the essence of the gameplay.
    
    <|channel|>analysis<|message|>User asks: "What is 2 + 2?" Simple arithmetic. Provide answer.<|end|>
    <|start|>assistant<|channel|>final<|message|>2 + 2 = 4.<|return|>
"""

# Compile the regex pattern once
# pattern = re.compile(r"<\|end\|><\|start\|>assistant<\|channel\|>final(?:<\|message\|>|\|assistant_output\|?|>| <\|constrain\|><\|assistant\|>)")
pattern = re.compile(r"final<\|message\|>")
    
chunk_size = 5
batch_number = 0
total_time = 0
total_tps = 0

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

    full_prompt = f"""<|start|>system<|message|>{system_prompt}<|end|><|start|>user<|message|>{prompt}<|end|>"""

    llm_start_time = time.time()
    # Generate Output
    output = llm(
        full_prompt,
        max_tokens=10000, 
        temperature=0.1,
        top_p=1.0,
        top_k=0,
        # stop=["User:"]
    )

    llm_end_time = time.time()
    elapsed = llm_end_time - llm_start_time

    #tokens accounting (llama-cpp-python provides this)
    usage = output.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", None)
    completion_tokens = usage.get("completion_tokens", None)
    total_tokens = usage.get("total_tokens", None)

    tps = (completion_tokens / elapsed) if (elapsed > 0 and completion_tokens is not None) else float("inf")

    print(f"Batch {batch_number} inference time: {elapsed:.2f} seconds")
    if completion_tokens is not None:
        print(f"Batch {batch_number} tokens: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")
        print(f"Batch {batch_number} speed: {tps:.2f} tokens/sec")
        total_tps += tps


    # 5. Use the pre-compiled regex pattern
    clean_output = output['choices'][0]['text'].strip()
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
average_tps = total_tps / batch_number if batch_number > 0 else 0
print(f"Average inference time for all batches: {average_time:.2f} seconds")
print(f"Average tokens/sec for all batches: {average_tps:.2f} tokens/sec")

print("Processing complete. Cleaning up...")
llm.close() 
del llm