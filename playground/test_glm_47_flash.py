import torch
import os 
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import time

# 1. Download the quantized GGUF model
model_repo = "unsloth/GLM-4.7-Flash-GGUF"
model_filename = "GLM-4.7-Flash-Q4_K_M.gguf"

local_model_dir = "/media/gatv-projects/ssd/AI_models" 

print(f"Downloading or loading {model_filename} from cache...")
model_path = hf_hub_download(
    repo_id = model_repo,
    filename = model_filename, 
    local_dir = local_model_dir,
    local_dir_use_symlinks = False  # Avoid symlinks for better compatibility
)

print(f"Model is stored at: {model_path}")

# 2. Load the model into the GPU
print("Loading model into GPU...")
llm = Llama(
    model_path = model_path,
    n_ctx = 16384,
    n_gpu_layers = -1,  # Offload all layers to GPU
    verbose = False # llama-cpp-python will print only loading info; set to True for more detailed logs
)

segment_description = """In this segment, the player is controlling the champion "Ahri" in the mid lane. The player is currently at level 6 with 1500 gold. The enemy champion "Zed" is also in the mid lane, at level 5 with 1200 gold. The player has just used Ahri's ultimate ability, "Spirit Rush," to engage on Zed, dealing significant damage. Zed has used his "Living Shadow" ability to create a shadow clone and escape towards the river. The player is now deciding whether to follow Zed into the river or to stay in the mid lane to farm minions and gain more gold. The player's team is currently behind in overall gold and objectives, so making the right decision in this moment is crucial for turning the game around."""

# 3. Setup the Knowledge Graph Extraction test prompt
messages = [
    {
        "role": "user",
        "content": "Extract key entities and relationships from the following VLM output describing a League of Legends gameplay segment:\n\n" + segment_description 
    }
] 

print("Generating response...")

# 4. Generate text (using recommended params)
start_inference = time.time()
reponse = llm.create_chat_completion(
    messages = messages,
    temperature=0.7,
    top_p = 1.0,
    min_p = 0.01,
    repeat_penalty = 1.0,
    max_tokens = 1024
)
end_inference = time.time()

print(f"Inference time: {end_inference - start_inference:.2f} seconds")
print("\n=== Generated Response ===")
print(reponse["choices"][0]["message"]["content"])