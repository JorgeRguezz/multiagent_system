from huggingface_hub import hf_hub_download

repo_id = "bartowski/ServiceNow-AI_Apriel-1.6-15b-Thinker-GGUF"
filename = "ServiceNow-AI_Apriel-1.6-15b-Thinker-Q6_K_L.gguf"

local_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir="./models",
    local_dir_use_symlinks=False,
)

print("Downloaded to:", local_path)

 