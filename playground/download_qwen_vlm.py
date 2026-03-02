from huggingface_hub import snapshot_download

print("Downloading Qwen2.5-VL... this may take a while (approx 15GB)")
snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-7B-Instruct", 
    local_dir="./qwen2.5-vl-7b"
)
print("Download complete!")