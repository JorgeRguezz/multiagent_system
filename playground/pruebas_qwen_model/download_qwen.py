from huggingface_hub import snapshot_download

# Nombre del repositorio del modelo cuantizado en Hugging Face
model_name = "btbtyler09/Qwen3-30B-A3B-Instruct-2507-gptq-4bit"

# Ruta local donde se guardará el modelo
local_dir = "./Qwen3-30B-A3B-Instruct-2507-gptq-4bit"

# Descarga el repositorio completo del modelo cuantizado
snapshot_download(
    repo_id=model_name,
    local_dir=local_dir,
    local_dir_use_symlinks=False, 
    revision="main",
    token=True  # Se recomienda token para repositorios privados o para evitar límites de descarga
)

print(f"✅ Modelo GPTQ (formato repositorio) descargado en: {local_dir}")