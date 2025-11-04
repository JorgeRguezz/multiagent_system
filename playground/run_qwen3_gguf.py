import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# -----------------------------------------------------
# 1. Configuración de Archivos y Repositorio (¡CORREGIDO!)
# -----------------------------------------------------

# Configuración del modelo GGUF en Hugging Face
REPO_ID = "bartowski/Qwen_Qwen3-30B-A3B-Instruct-2507-GGUF"
# FILENAME CORREGIDO: Añadido el prefijo "Qwen_"
FILENAME = "Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"

# Rutas locales
LOCAL_DIR = "./Qwen3-GGUF"
GGUF_PATH = os.path.join(LOCAL_DIR, FILENAME)

# -----------------------------------------------------
# 2. Función de Descarga (Si es necesario)
# -----------------------------------------------------

def download_model_if_needed():
    """Descarga el modelo GGUF de Hugging Face si no existe localmente."""
    if os.path.exists(GGUF_PATH):
        print(f"✅ Archivo GGUF ya existe: {GGUF_PATH}")
        return

    print(f"⏳ Archivo GGUF no encontrado. Iniciando descarga de {FILENAME}...")
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    try:
        # Nota sobre local_dir_use_symlinks: La advertencia es inofensiva.
        hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME, # <--- ¡Aquí está la corrección!
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False, 
            # Añadir una barra de progreso más visible:
            force_download=False 
        )
        print("\n✅ ¡Descarga completada con éxito!")
    except Exception as e:
        print(f"\n❌ Ocurrió un error durante la descarga desde Hugging Face: {e}")
        # Re-raise el error para ver la pila completa si falla de nuevo
        raise

# -----------------------------------------------------
# 3. Proceso Principal
# -----------------------------------------------------

if __name__ == "__main__":
    download_model_if_needed()

    try:
        print("\n⏳ Inicializando el modelo. Cargando capas a la GPU (Turing)...")
        
        llm = Llama(
            model_path=GGUF_PATH,
            n_gpu_layers=-1,        
            n_ctx=4096,             
            verbose=True
        )
        
        print("\n🎉 ¡Modelo Qwen3-30B-A3B-Instruct-2507 cargado con éxito en la GPU!")

    except Exception as e:
        print(f"\n❌ Fallo en la inicialización de Llama-CPP: {e}")
        print("Asegúrate de que 'llama-cpp-python[cuda]' está correctamente instalado.")
        exit()

    # ... (Resto del código de generación de prueba) ...
    messages = [
        {"role": "system", "content": "Eres un asistente de IA útil y honesto que opera en español."},
        {"role": "user", "content": "Explica la diferencia técnica clave que hace que Ollama/GGUF funcione en Turing (TITAN RTX) para MoE INT4, mientras que vLLM falla."}
    ]

    try:
        print("\n🚀 Iniciando la generación de respuesta. Verificando rendimiento en Turing...")
        
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=600,
            temperature=0.1 
        )

        print("\n--- Respuesta del Modelo Qwen3 (llama-cpp-python) ---")
        print(response['choices'][0]['message']['content'])
        print("-----------------------------------------------------")
        
    except Exception as e:
        print(f"\n❌ Error al generar la respuesta: {e}")