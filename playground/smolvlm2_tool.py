from mcp.server.fastmcp import FastMCP
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
import io
import base64

# =====================================================
# Configuración inicial
# =====================================================
mcp = FastMCP("vlm_server")

current_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"------------> VLM Server: Inicializando en dispositivo: {current_device}")

model_path = "HuggingFaceTB/SmolVLM2-256M-Instruct"

print("VLM Server: Cargando modelo...")
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
).to(current_device)
print("VLM Server: Modelo cargado correctamente ✅")

# =====================================================
# Función de inferencia
# =====================================================
def run_inference(model, processor, messages):
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=128)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_texts[0]


# =====================================================
# Herramienta MCP
# =====================================================
@mcp.tool()
def analyze_media(payload: dict) -> dict:
    """
    Procesa una conversación (texto, imágenes, video) y devuelve una respuesta del VLM.
    Espera un diccionario con un campo 'conversation'.
    """
    try:

        print("------> tool payload:", payload)
        print("------> tool payload type:", type(payload))

        conversation = payload.get("conversation")

        if not conversation:
            return {"error": "Falta el parámetro 'conversation'"}

        # Procesar imágenes base64 (si se envían así)
        for msg in conversation:
            if "content" in msg:
                for item in msg["content"]:
                    if item["type"] == "image":
                        if "base64" in item:
                            image_bytes = base64.b64decode(item["base64"])
                            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                            item["image"] = image
                            del item["base64"]
                    # The video path is handled by the main chatbot, here we just use the text query
                    elif item["type"] == "video":
                        # The model can't process the video directly, it will use the text part of the query
                        pass


        # Ejecutar inferencia
        result_text = run_inference(model, processor, conversation)

        return {"generated_text": result_text}

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    mcp.run(transport='stdio')