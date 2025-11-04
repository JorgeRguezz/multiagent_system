from flask import Flask, request, jsonify
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
import io
import base64
import os

# =====================================================
# Configuración inicial
# =====================================================
app = Flask(__name__)

current_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"------------> Inicializando en dispositivo: {current_device}")

model_path = "HuggingFaceTB/SmolVLM2-256M-Instruct"

print("Cargando modelo...")
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
).to(current_device)
print("Modelo cargado correctamente ✅")

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
# Endpoint principal
# =====================================================
@app.route("/generate", methods=["POST"])
def generate_text():
    """
    Espera un JSON con un campo 'conversation'.
    Puede contener tipos:
      - {"type": "text", "text": "..."}
      - {"type": "image", "url": "..."} o {"type": "image", "base64": "..."}
      - {"type": "video", "path": "..."}
    """
    try:
        data = request.get_json(force=True)
        conversation = data.get("conversation")

        if not conversation:
            return jsonify({"error": "Falta el parámetro 'conversation'"}), 400

        # Procesar imágenes base64 (si se envían así)
        for msg in conversation:
            if "content" in msg:
                for item in msg["content"]:
                    if item["type"] == "image":
                        if "base64" in item:
                            # Convertir base64 a imagen PIL
                            image_bytes = base64.b64decode(item["base64"])
                            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                            item["image"] = image
                            del item["base64"]

        # Ejecutar inferencia
        result_text = run_inference(model, processor, conversation)

        return jsonify({"generated_text": result_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =====================================================
# Endpoint de salud
# =====================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": current_device})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
