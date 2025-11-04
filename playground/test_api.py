import requests

# URL de tu API Flask
url = "http://localhost:5000/generate"

# Cuerpo de la petición (igual que en curl)
payload = {
    "conversation": [
        {
            "role": "user",
            "content": [
                {"type": "video", "path": "/home/gatv-projects/Desktop/project/downloads/zelda_gameplay_720p_10min.mp4"},
                 {"type": "text", "text": "Describe this video in detail"}
            ]
        }
    ]
}

# Enviar POST request
response = requests.post(url, json=payload)

# Verificar y mostrar la respuesta
if response.ok:
    data = response.json()
    print("✅ Respuesta del modelo:")
    print(data["generated_text"])
else:
    print("❌ Error:", response.status_code, response.text)
