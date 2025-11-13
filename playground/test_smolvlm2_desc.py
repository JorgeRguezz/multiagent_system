from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from PIL import Image

# 1. Setup
current_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"------------> Initializing on device: {current_device}")

model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
).to(current_device)


def run_inference(model, processor, messages, current_device):
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=2048)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return generated_texts[0]


# # 2. Image Test
# print("--- Image Test ---")
# image_path = "pokemon_gameplay.jpeg"
# image = Image.open(image_path).convert("RGB")

# messages_image = [
#     {

#         "role": "user",
#         "content": [
#             {"type": "image", "image": image},
#             {"type": "text", "text": "What is in the image?"}
#         ]
#     },
# ]

# decoded_text_image = run_inference(model, processor, messages_image, current_device)
# print(decoded_text_image)


# 3. Video Test
print("\n--- Video Test ---")
video_path = "/home/gatv-projects/Desktop/project/chatbot_system/downloads/My Nintendo Switch 2 Review.mp4"

messages_video = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": video_path},
            {"type": "text", "text": "Describe this video in detail"}
        ]
    },
]

decoded_text_video = run_inference(model, processor, messages_video, current_device)
vlm_response_parts = decoded_text_video.split("Assistant:")
vlm_response_clean = vlm_response_parts[1].strip()
print(">>> VLM response:")
print(vlm_response_clean)
