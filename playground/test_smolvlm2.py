
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from decord import VideoReader, cpu
import numpy as np
from tqdm import tqdm

# 1. Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print("------------> USING DEVICE: ", device)
# It can be changed to 256M-Instruct, 500M-Instruct or 2.2B-Instruct for more powerful model
model_id = "HuggingFaceTB/SmolVLM2-256M-Instruct"

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    trust_remote_code=True
).to(device)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


# 3. Video Test
print("\n--- Video Test ---")
video_path = "/home/gatv-projects/Desktop/project/playground/Penguins_720p_3min.mp4"

print("Type 'quit' to exit.")

while True:
    question = input("\nYour question: ")
    if question.lower() == 'quit':
        break

    messages = [
    {
        "role": "user",
            "content": [
                {"type": "video", "path": video_path},
                {"type": "text", "text": question}
            ] 
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device, torch.bfloat16)

    generation_args = {
        "max_new_tokens": 1024,
        "do_sample": False,
    }

    generate_ids = model.generate(**inputs, **generation_args)
    decoded_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(f"Assistant: {decoded_text}")
