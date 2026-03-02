
import os
import sys
import torch
from PIL import Image
import numpy as np

# Add project root to path
project_root = "/home/gatv-projects/Desktop/project"
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "sam3"))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Config from knowledge_extraction.config
MIDDLE_HUD = [500, 900, 1350, 1080]
IMAGE_PATH = os.path.join(project_root, "playground/lol_gameplay_image.png")

def test_sam3():
    print("Loading SAM3...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=0.1) # Lower threshold for debugging
    
    print(f"Opening image {IMAGE_PATH}...")
    full_image = Image.open(IMAGE_PATH).convert("RGB")
    roi = full_image.crop(MIDDLE_HUD)
    # roi.save("debug_roi_middle.png")
    # print("Saved debug_roi_middle.png")

    print("Running SAM3 on ROI...")
    state = processor.set_image(roi)
    
    prompts = ["Character.", "Champion.", "Face.", "Icon.", "Portrait."]
    for p in prompts:
        print(f"\n--- Testing prompt: '{p}' ---")
        output = processor.set_text_prompt(state=state, prompt=p)
        boxes = output["boxes"]
        scores = output["scores"]
        
        if boxes is not None and len(boxes) > 0:
            print(f"Detected {len(boxes)} boxes:")
            for i, (box, score) in enumerate(zip(boxes, scores)):
                x_min, y_min, x_max, y_max = map(int, box)
                w = x_max - x_min
                h = y_max - y_min
                aspect_ratio = w / h if h != 0 else 0
                print(f"  Box {i}: [{x_min}, {y_min}, {x_max}, {y_max}], Score: {score:.4f}, AR: {aspect_ratio:.2f}")
                
                # # Save cutout
                # cutout = roi.crop((x_min, y_min, x_max, y_max))
                # cutout.save(f"debug_cutout_{p}_{i}.png")
        else:
            print("No boxes detected.")

if __name__ == "__main__":
    test_sam3()
