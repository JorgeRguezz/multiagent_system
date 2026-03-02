import sys
import os
import shutil
from PIL import Image
from tqdm import tqdm
import torch

# Ensure we can import sam3
# We add the 'playground' and project root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
playground_dir = os.path.dirname(current_dir) # playground/
project_root = os.path.dirname(playground_dir) # project root
sam3_repo_root = os.path.join(project_root, "sam3")

sys.path.append(playground_dir)
sys.path.append(sam3_repo_root)

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# Configuration
ASSETS_ROOT = "/home/gatv-projects/Desktop/project/playground/lol_images_extraction/assets/champions"


def extract_cutout(image, box):
    """
    Extracts the image cutout based on bounding box.
    """
    if hasattr(box, 'cpu'):
        box = box.cpu().numpy()
    
    x_min, y_min, x_max, y_max = map(int, box)
    
    # Clip to image bounds
    width, height = image.size
    x_min = max(0, min(x_min, width - 1))
    x_max = max(0, min(x_max, width))
    y_min = max(0, min(y_min, height - 1))
    y_max = max(0, min(y_max, height))
    
    return image.crop((x_min, y_min, x_max, y_max))

def main():
    if not os.path.exists(ASSETS_ROOT):
        print(f"Error: Assets directory not found at {ASSETS_ROOT}")
        return

    # Load SAM3 Model
    print("--> Loading SAM3 Model...")
    try:
        model = build_sam3_image_model()
        processor = Sam3Processor(model)
        print("--> SAM3 Model Loaded.")
    except Exception as e:
        print(f"Failed to load SAM3 model: {e}")
        return

    # List all champions
    champions = [d for d in os.listdir(ASSETS_ROOT) if os.path.isdir(os.path.join(ASSETS_ROOT, d))]
    print(f"--> Found {len(champions)} champions.")

    for champion_name in tqdm(champions, desc="Processing Champions"):
        champion_dir = os.path.join(ASSETS_ROOT, champion_name)
        loading_screen_dir = os.path.join(champion_dir, "loading_screen")
        square_dir = os.path.join(champion_dir, "square")
        
        # Ensure square directory exists
        os.makedirs(square_dir, exist_ok=True)

        if not os.path.exists(loading_screen_dir):
            # print(f"Skipping {champion_name}: No loading_screen folder.")
            continue
            
        # Process images in loading_screen
        img_files = [f for f in os.listdir(loading_screen_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in img_files:
            img_path = os.path.join(loading_screen_dir, img_file)
            
            try:
                # Prepare filenames
                base_name, ext = os.path.splitext(img_file)
                base_save_path = os.path.join(square_dir, img_file) # The file from run #1
                
                # Load image
                image = Image.open(img_path).convert("RGB")
                
                # Run SAM3
                inference_state = processor.set_image(image)
                output = processor.set_text_prompt(state=inference_state, prompt="Head")
                
                masks = output["masks"]
                boxes = output["boxes"]
                
                if len(masks) > 0:
                    # SAM3 detected one or more heads
                    # Process all detected masks
                    for i, box in enumerate(boxes):
                        # Determine save path for this mask
                        if i == 0:
                            # Mask 0 corresponds to the original run's logic
                            save_path = base_save_path
                        else:
                            # Subsequent masks get an ID: name_1.png, name_2.png
                            save_path = os.path.join(square_dir, f"{base_name}_{i}{ext}")
                        
                        # Optimization: Skip saving if file already exists (User request)
                        if os.path.exists(save_path):
                            continue
                            
                        # Extract and Save
                        result_image = extract_cutout(image, box)
                        result_image.save(save_path)
                        # print(f"  [+] Saved head {i} for {img_file}")

                else:
                    # No mask detected. 
                    # If base file doesn't exist (missed in run #1?), save original.
                    if not os.path.exists(base_save_path):
                        image.save(base_save_path)
                        # print(f"  [-] Saved original for {img_file}")
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    main()
