import os
import sys
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Ensure we can import 'embed' from the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import embed_image from embed_dinov2.py
try:
    from embed_dinov2 import embed_image
except ImportError:
    sys.path.append(os.path.join(current_dir))
    from embed_dinov2 import embed_image

def preprocess_image(input_path, target_size=224, fill_color=(0, 0, 0)):
    """
    Resize an image to 224x224 for DINOv2:
    - Keep aspect ratio
    - Use bicubic interpolation
    - Pad to 224x224 with a solid background (default black)
    """
    img = Image.open(input_path).convert("RGB")

    # Original size
    w, h = img.size

    # Compute scale factor so that the longest side == TARGET_SIZE
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize with bicubic interpolation
    img_resized = img.resize((new_w, new_h), resample=Image.BICUBIC)

    # Create new 224x224 canvas and paste the resized image centered
    new_img = Image.new("RGB", (target_size, target_size), fill_color)
    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2
    new_img.paste(img_resized, (left, top))
    
    return new_img

def main():
    # Directory containing the images
    cutouts_dir = os.path.join(current_dir, "lol_cutouts")
    
    # Get all image files
    image_files = glob.glob(os.path.join(cutouts_dir, "*.png")) + \
                  glob.glob(os.path.join(cutouts_dir, "*.jpg")) + \
                  glob.glob(os.path.join(cutouts_dir, "*.jpeg"))
    
    if not image_files:
        print(f"No images found in {cutouts_dir}")
        return

    print(f"Found {len(image_files)} images.")

    embeddings = []
    filenames = []
    
    # Process images
    print("Processing images and generating embeddings...")
    for i, img_path in enumerate(image_files):
        try:
            if i % 10 == 0:
                print(f"Processing {i}/{len(image_files)}...", end="\r")
                
            # Preprocess
            pil_image = preprocess_image(img_path)
            
            # Embed
            emb = embed_image(pil_image)
            
            embeddings.append(emb)
            filenames.append(os.path.basename(img_path))
            
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")

    print(f"Processing {len(image_files)}/{len(image_files)}... Done.")

    if not embeddings:
        print("No embeddings generated.")
        return

    embeddings = np.array(embeddings)
    print(f"Embeddings shape: {embeddings.shape}")

    # Compute t-SNE
    print("Computing t-SNE...")
    n_samples = embeddings.shape[0]
    # Perplexity must be less than n_samples.
    perp = min(30, n_samples - 1) if n_samples > 1 else 1
    
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
    projections = tsne.fit_transform(embeddings)

    # Plot
    print("Plotting...")
    plt.figure(figsize=(14, 12))
    plt.scatter(projections[:, 0], projections[:, 1], alpha=0.7, c='royalblue', edgecolors='k')

    # Annotate points
    for i, filename in enumerate(filenames):
        # Simplify filename for label
        # Example: cutout_segment_1_frame_4_original_mask_28.png -> s1_f4_m28
        label = filename.replace("cutout_", "")\
                        .replace("segment_", "s")\
                        .replace("frame_", "f")\
                        .replace("original_", "")\
                        .replace("mask_", "m")\
                        .replace(".png", "")
        
        plt.annotate(label, (projections[i, 0], projections[i, 1]), 
                     fontsize=8, alpha=0.8, xytext=(5, 5), textcoords='offset points')

    plt.title("t-SNE Visualization of LoL Cutout Embeddings (DINOv2)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True, alpha=0.3)

    output_plot_path = os.path.join(current_dir, "embedding_visualization.png")
    plt.savefig(output_plot_path)
    print(f"Plot saved to {output_plot_path}")

if __name__ == "__main__":
    main()
