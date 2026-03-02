
import os
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image

VIDEO_PATH = "/home/gatv-projects/Desktop/project/downloads/The_COMPLETE_SMOLDER_STARTER_GUIDE_-_League_of_Legends.mp4"
PLAYER_CUTOUT = (600, 900, 1235, 1080) # [x0, y0, x1, y1]

def check_video():
    if not os.path.exists(VIDEO_PATH):
        print(f"Video not found: {VIDEO_PATH}")
        return

    print(f"Checking video: {VIDEO_PATH}")
    
    with VideoFileClip(VIDEO_PATH) as clip:
        w, h = clip.size
        print(f"Video Resolution: {w}x{h}")
        print(f"Crop Coordinates: {PLAYER_CUTOUT}")
        
        # Check bounds
        x0, y0, x1, y1 = PLAYER_CUTOUT
        if x1 > w or y1 > h:
            print("WARNING: Crop coordinates are OUT OF BOUNDS!")
            if x1 > w: print(f"  - Right edge ({x1}) > Video Width ({w})")
            if y1 > h: print(f"  - Bottom edge ({y1}) > Video Height ({h})")
        else:
            print("Crop coordinates are within bounds.")

        # Extract a test frame at 60 seconds (likely in game)
        print("Extracting test frame at t=60s...")
        frame = clip.get_frame(60)
        img = Image.fromarray(frame)
        
        # Save full frame
        img.save("debug_full_frame.jpg")
        print("Saved debug_full_frame.jpg")
        
        # Attempt crop
        crop = img.crop(PLAYER_CUTOUT)
        crop.save("debug_crop.jpg")
        print(f"Saved debug_crop.jpg (Size: {crop.size})")
        
        # Check if crop is black/empty
        extrema = crop.getextrema()
        print(f"Crop Pixel Extrema (Min/Max per channel): {extrema}")
        
        # Simple check for solid color
        if extrema:
            is_black = all(min_val == 0 and max_val == 0 for min_val, max_val in extrema)
            if is_black:
                print("ALERT: The cropped image is completely BLACK.")

if __name__ == "__main__":
    check_video()
