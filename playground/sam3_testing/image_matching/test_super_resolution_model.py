import os
import cv2

INPUT_IMAGE = "/home/gatv-projects/Desktop/project/playground/sam3_testing/lol_cutouts/champions/cutout_segment_1_frame_0_original_mask_26.png"           # your tiny LoL crop
OUTPUT_IMAGE = "cutout_segment_1_frame_0_original_mask_26_x4_lanczos.png"

def main():
    if not os.path.exists(INPUT_IMAGE):
        print(f"Input image not found: {INPUT_IMAGE}")
        return

    img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Failed to read {INPUT_IMAGE}")
        return

    print("Original size:", img.shape[1], "x", img.shape[0])

    # 4x upscaling using Lanczos (high-quality)
    sr = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)

    print("Upscaled size:", sr.shape[1], "x", sr.shape[0])
    cv2.imwrite(OUTPUT_IMAGE, sr)
    print("Saved:", OUTPUT_IMAGE)

    # Optional display
    # cv2.imshow("Original", img)
    # cv2.imshow("Lanczos x4", sr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
 