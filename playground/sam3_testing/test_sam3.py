import torch
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)
# Load an image
image = Image.open("/home/gatv-projects/Desktop/project/playground/lol_gameplay_image.png").convert("RGB")
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="Segment all champions in the image.")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

print(f"Detected {len(masks)} masks.")
for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
    print(f"Mask {i+1}: Box={box.tolist()}, Score={score.item():.4f}")  

#################################### For Video ####################################

# from sam3.model_builder import build_sam3_video_predictor

# video_predictor = build_sam3_video_predictor()
# video_path = "<YOUR_VIDEO_PATH>" # a JPEG folder or an MP4 video file
# # Start a session
# response = video_predictor.handle_request(
#     request=dict(
#         type="start_session",
#         resource_path=video_path,
#     )
# )
# response = video_predictor.handle_request(
#     request=dict(
#         type="add_prompt",
#         session_id=response["session_id"],
#         frame_index=0, # Arbitrary frame index
#         text="<YOUR_TEXT_PROMPT>",
#     )
# )
# output = response["outputs"]
