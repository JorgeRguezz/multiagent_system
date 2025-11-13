audio_path = "/home/gatv-projects/Desktop/project/chatbot_system/downloads/My_Nintendo_Switch_2_Review.mp4"

# -----------------------------------------------------

import whisper

# Load Whisper model (e.g., "small", "medium", "large", etc.)
model = whisper.load_model("base")

# Transcribe audio with segment timestamps
result = model.transcribe(audio_path)

# 'segments' is a list of dicts with 'start', 'end', and 'text' keys

print("-"*10, "Whisper inference", "-"*10)
for segment in result["segments"]:
    print(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}")

# ------------------------------------------------------
 
# faster-whisper -> Only able to run in the CPU due to pytorch dependencies error I am not willing to look into.

from faster_whisper import WhisperModel

model = WhisperModel("distil-large-v3", device="cpu", compute_type="int8")

segments, info = model.transcribe(audio_path, language="en", condition_on_previous_text=False)
print("-"*10, "Faster-whisper inference", "-"*10)
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
