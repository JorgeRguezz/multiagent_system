import whisper
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from datetime import datetime
import os, tempfile
from tqdm import tqdm
from PIL import Image
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from pathlib import Path
import json


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% KNOWLEDGE EXTRACTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import shutil
import time


def split_video(
    video_path,
    working_dir,
    segment_length,
    num_frames_per_segment,
    audio_output_format='mp3',
):  
    unique_timestamp = str(int(time.time() * 1000))
    video_name = os.path.basename(video_path).split('.')[0]
    video_segment_cache_path = os.path.join(working_dir, '_cache', video_name)
    if os.path.exists(video_segment_cache_path):
        shutil.rmtree(video_segment_cache_path)
    os.makedirs(video_segment_cache_path, exist_ok=False)
    
    segment_index = 0
    segment_index2name, segment_times_info = {}, {}
    with VideoFileClip(video_path) as video:
    
        total_video_length = int(video.duration)
        start_times = list(range(0, total_video_length, segment_length))
        # if the last segment is shorter than 5 seconds, we merged it to the last segment
        if len(start_times) > 1 and (total_video_length - start_times[-1]) < 5:
            start_times = start_times[:-1]
        
        for start in tqdm(start_times, desc=f"Spliting Video {video_name}"):
            if start != start_times[-1]:
                end = min(start + segment_length, total_video_length)
            else:
                end = total_video_length
            
            subvideo = video.subclip(start, end)
            subvideo_length = subvideo.duration
            frame_times = np.linspace(0, subvideo_length, num_frames_per_segment, endpoint=False)
            frame_times += start
            
            segment_index2name[f"{segment_index}"] = f"{unique_timestamp}-{segment_index}-{start}-{end}"
            segment_times_info[f"{segment_index}"] = {"frame_times": frame_times, "timestamp": (start, end)}
            
            # save audio
            audio_file_base_name = segment_index2name[f"{segment_index}"]
            audio_file = f'{audio_file_base_name}.{audio_output_format}'
            subaudio = subvideo.audio
            subaudio.write_audiofile(os.path.join(video_segment_cache_path, audio_file), codec='mp3', verbose=False, logger=None)

            segment_index += 1

    return segment_index2name, segment_times_info


# ASR set-up >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def speech_to_text(video_name, working_dir, segment_index2name, audio_output_format):
    model = whisper.load_model("base")
    
    cache_path = os.path.join(working_dir, '_cache', video_name)
    
    transcripts = {}
    for index in tqdm(segment_index2name, desc=f"Speech Recognition {video_name}"):
        segment_name = segment_index2name[index]
        audio_file = os.path.join(cache_path, f"{segment_name}.{audio_output_format}")

        # if the audio file does not exist, skip it
        if not os.path.exists(audio_file):
            transcripts[index] = ""
            continue
        
        asr_output = model.transcribe(audio_file)
        result = ""
        for segment in asr_output["segments"]:
            result += f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}"
        transcripts[index] = result
    
    return transcripts

# VLM set-up >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

## Configuración inicial ----------------------------------

MODEL_PATH = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"

## Función inferencia -------------------------------------
def encode_video(video, frame_times):
    frames = []
    for t in frame_times:
        frames.append(video.get_frame(t))
    frames = np.stack(frames, axis=0)
    frames = [Image.fromarray(v.astype('uint8')).resize((1280, 720)) for v in frames]
    return frames

def segment_caption(video_name, video_path, segment_index2name, transcripts, segment_times_info):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("------------> VLM Server: Inicializando en dispositivo:", device)
    print("VLM Server: Cargando modelo...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # use bf16 when supported, else fp16/cpu float32
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (
            torch.float16 if device == "cuda" else torch.float32
    )

    model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, torch_dtype=dtype).to(device)
    print("VLM Server: Modelo cargado correctamente ✅")

    local_captions = {}
    with VideoFileClip(video_path) as video:
        for idx in tqdm(segment_index2name.keys(), desc=f"Captioning Video {video_name}"):
            frame_times = segment_times_info[idx]["frame_times"]
            video_frames = encode_video(video, frame_times)  # -> list[PIL.Image]

            prompt_text = (
                f"The transcript of the current video:\n{transcripts[idx]}\n"
                "Now provide a description (caption) of the video in English."
            )

            # N image placeholders in the chat; actual pixels provided in `images=...`
            user_content = [{"type": "image"} for _ in range(len(video_frames))]
            user_content.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": user_content}]

            prompt_str = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,   # we’ll tokenize in the next call
            )

            batch = processor(
                text=[prompt_str],
                images=[video_frames],      # <-- frames in memory (no disk)
                return_tensors="pt",
            ).to(model.device, dtype=dtype)

            out_ids = model.generate(**batch, do_sample=False, max_new_tokens=128)
            text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
            cleaned = (text.split("Assistant:")[-1] or text).strip()
            local_captions[idx] = cleaned.replace("\n", "").replace("<|endoftext|>", "")

            del video_frames
            torch.cuda.empty_cache()

    return local_captions


# Video chunking >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## Params:
video_path = "/home/gatv-projects/Desktop/project/chatbot_system/downloads/My_Nintendo_Switch_2_Review.mp4"
working_dir: str = f"./videorag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"

# if working_dir inside of a class value should be assigned like this
# working_dir: str = field(
#     default_factory=lambda: f"./videorag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
# )

video_segment_length: int = 30 # 30 seconds each chunck
rough_num_frames_per_segment: int = 5 # 5 frames processed by the VLM in each 30s chunck
audio_output_format: str = "mp3"


## Split the videos
segment_index2name, segment_times_info = split_video(video_path, working_dir, video_segment_length, rough_num_frames_per_segment, audio_output_format)

# Models inference (ASR and VLM) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
video_name = os.path.basename(video_path).split('.')[0]
## ASR inference
transcripts = speech_to_text(video_name, working_dir, segment_index2name, audio_output_format)
## VLM inference
caption = segment_caption(video_name, video_path, segment_index2name, transcripts, segment_times_info)
## Unify info

def merge_segment_information(segment_index2name, segment_times_info, transcripts, captions):
    inserting_segments = {}
    for index in segment_index2name:
        inserting_segments[index] = {"content": None, "time": None}
        segment_name = segment_index2name[index]
        inserting_segments[index]["time"] = '-'.join(segment_name.split('-')[-2:])
        inserting_segments[index]["content"] = f"Caption:\n{captions[index]}\nTranscript:\n{transcripts[index]}\n\n"
        inserting_segments[index]["transcript"] = transcripts[index]
        inserting_segments[index]["frame_times"] = segment_times_info[index]["frame_times"].tolist()
    return inserting_segments

segments_information = merge_segment_information(segment_index2name, segment_times_info, transcripts, caption)

# Print results >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
print("<>"*20)
print(json.dumps(segments_information, indent=4, ensure_ascii=False))

