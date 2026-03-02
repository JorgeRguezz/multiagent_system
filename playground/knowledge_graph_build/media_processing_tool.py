import asyncio
import os
import json
import shutil
import sys
import gc
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Type

import torch
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image
import numpy as np
from tqdm import tqdm

from mcp.server.fastmcp import FastMCP

# Local project imports (assuming they are in the python path or same directory)
# Note: These are now minimal as most logic is in this file.
from chatbot_system.knowledge_graph._videoutil.split import split_video
from chatbot_system.knowledge_graph._storage.kv_json import JsonKVStorage
from chatbot_system.knowledge_graph.base import BaseKVStorage

# =====================================================
#  Configuration and Model Loading
# =====================================================
mcp = FastMCP("media_processing_server")
current_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--> Media Processing Server: Initializing on device: {current_device}", file=sys.stderr)

# Global variables for models
asr_model = None
vlm_processor = None
vlm_model = None

def load_models():
    """Loads ASR and VLM models into GPU memory if not already loaded."""
    global asr_model, vlm_processor, vlm_model, current_device
    
    # --- ASR Model ---
    if asr_model is None:
        print("--> Media Server: Loading ASR model (Whisper)...", file=sys.stderr)
        try:
            import whisper
            asr_model = whisper.load_model("base", device=current_device)
            print("--> Media Server: ASR model loaded successfully ✅", file=sys.stderr)
        except ImportError:
            print("--> Media Server: `whisper` not installed, ASR will not be available.", file=sys.stderr)
        except Exception as e:
            print(f"--> Media Server: Error loading ASR model: {e}", file=sys.stderr)

    # --- VLM Model ---
    if vlm_model is None:
        print("--> Media Server: Loading VLM model (SmolVLM2-Video)...", file=sys.stderr)
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
            vlm_model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
            vlm_processor = AutoProcessor.from_pretrained(vlm_model_path)
            vlm_model = AutoModelForImageTextToText.from_pretrained(
                vlm_model_path,
                torch_dtype=torch.bfloat16 if (current_device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32,
            ).to(current_device)
            print("--> Media Server: VLM model loaded successfully ✅", file=sys.stderr)
        except ImportError:
            print("--> Media Server: `transformers` not installed, VLM will not be available.", file=sys.stderr)
        except Exception as e:
            print(f"--> Media Server: Error loading VLM model: {e}", file=sys.stderr)

def unload_models():
    """Offloads ASR and VLM models from GPU memory."""
    global asr_model, vlm_processor, vlm_model
    
    print("--> Media Server: Unloading models to free GPU...", file=sys.stderr)
    
    if asr_model is not None:
        del asr_model
        asr_model = None
        
    if vlm_model is not None:
        del vlm_model
        vlm_model = None
        
    if vlm_processor is not None:
        del vlm_processor
        vlm_processor = None
        
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("--> Media Server: GPU memory cleared ✅", file=sys.stderr)


# =====================================================
#  Server-Side Processing Class
# =====================================================
@dataclass
class ServerSideProcessor:
    """
    Handles the ASR and VLM data extraction pipeline on the server.
    """
    video_path: str
    
    # Config
    working_dir: str = field(init=False)
    video_segment_length: int = 30
    rough_num_frames_per_segment: int = 5
    audio_output_format: str = "mp3"
    
    # Storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage

    def __post_init__(self):
        """Initializes working directory."""
        self.working_dir = f"./server_cache_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        os.makedirs(self.working_dir, exist_ok=True)
        
        # This DB will be created on the server and its path returned to the client
        self.video_segments_db = self.key_string_value_json_storage_cls(
            namespace="video_segments", global_config=asdict(self)
        )

    def speech_to_text(self, video_name, segment_index2name):
        if not asr_model:
             # Fallback: ensure models are loaded
             load_models()
             if not asr_model:
                raise RuntimeError("ASR model could not be loaded.")
        
        cache_path = os.path.join(self.working_dir, '_cache', video_name)
        transcripts = {}
        for index in tqdm(segment_index2name, desc=f"Speech Recognition for {video_name}"):
            segment_name = segment_index2name[index]
            audio_file = os.path.join(cache_path, f"{segment_name}.{self.audio_output_format}")
            if not os.path.exists(audio_file):
                transcripts[index] = ""
                continue
            result = asr_model.transcribe(audio_file)
            transcripts[index] = "".join([f"[{s['start']:.2f}s -> {s['end']:.2f}s] {s['text']}" for s in result["segments"]])
        return transcripts

    def segment_caption(self, video_name, transcripts, segment_times_info):
        if not vlm_model or not vlm_processor:
             # Fallback: ensure models are loaded
             load_models()
             if not vlm_model or not vlm_processor:
                raise RuntimeError("VLM model could not be loaded.")
        
        local_captions = {}
        with VideoFileClip(self.video_path) as video:
            for idx in tqdm(segment_times_info.keys(), desc=f"Captioning Video {video_name}"):
                frame_times = segment_times_info[idx]["frame_times"]
                
                frames = [Image.fromarray(video.get_frame(t).astype('uint8')).resize((1280, 720)) for t in frame_times]

                prompt_text = (
                    f"The transcript of the current video segment:\n{transcripts.get(idx, '')}\n\n"
                    "Provide a concise, one-sentence description of the video content."
                )
                
                messages = [{"role": "user", "content": [{"type": "image"}] * len(frames) + [{"type": "text", "text": prompt_text}]}]
                prompt_str = vlm_processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                
                batch = vlm_processor(text=[prompt_str], images=[frames], return_tensors="pt").to(vlm_model.device, dtype=vlm_model.dtype)

                out_ids = vlm_model.generate(**batch, do_sample=False, max_new_tokens=128)
                text = vlm_processor.batch_decode(out_ids, skip_special_tokens=True)[0]
                cleaned_response = text.split("Assistant:")[-1].strip()
                local_captions[idx] = cleaned_response

                del frames
                if current_device == "cuda":
                    torch.cuda.empty_cache()
        return local_captions

    @staticmethod
    def merge_segment_information(segment_index2name, segment_times_info, transcripts, captions):
        inserting_segments = {}
        for index in segment_index2name:
            segment_name = segment_index2name[index]
            inserting_segments[index] = {
                "time": '-'.join(segment_name.split('-')[-2:]),
                "content": f"Caption:\n{captions.get(index, '')}\nTranscript:\n{transcripts.get(index, '')}\n\n",
                "transcript": transcripts.get(index, ''),
                "frame_times": segment_times_info[index]["frame_times"].tolist()
            }
        return inserting_segments

    async def run_pipeline(self):
        """Runs the ASR/VLM data extraction pipeline."""
        video_name = os.path.basename(self.video_path).split('.')[0]
        
        print(f"--> Server: Splitting video '{video_name}'", file=sys.stderr)
        segment_index2name, segment_times_info = split_video(
            self.video_path, self.working_dir, self.video_segment_length, 
            self.rough_num_frames_per_segment, self.audio_output_format
        )
        
        print("--> Server: Running ASR and VLM...", file=sys.stderr)
        transcripts = self.speech_to_text(video_name, segment_index2name)
        captions = self.segment_caption(video_name, transcripts, segment_times_info)
        
        print("--> Server: Merging and saving segment information...", file=sys.stderr)
        segments_information = self.merge_segment_information(
            segment_index2name, segment_times_info, transcripts, captions
        )
        await self.video_segments_db.upsert({video_name: segments_information})
        await self.video_segments_db.index_done_callback()

        audio_cache_path = os.path.join(self.working_dir, '_cache', video_name)
        if os.path.exists(audio_cache_path):
            shutil.rmtree(audio_cache_path)
            
        db_paths = {
            "working_dir": os.path.abspath(self.working_dir),
            "video_segments_db_path": os.path.abspath(self.video_segments_db._file_name),
        }
        
        print(f"--> Server: Processing complete. Returning DB paths: {db_paths}", file=sys.stderr)
        return db_paths

# =====================================================
#  New High-Level MCP Tool
# =====================================================
@mcp.tool()
async def extract_video_knowledge(video_path: str) -> str:
    """
    Processes a video file to extract transcripts and captions.
    This tool performs video splitting, ASR, and VLM captioning,
    then saves the results into a `video_segments.json` file.
    It returns a JSON string containing the path to this file and the
    server-side working directory.
    """
    print(f"--> Tool: Received request to process video: {video_path}", file=sys.stderr)
    if not os.path.exists(video_path):
        return json.dumps({"error": f"Video file not found at: {video_path}"})

    try:
        # 1. Load models explicitly before processing
        load_models()
        
        processor = ServerSideProcessor(video_path=video_path)
        result_paths = await processor.run_pipeline()
        return json.dumps(result_paths)
    except Exception as e:
        print(f"--> Tool: Error during pipeline execution: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return json.dumps({"error": str(e)})
    finally:
        # 2. Unload models to free GPU resources for other tasks (e.g. LLM)
        unload_models()

# =====================================================
#  Server Execution
# =====================================================
if __name__ == "__main__":
    mcp.run(transport='stdio')
