"""
encapsulates the VLM logic, runs independently, and communicates its results via
standard output.
"""

import sys
import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import yt_dlp

class VLMProcessor:
    def __init__(self, model_id="HuggingFaceTB/SmolVLM2-256M-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.generation_args = {
            "max_new_tokens": 1024,
            "do_sample": False,
        }

    def _load_model(self):
        if self.model is None:
            print("[VLM Processor] Loading VLM model...")
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            print("[VLM Processor] VLM model loaded.")

    def download_youtube_video(self, url: str, output_dir: str = "downloads") -> str:
        os.makedirs(output_dir, exist_ok=True)
        ydl_opts = {
            'format': 'bestvideo+bestaudio',
            'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
            'merge_output_format': 'mp4',
            'quiet': True,
            'no_warnings': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return os.path.abspath(os.path.join(output_dir, f"{info['title']}.mp4"))

    def analyze_video(self, video_path: str, query: str) -> str:
        self._load_model()
        print(f"[VLM Processor] Analyzing video: {video_path}")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": video_path},
                    {"type": "text", "text": query}
                ]
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device, torch.bfloat16)

        generate_ids = self.model.generate(**inputs, **self.generation_args)
        decoded_text = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return decoded_text