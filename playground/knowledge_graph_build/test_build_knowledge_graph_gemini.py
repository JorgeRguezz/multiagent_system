import whisper
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from datetime import datetime
import os
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from dataclasses import dataclass, field, asdict
from typing import Type, cast, Callable, List, Optional, Dict, Union
import shutil
import asyncio
import tiktoken


# Assuming these are in a reachable path
from ._videoutil import split_video
from ._storage import NanoVectorDBVideoSegmentStorage, JsonKVStorage, NanoVectorDBStorage, NetworkXStorage
from .base import BaseVectorStorage, StorageNameSpace, BaseKVStorage, BaseGraphStorage
from ._utils import always_get_an_event_loop, get_chunks, limit_async_func_call, wrap_embedding_func_with_attrs
from ._op import chunking_by_video_segments, extract_entities

@dataclass
class VideoKnowledgeExtractor:
    """
    A class to extract knowledge (transcripts, captions, features) from a video file.
    """
    # --- Configuration ---
    video_path: str
    working_dir: str = field(init=False)
    video_segment_length: int = 30  # seconds
    rough_num_frames_per_segment: int = 5
    audio_output_format: str = "mp3"
    video_output_format: str = "mp4"

    # graph mode
    enable_local: bool = True
    enable_naive_rag: bool = True
    
    # --- Models ---
    asr_model_name: str = "base"
    vlm_model_path: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Internal State ---
    asr_model: "Whisper" = field(init=False, repr=False, default=None)
    vlm_model: "AutoModelForImageTextToText" = field(init=False, repr=False, default=None)
    vlm_processor: "AutoProcessor" = field(init=False, repr=False, default=None)
    video_segment_feature_vdb: "BaseVectorStorage" = field(init=False, repr=False, default=None)

    # text chunking
    chunk_func: Callable[
        [
            list[list[int]],
            List[str],
            tiktoken.Encoding,
            Optional[int],
        ],
        List[Dict[str, Union[str, int]]],
    ] = chunking_by_video_segments
    chunk_token_size: int = 1200

    # storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage

    # entity extraction
    entity_extraction_func: callable = extract_entities

    def __post_init__(self):
        """Initializes working directory and storage."""
        self.working_dir = f"./videorag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
        os.makedirs(self.working_dir, exist_ok=True)
        
        # Correctly initialize storage using 'self'
        self.video_segment_feature_vdb = NanoVectorDBVideoSegmentStorage(
            namespace="video_segment_feature",
            global_config=asdict(self),
            embedding_func=None,  # Embedding is handled inside the class
        )

        self.video_path_db = self.key_string_value_json_storage_cls(
            namespace="video_path", global_config=asdict(self)
        )

        self.video_segments = self.key_string_value_json_storage_cls(
            namespace="video_segments", global_config=asdict(self)
        )

        self.entities_vdb = (
            self.vector_db_storage_cls(
                namespace="entities",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"},
            )
            if self.enable_local
            else None
        )

        self.chunks_vdb = (
            self.vector_db_storage_cls(
                namespace="chunks",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
            )
            if self.enable_naive_rag
            else None
        )

        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=asdict(self)
        )

        self.embedding_func = limit_async_func_call(self.llm.embedding_func_max_async)(wrap_embedding_func_with_attrs(
                embedding_dim = self.llm.embedding_dim,
                max_token_size = self.llm.embedding_max_token_size,
                model_name = self.llm.embedding_model_name)(self.llm.embedding_func))

        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=asdict(self)
        )

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=asdict(self)
            )
            if self.enable_llm_cache
            else None
        )

    async def _save_video_segments(self):
        tasks = []
        for storage_inst in [
            self.video_segment_feature_vdb,
            self.video_segments,
            self.video_path_db,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def load_models(self):
        """Loads ASR and VLM models into memory."""
        print(f"Loading models to device: {self.device}")
        
        # Load ASR model
        print(f"Loading Whisper model: {self.asr_model_name}")
        self.asr_model = whisper.load_model(self.asr_model_name, device=self.device)
        
        # Load VLM model
        print(f"Loading VLM model: {self.vlm_model_path}")
        self.vlm_processor = AutoProcessor.from_pretrained(self.vlm_model_path)
        dtype = torch.bfloat16 if (self.device == "cuda" and torch.cuda.is_bf16_supported()) else (
            torch.float16 if self.device == "cuda" else torch.float32
        )
        self.vlm_model = AutoModelForImageTextToText.from_pretrained(
            self.vlm_model_path, torch_dtype=dtype
        ).to(self.device)
        print("Models loaded successfully ✅")

    def _speech_to_text(self, video_name, segment_index2name):
        """Extracts text from audio segments."""
        if not self.asr_model:
            raise RuntimeError("ASR model not loaded. Call load_models() first.")
            
        cache_path = os.path.join(self.working_dir, '_cache', video_name)
        transcripts = {}
        for index in tqdm(segment_index2name, desc=f"Speech Recognition for {video_name}"):
            segment_name = segment_index2name[index]
            audio_file = os.path.join(cache_path, f"{segment_name}.{self.audio_output_format}")

            if not os.path.exists(audio_file):
                transcripts[index] = ""
                continue
            
            asr_output = self.asr_model.transcribe(audio_file)
            result = "".join([f"[{s['start']:.2f}s -> {s['end']:.2f}s] {s['text']}" for s in asr_output["segments"]])
            transcripts[index] = result
        
        return transcripts

    def _segment_caption(self, video_name, segment_index2name, transcripts, segment_times_info):
        """Generates captions for video segments."""
        if not self.vlm_model or not self.vlm_processor:
            raise RuntimeError("VLM model not loaded. Call load_models() first.")

        local_captions = {}
        with VideoFileClip(self.video_path) as video:
            for idx in tqdm(segment_index2name.keys(), desc=f"Captioning Video {video_name}"):
                frame_times = segment_times_info[idx]["frame_times"]
                video_frames = self._encode_video_frames(video, frame_times)

                prompt_text = (
                    f"The transcript of the current video:\n{transcripts[idx]}\n"
                    "Now provide a description (caption) of the video in English."
                )
                user_content = [{"type": "image"} for _ in range(len(video_frames))]
                user_content.append({"type": "text", "text": prompt_text})
                messages = [{"role": "user", "content": user_content}]

                prompt_str = self.vlm_processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                
                dtype = self.vlm_model.dtype
                batch = self.vlm_processor(
                    text=[prompt_str], images=[video_frames], return_tensors="pt"
                ).to(self.device, dtype=dtype)

                out_ids = self.vlm_model.generate(**batch, do_sample=False, max_new_tokens=128)
                text = self.vlm_processor.batch_decode(out_ids, skip_special_tokens=True)[0]
                cleaned = (text.split("Assistant:")[-1] or text).strip()
                local_captions[idx] = cleaned.replace("\n", "").replace("<|endoftext|>", "")

                del video_frames
                if self.device == "cuda":
                    torch.cuda.empty_cache()
        return local_captions

    @staticmethod
    def _encode_video_frames(video, frame_times):
        """Helper to extract and resize frames from a video clip."""
        frames = [Image.fromarray(video.get_frame(t).astype('uint8')).resize((1280, 720)) for t in frame_times]
        return frames

    @staticmethod
    def _merge_segment_information(segment_index2name, segment_times_info, transcripts, captions):
        """Merges all extracted information into a single dictionary."""
        inserting_segments = {}
        for index in segment_index2name:
            segment_name = segment_index2name[index]
            inserting_segments[index] = {
                "time": '-'.join(segment_name.split('-')[-2:]),
                "content": f"Caption:\n{captions[index]}\nTranscript:\n{transcripts[index]}\n\n",
                "transcript": transcripts[index],
                "frame_times": segment_times_info[index]["frame_times"].tolist()
            }
        return inserting_segments

    def inference_process(self):
        """Main method to run the full knowledge extraction pipeline."""
        video_name = os.path.basename(self.video_path).split('.')[0]
        loop = always_get_an_event_loop()
        
        # 1. Split video into segments
        print("Step 1: Splitting video...")
        segment_index2name, segment_times_info = split_video(
            self.video_path, self.working_dir, self.video_segment_length, 
            self.rough_num_frames_per_segment, self.audio_output_format
        )
        
        # 2. ASR and VLM inference
        print("Step 2: Running ASR and VLM inference...")
        transcripts = self._speech_to_text(video_name, segment_index2name)
        captions = self._segment_caption(video_name, segment_index2name, transcripts, segment_times_info)
        
        # 3. Unify information
        print("Step 3: Merging segment information...")
        segments_information = self._merge_segment_information(
            segment_index2name, segment_times_info, transcripts, captions
        )    

        loop.run_until_complete(self.video_segments.upsert(
            {video_name: segments_information}
        )) 

        return segments_information, segment_index2name

    async def ainsert(self, new_video_segment):
        await self._insert_start()
        try:
            # ---------- chunking
            inserting_chunks = get_chunks(
                new_videos=new_video_segment,
                chunk_func=self.chunk_func,
                max_token_size=self.chunk_token_size,
            )
            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if self.enable_naive_rag:
                await self.chunks_vdb.upsert(inserting_chunks)

            # ---------- extract/summary entity and upsert to graph
            maybe_new_kg, _, _ = await self.entity_extraction_func(
                inserting_chunks,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                global_config=asdict(self),
            )
            self.chunk_entity_relation_graph = maybe_new_kg
            # ---------- commit upsertings and indexing
            await self.text_chunks.upsert(inserting_chunks)
        finally:
            await self._insert_done()

    async def _insert_start(self):
        tasks = []
        for storage_inst in [
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_start_callback())
        await asyncio.gather(*tasks)

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.text_chunks,
            self.llm_response_cache,
            self.entities_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
            self.video_segment_feature_vdb,
            self.video_segments,
            self.video_path_db,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)


    def build_graph_process(self, segment_index2name):

        video_name = os.path.basename(self.video_path).split('.')[0]

        # 1. Encode and store video segment features
        print("Step 4: Encoding and storing video features...")
        loop = always_get_an_event_loop()
        loop.run_until_complete(
            self.video_segment_feature_vdb.upsert(video_name, segment_index2name, self.video_output_format)
        )
        print("Video processing complete.")

        # 2. Delete the cache file
        print("Step 5: Deleting the cache file...")
        video_segment_cache_path = os.path.join(self.working_dir, '_cache', video_name)
        if os.path.exists(video_segment_cache_path):
            shutil.rmtree(video_segment_cache_path)
        print("Cache file deleted")

        # 3. Save video information
        print("Step 6: Saving video information...")
        loop.run_until_complete(self._save_video_segments())
        print("Video information saved")

        #
        loop.run_until_complete(self.ainsert(self.video_segments._data))   


if __name__ == '__main__':
    # Example of how to use the class
    VIDEO_FILE = "/home/gatv-projects/Desktop/project/chatbot_system/downloads/My_Nintendo_Switch_2_Review.mp4" 
    
    # 1. Initialize the extractor
    extractor = VideoKnowledgeExtractor(video_path=VIDEO_FILE) 
    
    # 2. Load the AI models
    extractor.load_models()
    
    # 3. Run the ineference processing pipeline
    final_segments_info, segment_index2name = extractor.inference_process()
    
    # 4. Print video info results
    print("\n" + "<>"*20)
    print("Final Extracted Information:")
    print(json.dumps(final_segments_info, indent=4, ensure_ascii=False))
    print("<>"*20 + "\n")

    # 5. Run the build graph processing pipeline
    extractor.build_graph_process(segment_index2name = segment_index2name)