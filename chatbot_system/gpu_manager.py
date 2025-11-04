import torch
import time
import gc
from vllm import LLM
from vlm_processor import VLMProcessor

class GPUModelManager:
    """
    Manages loading both LLM and VLM models to the GPU simultaneously,
    and handles cleanup in case of errors like out-of-memory.
    """
    def __init__(self, llm_config, vlm_config):
        self.llm = None
        self.vlm_processor = None
        self.llm_config = llm_config
        self.vlm_config = vlm_config

    def _load_llm(self):
        print("Loading LLM to GPU...")
        start_time = time.time()
        self.llm = LLM(
            model=self.llm_config['model_name'],
            dtype=self.llm_config['dtype'],
            max_model_len=self.llm_config['max_model_len'],
            gpu_memory_utilization=self.llm_config['gpu_memory_utilization'],
            load_format=self.llm_config['load_format']
        )
        end_time = time.time()
        print(f"LLM loaded to GPU in {end_time - start_time:.2f} seconds.")
        return self.llm

    def _load_vlm(self):
        print("Loading VLM to GPU...")
        start_time = time.time()
        self.vlm_processor = VLMProcessor(model_id=self.vlm_config['model_id'])
        self.vlm_processor._load_model()
        end_time = time.time()
        print(f"VLM loaded to GPU in {end_time - start_time:.2f} seconds.")
        return self.vlm_processor

    def load_models_to_gpu(self):
        try:
            self._load_llm()
            self._load_vlm()
            print("Both LLM and VLM loaded to GPU successfully.")
        except torch.cuda.OutOfMemoryError:
            print("GPU out of memory. Cleaning up and stopping.")
            self.cleanup()
            raise
        except Exception as e:
            print(f"An error occurred during model loading: {e}")
            self.cleanup()
            raise

    def cleanup(self):
        print("Cleaning up models from GPU...")
        if self.llm is not None:
            del self.llm
            self.llm = None
        if self.vlm_processor is not None:
            if self.vlm_processor.model:
                del self.vlm_processor.model
            if self.vlm_processor.processor:
                del self.vlm_processor.processor
            del self.vlm_processor
            self.vlm_processor = None
        
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()
        print("Cleanup complete.")

    def get_llm(self):
        return self.llm

    def get_vlm_processor(self):
        return self.vlm_processor
