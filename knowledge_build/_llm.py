import numpy as np
import torch
import torch.distributed
from dataclasses import asdict, dataclass, field

import os
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer
import asyncio

from ._utils import compute_args_hash, wrap_embedding_func_with_attrs
from .base import BaseKVStorage
from ._utils import EmbeddingFunc

# GPT-OSS-20B (GGUF) via llama-cpp-python
from huggingface_hub import snapshot_download
from llama_cpp import Llama

# Setup LLM Configuration.
@dataclass
class LLMConfig:
    # To be set
    embedding_func_raw: callable
    embedding_model_name: str
    embedding_dim: int
    embedding_max_token_size: int
    embedding_batch_num: int    
    embedding_func_max_async: int 
    query_better_than_threshold: float
    
    best_model_func_raw: callable
    best_model_name: str    
    best_model_max_token_size: int
    best_model_max_async: int
    
    cheap_model_func_raw: callable
    cheap_model_name: str
    cheap_model_max_token_size: int
    cheap_model_max_async: int

    # Assigned in post init
    embedding_func: EmbeddingFunc  = None    
    best_model_func: callable = None    
    cheap_model_func: callable = None
    

    def __post_init__(self):
        embedding_wrapper = wrap_embedding_func_with_attrs(
            embedding_dim = self.embedding_dim,
            max_token_size = self.embedding_max_token_size,
            model_name = self.embedding_model_name)
        self.embedding_func = embedding_wrapper(self.embedding_func_raw)
        self.best_model_func = lambda prompt, *args, **kwargs: self.best_model_func_raw(
            self.best_model_name, prompt, *args, **kwargs
        )

        self.cheap_model_func = lambda prompt, *args, **kwargs: self.cheap_model_func_raw(
            self.cheap_model_name, prompt, *args, **kwargs
        )

###### Local VLLM Configuration
# NOTE: You need to install vllm and sentence-transformers in your venv_llm:
# pip install vllm sentence-transformers

global_local_llm = None
global_local_embedding_model = None
global_oss_llm = None

# GPT-OSS-20B config (mirrors playground/test_gpt_oss_20b.py)
OSS_MODEL_ID = "unsloth/gpt-oss-20b-GGUF"
OSS_QUANT_FILE = "gpt-oss-20b-Q4_K_M.gguf"
OSS_LOCAL_DIR = "./gpt-oss-20b"
OSS_N_GPU_LAYERS = -1
OSS_N_CTX = 20000
OSS_N_BATCH = 512
OSS_F16_KV = True

def set_global_llm_instance(llm):
    """Sets the global LLM instance from an external source."""
    global global_local_llm
    if global_local_llm is None:
        print("Setting global LLM instance from external source.")
        global_local_llm = llm

def get_local_llm_instance(model_path):
    """
    Gets the global LLM instance.
    If the global instance is not set, it will load a new one as a fallback.
    """
    global global_local_llm
    if global_local_llm is None:
        print(f"WARNING: Global LLM not set. Loading a new local LLM instance from: {model_path}")
        # You might need to adjust parameters like tensor_parallel_size
        # depending on your hardware.
        global_local_llm = LLM(model=model_path, trust_remote_code=True, max_model_len=8192, max_num_seqs=32)
    return global_local_llm

def get_local_embedding_model_instance(model_name):
    global global_local_embedding_model
    if global_local_embedding_model is None:
        print(f"Loading local embedding model: {model_name}")
        global_local_embedding_model = SentenceTransformer(model_name)
    return global_local_embedding_model

def get_oss_llm_instance():
    global global_oss_llm
    if global_oss_llm is None:
        model_path = snapshot_download(
            repo_id=OSS_MODEL_ID,
            local_dir=OSS_LOCAL_DIR,
            allow_patterns=[OSS_QUANT_FILE],
        )
        full_path = os.path.join(model_path, OSS_QUANT_FILE)
        global_oss_llm = Llama(
            model_path=full_path,
            n_gpu_layers=OSS_N_GPU_LAYERS,
            n_ctx=OSS_N_CTX,
            n_batch=OSS_N_BATCH,
            f16_kv=OSS_F16_KV,
            verbose=False,
        )
    return global_oss_llm

async def local_llm_complete_if_cache(
    model_path, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None:
        args_hash = compute_args_hash(model_path, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    llm = get_local_llm_instance(model_path)
    
    # Constructing a single prompt string. This might need adjustment
    # based on the specific model's expected chat format.
    full_prompt = ""
    if system_prompt:
        full_prompt += f"System: {system_prompt}\n\n"
    for msg in history_messages:
        full_prompt += f"{msg['role'].capitalize()}: {msg['content']}\n\n"
    full_prompt += f"User: {prompt}\n\nAssistant:"

    sampling_params = SamplingParams(temperature=0.7, max_tokens=4096) # Adjust as needed

    loop = asyncio.get_running_loop()
    
    def generate():
        # llm.generate is synchronous
        outputs = llm.generate(full_prompt, sampling_params)
        return outputs[0].outputs[0].text

    response_text = await loop.run_in_executor(None, generate)

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response_text, "model": model_path}}
        )
        await hashing_kv.index_done_callback()
        
    return response_text

async def local_llm_complete(model_name, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await local_llm_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )

async def oss_llm_complete(model_name, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    # model_name unused; kept for interface parity
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None:
        args_hash = compute_args_hash(OSS_MODEL_ID, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    full_prompt = ""
    if system_prompt:
        full_prompt += f"<s><|begin_system|>\n{system_prompt}\n"
    full_prompt += "<|begin_user|>\n" + prompt + "\n<|begin_assistant|>\n"

    llm = get_oss_llm_instance()
    loop = asyncio.get_running_loop()

    def generate():
        output = llm(
            full_prompt,
            max_tokens=kwargs.get("max_tokens", 10000),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            stop=kwargs.get("stop", ["User:"]),
        )
        return output["choices"][0]["text"].strip()

    response_text = await loop.run_in_executor(None, generate)

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response_text, "model": OSS_MODEL_ID}}
        )
        await hashing_kv.index_done_callback()

    return response_text

async def local_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    model = get_local_embedding_model_instance(model_name)
    loop = asyncio.get_running_loop()
    
    def encode():
        # encode is synchronous
        return model.encode(texts)
        
    embeddings = await loop.run_in_executor(None, encode)
    return np.array(embeddings)

local_llm_config = LLMConfig(
    embedding_func_raw = local_embedding,
    embedding_model_name = "all-MiniLM-L6-v2", # Good general-purpose small model
    embedding_dim = 384,
    embedding_max_token_size=512,
    embedding_batch_num = 32,
    embedding_func_max_async = 4,
    query_better_than_threshold = 0.2,

    best_model_func_raw = local_llm_complete,
    best_model_name = "/home/gatv-projects/Desktop/project/llama-3.2-3B-Instruct", 
    best_model_max_token_size = 4096, # Adjust to your model's context window
    best_model_max_async  = 1,
    
    cheap_model_func_raw = oss_llm_complete,
    cheap_model_name = OSS_MODEL_ID,
    cheap_model_max_token_size = OSS_N_CTX,
    cheap_model_max_async = 1
    )

async def local_llm_batch_generate(model_path: str, prompts: list[str]) -> list[str]:
    """
    Generates responses for a batch of prompts using a single, efficient call to vLLM.
    This is the recommended approach for local inference to maximize throughput.
    """
    llm = get_local_llm_instance(model_path)
    # Using low temperature for deterministic, repeatable entity extraction.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=4096) 

    loop = asyncio.get_running_loop()

    def generate():
        # vLLM's generate method is optimized for processing a list of prompts.
        outputs = llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]

    # Run the single, batched generation call in an executor to be async-friendly.
    # This is safe because it's one call, not many concurrent ones.
    response_texts = await loop.run_in_executor(None, generate)
    
    return response_texts

async def oss_llm_batch_generate(prompts: list[str]) -> list[str]:
    llm = get_oss_llm_instance()
    loop = asyncio.get_running_loop()

    def generate_one(prompt: str) -> str:
        full_prompt = (
            "<s><|begin_system|>\n"
            "You are a helpful assistant.\n"
            "<|begin_user|>\n"
            f"{prompt}\n"
            "<|begin_assistant|>\n"
        )
        output = llm(
            full_prompt,
            max_tokens=10000,
            temperature=0.7,
            top_p=0.9,
            stop=["User:"],
        )
        return output["choices"][0]["text"].strip()

    async def run_batch():
        results = []
        for p in prompts:
            results.append(await loop.run_in_executor(None, generate_one, p))
        return results

    return await run_batch()


def shutdown_local_llm():
    """
    Shuts down the global vLLM instance and PyTorch distributed process group to release resources.
    """
    global global_local_llm
    if global_local_llm is not None:
        print("Attempting to shut down local LLM and distributed process group...")
        
        # Try vllm's own shutdown first
        if hasattr(global_local_llm, 'llm_engine') and hasattr(global_local_llm.llm_engine, 'shutdown'):
            print("Shutting down the vLLM engine via engine.shutdown()...")
            global_local_llm.llm_engine.shutdown()
        elif hasattr(global_local_llm, 'shutdown'):
            print("Shutting down the vLLM engine via llm.shutdown()...")
            global_local_llm.shutdown()
        else:
            print("vLLM shutdown method not found.")

        # Explicitly destroy the process group as a fallback/cleanup
        if torch.distributed.is_initialized():
            print("PyTorch distributed is initialized. Destroying process group...")
            torch.distributed.destroy_process_group()
            print("Process group destroyed.")
        else:
            print("PyTorch distributed was not initialized, no need to destroy process group.")

        global_local_llm = None
        print("Local LLM resources have been released.")
    else:
        print("No local LLM instance to shut down.")
