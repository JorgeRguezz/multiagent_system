import numpy as np

from dataclasses import asdict, dataclass, field

import os
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer
import asyncio

from ._utils import compute_args_hash, wrap_embedding_func_with_attrs
from .base import BaseKVStorage
from ._utils import EmbeddingFunc

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

def get_local_llm_instance(model_path):
    global global_local_llm
    if global_local_llm is None:
        print(f"Loading local LLM from: {model_path}")
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
    
    cheap_model_func_raw = local_llm_complete,
    cheap_model_name = "/home/gatv-projects/Desktop/project/llama-3.2-3B-Instruct", 
    cheap_model_max_token_size = 4096, # Adjust to your model's context window
    cheap_model_max_async = 1
)