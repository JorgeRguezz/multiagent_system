import numpy as np
import torch
import torch.distributed
import gc
from dataclasses import asdict, dataclass, field

import os
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer
import asyncio

from ._utils import compute_args_hash, wrap_embedding_func_with_attrs
from .base import BaseKVStorage
from ._utils import EmbeddingFunc

import re

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
FINAL_MARKER_PATTERN = re.compile(r"final<\|message\|>")
FINAL_CHANNEL_MARKER = "<|start|>assistant<|channel|>final<|message|>"

# GPT-OSS-20B config (mirrors playground/test_gpt_oss_20b.py)
OSS_MODEL_ID = "unsloth/gpt-oss-20b-GGUF"
OSS_QUANT_FILE = "gpt-oss-20b-F16.gguf"
OSS_LOCAL_DIR = "/home/gatv-projects/Desktop/project/gpt-oss-20b"
OSS_N_GPU_LAYERS = -1
OSS_N_CTX = 16384
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


def _format_chat_prompt(system_prompt: str | None, user_prompt: str, history_messages: list[dict]) -> str:
    parts = []
    if system_prompt:
        parts.append(f"<|start|>system<|message|>{system_prompt}<|end|>")
    for msg in history_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|start|>{role}<|message|>{content}<|end|>")
    parts.append(f"<|start|>user<|message|>{user_prompt}<|end|>")
    return "".join(parts)


def _split_thought_and_answer(raw_text: str) -> tuple[str, str, bool]:
    clean_output = (raw_text or "").strip()
    if not clean_output:
        return "", "", False

    if FINAL_CHANNEL_MARKER in clean_output:
        idx = clean_output.rfind(FINAL_CHANNEL_MARKER)
        thought_process = clean_output[:idx].strip()
        answer = clean_output[idx + len(FINAL_CHANNEL_MARKER):].strip()
        answer = answer.replace("<|end|>", "").strip()
        return thought_process, answer, True

    matches = list(FINAL_MARKER_PATTERN.finditer(clean_output))
    if matches:
        last = matches[-1]
        thought_process = clean_output[: last.start()].strip()
        answer = clean_output[last.end() :].strip()
        answer = answer.replace("<|end|>", "").strip()
        return thought_process, answer, True

    answer = clean_output
    for marker in (
        "<|start|>assistant<|channel|>analysis<|message|>",
        "<|start|>assistant<|message|>",
    ):
        if marker in answer:
            answer = answer.split(marker, maxsplit=1)[-1].strip()
    answer = answer.replace("<|end|>", "").strip()
    return "", answer, False


def _trim_to_extraction_payload(text: str) -> str:
    if not text:
        return text
    starts = []
    for marker in ('("entity"', '("relationship"', "<|COMPLETE|>"):
        pos = text.find(marker)
        if pos != -1:
            starts.append(pos)
    if not starts:
        return text.strip()
    return text[min(starts):].strip()


def _truncate_on_repetition(text: str, window: int = 30, repeat_threshold: int = 3) -> str:
    if not text:
        return text
    tokens = text.split()
    if len(tokens) < window * (repeat_threshold + 1):
        return text

    repeat_count = 1
    for i in range(window, len(tokens), window):
        prev = tokens[i - window : i]
        cur = tokens[i : i + window]
        if len(cur) < window:
            break
        if cur == prev:
            repeat_count += 1
            if repeat_count >= repeat_threshold:
                return " ".join(tokens[:i]).strip()
        else:
            repeat_count = 1
    return text

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
    return_metadata = bool(kwargs.pop("return_metadata", False))
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None and not return_metadata:
        args_hash = compute_args_hash(OSS_MODEL_ID, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    full_prompt = _format_chat_prompt(system_prompt, prompt, history_messages)

    llm = get_oss_llm_instance()
    loop = asyncio.get_running_loop()

    def generate():
        output = llm(
            full_prompt,
            max_tokens=kwargs.get("max_tokens", 3500),
            temperature=kwargs.get("temperature", 0.1),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", 0),
            repeat_penalty=kwargs.get("repeat_penalty", 1.12),
        )
        raw_text = output["choices"][0]["text"]
        thought_process, answer, has_final_marker = _split_thought_and_answer(raw_text)
        cleaned = _trim_to_extraction_payload(answer)
        cleaned = _truncate_on_repetition(cleaned)
        if return_metadata:
            return {
                "raw_text": (raw_text or "").strip(),
                "thoughts": thought_process,
                "answer": cleaned,
                "has_final_marker": has_final_marker,
            }
        return cleaned

    response_text = await loop.run_in_executor(None, generate)

    if hashing_kv is not None and not return_metadata:
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

    best_model_func_raw = oss_llm_complete,
    best_model_name = OSS_MODEL_ID,
    best_model_max_token_size = OSS_N_CTX,
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

async def oss_llm_batch_generate(
    prompts: list[str],
    system_prompt: str | None = None,
    max_tokens: int = 3000,
) -> list[str]:
    llm = get_oss_llm_instance()
    loop = asyncio.get_running_loop()

    def generate_one(prompt: str) -> str:
        full_prompt = _format_chat_prompt(system_prompt, prompt, [])
        output = llm(
            full_prompt,
            max_tokens=max_tokens,
            temperature=0.1,
            top_p=1.0,
            top_k=0,
            repeat_penalty=1.12,
        )
        raw_text = output["choices"][0]["text"]
        _, answer, _ = _split_thought_and_answer(raw_text)
        cleaned = _trim_to_extraction_payload(answer)
        cleaned = _truncate_on_repetition(cleaned)
        return cleaned

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


def shutdown_oss_llm():
    """
    Shuts down the global llama.cpp GPT-OSS instance and frees related resources.
    """
    global global_oss_llm
    if global_oss_llm is not None:
        print("Attempting to shut down GPT-OSS llama.cpp instance...")
        try:
            if hasattr(global_oss_llm, "close"):
                global_oss_llm.close()
        except Exception as exc:
            print(f"Warning: failed to close GPT-OSS instance cleanly: {exc}")
        global_oss_llm = None
        print("GPT-OSS resources have been released.")
    else:
        print("No GPT-OSS instance to shut down.")


def shutdown_embedding_model():
    """
    Releases the global sentence-transformers embedding model instance.
    """
    global global_local_embedding_model
    if global_local_embedding_model is not None:
        print("Releasing embedding model instance...")
        global_local_embedding_model = None
    else:
        print("No embedding model instance to release.")


def shutdown_all_llm_resources():
    """
    Best-effort release of all LLM/embedding resources and CUDA cache.
    """
    shutdown_local_llm()
    shutdown_oss_llm()
    shutdown_embedding_model()

    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception as exc:
        print(f"Warning: cleanup encountered an error: {exc}")
