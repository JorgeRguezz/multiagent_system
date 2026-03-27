from __future__ import annotations

import asyncio
import gc
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from huggingface_hub import snapshot_download

# Force CUDA runtime initialization before importing llama_cpp.
# In the current env layout the required CUDA libs are exposed through the
# nvidia wheels, and touching torch.cuda first matches the working playground path.
if torch.cuda.is_available():
    torch.cuda.get_device_properties(0)

from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

from ._utils import EmbeddingFunc, compute_args_hash, wrap_embedding_func_with_attrs
from .base import BaseKVStorage


@dataclass
class LLMConfig:
    embedding_func_raw: Callable[..., Any]
    embedding_model_name: str
    embedding_dim: int
    embedding_max_token_size: int
    embedding_batch_num: int
    embedding_func_max_async: int
    query_better_than_threshold: float

    best_model_func_raw: Callable[..., Any]
    best_model_name: str
    best_model_max_token_size: int
    best_model_max_async: int

    cheap_model_func_raw: Callable[..., Any]
    cheap_model_name: str
    cheap_model_max_token_size: int
    cheap_model_max_async: int

    embedding_func: EmbeddingFunc | None = None
    best_model_func: Callable[..., Any] | None = None
    cheap_model_func: Callable[..., Any] | None = None

    def __post_init__(self) -> None:
        embedding_wrapper = wrap_embedding_func_with_attrs(
            embedding_dim=self.embedding_dim,
            max_token_size=self.embedding_max_token_size,
            model_name=self.embedding_model_name,
        )
        self.embedding_func = embedding_wrapper(self.embedding_func_raw)
        self.best_model_func = lambda prompt, *args, **kwargs: self.best_model_func_raw(
            self.best_model_name, prompt, *args, **kwargs
        )
        self.cheap_model_func = lambda prompt, *args, **kwargs: self.cheap_model_func_raw(
            self.cheap_model_name, prompt, *args, **kwargs
        )


_embedding_model: SentenceTransformer | None = None
_oss_llm: Llama | None = None
_final_marker_pattern = re.compile(r"final<\|message\|>")
_final_channel_marker = "<|start|>assistant<|channel|>final<|message|>"

# GPT-OSS-20B config used by both build and inference.
OSS_MODEL_ID = "unsloth/gpt-oss-20b-GGUF"
OSS_QUANT_FILE = "gpt-oss-20b-F16.gguf"
OSS_LOCAL_DIR = str(Path(__file__).resolve().parents[1] / "gpt-oss-20b")
OSS_N_GPU_LAYERS = -1
OSS_N_CTX = 16384
OSS_N_BATCH = 512
OSS_F16_KV = True


def _get_embedding_model_instance(model_name: str) -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model


def _get_oss_llm_instance() -> Llama:
    global _oss_llm
    if _oss_llm is None:
        model_path = snapshot_download(
            repo_id=OSS_MODEL_ID,
            local_dir=OSS_LOCAL_DIR,
            allow_patterns=[OSS_QUANT_FILE],
        )
        full_path = os.path.join(model_path, OSS_QUANT_FILE)
        _oss_llm = Llama(
            model_path=full_path,
            n_gpu_layers=OSS_N_GPU_LAYERS,
            n_ctx=OSS_N_CTX,
            n_batch=OSS_N_BATCH,
            f16_kv=OSS_F16_KV,
            verbose=False,
        )
    return _oss_llm


def _format_chat_prompt(
    system_prompt: str | None,
    user_prompt: str,
    history_messages: list[dict],
) -> str:
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

    if _final_channel_marker in clean_output:
        idx = clean_output.rfind(_final_channel_marker)
        thought_process = clean_output[:idx].strip()
        answer = clean_output[idx + len(_final_channel_marker) :].strip()
        answer = answer.replace("<|end|>", "").strip()
        return thought_process, answer, True

    matches = list(_final_marker_pattern.finditer(clean_output))
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
    return text[min(starts) :].strip()


def _truncate_on_repetition(
    text: str,
    window: int = 30,
    repeat_threshold: int = 3,
) -> str:
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


async def oss_llm_complete(
    model_name,
    prompt,
    system_prompt=None,
    history_messages=None,
    **kwargs,
) -> str | dict[str, Any]:
    del model_name  # kept for interface parity with LLMConfig wrappers

    hashing_kv: BaseKVStorage | None = kwargs.pop("hashing_kv", None)
    return_metadata = bool(kwargs.pop("return_metadata", False))
    history_messages = history_messages or []
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
    llm = _get_oss_llm_instance()
    loop = asyncio.get_running_loop()

    def generate() -> str | dict[str, Any]:
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
    model = _get_embedding_model_instance(model_name)
    loop = asyncio.get_running_loop()

    def encode() -> np.ndarray:
        return np.array(model.encode(texts))

    return await loop.run_in_executor(None, encode)


local_llm_config = LLMConfig(
    embedding_func_raw=local_embedding,
    embedding_model_name="all-MiniLM-L6-v2",
    embedding_dim=384,
    embedding_max_token_size=512,
    embedding_batch_num=32,
    embedding_func_max_async=4,
    query_better_than_threshold=0.2,
    best_model_func_raw=oss_llm_complete,
    best_model_name=OSS_MODEL_ID,
    best_model_max_token_size=OSS_N_CTX,
    best_model_max_async=1,
    cheap_model_func_raw=oss_llm_complete,
    cheap_model_name=OSS_MODEL_ID,
    cheap_model_max_token_size=OSS_N_CTX,
    cheap_model_max_async=1,
)


async def oss_llm_batch_generate(
    prompts: list[str],
    system_prompt: str | None = None,
    max_tokens: int = 3000,
) -> list[str]:
    llm = _get_oss_llm_instance()
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
        return _truncate_on_repetition(cleaned)

    async def run_batch() -> list[str]:
        results = []
        for prompt in prompts:
            results.append(await loop.run_in_executor(None, generate_one, prompt))
        return results

    return await run_batch()


def shutdown_all_llm_resources() -> None:
    global _embedding_model, _oss_llm

    if _oss_llm is not None:
        close_fn = getattr(_oss_llm, "close", None)
        if callable(close_fn):
            close_fn()
        _oss_llm = None

    _embedding_model = None
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


__all__ = [
    "LLMConfig",
    "local_llm_config",
    "local_embedding",
    "oss_llm_batch_generate",
    "oss_llm_complete",
    "shutdown_all_llm_resources",
]
