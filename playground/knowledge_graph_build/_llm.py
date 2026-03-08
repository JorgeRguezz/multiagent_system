import asyncio
import os
import re
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed
from huggingface_hub import snapshot_download
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

from ._utils import compute_args_hash, wrap_embedding_func_with_attrs
from .base import BaseKVStorage
from ._utils import EmbeddingFunc


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
    embedding_func: EmbeddingFunc = None
    best_model_func: callable = None
    cheap_model_func: callable = None

    def __post_init__(self):
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


# GPT-OSS-20B GGUF config (aligned with playground/test_gpt_oss_20b.py)
OSS_MODEL_ID = "unsloth/gpt-oss-20b-GGUF"
OSS_QUANT_FILE = "gpt-oss-20b-F16.gguf"
OSS_LOCAL_DIR = "/home/gatv-projects/Desktop/project/gpt-oss-20b"
OSS_N_GPU_LAYERS = -1
OSS_N_CTX = 16384
OSS_VERBOSE = False


global_local_llm = None
global_local_embedding_model = None
FINAL_MARKER_PATTERN = re.compile(r"final<\\|message\\|>")
FINAL_CHANNEL_MARKER = "<|start|>assistant<|channel|>final<|message|>"


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

    # Prefer the explicit final channel marker and use the last one in case the model
    # echoes template examples earlier in the generation.
    if FINAL_CHANNEL_MARKER in clean_output:
        idx = clean_output.rfind(FINAL_CHANNEL_MARKER)
        thought_process = clean_output[:idx].strip()
        answer = clean_output[idx + len(FINAL_CHANNEL_MARKER):].strip()
        answer = answer.replace("<|end|>", "").strip()
        return thought_process, answer, True

    # Fallback to plain "final<|message|>" marker; also use the last occurrence.
    matches = list(FINAL_MARKER_PATTERN.finditer(clean_output))
    if matches:
        last = matches[-1]
        thought_process = clean_output[: last.start()].strip()
        answer = clean_output[last.end() :].strip()
        answer = answer.replace("<|end|>", "").strip()
        return thought_process, answer, True

    # Final fallback: keep output, then trim obvious leading assistant wrappers.
    answer = clean_output
    for marker in (
        "<|start|>assistant<|channel|>analysis<|message|>",
        "<|start|>assistant<|message|>",
    ):
        if marker in answer:
            answer = answer.split(marker, maxsplit=1)[-1].strip()
    answer = answer.replace("<|end|>", "").strip()
    return "", answer, False


def _debug_print_thought_and_answer(stage: str, thought: str, answer: str, malformed: bool = False):
    print("\n" + "=" * 24 + f" {stage} " + "=" * 24)
    if malformed:
        print("STATUS: MALFORMED OUTPUT (final<|message|> marker not found)")
    print(">>>>>>>>>> THOUGHT TOKENS <<<<<<<<<<")
    print(thought if thought else "<EMPTY>")
    print(">>>>>>>>>> ANSWER TOKENS <<<<<<<<<<")
    print(answer if answer else "<EMPTY>")
    print("=" * 60)


def _trim_to_extraction_payload(text: str) -> str:
    """
    Keep only the extraction payload when present:
    starts from first tuple or completion marker, whichever appears first.
    """
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
    """
    Guard against degenerate loops by cutting output when the same token window
    starts repeating too many times consecutively.
    """
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
                cut_idx = i
                return " ".join(tokens[:cut_idx]).strip()
        else:
            repeat_count = 1
    return text


def get_local_llm_instance(model_path):
    global global_local_llm
    if global_local_llm is None:
        downloaded_dir = snapshot_download(
            repo_id=OSS_MODEL_ID,
            local_dir=OSS_LOCAL_DIR,
            allow_patterns=[OSS_QUANT_FILE],
        )
        full_path = os.path.join(downloaded_dir, OSS_QUANT_FILE)
        print(f"Loading local GPT-OSS-20B GGUF from: {full_path}")
        global_local_llm = Llama(
            model_path=full_path,
            n_gpu_layers=OSS_N_GPU_LAYERS,
            n_ctx=OSS_N_CTX,
            verbose=OSS_VERBOSE,
        )
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
    full_prompt = _format_chat_prompt(system_prompt, prompt, history_messages)

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
        thought, answer, found = _split_thought_and_answer(raw_text)
        _debug_print_thought_and_answer(
            stage="GPT-OSS SINGLE INFERENCE",
            thought=thought,
            answer=answer,
            malformed=not found,
        )
        cleaned = _trim_to_extraction_payload(answer)
        cleaned = _truncate_on_repetition(cleaned)
        return cleaned

    response_text = await loop.run_in_executor(None, generate)

    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": response_text, "model": model_path}})
        await hashing_kv.index_done_callback()

    return response_text


async def local_llm_complete(model_name, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await local_llm_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def local_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    model = get_local_embedding_model_instance(model_name)
    loop = asyncio.get_running_loop()

    def encode():
        return model.encode(texts)

    embeddings = await loop.run_in_executor(None, encode)
    return np.array(embeddings)


local_llm_config = LLMConfig(
    embedding_func_raw=local_embedding,
    embedding_model_name="all-MiniLM-L6-v2",
    embedding_dim=384,
    embedding_max_token_size=512,
    embedding_batch_num=32,
    embedding_func_max_async=4,
    query_better_than_threshold=0.2,
    best_model_func_raw=local_llm_complete,
    best_model_name=OSS_MODEL_ID,
    best_model_max_token_size=OSS_N_CTX,
    best_model_max_async=1,
    cheap_model_func_raw=local_llm_complete,
    cheap_model_name=OSS_MODEL_ID,
    cheap_model_max_token_size=OSS_N_CTX,
    cheap_model_max_async=1,
)


async def local_llm_batch_generate(
    model_path: str,
    prompts: list[str],
    system_prompt: str | None = None,
    max_tokens: int = 1024,
) -> list[str]:
    """
    Batch helper for GPT-OSS-20B GGUF via llama-cpp.
    llama-cpp does not expose the same multi-prompt API as vLLM, so this loops.
    """
    llm = get_local_llm_instance(model_path)
    loop = asyncio.get_running_loop()

    def generate():
        outputs = []
        for idx, prompt in enumerate(prompts):
            full_prompt = _format_chat_prompt(system_prompt, prompt, [])
            out = llm(
                full_prompt,
                max_tokens=max_tokens,
                temperature=0.1,
                top_p=1.0,
                top_k=0,
                repeat_penalty=1.12,
            )
            raw_text = out["choices"][0]["text"]
            thought, answer, found = _split_thought_and_answer(raw_text)
            _debug_print_thought_and_answer(
                stage=f"GPT-OSS BATCH INFERENCE #{idx}",
                thought=thought,
                answer=answer,
                malformed=not found,
            )
            cleaned = _trim_to_extraction_payload(answer)
            cleaned = _truncate_on_repetition(cleaned)
            outputs.append(cleaned)
        return outputs

    return await loop.run_in_executor(None, generate)


def shutdown_local_llm():
    """
    Shuts down the global llama-cpp instance and PyTorch distributed process group.
    """
    global global_local_llm
    if global_local_llm is not None:
        print("Attempting to shut down local LLM and distributed process group...")

        if hasattr(global_local_llm, "close"):
            print("Closing llama-cpp model...")
            global_local_llm.close()
        elif hasattr(global_local_llm, "shutdown"):
            print("Shutting down model via shutdown()...")
            global_local_llm.shutdown()
        else:
            print("Model shutdown method not found.")

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
