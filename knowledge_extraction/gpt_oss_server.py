"""
MCP server that runs GPT-OSS-20B (GGUF via llama-cpp-python) for segment caption summarization.
"""
import os
import time
import re
import ctypes
from typing import List

import torch
from huggingface_hub import snapshot_download
from mcp.server.fastmcp import FastMCP

# Ensure torch's bundled CUDA libs are on the loader path before importing llama_cpp
_torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
_existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
if _torch_lib_dir not in _existing_ld:
    os.environ["LD_LIBRARY_PATH"] = f"{_torch_lib_dir}:{_existing_ld}" if _existing_ld else _torch_lib_dir
try:
    # Preload libcudart from torch if available
    ctypes.CDLL(os.path.join(_torch_lib_dir, "libcudart.so.12"))
except OSError:
    # If not found, llama_cpp import will fail and surface the error
    pass

from llama_cpp import Llama

mcp = FastMCP("gpt_oss_server")

# GPT-OSS-20B config (mirrors playground/test_gpt_oss_20b.py)
OSS_MODEL_ID = "unsloth/gpt-oss-20b-GGUF"
OSS_QUANT_FILE = "gpt-oss-20b-Q4_K_M.gguf"
OSS_LOCAL_DIR = "./gpt-oss-20b"
OSS_N_GPU_LAYERS = -1
OSS_N_CTX = 20000
OSS_N_BATCH = 512
OSS_F16_KV = True

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        model_path = snapshot_download(
            repo_id=OSS_MODEL_ID,
            local_dir=OSS_LOCAL_DIR,
            allow_patterns=[OSS_QUANT_FILE],
        )
        full_path = os.path.join(model_path, OSS_QUANT_FILE)
        _llm = Llama(
            model_path=full_path,
            n_gpu_layers=OSS_N_GPU_LAYERS,
            n_ctx=OSS_N_CTX,
            n_batch=OSS_N_BATCH,
            f16_kv=OSS_F16_KV,
            verbose=False,
        )
    return _llm


SYSTEM_PROMPT = """
You are an expert on summarizing League of Legends gameplay based on visual descriptions.
Given the VLM outputs describing the visual content of consecutive video segments, your task is to generate a concise summary of the key events and actions.
Focus on identifying the main champion(s) involved, their actions, and any significant interactions.
Use the VLM output as your primary source of information, and infer the most likely gameplay events based on the visual cues provided. Your summary should be clear, informative, and capture the essence of the gameplay.
"""

# Same cleanup pattern as playground/test_gpt_oss_20b.py
_CLEAN_PATTERN = re.compile(r"<\|end\|><\|start\|>assistant<\|channel\|>final(?:<\|message\|>|\|assistant_output\|?|>| <\|constrain\|><\|assistant\|>)")

@mcp.tool()
async def summarize_segment_captions(captions: List[str]) -> str:
    """
    Summarize a list of per-frame captions into a single segment caption.
    """
    llm = get_llm()
    merged_content = "\n\n--- Next Segment ---\n\n".join(captions)
    prompt = f"""
    Based on the following VLM descriptions of a League of Legends gameplay segment, provide a detailed description of what is happening.

    VLM Outputs of the merged frames:
    {merged_content}
    """

    full_prompt = f"""<s><|begin_system|>
    {SYSTEM_PROMPT}
    <|begin_user|>
    {prompt}
    <|begin_assistant|>
    """

    start = time.time()
    output = llm(
        full_prompt,
        max_tokens=10000,
        temperature=0.7,
        top_p=0.9,
        stop=["User:"],
    )
    _ = time.time() - start
    clean_output = output["choices"][0]["text"].strip()
    if _CLEAN_PATTERN.search(clean_output):
        response = _CLEAN_PATTERN.split(clean_output, maxsplit=1)
        return response[1].strip() if len(response) > 1 else clean_output
    return clean_output


if __name__ == "__main__":
    mcp.run(transport="stdio")
