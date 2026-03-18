from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# Keep these JSON paths visible and easy to edit.
INPUT_JSON_PATH = Path(__file__).resolve().parent / "report_eval_test_rag_merged.json"
OUTPUT_JSON_PATH = Path(__file__).resolve().parent / "report_eval_test_rag_merged_done.json"
TEST_OUTPUT_JSON_PATH = Path(__file__).resolve().parent / "report_eval_test_rag_weak_llm_test5.json"

# Test run configuration.
TEST_RUN_LIMIT = 5

# GPT-OSS-20B config copied from playground/test_gpt_oss_20b.py.
MODEL_ID = "unsloth/gpt-oss-20b-GGUF"
QUANT_FILE = "gpt-oss-20b-F16.gguf"
MODEL_LOCAL_DIR = Path(__file__).resolve().parents[1] / "gpt-oss-20b"
N_GPU_LAYERS = -1
N_CTX = 16384
VERBOSE = False

# Optional throughput tuning. Left visible for manual adjustment.
N_BATCH: int | None = None
N_UBATCH: int | None = None
F16_KV: bool | None = None
N_THREADS: int | None = None
N_THREADS_BATCH: int | None = None

MAX_TOKENS = 1000
TEMPERATURE = 0.1
TOP_P = 1.0
TOP_K = 0
CHECKPOINT_EVERY = 1

SYSTEM_PROMPT = """
You are a helpful assistant answering questions.

Reasoning: low

Answer the user's question directly and concisely. If the prompt does not provide enough evidence, answer with the best general response you can.

<|channel|>analysis<|message|>User asks: "What is 2 + 2?" Simple arithmetic. Provide answer.<|end|>
<|start|>assistant<|channel|>final<|message|>2 + 2 = 4.<|return|>
""".strip()

_CLEAN_PATTERN = re.compile(r"final<\|message\|>")
_LLM = None


def _ensure_torch_cuda_libs_visible() -> None:
    import torch

    torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
    existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if torch_lib_dir not in existing_ld:
        os.environ["LD_LIBRARY_PATH"] = (
            f"{torch_lib_dir}:{existing_ld}" if existing_ld else torch_lib_dir
        )
    try:
        ctypes.CDLL(os.path.join(torch_lib_dir, "libcudart.so.12"))
    except OSError:
        pass


def _build_llama_kwargs(model_path: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model_path": model_path,
        "n_gpu_layers": N_GPU_LAYERS,
        "n_ctx": N_CTX,
        "verbose": VERBOSE,
    }
    if N_BATCH is not None:
        kwargs["n_batch"] = N_BATCH
    if N_UBATCH is not None:
        kwargs["n_ubatch"] = N_UBATCH
    if F16_KV is not None:
        kwargs["f16_kv"] = F16_KV
    if N_THREADS is not None:
        kwargs["n_threads"] = N_THREADS
    if N_THREADS_BATCH is not None:
        kwargs["n_threads_batch"] = N_THREADS_BATCH
    return kwargs


def get_llm():
    global _LLM
    if _LLM is not None:
        return _LLM

    import torch
    from huggingface_hub import snapshot_download

    _ensure_torch_cuda_libs_visible()
    from llama_cpp import Llama

    model_path = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(MODEL_LOCAL_DIR),
        allow_patterns=[QUANT_FILE],
    )
    full_path = os.path.join(model_path, QUANT_FILE)

    print(f"Model path: {full_path}")
    if torch.cuda.is_available():
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM check: {total_vram_gb:.1f}GB available")

    _LLM = Llama(**_build_llama_kwargs(full_path))
    return _LLM


def _render_progress(current: int, total: int, width: int = 32) -> str:
    if total <= 0:
        return "[no work]"
    ratio = min(1.0, max(0.0, current / total))
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total} ({ratio * 100:5.1f}%)"


def _print_progress(current: int, total: int) -> None:
    end = "\n" if current >= total else "\r"
    print(_render_progress(current, total), end=end, flush=True)


def _load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a top-level JSON list in {path}")
    if not all(isinstance(item, dict) for item in data):
        raise ValueError(f"Expected every item in {path} to be a JSON object")
    return data


def _write_cases(path: Path, cases: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(cases, f, indent=4, ensure_ascii=True)
        f.write("\n")


def _resolve_starting_cases(input_path: Path, output_path: Path) -> list[dict[str, Any]]:
    source_path = output_path if output_path.exists() else input_path
    print(f"Loading cases from: {source_path}")
    return _load_cases(source_path)


def _build_prompt(question: str) -> str:
    user_prompt = f"Question:\n{question.strip()}\n"
    return (
        f"<|start|>system<|message|>{SYSTEM_PROMPT}<|end|>"
        f"<|start|>user<|message|>{user_prompt}<|end|>"
    )


def _extract_answer(raw_text: str) -> str:
    clean_output = raw_text.strip()
    if _CLEAN_PATTERN.search(clean_output):
        parts = _CLEAN_PATTERN.split(clean_output, maxsplit=1)
        if len(parts) > 1:
            return parts[1].strip()
    return clean_output


def _generate_answer(llm, question: str) -> tuple[str, dict[str, Any], float]:
    start = time.time()
    output = llm(
        _build_prompt(question),
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
    )
    elapsed = time.time() - start
    text = output["choices"][0]["text"]
    usage = output.get("usage", {})
    return _extract_answer(text), usage, elapsed


def run(
    input_path: Path,
    output_path: Path,
    limit: int | None = None,
    skip_filled: bool = True,
) -> None:
    cases = _resolve_starting_cases(input_path=input_path, output_path=output_path)
    total = len(cases) if limit is None else min(len(cases), max(0, limit))
    llm = get_llm()

    processed = 0
    _print_progress(0, total)

    for idx in range(total):
        item = cases[idx]
        question = str(item.get("question", "")).strip()
        existing_answer = str(item.get("weak_llm_answer", "")).strip()

        if not question:
            item["weak_llm_answer"] = ""
            processed += 1
            _print_progress(processed, total)
            continue

        if skip_filled and existing_answer:
            processed += 1
            _print_progress(processed, total)
            continue

        answer, usage, elapsed = _generate_answer(llm=llm, question=question)
        item["weak_llm_answer"] = answer

        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        print(
            f"\n[{idx + 1}/{total}] {elapsed:.2f}s"
            f" prompt_tokens={prompt_tokens} completion_tokens={completion_tokens}"
        )

        if (idx + 1) % CHECKPOINT_EVERY == 0:
            _write_cases(output_path, cases)

        processed += 1
        _print_progress(processed, total)

    _write_cases(output_path, cases)
    print(f"Saved output to: {output_path}")

    if _LLM is not None:
        _LLM.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fill weak_llm_answer in the evaluation JSON using local GPT-OSS-20B."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_JSON_PATH,
        help="Source evaluation JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_JSON_PATH,
        help="Destination JSON file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N objects",
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help=f"Process only the first {TEST_RUN_LIMIT} questions and use the test output path by default",
    )
    parser.add_argument(
        "--no-skip-filled",
        action="store_true",
        help="Recompute rows even if weak_llm_answer is already populated",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    limit = args.limit

    if args.test_run:
        limit = TEST_RUN_LIMIT if limit is None else min(limit, TEST_RUN_LIMIT)
        if args.output == OUTPUT_JSON_PATH:
            output_path = TEST_OUTPUT_JSON_PATH.resolve()

    run(
        input_path=input_path,
        output_path=output_path,
        limit=limit,
        skip_filled=not args.no_skip_filled,
    )


if __name__ == "__main__":
    main()
