from __future__ import annotations

from knowledge_build._llm import local_llm_config

from . import config
from .prompts import USER_QA_TEMPLATE, get_system_grounded_qa_prompt
from .types import GenerationResult


MALFORMED_GENERATION_ANSWER = (
    "The local answer model did not return a valid final response."
)
RETRY_INSTRUCTION = """

Output requirement:
Keep analysis under 100 words, then always provide a concise answer in the final
channel. Do not repeat the question, evidence, or analysis.
""".rstrip()


async def generate_answer(query: str, context: str) -> GenerationResult:
    user_prompt = USER_QA_TEMPLATE.format(question=query, context=context)
    system_prompt = get_system_grounded_qa_prompt()
    generation_kwargs = {
        "system_prompt": system_prompt,
        "max_tokens": config.MAX_ANSWER_TOKENS,
        "temperature": config.GEN_TEMPERATURE,
        "top_p": config.GEN_TOP_P,
        "top_k": config.GEN_TOP_K,
        "repeat_penalty": config.GEN_REPEAT_PENALTY,
        "return_metadata": True,
    }
    result = await local_llm_config.best_model_func(user_prompt, **generation_kwargs)
    if not bool(result.get("has_final_marker", False)):
        generation_kwargs["system_prompt"] = system_prompt.replace(
            "Reasoning: medium",
            "Reasoning: low",
        )
        generation_kwargs["max_tokens"] = min(config.MAX_ANSWER_TOKENS, 1200)
        result = await local_llm_config.best_model_func(
            f"{user_prompt}{RETRY_INSTRUCTION}",
            **generation_kwargs,
        )

    raw_text = str(result.get("raw_text", "")).strip()
    thoughts = str(result.get("thoughts", "")).strip()
    has_final_marker = bool(result.get("has_final_marker", False))
    if has_final_marker:
        answer = str(result.get("answer", "")).replace("<|return|>", "").strip()
    else:
        thoughts = thoughts or raw_text
        answer = MALFORMED_GENERATION_ANSWER

    return GenerationResult(
        answer=answer,
        thoughts=thoughts,
        has_final_marker=has_final_marker,
        raw_text=raw_text,
    )
