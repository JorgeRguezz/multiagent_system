from __future__ import annotations

from knowledge_build._llm import local_llm_config

from . import config
from .prompts import USER_QA_TEMPLATE, get_system_grounded_qa_prompt
from .types import GenerationResult


async def generate_answer(query: str, context: str) -> GenerationResult:
    user_prompt = USER_QA_TEMPLATE.format(question=query, context=context)
    result = await local_llm_config.best_model_func(
        user_prompt,
        system_prompt=get_system_grounded_qa_prompt(),
        max_tokens=config.MAX_ANSWER_TOKENS,
        temperature=config.GEN_TEMPERATURE,
        top_p=config.GEN_TOP_P,
        top_k=config.GEN_TOP_K,
        repeat_penalty=config.GEN_REPEAT_PENALTY,
        return_metadata=True,
    )
    return GenerationResult(
        answer=str(result.get("answer", "")).strip(),
        thoughts=str(result.get("thoughts", "")).strip(),
        has_final_marker=bool(result.get("has_final_marker", False)),
        raw_text=str(result.get("raw_text", "")).strip(),
    )
