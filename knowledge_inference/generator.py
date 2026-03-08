from __future__ import annotations

from knowledge_build._llm import local_llm_config

from . import config
from .prompts import SYSTEM_GROUNDED_QA, USER_QA_TEMPLATE


async def generate_answer(query: str, context: str) -> str:
    user_prompt = USER_QA_TEMPLATE.format(question=query, context=context)
    return await local_llm_config.best_model_func(
        user_prompt,
        system_prompt=SYSTEM_GROUNDED_QA,
        max_tokens=config.MAX_ANSWER_TOKENS,
        temperature=config.GEN_TEMPERATURE,
        top_p=config.GEN_TOP_P,
        top_k=config.GEN_TOP_K,
        repeat_penalty=config.GEN_REPEAT_PENALTY,
    )
