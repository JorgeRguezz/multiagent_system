from knowledge_pipeline.game_profiles import get_active_game_profile


def get_system_grounded_qa_prompt(video_game: str | None = None) -> str:
    return get_active_game_profile(video_game).grounded_qa_system_prompt

USER_QA_TEMPLATE = """Question:
{question}

Evidence Context:
{context}

Instruction:
Answer using only the evidence above. If the evidence is weak, respond with uncertainty and explain what is missing.
"""

VERIFIER_TEMPLATE = """You are a factual verifier.
Classify each claim against the evidence as one of: supported, unsupported, uncertain.
Return STRICT JSON with this shape:
{{
  "claims": [
    {{"index": 1, "label": "supported|unsupported|uncertain", "reason": "short"}}
  ],
  "summary": "short summary"
}}

Evidence:
{context}

Claims:
{claims}
"""
