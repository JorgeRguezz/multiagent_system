import asyncio

from knowledge_inference.generator import MALFORMED_GENERATION_ANSWER, generate_answer


def test_generate_answer_returns_structured_generation_result(monkeypatch):
    async def fake_best_model_func(*args, **kwargs):
        return {
            "answer": "Final answer text<|return|>",
            "thoughts": "Reasoning tokens",
            "has_final_marker": True,
            "raw_text": "Reasoning tokens final<|message|>Final answer text",
        }

    monkeypatch.setattr(
        "knowledge_inference.generator.local_llm_config.best_model_func",
        fake_best_model_func,
    )

    result = asyncio.run(generate_answer("What happened?", "Evidence block"))

    assert result.answer == "Final answer text"
    assert result.thoughts == "Reasoning tokens"
    assert result.has_final_marker is True
    assert "final<|message|>" in result.raw_text


def test_generate_answer_does_not_expose_reasoning_without_final_marker(monkeypatch):
    async def fake_best_model_func(*args, **kwargs):
        return {
            "answer": "Fallback answer",
            "thoughts": "",
            "has_final_marker": False,
            "raw_text": "Fallback answer",
        }

    monkeypatch.setattr(
        "knowledge_inference.generator.local_llm_config.best_model_func",
        fake_best_model_func,
    )

    result = asyncio.run(generate_answer("What happened?", "Evidence block"))

    assert result.answer == MALFORMED_GENERATION_ANSWER
    assert result.thoughts == "Fallback answer"
    assert result.has_final_marker is False
    assert result.raw_text == "Fallback answer"


def test_generate_answer_retries_malformed_output_once(monkeypatch):
    responses = [
        {
            "answer": "Private reasoning",
            "thoughts": "",
            "has_final_marker": False,
            "raw_text": "Private reasoning",
        },
        {
            "answer": "Recovered answer<|return|>",
            "thoughts": "Brief reasoning",
            "has_final_marker": True,
            "raw_text": "Brief reasoning final<|message|>Recovered answer<|return|>",
        },
    ]
    calls = []

    async def fake_best_model_func(*args, **kwargs):
        calls.append((args, kwargs))
        return responses.pop(0)

    monkeypatch.setattr(
        "knowledge_inference.generator.local_llm_config.best_model_func",
        fake_best_model_func,
    )

    result = asyncio.run(generate_answer("What happened?", "Evidence block"))

    assert result.answer == "Recovered answer"
    assert result.has_final_marker is True
    assert len(calls) == 2
    assert "Reasoning: low" in calls[1][1]["system_prompt"]
    assert calls[1][1]["max_tokens"] == 1200
