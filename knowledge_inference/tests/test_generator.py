import asyncio

from knowledge_inference.generator import generate_answer


def test_generate_answer_returns_structured_generation_result(monkeypatch):
    async def fake_best_model_func(*args, **kwargs):
        return {
            "answer": "Final answer text",
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


def test_generate_answer_handles_missing_final_marker(monkeypatch):
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

    assert result.answer == "Fallback answer"
    assert result.thoughts == ""
    assert result.has_final_marker is False
    assert result.raw_text == "Fallback answer"
