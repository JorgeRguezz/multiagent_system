import asyncio

from knowledge_inference.service import InferenceService
from knowledge_inference.types import EvidenceBlock, GenerationResult, QueryIntent


def _intent() -> QueryIntent:
    return QueryIntent(
        normalized_query="what happened",
        is_cross_video=False,
        is_temporal=False,
        is_visual_detail=False,
        entity_focus_terms=["what", "happened"],
    )


def _evidence() -> list[EvidenceBlock]:
    return [
        EvidenceBlock(
            video_name="video_a",
            time_span="0:00-0:10",
            chunk_id="chunk-1",
            source="dense_chunk",
            text="Relevant evidence text",
            final_score=0.9,
        )
    ]


def test_service_uses_verified_answer_and_keeps_generation_thoughts(monkeypatch):
    monkeypatch.setattr("knowledge_inference.service.analyze_query", lambda query: _intent())
    monkeypatch.setattr("knowledge_inference.service.retrieve_all", _async_value([]))
    monkeypatch.setattr("knowledge_inference.service.rerank_hits", lambda **kwargs: [])
    monkeypatch.setattr("knowledge_inference.service.make_evidence_blocks", lambda **kwargs: _evidence())
    monkeypatch.setattr(
        "knowledge_inference.service.render_context_for_prompt",
        lambda evidence: "rendered context",
    )
    monkeypatch.setattr(
        "knowledge_inference.service.generate_answer",
        _async_value(
            GenerationResult(
                answer="Raw answer",
                thoughts="Hidden reasoning",
                has_final_marker=True,
                raw_text="Hidden reasoning final<|message|>Raw answer",
            )
        ),
    )
    monkeypatch.setattr(
        "knowledge_inference.service.verify_answer",
        _async_value(
            (
                "Verified answer",
                1.0,
                {
                    "supported_ratio": 1.0,
                    "unsupported_ratio": 0.0,
                    "uncertain_ratio": 0.0,
                    "claims_total": 1,
                    "labels": ["supported"],
                    "summary": "ok",
                    "enabled": True,
                },
            )
        ),
    )
    monkeypatch.setattr("knowledge_inference.service.inject_video_urls", lambda answer, registry: f"{answer} [post]")
    monkeypatch.setattr("knowledge_inference.service.config.ENABLE_VERIFIER", True)

    service = InferenceService()
    result = asyncio.run(service._answer_async("what happened", debug=True))

    assert result.answer == "Verified answer [post]"
    assert result.debug["generation"]["thoughts"] == "Hidden reasoning"
    assert result.debug["generation"]["answer_raw"] == "Raw answer"
    assert result.debug["generation"]["has_final_marker"] is True
    assert result.debug["generation"]["verifier_enabled"] is True
    assert result.debug["generation"]["final_answer_source"] == "verified"


def test_service_uses_raw_generated_answer_when_verifier_disabled(monkeypatch):
    monkeypatch.setattr("knowledge_inference.service.analyze_query", lambda query: _intent())
    monkeypatch.setattr("knowledge_inference.service.retrieve_all", _async_value([]))
    monkeypatch.setattr("knowledge_inference.service.rerank_hits", lambda **kwargs: [])
    monkeypatch.setattr("knowledge_inference.service.make_evidence_blocks", lambda **kwargs: _evidence())
    monkeypatch.setattr(
        "knowledge_inference.service.render_context_for_prompt",
        lambda evidence: "rendered context",
    )
    monkeypatch.setattr(
        "knowledge_inference.service.generate_answer",
        _async_value(
            GenerationResult(
                answer="Raw answer",
                thoughts="Hidden reasoning",
                has_final_marker=False,
                raw_text="Raw answer",
            )
        ),
    )

    async def fail_verify_answer(*args, **kwargs):
        raise AssertionError("verify_answer should not be called when verifier is disabled")

    monkeypatch.setattr("knowledge_inference.service.verify_answer", fail_verify_answer)
    monkeypatch.setattr("knowledge_inference.service.inject_video_urls", lambda answer, registry: f"{answer} [post]")
    monkeypatch.setattr("knowledge_inference.service.config.ENABLE_VERIFIER", False)

    service = InferenceService()
    result = asyncio.run(service._answer_async("what happened", debug=True))

    assert result.answer == "Raw answer [post]"
    assert result.debug["generation"]["thoughts"] == "Hidden reasoning"
    assert result.debug["generation"]["answer_raw"] == "Raw answer"
    assert result.debug["generation"]["has_final_marker"] is False
    assert result.debug["generation"]["verifier_enabled"] is False
    assert result.debug["generation"]["final_answer_source"] == "raw_generation"
    assert result.debug["verification"]["enabled"] is False


def _async_value(value):
    async def _inner(*args, **kwargs):
        return value

    return _inner
