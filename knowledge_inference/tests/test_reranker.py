from knowledge_inference.reranker import rerank_hits
from knowledge_inference.types import QueryIntent, RetrievalHit


def test_reranker_dedupe_and_sorting():
    intent = QueryIntent(
        normalized_query="pyke combo",
        is_cross_video=False,
        is_temporal=False,
        is_visual_detail=False,
        entity_focus_terms=["pyke", "combo"],
    )

    hits = [
        RetrievalHit(
            chunk_id="c1",
            video_name="video_a",
            source="dense_chunk",
            chunk_text="Pyke combo details",
            score_semantic=0.8,
        ),
        RetrievalHit(
            chunk_id="c1",
            video_name="video_a",
            source="entity_graph",
            chunk_text="Pyke combo details extended",
            score_entity=0.7,
            score_graph=0.7,
        ),
        RetrievalHit(
            chunk_id="c2",
            video_name="video_b",
            source="dense_chunk",
            chunk_text="General support notes",
            score_semantic=0.2,
        ),
    ]

    ranked = rerank_hits(hits, "pyke combo", intent, ["video_a", "video_b"])
    assert len(ranked) >= 2
    assert ranked[0].chunk_id == "c1"
    assert "dense_chunk" in ranked[0].source and "entity_graph" in ranked[0].source
