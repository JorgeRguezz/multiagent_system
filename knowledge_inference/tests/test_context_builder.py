import networkx as nx

from knowledge_inference.context_builder import make_evidence_blocks, resolve_time_span
from knowledge_inference.types import RetrievalHit, VideoStore


def _mock_store() -> VideoStore:
    video_name = "video_x"
    chunks = {
        "chunk-1": {
            "content": "A" * 4000,
            "video_segment_id": ["video_x_0"],
        }
    }
    segments = {video_name: {"0": {"time": "0-30", "content": "seg"}}}
    frames = {video_name: {}}
    graph = nx.Graph()
    # In tests we do not query VDB clients; only placeholders are needed.
    dummy_vdb = object()
    return VideoStore(
        video_name=video_name,
        chunks_vdb=dummy_vdb,
        entities_vdb=dummy_vdb,
        chunks_kv=chunks,
        segments_kv=segments,
        frames_kv=frames,
        graph=graph,
    )


def test_resolve_time_span():
    store = _mock_store()
    span = resolve_time_span(store, ["video_x_0"])
    assert span == "0:00-0:30"


def test_context_budget_truncation():
    store = _mock_store()
    hit = RetrievalHit(
        chunk_id="chunk-1",
        video_name="video_x",
        source="dense_chunk",
        chunk_text="A" * 8000,
        segment_ids=["video_x_0"],
        final_score=0.9,
    )
    blocks = make_evidence_blocks([hit], {"video_x": store}, budget_tokens=100)
    assert len(blocks) == 1
    assert "[truncated]" in blocks[0].text
