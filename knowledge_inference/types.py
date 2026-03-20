from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
from nano_vectordb import NanoVectorDB


@dataclass
class VideoStore:
    video_name: str
    chunks_vdb: NanoVectorDB
    entities_vdb: NanoVectorDB
    chunks_kv: dict[str, dict[str, Any]]
    segments_kv: dict[str, dict[str, Any]]
    frames_kv: dict[str, dict[str, Any]]
    graph: nx.Graph


@dataclass
class QueryIntent:
    normalized_query: str
    is_cross_video: bool
    is_temporal: bool
    is_visual_detail: bool
    entity_focus_terms: list[str]


@dataclass
class RetrievalHit:
    chunk_id: str
    video_name: str
    source: str
    chunk_text: str
    segment_ids: list[str] = field(default_factory=list)
    score_semantic: float = 0.0
    score_entity: float = 0.0
    score_graph: float = 0.0
    final_score: float = 0.0


@dataclass
class EvidenceBlock:
    video_name: str
    time_span: str
    chunk_id: str
    source: str
    text: str
    final_score: float


@dataclass
class AnswerResult:
    answer: str
    evidence: list[EvidenceBlock]
    context: str
    confidence: float
    debug: dict[str, Any]


@dataclass
class GenerationResult:
    answer: str
    thoughts: str
    has_final_marker: bool
    raw_text: str = ""
