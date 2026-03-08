from __future__ import annotations

import asyncio
import logging
import math
import re
from typing import Iterable

import networkx as nx

from knowledge_build._llm import local_llm_config

from . import config
from .types import QueryIntent, RetrievalHit, VideoStore

logger = logging.getLogger(config.LOGGER_NAME)


def _split_source_ids(source_id: str | None) -> list[str]:
    if not source_id:
        return []
    return [p.strip() for p in str(source_id).split("<SEP>") if p.strip()]


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]+", text.lower()))


async def _embed_query(query: str):
    emb = await local_llm_config.embedding_func([query])
    return emb[0]


def _resolve_chunk_hit(store: VideoStore, chunk_id: str, source: str, semantic: float = 0.0, entity: float = 0.0, graph: float = 0.0) -> RetrievalHit | None:
    chunk = store.chunks_kv.get(chunk_id)
    if not chunk:
        return None
    seg_ids = chunk.get("video_segment_id", [])
    if isinstance(seg_ids, str):
        seg_ids = [seg_ids]
    return RetrievalHit(
        chunk_id=chunk_id,
        video_name=store.video_name,
        source=source,
        chunk_text=str(chunk.get("content", "")),
        segment_ids=[str(x) for x in seg_ids],
        score_semantic=float(semantic),
        score_entity=float(entity),
        score_graph=float(graph),
    )


async def retrieve_chunks_dense(query: str, stores: dict[str, VideoStore], k: int) -> list[RetrievalHit]:
    query_embedding = await _embed_query(query)
    hits: list[RetrievalHit] = []
    for store in stores.values():
        try:
            results = store.chunks_vdb.query(query=query_embedding, top_k=k)
        except Exception as exc:
            logger.warning("dense_chunk retrieval failed for video '%s': %s", store.video_name, exc)
            continue
        for row in results:
            chunk_id = row.get("__id__")
            if not chunk_id:
                continue
            hit = _resolve_chunk_hit(
                store,
                chunk_id=chunk_id,
                source="dense_chunk",
                semantic=float(row.get("__metrics__", 0.0)),
            )
            if hit:
                hits.append(hit)
    return hits


def _graph_nodes_for_entity(store: VideoStore, entity_name: str) -> list[str]:
    candidates = [entity_name, entity_name.strip('"'), entity_name.upper(), entity_name.lower()]
    nodes = []
    node_set = set(store.graph.nodes)
    for cand in candidates:
        if cand in node_set:
            nodes.append(cand)
    if nodes:
        return list(dict.fromkeys(nodes))

    stripped = entity_name.strip('"').lower()
    for node in node_set:
        node_l = str(node).lower().strip('"')
        if node_l == stripped:
            nodes.append(str(node))
    return list(dict.fromkeys(nodes))


async def retrieve_entity_graph(
    query: str,
    stores: dict[str, VideoStore],
    k_entities: int,
    k_graph_chunks: int,
) -> list[RetrievalHit]:
    query_embedding = await _embed_query(query)
    hits: list[RetrievalHit] = []

    for store in stores.values():
        try:
            entity_results = store.entities_vdb.query(query=query_embedding, top_k=k_entities)
        except Exception as exc:
            logger.warning("entity_graph entity query failed for video '%s': %s", store.video_name, exc)
            continue

        chunk_scores: dict[str, float] = {}
        for row in entity_results:
            entity_name = str(row.get("entity_name", "")).strip()
            if not entity_name:
                continue
            base_score = float(row.get("__metrics__", 0.0))
            nodes = _graph_nodes_for_entity(store, entity_name)
            for node in nodes:
                node_data = store.graph.nodes[node]
                for cid in _split_source_ids(node_data.get("source_id")):
                    chunk_scores[cid] = max(chunk_scores.get(cid, 0.0), base_score)

                for neighbor in store.graph.neighbors(node):
                    edge_data = store.graph.get_edge_data(node, neighbor) or {}
                    weight = float(edge_data.get("weight", 1.0) or 1.0)
                    neighbor_boost = min(1.0, base_score + math.log1p(max(0.0, weight)) * 0.05)
                    neighbor_data = store.graph.nodes[neighbor]
                    for cid in _split_source_ids(neighbor_data.get("source_id")):
                        chunk_scores[cid] = max(chunk_scores.get(cid, 0.0), neighbor_boost)

                    for cid in _split_source_ids(edge_data.get("source_id")):
                        chunk_scores[cid] = max(chunk_scores.get(cid, 0.0), neighbor_boost)

        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:k_graph_chunks]
        for chunk_id, graph_score in sorted_chunks:
            hit = _resolve_chunk_hit(
                store,
                chunk_id=chunk_id,
                source="entity_graph",
                semantic=0.0,
                entity=graph_score,
                graph=graph_score,
            )
            if hit:
                hits.append(hit)

    return hits


def _lexical_match_score(query_tokens: set[str], node_name: str, description: str) -> float:
    name = node_name.lower().strip('"')
    desc = description.lower()
    text_tokens = _tokenize(f"{name} {desc}")
    if not text_tokens or not query_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens)
    jacc = overlap / len(query_tokens | text_tokens)
    contains_bonus = 0.15 if any(tok in name for tok in query_tokens) else 0.0
    return jacc + contains_bonus


def _build_chunk_to_video_index(stores: dict[str, VideoStore]) -> dict[str, str]:
    index: dict[str, str] = {}
    for video_name, store in stores.items():
        for chunk_id in store.chunks_kv.keys():
            index[chunk_id] = video_name
    return index


def retrieve_global_graph(query: str, global_graph: nx.Graph, stores: dict[str, VideoStore], k: int) -> list[RetrievalHit]:
    query_tokens = _tokenize(query)
    chunk_to_video = _build_chunk_to_video_index(stores)

    scored_nodes: list[tuple[str, float]] = []
    for node, attrs in global_graph.nodes(data=True):
        score = _lexical_match_score(query_tokens, str(node), str(attrs.get("description", "")))
        if score > 0.0:
            scored_nodes.append((str(node), score))

    scored_nodes.sort(key=lambda x: x[1], reverse=True)
    hits: list[RetrievalHit] = []

    for node, node_score in scored_nodes[:k]:
        attrs = global_graph.nodes[node]
        source_ids = _split_source_ids(attrs.get("source_id"))
        for chunk_id in source_ids:
            video_name = chunk_to_video.get(chunk_id)
            if not video_name:
                continue
            store = stores[video_name]
            hit = _resolve_chunk_hit(
                store,
                chunk_id=chunk_id,
                source="global_graph",
                graph=node_score,
                entity=0.0,
                semantic=0.0,
            )
            if hit:
                hits.append(hit)

    return hits


def _segment_to_chunk_ids(store: VideoStore) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for chunk_id, chunk in store.chunks_kv.items():
        seg_ids = chunk.get("video_segment_id", [])
        if isinstance(seg_ids, str):
            seg_ids = [seg_ids]
        for seg_id in seg_ids:
            mapping.setdefault(str(seg_id), []).append(chunk_id)
    return mapping


def retrieve_visual_support(query: str, intent: QueryIntent, stores: dict[str, VideoStore], per_video_k: int = 6) -> list[RetrievalHit]:
    if not intent.is_visual_detail:
        return []

    query_tokens = _tokenize(query)
    hits: list[RetrievalHit] = []

    for store in stores.values():
        frame_root = store.frames_kv.get(store.video_name, {}) if isinstance(store.frames_kv, dict) else {}
        seg_to_chunks = _segment_to_chunk_ids(store)

        scored_frames: list[tuple[str, float, dict]] = []
        for frame_key, frame in frame_root.items():
            visual_text = f"{frame.get('vlm_output', '')} {frame.get('transcript', '')}".strip()
            if not visual_text:
                continue
            toks = _tokenize(visual_text)
            if not toks:
                continue
            overlap = len(query_tokens & toks)
            if overlap <= 0:
                continue
            score = overlap / max(1, len(query_tokens))
            scored_frames.append((frame_key, score, frame))

        scored_frames.sort(key=lambda x: x[1], reverse=True)

        for _, frame_score, frame in scored_frames[:per_video_k]:
            seg_idx = str(frame.get("segment_idx", "")).strip()
            if not seg_idx:
                continue
            segment_id = f"{store.video_name}_{seg_idx}"
            chunk_ids = seg_to_chunks.get(segment_id, [])
            for chunk_id in chunk_ids:
                hit = _resolve_chunk_hit(
                    store,
                    chunk_id=chunk_id,
                    source="visual_support",
                    semantic=0.0,
                    entity=frame_score,
                    graph=0.0,
                )
                if hit:
                    hits.append(hit)

    return hits


async def retrieve_all(
    query: str,
    intent: QueryIntent,
    stores: dict[str, VideoStore],
    global_graph: nx.Graph,
) -> list[RetrievalHit]:
    tasks = [
        retrieve_chunks_dense(query, stores, config.TOP_K_CHUNKS_DENSE),
        retrieve_entity_graph(query, stores, config.TOP_K_ENTITIES_DENSE, config.TOP_K_GRAPH_CHUNKS),
    ]

    async def run_global():
        try:
            return retrieve_global_graph(query, global_graph, stores, config.TOP_K_GRAPH_CHUNKS)
        except Exception as exc:
            logger.warning("global_graph retrieval failed: %s", exc)
            return []

    async def run_visual():
        try:
            return retrieve_visual_support(query, intent, stores)
        except Exception as exc:
            logger.warning("visual_support retrieval failed: %s", exc)
            return []

    tasks.append(run_global())
    tasks.append(run_visual())

    results = await asyncio.gather(*tasks, return_exceptions=True)

    merged: list[RetrievalHit] = []
    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning("retrieval branch %s failed with exception: %s", idx, result)
            continue
        merged.extend(result)
    return merged
