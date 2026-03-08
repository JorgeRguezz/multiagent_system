from __future__ import annotations

import re
from collections import defaultdict

from . import config
from .types import QueryIntent, RetrievalHit


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]+", text.lower()))


def dedupe_hits(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    merged: dict[tuple[str, str], RetrievalHit] = {}
    for hit in hits:
        key = (hit.video_name, hit.chunk_id)
        if key not in merged:
            merged[key] = hit
            continue
        existing = merged[key]
        existing.score_semantic = max(existing.score_semantic, hit.score_semantic)
        existing.score_entity = max(existing.score_entity, hit.score_entity)
        existing.score_graph = max(existing.score_graph, hit.score_graph)
        if hit.final_score > existing.final_score:
            existing.final_score = hit.final_score
        if len(hit.chunk_text) > len(existing.chunk_text):
            existing.chunk_text = hit.chunk_text
        if hit.source != existing.source:
            existing.source = f"{existing.source}|{hit.source}"
        if hit.segment_ids:
            existing.segment_ids = sorted(set(existing.segment_ids + hit.segment_ids))
    return list(merged.values())


def compute_component_scores(hits: list[RetrievalHit], query: str, intent: QueryIntent) -> list[RetrievalHit]:
    q_tokens = _tokenize(query)

    for hit in hits:
        h_tokens = _tokenize(hit.chunk_text)
        overlap = len(q_tokens & h_tokens)
        lexical_overlap = overlap / max(1, len(q_tokens))

        # Entity score strengthened by explicit entity terms.
        if intent.entity_focus_terms:
            entity_overlap = sum(1 for term in intent.entity_focus_terms if term.lower() in hit.chunk_text.lower())
            entity_bonus = entity_overlap / max(1, len(intent.entity_focus_terms))
        else:
            entity_bonus = 0.0

        hit.score_entity = max(hit.score_entity, 0.65 * lexical_overlap + 0.35 * entity_bonus)

        # Graph score gets a weak lexical backoff only when no graph signal exists.
        if hit.score_graph <= 0.0:
            if "graph" in hit.source:
                hit.score_graph = min(1.0, 0.25 + 0.75 * lexical_overlap)
            else:
                hit.score_graph = 0.20 * lexical_overlap

        # Semantic score defaults to lexical backoff if branch didn't set it.
        if hit.score_semantic <= 0.0:
            hit.score_semantic = lexical_overlap

    return hits


def apply_weighted_score(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    for hit in hits:
        diversity_component = 1.0
        hit.final_score = (
            config.W_SEMANTIC * hit.score_semantic
            + config.W_ENTITY * hit.score_entity
            + config.W_GRAPH * hit.score_graph
            + config.W_DIVERSITY * diversity_component
        )
    return hits


def apply_diversity(hits: list[RetrievalHit], max_per_video: int) -> list[RetrievalHit]:
    hits_sorted = sorted(hits, key=lambda h: h.final_score, reverse=True)
    selected: list[RetrievalHit] = []
    per_video_count: defaultdict[str, int] = defaultdict(int)

    top_kept = False
    for hit in hits_sorted:
        if not top_kept:
            selected.append(hit)
            per_video_count[hit.video_name] += 1
            top_kept = True
            continue

        if per_video_count[hit.video_name] >= max_per_video:
            continue

        selected.append(hit)
        per_video_count[hit.video_name] += 1

        if len(selected) >= config.FINAL_EVIDENCE_K:
            break

    return selected


def _infer_single_video_focus(query: str, available_videos: list[str]) -> bool:
    q = query.lower()
    mentions = 0
    for video_name in available_videos:
        slug = video_name.lower().replace("_", " ")
        if slug in q:
            mentions += 1
    return mentions == 1


def rerank_hits(
    hits: list[RetrievalHit],
    query: str,
    intent: QueryIntent,
    available_videos: list[str],
) -> list[RetrievalHit]:
    unique_hits = dedupe_hits(hits)
    if not unique_hits:
        return []

    scored = compute_component_scores(unique_hits, query, intent)
    scored = apply_weighted_score(scored)

    max_per_video = config.DEFAULT_MAX_PER_VIDEO
    if intent.is_cross_video:
        max_per_video = config.CROSS_VIDEO_MAX_PER_VIDEO
    if _infer_single_video_focus(query, available_videos):
        max_per_video = config.FINAL_EVIDENCE_K

    diverse = apply_diversity(scored, max_per_video=max_per_video)
    diverse.sort(key=lambda h: h.final_score, reverse=True)
    return diverse[: config.FINAL_EVIDENCE_K]
