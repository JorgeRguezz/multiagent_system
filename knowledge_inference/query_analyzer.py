from __future__ import annotations

import re
from collections import Counter

from .types import QueryIntent

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "to", "for", "of", "in", "on", "at", "by",
    "with", "from", "as", "it", "this", "that", "these", "those", "what", "which", "who", "when",
    "where", "why", "how", "does", "do", "did", "can", "could", "would", "should", "about", "into",
    "and", "or", "but", "than", "then", "if", "across", "video", "videos",
}

_CHAMPION_ALIASES = {
    "pike": "pyke",
    "ahry": "ahri",
    "smoulder": "smolder",
    "zaahen": "zaahen",
}

_CROSS_VIDEO_TERMS = [
    "across videos", "compare", "versus", "vs", "which video", "most videos", "difference between",
]

_TEMPORAL_TERMS = [
    "when", "timeline", "before", "after", "at what time", "earlier", "later", "first", "then", "timestamp",
]

_VISUAL_TERMS = [
    "what appears", "what is shown", "in the frame", "on screen", "visual", "looks like", "shown in", "color",
]


def _normalize_query(text: str) -> str:
    return " ".join(text.strip().split())


def _contains_count(haystack: str, terms: list[str]) -> int:
    return sum(1 for term in terms if term in haystack)


def _extract_entity_terms(normalized_query: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]{1,}", normalized_query.lower())
    normalized_tokens = []
    for token in tokens:
        token = _CHAMPION_ALIASES.get(token, token)
        if token in _STOPWORDS:
            continue
        normalized_tokens.append(token)
    freq = Counter(normalized_tokens)
    return [t for t, _ in freq.most_common(12)]


def analyze_query(query: str) -> QueryIntent:
    normalized = _normalize_query(query)
    lower = normalized.lower()

    cross_score = _contains_count(lower, _CROSS_VIDEO_TERMS)
    temporal_score = _contains_count(lower, _TEMPORAL_TERMS)
    visual_score = _contains_count(lower, _VISUAL_TERMS)

    # Bonus cues used in our heuristic-first policy.
    if re.search(r"\b\d{1,2}:\d{2}\b", lower) or any(x in lower for x in ["second", "seconds", "minute", "minutes"]):
        temporal_score += 1

    entity_terms = _extract_entity_terms(normalized)
    if len(entity_terms) >= 2 and any(t in lower for t in ["compare", "versus", "vs", "difference"]):
        cross_score += 1

    if any(t in lower for t in ["color", "icon", "ui", "hud", "left side", "right side"]):
        visual_score += 1

    return QueryIntent(
        normalized_query=normalized,
        is_cross_video=cross_score >= 1,
        is_temporal=temporal_score >= 1,
        is_visual_detail=visual_score >= 1,
        entity_focus_terms=entity_terms,
    )
