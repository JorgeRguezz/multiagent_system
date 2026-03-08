from __future__ import annotations

import math
import re

from .types import EvidenceBlock, RetrievalHit, VideoStore


def _parse_seconds(raw: str) -> float | None:
    try:
        return float(str(raw).strip())
    except Exception:
        return None


def _fmt_hms(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def resolve_time_span(store: VideoStore, segment_ids: list[str]) -> str:
    if not segment_ids:
        return "unknown"

    seg_map = store.segments_kv.get(store.video_name, {}) if isinstance(store.segments_kv, dict) else {}
    starts: list[float] = []
    ends: list[float] = []

    for segment_id in segment_ids:
        seg_idx = str(segment_id).rsplit("_", 1)[-1]
        seg = seg_map.get(seg_idx)
        if not isinstance(seg, dict):
            continue
        raw_time = str(seg.get("time", ""))
        if "-" not in raw_time:
            continue
        raw_s, raw_e = raw_time.split("-", 1)
        s = _parse_seconds(raw_s)
        e = _parse_seconds(raw_e)
        if s is None or e is None:
            continue
        starts.append(s)
        ends.append(e)

    if not starts or not ends:
        return "unknown"

    return f"{_fmt_hms(min(starts))}-{_fmt_hms(max(ends))}"


def _truncate_text_to_budget(text: str, available_tokens: int) -> tuple[str, int]:
    if available_tokens <= 0:
        return "", 0

    max_chars = available_tokens * 4
    if len(text) <= max_chars:
        used_tokens = math.ceil(len(text) / 4)
        return text, used_tokens

    clipped = text[: max(0, max_chars - len(" [truncated]"))].rstrip() + " [truncated]"
    used_tokens = math.ceil(len(clipped) / 4)
    return clipped, used_tokens


def make_evidence_blocks(
    hits: list[RetrievalHit],
    stores: dict[str, VideoStore],
    budget_tokens: int,
) -> list[EvidenceBlock]:
    blocks: list[EvidenceBlock] = []
    used_tokens = 0

    for hit in hits:
        if used_tokens >= budget_tokens:
            break
        store = stores.get(hit.video_name)
        if not store:
            continue

        # Keep medium-size blocks so we can include more sources.
        remaining = budget_tokens - used_tokens
        block_budget = min(900, max(240, remaining))
        text, spent = _truncate_text_to_budget(hit.chunk_text.strip(), block_budget)
        if not text:
            continue

        block = EvidenceBlock(
            video_name=hit.video_name,
            time_span=resolve_time_span(store, hit.segment_ids),
            chunk_id=hit.chunk_id,
            source=hit.source,
            text=text,
            final_score=hit.final_score,
        )
        blocks.append(block)
        used_tokens += spent

    return blocks


def render_context_for_prompt(evidence_blocks: list[EvidenceBlock]) -> str:
    parts: list[str] = []
    for block in evidence_blocks:
        header = (
            f"[video={block.video_name} | time={block.time_span} | "
            f"chunk={block.chunk_id} | source={block.source}]"
        )
        parts.append(f"{header}\n{block.text}")
    return "\n\n".join(parts)
