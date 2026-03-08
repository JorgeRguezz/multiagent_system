from __future__ import annotations

import json
import re
from typing import Any

from knowledge_build._llm import local_llm_config

from . import config
from .prompts import VERIFIER_TEMPLATE
from .types import EvidenceBlock


def _split_claims(answer: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+", answer.strip())
    return [p.strip() for p in pieces if p.strip()]


def _render_claims(claims: list[str]) -> str:
    return "\n".join(f"{i + 1}. {claim}" for i, claim in enumerate(claims))


def _render_evidence(evidence_blocks: list[EvidenceBlock]) -> str:
    rows = []
    for i, block in enumerate(evidence_blocks, start=1):
        rows.append(
            f"[{i}] video={block.video_name} time={block.time_span} chunk={block.chunk_id} source={block.source}\n{block.text}"
        )
    return "\n\n".join(rows)


def _parse_verifier_json(raw: str) -> dict[str, Any]:
    raw = (raw or "").strip()
    if not raw:
        return {"claims": [], "summary": "empty verifier output"}
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {"claims": [], "summary": "verifier output was not JSON"}
    payload = raw[start : end + 1]
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return {"claims": [], "summary": "verifier JSON parse failure"}
    if "claims" not in data or not isinstance(data["claims"], list):
        data["claims"] = []
    if "summary" not in data:
        data["summary"] = ""
    return data


def _prune_unsupported_sentences(answer: str, unsupported_indexes: set[int]) -> str:
    sentences = _split_claims(answer)
    if not sentences:
        return answer
    kept = [s for i, s in enumerate(sentences, start=1) if i not in unsupported_indexes]
    return " ".join(kept).strip() or answer


async def verify_answer(answer: str, evidence_blocks: list[EvidenceBlock]) -> tuple[str, float, dict[str, Any]]:
    claims = _split_claims(answer)
    if not claims:
        return answer, 0.0, {
            "supported_ratio": 0.0,
            "unsupported_ratio": 0.0,
            "uncertain_ratio": 1.0,
            "claims_total": 0,
            "summary": "no claims",
        }

    verifier_prompt = VERIFIER_TEMPLATE.format(
        context=_render_evidence(evidence_blocks),
        claims=_render_claims(claims),
    )

    verifier_raw = await local_llm_config.best_model_func(
        verifier_prompt,
        max_tokens=config.VERIFIER_MAX_TOKENS,
        temperature=config.VERIFIER_TEMPERATURE,
        top_p=config.VERIFIER_TOP_P,
        top_k=config.VERIFIER_TOP_K,
        repeat_penalty=config.VERIFIER_REPEAT_PENALTY,
    )

    parsed = _parse_verifier_json(verifier_raw)
    labels = [str(c.get("label", "uncertain")).lower().strip() for c in parsed.get("claims", [])]

    # Align classifier length to claims length.
    if len(labels) < len(claims):
        labels.extend(["uncertain"] * (len(claims) - len(labels)))
    labels = labels[: len(claims)]

    supported = sum(1 for l in labels if l == "supported")
    unsupported = sum(1 for l in labels if l == "unsupported")
    uncertain = len(labels) - supported - unsupported

    total = max(1, len(labels))
    supported_ratio = supported / total
    unsupported_ratio = unsupported / total
    uncertain_ratio = uncertain / total

    adjusted_answer = answer
    unsupported_indexes = {i + 1 for i, l in enumerate(labels) if l == "unsupported"}

    # Soft-verification policy: usually preserve answer and annotate uncertainty.
    if unsupported_ratio > 0.40:
        adjusted_answer = (
            "Evidence is partially conflicting or insufficient for parts of this answer. "
            + answer.strip()
        ).strip()
    elif unsupported_indexes and len(unsupported_indexes) <= 2:
        adjusted_answer = _prune_unsupported_sentences(answer, unsupported_indexes)

    debug = {
        "supported_ratio": supported_ratio,
        "unsupported_ratio": unsupported_ratio,
        "uncertain_ratio": uncertain_ratio,
        "claims_total": len(labels),
        "labels": labels,
        "summary": parsed.get("summary", ""),
    }
    return adjusted_answer, supported_ratio, debug
