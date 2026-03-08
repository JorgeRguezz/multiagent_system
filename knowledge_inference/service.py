from __future__ import annotations

import asyncio
import logging
import time
import uuid

import networkx as nx

from knowledge_build._utils import always_get_an_event_loop

from . import config
from .context_builder import make_evidence_blocks, render_context_for_prompt
from .generator import generate_answer
from .query_analyzer import analyze_query
from .reranker import rerank_hits
from .retrievers import retrieve_all
from .store_loader import warmup
from .types import AnswerResult, EvidenceBlock, QueryIntent, VideoStore
from .verifier import verify_answer

logger = logging.getLogger(config.LOGGER_NAME)


class InferenceService:
    def __init__(self) -> None:
        self.stores: dict[str, VideoStore] = {}
        self.global_graph: nx.Graph | None = None
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        self.stores, self.global_graph = warmup()
        self._initialized = True

    def answer(self, query: str, debug: bool = False) -> AnswerResult:
        self.initialize()
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self._answer_async(query=query, debug=debug))

    async def _answer_async(self, query: str, debug: bool = False) -> AnswerResult:
        query_id = str(uuid.uuid4())
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        intent = analyze_query(query)
        timings["analyze_query_s"] = time.perf_counter() - t0

        t1 = time.perf_counter()
        hits = await retrieve_all(
            query=query,
            intent=intent,
            stores=self.stores,
            global_graph=self.global_graph if self.global_graph is not None else nx.Graph(),
        )
        timings["retrieve_s"] = time.perf_counter() - t1

        t2 = time.perf_counter()
        ranked_hits = rerank_hits(
            hits=hits,
            query=query,
            intent=intent,
            available_videos=list(self.stores.keys()),
        )
        timings["rerank_s"] = time.perf_counter() - t2

        t3 = time.perf_counter()
        evidence = make_evidence_blocks(
            hits=ranked_hits,
            stores=self.stores,
            budget_tokens=config.MAX_CONTEXT_TOKENS,
        )
        context = render_context_for_prompt(evidence)
        timings["context_build_s"] = time.perf_counter() - t3

        if not evidence or (ranked_hits and ranked_hits[0].final_score < config.MIN_EVIDENCE_SCORE):
            timings["total_s"] = sum(timings.values())
            answer_text = self._uncertainty_answer(query)
            debug_payload = {
                "query_id": query_id,
                "intent": intent.__dict__,
                "timings": timings,
                "retrieval_counts": self._counts_by_source(hits),
                "final_evidence_count": len(evidence),
                "verification": {"supported_ratio": 0.0, "reason": "insufficient_evidence"},
            }
            self._log_query(query_id, hits, evidence, timings, debug_payload["verification"])
            return AnswerResult(answer=answer_text, evidence=evidence, confidence=0.2, debug=debug_payload if debug else {})

        t4 = time.perf_counter()
        generated = await generate_answer(query=query, context=context)
        timings["generation_s"] = time.perf_counter() - t4

        t5 = time.perf_counter()
        verified_answer, supported_ratio, verify_debug = await verify_answer(generated, evidence)
        timings["verification_s"] = time.perf_counter() - t5

        confidence = self._compute_confidence(
            query=query,
            evidence=evidence,
            supported_ratio=supported_ratio,
            verify_debug=verify_debug,
        )

        if supported_ratio < config.MIN_SUPPORTED_CLAIMS_RATIO or confidence < 0.45:
            verified_answer = (
                "The available evidence supports only part of the answer. "
                "I may be missing additional clips or stronger corroboration.\n\n"
                + verified_answer
            ).strip()

        timings["total_s"] = sum(timings.values())
        debug_payload = {
            "query_id": query_id,
            "intent": intent.__dict__,
            "timings": timings,
            "retrieval_counts": self._counts_by_source(hits),
            "final_evidence_count": len(evidence),
            "verification": verify_debug,
            "confidence_band": self._confidence_band(confidence),
        }

        self._log_query(query_id, hits, evidence, timings, verify_debug)

        return AnswerResult(
            answer=verified_answer,
            evidence=evidence,
            confidence=confidence,
            debug=debug_payload if debug else {},
        )

    @staticmethod
    def _counts_by_source(hits) -> dict[str, int]:
        counts: dict[str, int] = {}
        for hit in hits:
            for source in hit.source.split("|"):
                counts[source] = counts.get(source, 0) + 1
        return counts

    def _compute_confidence(
        self,
        query: str,
        evidence: list[EvidenceBlock],
        supported_ratio: float,
        verify_debug: dict,
    ) -> float:
        if not evidence:
            return 0.0

        evidence_strength = sum(max(0.0, min(1.0, b.final_score)) for b in evidence) / len(evidence)

        source_count = len({b.source.split("|")[0] for b in evidence})
        retrieval_agreement = min(1.0, source_count / 3.0)

        confidence = 0.5 * evidence_strength + 0.35 * supported_ratio + 0.15 * retrieval_agreement

        if len(evidence) < 2:
            confidence -= 0.15

        unsupported_ratio = float(verify_debug.get("unsupported_ratio", 0.0))
        if unsupported_ratio > 0.40:
            confidence -= 0.25

        query_tokens = {t.lower() for t in query.split() if t.strip()}
        top_text = " ".join(b.text.lower() for b in evidence[:3])
        if query_tokens and not any(tok in top_text for tok in query_tokens):
            confidence -= 0.10

        return max(0.0, min(1.0, confidence))

    @staticmethod
    def _confidence_band(confidence: float) -> str:
        if confidence >= 0.75:
            return "high"
        if confidence >= 0.45:
            return "medium"
        return "low"

    @staticmethod
    def _uncertainty_answer(query: str) -> str:
        return (
            "I do not have enough grounded evidence in the sanitized knowledge cache to answer this confidently. "
            "Please rephrase the question with more specific entities, time windows, or video context."
        )

    def _log_query(
        self,
        query_id: str,
        hits,
        evidence: list[EvidenceBlock],
        timings: dict[str, float],
        verification: dict,
    ) -> None:
        logger.info(
            "query_id=%s retrieval_counts=%s final_evidence_count=%s generation_time=%.3fs verification=%s total_time=%.3fs",
            query_id,
            self._counts_by_source(hits),
            len(evidence),
            timings.get("generation_s", 0.0),
            verification,
            timings.get("total_s", 0.0),
        )
