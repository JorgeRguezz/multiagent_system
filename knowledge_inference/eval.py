from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .service import InferenceService


def _read_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "cases" in data:
        data = data["cases"]
    if not isinstance(data, list):
        raise ValueError("Dataset must be a list or {'cases': [...]} JSON.")
    return data


def _contains_any(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    return any(k.lower() in t for k in keywords)


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, int(round(q * (len(sorted_values) - 1))))
    return sorted_values[idx]


def run_eval(dataset_path: Path) -> dict[str, Any]:
    cases = _read_dataset(dataset_path)
    svc = InferenceService()

    per_case: list[dict[str, Any]] = []
    latencies: list[float] = []
    retrieval_proxy_scores: list[float] = []
    groundedness_scores: list[float] = []

    for i, case in enumerate(cases, start=1):
        question = str(case.get("question", "")).strip()
        if not question:
            continue

        t0 = time.perf_counter()
        result = svc.answer(question, debug=True)
        latency = time.perf_counter() - t0
        latencies.append(latency)

        expected_keywords = case.get("expected_answer_keywords", []) or []
        expected_videos = case.get("expected_videos", []) or case.get("expected_champions", []) or []

        keyword_hit = 1.0
        if expected_keywords:
            keyword_hit = 1.0 if _contains_any(result.answer, [str(k) for k in expected_keywords]) else 0.0

        video_hit = 1.0
        if expected_videos:
            videos_in_evidence = {b.video_name.lower() for b in result.evidence}
            expected = {str(v).lower() for v in expected_videos}
            overlap = len(videos_in_evidence & expected)
            video_hit = overlap / max(1, len(expected))

        retrieval_proxy = 0.5 * keyword_hit + 0.5 * video_hit
        retrieval_proxy_scores.append(retrieval_proxy)

        supported_ratio = float(result.debug.get("verification", {}).get("supported_ratio", 0.0))
        groundedness_scores.append(supported_ratio)

        per_case.append(
            {
                "index": i,
                "question": question,
                "answer": result.answer,
                "confidence": result.confidence,
                "latency_s": latency,
                "retrieval_recall_proxy": retrieval_proxy,
                "groundedness_proxy": supported_ratio,
                "evidence": [
                    {
                        "video": b.video_name,
                        "time": b.time_span,
                        "chunk": b.chunk_id,
                        "source": b.source,
                        "score": b.final_score,
                    }
                    for b in result.evidence
                ],
            }
        )

    latencies_sorted = sorted(latencies)
    summary = {
        "cases_total": len(per_case),
        "retrieval_recall_proxy_mean": statistics.fmean(retrieval_proxy_scores) if retrieval_proxy_scores else 0.0,
        "groundedness_proxy_mean": statistics.fmean(groundedness_scores) if groundedness_scores else 0.0,
        "latency_p50_s": _percentile(latencies_sorted, 0.50),
        "latency_p95_s": _percentile(latencies_sorted, 0.95),
    }

    return {"summary": summary, "cases": per_case}


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline evaluator for knowledge inference QA.")
    parser.add_argument("--dataset", required=True, help="Path to evaluation JSON")
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    report = run_eval(dataset_path)

    out_dir = Path(__file__).resolve().parent / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)

    print(f"Saved report: {out_path}")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
