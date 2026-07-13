from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_inference.query_analyzer import analyze_query
from knowledge_inference.service import InferenceService


QUESTIONS_PATH = PROJECT_ROOT / "knowledge_system_evaluation_v2" / "phase7_positive_control_questions.json"
RESULTS_PATH = PROJECT_ROOT / "knowledge_system_evaluation_v2" / "phase7_positive_control_results.json"
REQUIRED_FIELDS = {
    "question_id",
    "case_type",
    "question_body",
    "answer_gold",
    "source_video",
    "time_span",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 7 qualitative positive controls.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate questions and print detected intents without running inference.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_PATH,
        help=f"Write results to this new path (default: {RESULTS_PATH}).",
    )
    return parser


def load_questions() -> list[dict[str, Any]]:
    with QUESTIONS_PATH.open("r", encoding="utf-8") as handle:
        questions = json.load(handle)

    if not isinstance(questions, list) or not questions:
        raise ValueError("Phase 7 questions must be a non-empty JSON array.")

    seen_ids: set[str] = set()
    for index, question in enumerate(questions):
        if not isinstance(question, dict):
            raise ValueError(f"Question at index {index} must be an object.")
        missing = REQUIRED_FIELDS - question.keys()
        if missing:
            raise ValueError(f"Question at index {index} is missing fields: {sorted(missing)}")

        question_id = str(question["question_id"]).strip()
        question_body = str(question["question_body"]).strip()
        if not question_id or not question_body:
            raise ValueError(f"Question at index {index} has an empty ID or body.")
        if question_id in seen_ids:
            raise ValueError(f"Duplicate question_id: {question_id}")
        seen_ids.add(question_id)

    return questions


def parse_time_point(value: str) -> float:
    parts = value.strip().split(":")
    if not parts or any(not re.fullmatch(r"\d+(?:\.\d+)?", part) for part in parts):
        raise ValueError(f"Invalid time point: {value}")

    total = 0.0
    for part in parts:
        total = total * 60 + float(part)
    return total


def parse_time_span(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, str):
        return None
    parts = re.split(r"\s*[-\u2013\u2014]\s*", value.strip(), maxsplit=1)
    if len(parts) != 2:
        return None
    try:
        start, end = (parse_time_point(part) for part in parts)
    except ValueError:
        return None
    return (min(start, end), max(start, end))


def time_spans_overlap(gold_span: Any, evidence_span: Any) -> bool:
    gold = parse_time_span(gold_span)
    evidence = parse_time_span(evidence_span)
    if gold is None or evidence is None:
        return str(gold_span).strip() == str(evidence_span).strip()
    return max(gold[0], evidence[0]) < min(gold[1], evidence[1])


def gold_evidence_rank(question: dict[str, Any], evidence: list[Any]) -> int | None:
    source_video = question.get("source_video")
    time_span = question.get("time_span")
    if not source_video or not time_span:
        return None

    for rank, block in enumerate(evidence, start=1):
        if block.video_name != source_video:
            continue
        if time_spans_overlap(time_span, block.time_span):
            return rank
    return None


def serialize_evidence(evidence: list[Any], limit: int = 5) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "video": block.video_name,
            "time_span": block.time_span,
            "chunk_id": block.chunk_id,
            "source": block.source,
            "score": block.final_score,
            "text": block.text,
        }
        for rank, block in enumerate(evidence[:limit], start=1)
    ]


def print_dry_run(questions: list[dict[str, Any]]) -> None:
    output = []
    for question in questions:
        intent = analyze_query(question["question_body"])
        output.append(
            {
                "question_id": question["question_id"],
                "case_type": question["case_type"],
                "question_body": question["question_body"],
                "intent": intent.__dict__,
            }
        )
    print(json.dumps(output, indent=2, ensure_ascii=True))


def run_evaluation(questions: list[dict[str, Any]], results_path: Path) -> None:
    if results_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing results: {results_path}")

    service = InferenceService()
    service.initialize()
    results: list[dict[str, Any]] = []

    for index, question in enumerate(questions, start=1):
        logging.info("Running Phase 7 question %d/%d: %s", index, len(questions), question["question_id"])
        result = service.answer(question["question_body"], debug=True)
        rank = gold_evidence_rank(question, result.evidence)
        has_gold_source = bool(question.get("source_video") and question.get("time_span"))

        results.append(
            {
                **question,
                "system_answer": result.answer,
                "retrieved_evidence": serialize_evidence(result.evidence),
                "retrieval_counts": result.debug.get("retrieval_counts", {}),
                "query_intent": result.debug.get("intent", {}),
                "generation_valid": result.debug.get("generation", {}).get("has_final_marker"),
                "gold_evidence_hit": rank is not None if has_gold_source else None,
                "gold_evidence_rank": rank,
            }
        )

    with results_path.open("x", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    print(f"Saved {len(results)} results to {results_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    questions = load_questions()

    if args.dry_run:
        print_dry_run(questions)
        return

    run_evaluation(questions, args.output)


if __name__ == "__main__":
    main()
