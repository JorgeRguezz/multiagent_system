#!/usr/bin/env python3
"""Compute paired bootstrap CIs for Graph-RAG minus Vector-only RAG.

This is local post-processing over already evaluated per-question metrics.
It does not call inference, RAGAS, BERTScore, OpenAI, or any external API.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "knowledge_system_evaluation_v2"
    / "evaluated_datasets"
    / "community_qa_dataset_evaluated.json"
)
FALLBACK_INPUTS = [
    PROJECT_ROOT / "knowledge_system_evaluation_v2" / "community_qa_dataset_evaluated.json",
]
DEFAULT_JSON_OUT = Path(__file__).resolve().parent / "bootstrap_graph_vs_vector.json"
DEFAULT_CSV_OUT = Path(__file__).resolve().parent / "bootstrap_graph_vs_vector.csv"

GRAPH_KEY = "graph_rag"
VECTOR_KEY = "vector_only"


MetricExtractor = Callable[[dict[str, Any]], float | None]


def numeric_metric(field: str) -> MetricExtractor:
    def extract(ablation: dict[str, Any]) -> float | None:
        value = ablation.get(field)
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        return None

    return extract


def bool_metric(field: str) -> MetricExtractor:
    def extract(ablation: dict[str, Any]) -> float | None:
        value = ablation.get(field)
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        return None

    return extract


METRICS: dict[str, MetricExtractor] = {
    "bertscore_f1": numeric_metric("bertscore_f1"),
    "correctness_score": numeric_metric("correctness_score"),
    "faithfulness": numeric_metric("faithfulness"),
    "relevance": numeric_metric("relevance"),
    "refusal_rate": bool_metric("is_refusal"),
    "overrefusal_rate": bool_metric("is_overrefusal"),
}


def load_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists() and path == DEFAULT_INPUT:
        for candidate in FALLBACK_INPUTS:
            if candidate.exists():
                path = candidate
                break

    if not path.exists():
        fallback_text = "\n".join(f"- {candidate}" for candidate in [DEFAULT_INPUT, *FALLBACK_INPUTS])
        raise FileNotFoundError(
            f"Could not find evaluated dataset at {path}.\n"
            "Pass --input /path/to/community_qa_dataset_evaluated.json or place it in one of:\n"
            f"{fallback_text}"
        )

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of question records in {path}")
    return data


def build_slices(data: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    slices: dict[str, list[dict[str, Any]]] = {"all": data}

    for field, prefix in (
        ("answerability_status", "answerability"),
        ("temporal_status", "temporal"),
        ("source", "source"),
    ):
        values = sorted({str(item.get(field, "missing")) for item in data})
        for value in values:
            slices[f"{prefix}:{value}"] = [
                item for item in data if str(item.get(field, "missing")) == value
            ]

    return slices


def paired_values(
    rows: list[dict[str, Any]],
    extractor: MetricExtractor,
) -> tuple[np.ndarray, np.ndarray]:
    graph_values: list[float] = []
    vector_values: list[float] = []

    for item in rows:
        ablations = item.get("ablations", {})
        graph_ablation = ablations.get(GRAPH_KEY, {})
        vector_ablation = ablations.get(VECTOR_KEY, {})

        graph_value = extractor(graph_ablation)
        vector_value = extractor(vector_ablation)

        if graph_value is None or vector_value is None:
            continue

        graph_values.append(graph_value)
        vector_values.append(vector_value)

    return np.asarray(graph_values, dtype=float), np.asarray(vector_values, dtype=float)


def bootstrap_ci(
    diffs: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    if diffs.size == 0:
        return float("nan"), float("nan"), float("nan")

    sample_indices = rng.integers(0, diffs.size, size=(iterations, diffs.size))
    boot_means = diffs[sample_indices].mean(axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    prob_gt_zero = float(np.mean(boot_means > 0.0))
    return float(ci_low), float(ci_high), prob_gt_zero


def interpret(metric: str, ci_low: float, ci_high: float) -> str:
    if np.isnan(ci_low) or np.isnan(ci_high):
        return "insufficient_data"

    if ci_low > 0:
        if metric in {"refusal_rate", "overrefusal_rate"}:
            return "graph_rag_higher_rate"
        return "graph_rag_stable_improvement"

    if ci_high < 0:
        if metric in {"refusal_rate", "overrefusal_rate"}:
            return "graph_rag_lower_rate"
        return "vector_only_stable_improvement"

    return "ci_crosses_zero"


def compute_results(
    data: list[dict[str, Any]],
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    results: list[dict[str, Any]] = []

    for slice_name, rows in build_slices(data).items():
        for metric_name, extractor in METRICS.items():
            graph_values, vector_values = paired_values(rows, extractor)
            if graph_values.size == 0:
                continue

            diffs = graph_values - vector_values
            ci_low, ci_high, prob_gt_zero = bootstrap_ci(diffs, iterations, rng)

            results.append(
                {
                    "comparison": f"{GRAPH_KEY}_minus_{VECTOR_KEY}",
                    "slice": slice_name,
                    "metric": metric_name,
                    "n": int(diffs.size),
                    "graph_mean": float(graph_values.mean()),
                    "vector_mean": float(vector_values.mean()),
                    "mean_diff": float(diffs.mean()),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "prob_diff_gt_0": prob_gt_zero,
                    "bootstrap_iterations": iterations,
                    "seed": seed,
                    "interpretation": interpret(metric_name, ci_low, ci_high),
                }
            )

    return results


def save_json(results: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def save_csv(results: list[dict[str, Any]], path: Path) -> None:
    if not results:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(results[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_summary(results: list[dict[str, Any]]) -> None:
    preferred_order = [
        "bertscore_f1",
        "correctness_score",
        "faithfulness",
        "relevance",
        "refusal_rate",
        "overrefusal_rate",
    ]

    all_results = {row["metric"]: row for row in results if row["slice"] == "all"}
    print("\nGraph-RAG minus Vector-only, all questions")
    print("metric,n,graph_mean,vector_mean,diff,ci_low,ci_high,interpretation")
    for metric in preferred_order:
        row = all_results.get(metric)
        if not row:
            continue
        print(
            f"{metric},{row['n']},"
            f"{row['graph_mean']:.6f},{row['vector_mean']:.6f},"
            f"{row['mean_diff']:.6f},{row['ci_low']:.6f},{row['ci_high']:.6f},"
            f"{row['interpretation']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute paired bootstrap CIs for Graph-RAG minus Vector-only RAG."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV_OUT)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260709)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")

    data = load_dataset(args.input)
    results = compute_results(data, iterations=args.iterations, seed=args.seed)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, args.json_out)
    save_csv(results, args.csv_out)
    print_summary(results)
    print(f"\nWrote {len(results)} rows to:")
    print(f"- {args.json_out}")
    print(f"- {args.csv_out}")


if __name__ == "__main__":
    main()

