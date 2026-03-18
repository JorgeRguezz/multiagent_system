from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TARGET_JSON_PATH = (
    Path(__file__).resolve().parent / "report_eval_test_rag_merged_done.json"
)
SOURCE_JSON_PATH = (
    Path(__file__).resolve().parent
    / "report_eval_test_rag_merged_valid_strong_llm_answers.json"
)
OUTPUT_JSON_PATH = (
    Path(__file__).resolve().parent / "report_eval_test_rag_merged_done_updated.json"
)


def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a top-level JSON list in {path}")
    if not all(isinstance(item, dict) for item in data):
        raise ValueError(f"Expected every item in {path} to be a JSON object")
    return data


def _write_json(path: Path, data: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=True)
        f.write("\n")


def _validate_alignment(
    target_items: list[dict[str, Any]],
    source_items: list[dict[str, Any]],
) -> None:
    if len(target_items) != len(source_items):
        raise ValueError(
            f"Input lengths differ: target={len(target_items)} source={len(source_items)}"
        )

    for idx, (target, source) in enumerate(zip(target_items, source_items), start=1):
        target_question = str(target.get("question", "")).strip()
        source_question = str(source.get("question", "")).strip()
        if target_question != source_question:
            raise ValueError(
                f'Question mismatch at index {idx}: target="{target_question}" source="{source_question}"'
            )


def replace_strong_llm_answers(
    target_path: Path,
    source_path: Path,
    output_path: Path,
) -> None:
    target_items = _load_json(target_path)
    source_items = _load_json(source_path)

    _validate_alignment(target_items=target_items, source_items=source_items)

    merged_items: list[dict[str, Any]] = []
    for target, source in zip(target_items, source_items):
        updated = dict(target)
        updated["strong_llm_answer"] = source.get("strong_llm_answer", "")
        merged_items.append(updated)

    _write_json(output_path, merged_items)
    print(f"Copied strong_llm_answer into: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replace strong_llm_answer values in one evaluation JSON from another."
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=TARGET_JSON_PATH,
        help="JSON file whose strong_llm_answer values will be replaced",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=SOURCE_JSON_PATH,
        help="JSON file providing the valid strong_llm_answer values",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_JSON_PATH,
        help="Destination JSON file",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    replace_strong_llm_answers(
        target_path=args.target.resolve(),
        source_path=args.source.resolve(),
        output_path=args.output.resolve(),
    )


if __name__ == "__main__":
    main()
