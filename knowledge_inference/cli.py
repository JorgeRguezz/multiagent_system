from __future__ import annotations

import argparse
import json
import logging

from . import config
from .service import InferenceService


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Knowledge inference CLI over sanitized caches.")
    parser.add_argument("--query", required=True, help="User question to answer")
    parser.add_argument("--debug", action="store_true", help="Show debug timings and branch statistics")
    parser.add_argument("--max-evidence", type=int, default=5, help="Number of top evidence blocks to print")
    return parser


def main() -> None:
    _configure_logging()
    parser = build_parser()
    args = parser.parse_args()

    service = InferenceService()
    result = service.answer(args.query, debug=args.debug)

    print("Answer:\n")
    print(result.answer)
    print("\nConfidence:", f"{result.confidence:.3f}")

    print("\nTop Evidence:")
    if not result.evidence:
        print("- None")
    else:
        for block in result.evidence[: max(1, args.max_evidence)]:
            print(
                f"- video={block.video_name} time={block.time_span} "
                f"chunk={block.chunk_id} source={block.source} score={block.final_score:.3f}"
            )

    if args.debug:
        print("\nDebug:")
        print(json.dumps(result.debug, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
