from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


EVAL_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REPORT_PATH = EVAL_ROOT / "report_eval_test.json"
DEFAULT_KNOWN_CHAMPIONS_PATH = EVAL_ROOT / "system_known_champions_v2.json"
DEFAULT_OUTPUT_PATH = EVAL_ROOT / "filtered_report_eval_test.json"


# Official display-name variants for Riot/DataDragon compact IDs. These are
# not misspelling aliases; they only bridge compact IDs and player-facing names.
OFFICIAL_DISPLAY_VARIANTS = {
    "AURELIONSOL": ["AURELION SOL"],
    "BELVETH": ["BEL VETH"],
    "CHOGATH": ["CHO GATH"],
    "DRMUNDO": ["DR MUNDO"],
    "JARVANIV": ["JARVAN IV"],
    "KAISA": ["KAI SA"],
    "KHAZIX": ["KHA ZIX"],
    "KOGMAW": ["KOG MAW"],
    "KSANTE": ["K SANTE"],
    "LEBLANC": ["LE BLANC"],
    "LEESIN": ["LEE SIN"],
    "MASTERYI": ["MASTER YI"],
    "MISSFORTUNE": ["MISS FORTUNE"],
    "MONKEYKING": ["WUKONG"],
    "NUNU": ["NUNU WILLUMP", "NUNU AND WILLUMP"],
    "REKSAI": ["REK SAI"],
    "RENATA": ["RENATA GLASC"],
    "TAHMKENCH": ["TAHM KENCH"],
    "TWISTEDFATE": ["TWISTED FATE"],
    "VELKOZ": ["VEL KOZ"],
    "XINZHAO": ["XIN ZHAO"],
}


def normalize_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    without_marks = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return re.sub(r"[^A-Z0-9]+", "", without_marks.upper())


def normalize_words(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    without_marks = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[^A-Z0-9]+", " ", without_marks.upper()).split())


def split_camel_name(name: str) -> str:
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|[0-9]+", name)
    return " ".join(parts) if parts else name


def champion_phrase_variants(champion: str) -> list[str]:
    key = normalize_key(champion)
    variants = {
        normalize_words(champion),
        normalize_words(split_camel_name(champion)),
    }
    variants.update(normalize_words(name) for name in OFFICIAL_DISPLAY_VARIANTS.get(key, []))
    return sorted(variant for variant in variants if variant)


def phrase_in_text(text_words: str, phrase: str) -> bool:
    tokens = phrase.split()
    if not tokens:
        return False
    pattern = r"(?<![A-Z0-9])" + r"\s+".join(re.escape(tok) for tok in tokens) + r"(?![A-Z0-9])"
    return re.search(pattern, text_words) is not None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_known_champions(path: Path) -> list[str]:
    data = load_json(path)
    champions = data.get("known_champions", [])
    if not isinstance(champions, list):
        raise ValueError(f"Expected known_champions list in {path}")

    names = []
    for item in champions:
        if isinstance(item, dict) and item.get("champion"):
            names.append(str(item["champion"]))
    if not names:
        raise ValueError(f"No known champions found in {path}")
    return names


def matched_known_champions(question: str, known_champions: list[str]) -> list[str]:
    question_words = normalize_words(question)
    matches = []
    for champion in known_champions:
        variants = champion_phrase_variants(champion)
        if any(phrase_in_text(question_words, variant) for variant in variants):
            matches.append(champion)
    return matches


def filter_report(
    report_rows: list[dict[str, Any]],
    known_champions: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept = []
    excluded = []

    for index, row in enumerate(report_rows, start=1):
        question = str(row.get("question", ""))
        matches = matched_known_champions(question, known_champions)
        if matches:
            kept.append(dict(row))
        else:
            excluded.append(
                {
                    "index": index,
                    "question": question,
                    "reason": "no_known_champion_mentioned",
                }
            )

    return kept, excluded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter report_eval_test.json to questions mentioning system-known champions."
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--known-champions", type=Path, default=DEFAULT_KNOWN_CHAMPIONS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--excluded-output",
        type=Path,
        default=None,
        help="Optional JSON file listing excluded questions and reasons.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    report = load_json(args.report)
    if not isinstance(report, list):
        raise ValueError(f"Expected a top-level list in {args.report}")

    known_champions = load_known_champions(args.known_champions)
    filtered, excluded = filter_report(report, known_champions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(filtered, indent=4, ensure_ascii=True) + "\n", encoding="utf-8")

    if args.excluded_output:
        args.excluded_output.parent.mkdir(parents=True, exist_ok=True)
        args.excluded_output.write_text(
            json.dumps(excluded, indent=4, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

    print(f"Known champions loaded: {len(known_champions)}")
    print(f"Input questions: {len(report)}")
    print(f"Kept questions: {len(filtered)}")
    print(f"Excluded questions: {len(excluded)}")
    print(f"Wrote: {args.output}")
    if args.excluded_output:
        print(f"Wrote excluded questions: {args.excluded_output}")


if __name__ == "__main__":
    main()
