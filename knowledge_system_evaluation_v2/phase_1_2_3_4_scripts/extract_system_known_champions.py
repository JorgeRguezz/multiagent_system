from __future__ import annotations

import argparse
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CHAMPIONS_PATH = EVAL_ROOT / "lol_champion_names.txt"
DEFAULT_GRAPH_PATH = (
    PROJECT_ROOT
    / "knowledge_sanitization"
    / "cache"
    / "sanitized_global"
    / "graph_AetherNexus.graphml"
)
DEFAULT_OUTPUT_PATH = EVAL_ROOT / "system_known_champions_v2.json"

GRAPHML_NS = {"g": "http://graphml.graphdrawing.org/xmlns"}
GRAPH_FIELD_SEP = "<SEP>"

# These are official display-name variants for Riot/DataDragon compact IDs.
# They are not misspelling aliases.
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
    "NUNUWILLUMP": ["NUNU WILLUMP"],
    "REKSAI": ["REK SAI"],
    "RENATAGLASC": ["RENATA GLASC"],
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
    # Split Riot-style compact IDs such as MissFortune and TwistedFate.
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|[0-9]+", name)
    return " ".join(parts) if parts else name


def champion_phrase_variants(champion: str) -> list[str]:
    key = normalize_key(champion)
    variants = {normalize_words(champion), normalize_words(split_camel_name(champion))}
    variants.update(normalize_words(v) for v in OFFICIAL_DISPLAY_VARIANTS.get(key, []))
    variants.add(key)
    return sorted(v for v in variants if v)


def count_phrase(text_words: str, phrase: str) -> int:
    tokens = phrase.split()
    if not tokens:
        return 0
    pattern = r"(?<![A-Z0-9])" + r"\s+".join(re.escape(tok) for tok in tokens) + r"(?![A-Z0-9])"
    return len(re.findall(pattern, text_words))


def split_source_ids(source_id: str | None) -> list[str]:
    if not source_id:
        return []
    return [part.strip() for part in str(source_id).split(GRAPH_FIELD_SEP) if part.strip()]


def load_champions(path: Path) -> list[str]:
    champions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name:
            champions.append(name)
    return champions


def graph_key_map(root: ET.Element) -> dict[str, str]:
    mapping = {}
    for key in root.findall("g:key", GRAPHML_NS):
        key_id = key.attrib.get("id")
        attr_name = key.attrib.get("attr.name")
        if key_id and attr_name:
            mapping[key_id] = attr_name
    return mapping


def load_graph_nodes(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    root = ET.parse(path).getroot()
    key_map = graph_key_map(root)
    graph = root.find("g:graph", GRAPHML_NS)
    if graph is None:
        raise ValueError(f"No <graph> element found in {path}")

    nodes = []
    degree: dict[str, int] = defaultdict(int)

    for node in graph.findall("g:node", GRAPHML_NS):
        node_id = node.attrib.get("id", "")
        attrs = {}
        for data in node.findall("g:data", GRAPHML_NS):
            attr_name = key_map.get(data.attrib.get("key", ""))
            if attr_name:
                attrs[attr_name] = data.text or ""
        nodes.append(
            {
                "id": node_id,
                "normalized_id": normalize_key(node_id),
                "description": attrs.get("description", ""),
                "source_id": attrs.get("source_id", ""),
                "entity_type": attrs.get("entity_type", ""),
            }
        )

    for edge in graph.findall("g:edge", GRAPHML_NS):
        source = edge.attrib.get("source")
        target = edge.attrib.get("target")
        if source:
            degree[source] += 1
        if target:
            degree[target] += 1

    return nodes, dict(degree)


def extract_known_champions(
    champions: list[str],
    graph_nodes: list[dict[str, Any]],
    degree: dict[str, int],
    min_occurrences: int,
) -> dict[str, Any]:
    results = []

    for champion in champions:
        champion_key = normalize_key(champion)
        phrase_variants = champion_phrase_variants(champion)

        node_id_occurrences = 0
        description_occurrences = 0
        matched_nodes: set[str] = set()
        matched_source_chunks: set[str] = set()
        matched_degree = 0

        for node in graph_nodes:
            node_id = node["id"]
            node_matched = False

            if node["normalized_id"] == champion_key:
                node_id_occurrences += 1
                node_matched = True

            description_words = normalize_words(node.get("description", ""))
            description_count = 0
            if description_words:
                # Count the maximum across official variants to avoid double-counting
                # the same text span through compact and spaced variants.
                description_count = max(
                    count_phrase(description_words, variant) for variant in phrase_variants
                )

            if description_count:
                description_occurrences += description_count
                node_matched = True

            if node_matched:
                matched_nodes.add(node_id)
                matched_degree += degree.get(node_id, 0)
                matched_source_chunks.update(split_source_ids(node.get("source_id")))

        occurrence_count = node_id_occurrences + description_occurrences
        if occurrence_count >= min_occurrences:
            results.append(
                {
                    "champion": champion,
                    "canonical_key": champion_key,
                    "occurrence_count": occurrence_count,
                    "node_id_occurrences": node_id_occurrences,
                    "description_occurrences": description_occurrences,
                    "matched_node_count": len(matched_nodes),
                    "matched_nodes": sorted(matched_nodes),
                    "source_chunk_count": len(matched_source_chunks),
                    "source_chunks": sorted(matched_source_chunks),
                    "degree_sum": matched_degree,
                }
            )

    return {
        "metadata": {
            "min_occurrences": min_occurrences,
            "total_champions_in_reference_list": len(champions),
            "known_champions_count": len(results),
            "matching_policy": (
                "Exact normalized node-id matches plus exact normalized champion-name "
                "phrase matches in node descriptions. No fuzzy matching and no "
                "misspelling aliases."
            ),
        },
        "known_champions": sorted(results, key=lambda item: item["champion"].lower()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract LoL champions known by the system from the sanitized global graph."
    )
    parser.add_argument("--champions", type=Path, default=DEFAULT_CHAMPIONS_PATH)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--min-occurrences",
        type=int,
        default=10,
        help="Minimum total node-id + description occurrences required to count a champion.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    champions = load_champions(args.champions)
    graph_nodes, degree = load_graph_nodes(args.graph)
    output = extract_known_champions(
        champions=champions,
        graph_nodes=graph_nodes,
        degree=degree,
        min_occurrences=max(1, args.min_occurrences),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    metadata = output["metadata"]
    print(
        "Known champions: "
        f"{metadata['known_champions_count']}/"
        f"{metadata['total_champions_in_reference_list']} "
        f"(min_occurrences={metadata['min_occurrences']})"
    )
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
