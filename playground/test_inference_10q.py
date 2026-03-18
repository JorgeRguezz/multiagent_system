from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from knowledge_inference.service import InferenceService


TEST_CASES: list[dict[str, Any]] = [
    {
        "id": "pyke_topics",
        "question": "What major topics does the guide say it will cover for learning Pyke?",
        "expected_videos": ["How_to_Play_Like_a_PYKE_MAIN_-_ULTIMATE_PYKE_GUIDE"],
        "expected_answer_keywords": ["playstyle", "power spikes", "combos", "tips", "build"],
    },
    {
        "id": "pyke_carry_support",
        "question": "Why does the video describe Pyke as a support who can still carry games?",
        "expected_videos": ["How_to_Play_Like_a_PYKE_MAIN_-_ULTIMATE_PYKE_GUIDE"],
        "expected_answer_keywords": ["carry", "support", "mid lane assassin", "farm", "flashy"],
    },
    {
        "id": "smolder_q_sheen",
        "question": "What does the guide recommend doing when using Q on a minion wave with a Sheen item?",
        "expected_videos": ["The_COMPLETE_SMOLDER_STARTER_GUIDE_-_League_of_Legends"],
        "expected_answer_keywords": ["highest health minion", "splash damage", "Sheen", "Q"],
    },
    {
        "id": "smolder_w_trade",
        "question": "How should W be used to trade while also preparing minions for farming?",
        "expected_videos": ["The_COMPLETE_SMOLDER_STARTER_GUIDE_-_League_of_Legends"],
        "expected_answer_keywords": ["throw", "through", "creeps", "prepare", "Q"],
    },
    {
        "id": "ahri_level1",
        "question": "What level-1 laning advice is given for Ahri's early trading or wave control?",
        "expected_videos": ["S+_BUFFED_AHRI_MID_IS_GOD_TIER_Best_Build_&_Runes_How_to_Carry_with_Ahri_League_of_Legends"],
        "expected_answer_keywords": ["level 1", "W", "auto attack", "close enough", "minions"],
    },
    {
        "id": "ahri_shop_items",
        "question": "Which items are considered during Ahri's recall or shop segment, and what are they meant to help with?",
        "expected_videos": ["S+_BUFFED_AHRI_MID_IS_GOD_TIER_Best_Build_&_Runes_How_to_Carry_with_Ahri_League_of_Legends"],
        "expected_answer_keywords": ["Blackfire Torch", "Luden", "Malignance", "damage", "burst"],
    },
    {
        "id": "low_elo_rule1",
        "question": "What is presented as the first and most important rule?",
        "expected_videos": ["The_10_RULES_for_ESCAPING_LOW_ELO_(NOT_CLICKBAIT_)_-_League_of_Legends"],
        "expected_answer_keywords": ["never help your losing teammates", "losing teammates"],
    },
    {
        "id": "low_elo_dead_time",
        "question": "What does the guide mean by dead time, and how should players use it?",
        "expected_videos": ["The_10_RULES_for_ESCAPING_LOW_ELO_(NOT_CLICKBAIT_)_-_League_of_Legends"],
        "expected_answer_keywords": ["dead time", "clear", "wave", "roam", "push", "side lane"],
    },
    {
        "id": "ahri_lore_origin",
        "question": "What does the cache say about Ahri's origin, her connection to ice foxes, and the spirit realm?",
        "expected_videos": ["Toda_la_historia_de_Ahri_League_of_Legends"],
        "expected_answer_keywords": ["Bastaya", "ice foxes", "spirit realm", "origin", "tribe"],
    },
    {
        "id": "zaahen_passive",
        "question": "How does Zaahen's passive work, including stacks and the revive mechanic at maximum stacks?",
        "expected_videos": ["How_To_Dominate_with_Zaahen_League_of_Legends"],
        "expected_answer_keywords": ["stacks", "12", "attack damage", "revive", "full stacks"],
    },
]


def _contains_any(text: str, keywords: list[str]) -> bool:
    haystack = text.lower()
    return any(keyword.lower() in haystack for keyword in keywords)


def _evidence_overlap(result_videos: set[str], expected_videos: list[str]) -> float:
    if not expected_videos:
        return 1.0
    expected = {video.lower() for video in expected_videos}
    overlap = len({video.lower() for video in result_videos} & expected)
    return overlap / max(1, len(expected))


def run_test_cases() -> dict[str, Any]:
    service = InferenceService()
    results: list[dict[str, Any]] = []

    for index, case in enumerate(TEST_CASES, start=1):
        started = time.perf_counter()
        answer = service.answer(case["question"], debug=True)
        latency_s = time.perf_counter() - started

        evidence_videos = {block.video_name for block in answer.evidence}
        keyword_hit = _contains_any(answer.answer, case.get("expected_answer_keywords", []))
        video_overlap = _evidence_overlap(evidence_videos, case.get("expected_videos", []))

        results.append(
            {
                "index": index,
                "id": case["id"],
                "question": case["question"],
                "expected_videos": case["expected_videos"],
                "expected_answer_keywords": case["expected_answer_keywords"],
                "answer": answer.answer,
                "confidence": answer.confidence,
                "latency_s": latency_s,
                "keyword_hit": keyword_hit,
                "video_overlap": video_overlap,
                "supported_ratio": float(answer.debug.get("verification", {}).get("supported_ratio", 0.0)),
                "evidence": [
                    {
                        "video": block.video_name,
                        "time": block.time_span,
                        "chunk": block.chunk_id,
                        "source": block.source,
                        "score": block.final_score,
                    }
                    for block in answer.evidence
                ],
                "debug": answer.debug,
            }
        )

    avg_confidence = sum(item["confidence"] for item in results) / max(1, len(results))
    avg_supported = sum(item["supported_ratio"] for item in results) / max(1, len(results))
    avg_video_overlap = sum(item["video_overlap"] for item in results) / max(1, len(results))
    keyword_hits = sum(1 for item in results if item["keyword_hit"])

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "case_count": len(results),
        "summary": {
            "avg_confidence": avg_confidence,
            "avg_supported_ratio": avg_supported,
            "avg_video_overlap": avg_video_overlap,
            "keyword_hit_count": keyword_hits,
        },
        "cases": results,
    }


def save_report(report: dict[str, Any]) -> Path:
    out_dir = Path(__file__).resolve().parent / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"test_inference_10q_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return out_path


def main() -> None:
    report = run_test_cases()
    save_report(report)

    for case in report["cases"]:
        print(f"Question: {case['question']}")
        print(f"Answer: {case['answer']}")
        print(f"Time: {case['latency_s']:.3f}s")

        resources = case["evidence"]
        if resources:
            rendered = [
                (
                    f"video={item['video']} "
                    f"time={item['time']} "
                    f"chunk={item['chunk']} "
                    f"source={item['source']}"
                )
                for item in resources
            ]
            print("Resources:")
            for line in rendered:
                print(line)
        else:
            print("Resources: None")

        print()


if __name__ == "__main__":
    main()
