import argparse
import os
from collections import defaultdict

from .config import (
    EXTRACTED_DATA_ROOT,
    SANITIZED_EXTRACTED_DATA_ROOT,
    REPORT_PRE_ROOT,
    QUAR_PRE_ROOT,
    SPEC_ROOT,
    REQUIRED_EXTRACTION_FILES,
)
from .utils import (
    ensure_dir,
    load_json,
    save_json,
    append_jsonl,
    clean_text,
    parse_segment_time,
    normalize_name,
)


def _load_specs():
    blocked_meta = load_json(os.path.join(SPEC_ROOT, "blocked_meta_patterns.json"), default=[])
    alias_map = {}
    for fname in ("alias_champions.json", "alias_items.json", "alias_objectives.json"):
        alias_map.update(load_json(os.path.join(SPEC_ROOT, fname), default={}))
    return blocked_meta, alias_map


def sanitize_video_folder(video_dir: str) -> dict:
    video_name = os.path.basename(video_dir)
    out_dir = os.path.join(SANITIZED_EXTRACTED_DATA_ROOT, video_name)
    quar_dir = os.path.join(QUAR_PRE_ROOT, video_name)
    ensure_dir(out_dir)
    ensure_dir(quar_dir)

    blocked_meta, alias_map = _load_specs()
    report = {
        "video": video_name,
        "status": "pass",
        "input_dir": video_dir,
        "output_dir": out_dir,
        "files": defaultdict(lambda: {"in": 0, "out": 0, "dropped": 0}),
        "normalizations": defaultdict(int),
        "contamination_hits": 0,
        "warnings": [],
    }

    for fname in REQUIRED_EXTRACTION_FILES:
        if not os.path.exists(os.path.join(video_dir, fname)):
            report["status"] = "fail"
            report["warnings"].append(f"missing file: {fname}")

    segments = load_json(os.path.join(video_dir, "kv_store_video_segments.json"), default={}) or {}
    frames = load_json(os.path.join(video_dir, "kv_store_video_frames.json"), default={}) or {}
    video_path = load_json(os.path.join(video_dir, "kv_store_video_path.json"), default={}) or {}

    clean_segments = {video_name: {}}
    clean_frames = {video_name: {}}

    seg_map = segments.get(video_name, {})
    report["files"]["kv_store_video_segments.json"]["in"] = len(seg_map)

    valid_segments = set()
    for seg_idx, seg in seg_map.items():
        rec = dict(seg)
        parsed = parse_segment_time(str(rec.get("time", "")))
        if parsed is None:
            append_jsonl(os.path.join(quar_dir, "video_segments_invalid.jsonl"), {
                "segment_idx": seg_idx,
                "reason": "invalid_time",
                "value": rec.get("time"),
            })
            report["files"]["kv_store_video_segments.json"]["dropped"] += 1
            continue

        content, cstats = clean_text(str(rec.get("content", "")), blocked_meta)
        transcript, tstats = clean_text(str(rec.get("transcript", "")), blocked_meta)
        report["contamination_hits"] += cstats.get("meta_tags_removed", 0) + cstats.get("meta_patterns_removed", 0)
        report["contamination_hits"] += tstats.get("meta_tags_removed", 0) + tstats.get("meta_patterns_removed", 0)

        if not content and not transcript:
            append_jsonl(os.path.join(quar_dir, "video_segments_invalid.jsonl"), {
                "segment_idx": seg_idx,
                "reason": "empty_after_clean",
            })
            report["files"]["kv_store_video_segments.json"]["dropped"] += 1
            continue

        frame_times = rec.get("frame_times", [])
        if not isinstance(frame_times, list):
            frame_times = []
        start, end = parsed
        clean_frame_times = []
        for t in frame_times:
            try:
                tf = float(t)
            except Exception:
                continue
            if tf < start:
                tf = start
            if tf > end:
                tf = end
            clean_frame_times.append(tf)
        clean_frame_times = sorted(clean_frame_times)

        clean_segments[video_name][str(seg_idx)] = {
            "time": f"{start:g}-{end:g}",
            "content": content,
            "transcript": transcript,
            "frame_times": clean_frame_times,
        }
        valid_segments.add(str(seg_idx))

    report["files"]["kv_store_video_segments.json"]["out"] = len(clean_segments[video_name])

    frame_map = frames.get(video_name, {})
    report["files"]["kv_store_video_frames.json"]["in"] = len(frame_map)
    for frame_key, fr in frame_map.items():
        if "_" not in frame_key:
            report["files"]["kv_store_video_frames.json"]["dropped"] += 1
            append_jsonl(os.path.join(quar_dir, "video_frames_invalid.jsonl"), {
                "frame_key": frame_key,
                "reason": "invalid_key",
            })
            continue

        seg_idx = str(fr.get("segment_idx", frame_key.split("_")[0]))
        if seg_idx not in valid_segments:
            report["files"]["kv_store_video_frames.json"]["dropped"] += 1
            continue

        transcript, tstats = clean_text(str(fr.get("transcript", "")), blocked_meta)
        vlm_output, vstats = clean_text(str(fr.get("vlm_output", "")), blocked_meta)
        report["contamination_hits"] += tstats.get("meta_tags_removed", 0) + tstats.get("meta_patterns_removed", 0)
        report["contamination_hits"] += vstats.get("meta_tags_removed", 0) + vstats.get("meta_patterns_removed", 0)

        entities = fr.get("entities", [])
        if not isinstance(entities, list):
            entities = []
        norm_entities = []
        for entity in entities:
            normalized_entity = normalize_name(str(entity), alias_map)
            if normalized_entity and normalized_entity != "UNKNOWN" and normalized_entity not in norm_entities:
                norm_entities.append(normalized_entity)

        clean_frame = {
            **fr,
            "segment_idx": seg_idx,
            "entities": norm_entities,
            "transcript": transcript,
            "vlm_output": vlm_output,
        }
        if "main_champ" in fr:
            main_champ = normalize_name(str(fr.get("main_champ", "Unknown")), alias_map)
            if main_champ != str(fr.get("main_champ", "Unknown")).upper():
                report["normalizations"]["main_champ"] += 1
            clean_frame["main_champ"] = main_champ

        if "partners" in fr:
            partners = fr.get("partners", [])
            if not isinstance(partners, list):
                partners = []
            norm_partners = []
            for p in partners:
                np = normalize_name(str(p), alias_map)
                if np and np != "UNKNOWN" and np not in norm_partners:
                    norm_partners.append(np)
            clean_frame["partners"] = norm_partners

        clean_frames[video_name][frame_key] = clean_frame

    report["files"]["kv_store_video_frames.json"]["out"] = len(clean_frames[video_name])

    report["files"]["kv_store_video_path.json"]["in"] = len(video_path)
    clean_video_path = {}
    if video_name in video_path and isinstance(video_path[video_name], str):
        clean_video_path[video_name] = video_path[video_name]
        if not os.path.isabs(video_path[video_name]):
            report["warnings"].append("video path is not absolute")
        if not os.path.exists(video_path[video_name]):
            report["warnings"].append("video path does not exist on disk")
    else:
        report["files"]["kv_store_video_path.json"]["dropped"] = max(1, len(video_path))
        report["warnings"].append("video path key mismatch or invalid")
    report["files"]["kv_store_video_path.json"]["out"] = len(clean_video_path)

    if report["warnings"] and report["status"] != "fail":
        report["status"] = "warn"

    save_json(clean_segments, os.path.join(out_dir, "kv_store_video_segments.json"))
    save_json(clean_frames, os.path.join(out_dir, "kv_store_video_frames.json"))
    save_json(clean_video_path, os.path.join(out_dir, "kv_store_video_path.json"))
    save_json(report, os.path.join(REPORT_PRE_ROOT, f"{video_name}.json"))

    return report


def discover_video_dirs(extracted_data_root: str):
    if not os.path.isdir(extracted_data_root):
        return []
    return sorted([e.path for e in os.scandir(extracted_data_root) if e.is_dir()])


def main() -> int:
    parser = argparse.ArgumentParser(description="Sanitize extracted artifacts before knowledge build.")
    parser.add_argument("--input-root", default=EXTRACTED_DATA_ROOT)
    parser.add_argument("--video", default=None, help="Optional video folder name to process only one")
    args = parser.parse_args()

    ensure_dir(SANITIZED_EXTRACTED_DATA_ROOT)
    ensure_dir(REPORT_PRE_ROOT)

    if args.video:
        dirs = [os.path.join(args.input_root, args.video)]
    else:
        dirs = discover_video_dirs(args.input_root)

    if not dirs:
        print(f"No extracted video folders found in {args.input_root}")
        return 1

    failed = 0
    for d in dirs:
        report = sanitize_video_folder(d)
        print(f"[pre-build] {report['video']}: {report['status']}")
        if report["status"] == "fail":
            failed += 1

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
