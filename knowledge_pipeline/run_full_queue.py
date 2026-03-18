"""
Run full per-video workflow for downloads/queue videos that are not fully processed yet:
extractor -> pre-build sanitization -> build -> post-build sanitization

Usage:
  python3 -m pipeline.run_full_queue
  python3 -m pipeline.run_full_queue --dry-run
  python3 -m pipeline.run_full_queue --force
  python3 -m pipeline.run_full_queue --continue-on-error
  python3 -m pipeline.run_full_queue --video <VIDEO_BASENAME>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
PROJECT_ROOT = Path("/home/gatv-projects/Desktop/project")
DOWNLOADS_QUEUE_DIR = PROJECT_ROOT / "downloads" / "queue"

EXTRACTION_DIR = PROJECT_ROOT / "knowledge_extraction" / "cache" / "extracted_data"
SANITIZED_EXTRACTION_DIR = PROJECT_ROOT / "knowledge_sanitization" / "cache" / "sanitized_extracted_data"
SANITIZED_BUILD_ROOT = PROJECT_ROOT / "knowledge_sanitization" / "cache"
REPORT_ROOT = PROJECT_ROOT / "knowledge_sanitization" / "cache" / "reports"

REQUIRED_EXTRACTION_FILES = {
    "kv_store_video_segments.json",
    "kv_store_video_frames.json",
    "kv_store_video_path.json",
}

REQUIRED_BUILD_FILES = {
    "kv_store_text_chunks.json",
    "kv_store_video_segments.json",
    "kv_store_video_frames.json",
    "kv_store_video_path.json",
    "vdb_chunks.json",
    "vdb_entities.json",
    "graph_chunk_entity_relation.graphml",
    "graph_chunk_entity_relation_clean.graphml",
}


def _discover_videos(downloads_dir: Path) -> list[Path]:
    if not downloads_dir.is_dir():
        return []
    videos = []
    for entry in os.scandir(downloads_dir):
        if not entry.is_file():
            continue
        ext = Path(entry.name).suffix.lower()
        if ext in VIDEO_EXTENSIONS:
            videos.append(Path(entry.path))
    return sorted(videos)


def _dir_has_files(root: Path, file_names: set[str]) -> bool:
    if not root.is_dir():
        return False
    return all((root / fn).exists() for fn in file_names)


def _build_cache_dir(video_basename: str) -> Path:
    return PROJECT_ROOT / f"knowledge_build_cache_{video_basename}"


def _sanitized_build_cache_dir(video_basename: str) -> Path:
    return SANITIZED_BUILD_ROOT / f"sanitized_build_cache_{video_basename}"


def _is_fully_processed(video_basename: str) -> bool:
    extraction_ok = _dir_has_files(EXTRACTION_DIR / video_basename, REQUIRED_EXTRACTION_FILES)
    pre_ok = _dir_has_files(SANITIZED_EXTRACTION_DIR / video_basename, REQUIRED_EXTRACTION_FILES)
    build_ok = _dir_has_files(_build_cache_dir(video_basename), REQUIRED_BUILD_FILES)
    post_ok = _dir_has_files(_sanitized_build_cache_dir(video_basename), REQUIRED_BUILD_FILES)
    return extraction_ok and pre_ok and build_ok and post_ok


def _run(cmd: list[str], dry_run: bool = False) -> None:
    print("$", " ".join(cmd))
    if dry_run:
        return
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def _copy_sanitized_video_to_temp(video_basename: str, dry_run: bool = False) -> Path:
    src_video_dir = SANITIZED_EXTRACTION_DIR / video_basename
    if dry_run:
        return Path(f"/tmp/dryrun_kg_single_build_{video_basename}")

    if not src_video_dir.exists():
        raise FileNotFoundError(f"Sanitized extraction folder not found: {src_video_dir}")

    tmp_root = Path(tempfile.mkdtemp(prefix=f"kg_full_queue_build_{video_basename}_"))
    dst_root = tmp_root / "sanitized_extracted_data"
    dst_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_video_dir, dst_root / video_basename)
    return tmp_root


async def _run_build_for_video(isolated_extraction_root: Path, target_basename: str, dry_run: bool = False) -> str:
    if dry_run:
        return str(_build_cache_dir(target_basename))

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from knowledge_build.builder import KnowledgeBuilder  # noqa: WPS433

    builder = KnowledgeBuilder(extraction_dir=str(isolated_extraction_root))
    selected = Path(builder.artifact_dir).name
    if selected != target_basename:
        raise RuntimeError(
            f"Build selected unexpected artifact '{selected}' (expected '{target_basename}')"
        )

    await builder.build()
    return builder.working_dir


def _full_pipeline_for_video(video_path: Path, dry_run: bool = False) -> dict[str, Any]:
    video_basename = video_path.stem
    status: dict[str, Any] = {
        "video_basename": video_basename,
        "video_path": str(video_path),
        "steps": [],
        "result": "success",
        "error": None,
    }

    tmp_root: Path | None = None
    try:
        # 1) Extraction
        cmd = [
            sys.executable,
            "-m",
            "knowledge_extraction.extractor",
            "--video-path",
            str(video_path),
        ]
        _run(cmd, dry_run=dry_run)
        status["steps"].append("extraction")

        # 2) Pre-build sanitization
        cmd = [
            sys.executable,
            "-m",
            "knowledge_sanitization.pre_build",
            "--video",
            video_basename,
        ]
        _run(cmd, dry_run=dry_run)
        status["steps"].append("pre_build_sanitization")

        # 3) Build (isolated)
        tmp_root = _copy_sanitized_video_to_temp(video_basename, dry_run=dry_run)
        build_output = asyncio.run(_run_build_for_video(tmp_root, video_basename, dry_run=dry_run))
        status["steps"].append("build")
        status["build_output_dir"] = build_output

        # 4) Post-build sanitization
        cmd = [
            sys.executable,
            "-m",
            "knowledge_sanitization.post_build",
            "--video",
            video_basename,
        ]
        _run(cmd, dry_run=dry_run)
        status["steps"].append("post_build_sanitization")

    except Exception as exc:  # noqa: BLE001
        status["result"] = "failed"
        status["error"] = str(exc)
    finally:
        if tmp_root and not dry_run:
            shutil.rmtree(tmp_root, ignore_errors=True)

    return status


def _write_summary_report(summary: dict[str, Any]) -> Path:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    out_path = REPORT_ROOT / f"full_pipeline_queue_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return out_path


def _cleanup_in_process_vram() -> None:
    """
    Best-effort cleanup between queue items so next video starts with fresh VRAM.
    """
    try:
        from knowledge_build._llm import shutdown_all_llm_resources  # noqa: WPS433

        shutdown_all_llm_resources()
    except Exception as exc:  # noqa: BLE001
        print(f"[cleanup] Warning: could not run LLM cleanup: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full extraction->sanitize->build->sanitize queue.")
    parser.add_argument("--downloads-dir", default=str(DOWNLOADS_QUEUE_DIR))
    parser.add_argument("--video", default=None, help="Optional video basename to process only one item")
    parser.add_argument("--force", action="store_true", help="Process even if already fully processed")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without executing")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining videos if one video fails",
    )
    args = parser.parse_args()

    downloads_dir = Path(args.downloads_dir).expanduser().resolve()
    if not downloads_dir.is_dir():
        print(f"Downloads queue directory not found: {downloads_dir}")
        return 1

    videos = _discover_videos(downloads_dir)
    if args.video:
        videos = [v for v in videos if v.stem == args.video]

    if not videos:
        print(f"No videos found to process in {downloads_dir}")
        return 1

    pending: list[Path] = []
    skipped: list[str] = []
    for v in videos:
        if args.force or not _is_fully_processed(v.stem):
            pending.append(v)
        else:
            skipped.append(v.stem)

    print(f"Discovered videos: {len(videos)}")
    print(f"Skipped (already fully processed): {len(skipped)}")
    print(f"Pending: {len(pending)}")

    summary: dict[str, Any] = {
        "downloads_dir": str(downloads_dir),
        "force": args.force,
        "dry_run": args.dry_run,
        "continue_on_error": args.continue_on_error,
        "discovered": len(videos),
        "skipped": skipped,
        "results": [],
    }

    failures = 0
    for idx, video in enumerate(pending, start=1):
        print("\n" + "=" * 80)
        print(f"[{idx}/{len(pending)}] Processing {video.name}")
        result = _full_pipeline_for_video(video, dry_run=args.dry_run)
        summary["results"].append(result)
        if result["result"] == "failed":
            failures += 1
            print(f"FAILED: {video.name}: {result['error']}")
            _cleanup_in_process_vram()
            if not args.continue_on_error:
                break
        else:
            print(f"SUCCESS: {video.name}")
            _cleanup_in_process_vram()

    report_path = _write_summary_report(summary)
    print("\nSummary report:", report_path)

    if failures > 0:
        print(f"Completed with failures: {failures}")
        return 1

    print("Completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
