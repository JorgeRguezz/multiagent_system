"""
Single-video end-to-end pipeline runner:
extraction -> pre-build sanitization -> build -> post-build sanitization

Set MP4_PATH below and run:
    python3 playground/test_single_video_pipeline.py
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# =========================
# USER CONFIG
# =========================
MP4_PATH = "/home/gatv-projects/Desktop/project/downloads/queue/How_to_Play_Like_a_PYKE_MAIN_-_ULTIMATE_PYKE_GUIDE.mp4"
CLEANUP_TEMP_BUILD_INPUT = True


PROJECT_ROOT = Path("/home/gatv-projects/Desktop/project")
SANITIZED_EXTRACTION_ROOT = PROJECT_ROOT / "knowledge_sanitization" / "cache" / "sanitized_extracted_data"


def _run(cmd: list[str], cwd: Path = PROJECT_ROOT) -> None:
    print("\n$", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def _validate_paths(mp4_path: Path) -> None:
    if not mp4_path.exists():
        raise FileNotFoundError(f"MP4 path does not exist: {mp4_path}")
    if mp4_path.suffix.lower() != ".mp4":
        raise ValueError(f"Expected an .mp4 file, got: {mp4_path}")


def _copy_sanitized_video_to_temp(video_basename: str) -> Path:
    src_video_dir = SANITIZED_EXTRACTION_ROOT / video_basename
    if not src_video_dir.exists():
        raise FileNotFoundError(f"Sanitized extraction folder not found: {src_video_dir}")

    tmp_root = Path(tempfile.mkdtemp(prefix=f"kg_single_build_{video_basename}_"))
    dst_root = tmp_root / "sanitized_extracted_data"
    dst_root.mkdir(parents=True, exist_ok=True)
    dst_video_dir = dst_root / video_basename
    shutil.copytree(src_video_dir, dst_video_dir)

    print(f"\nTemporary isolated build input: {tmp_root}")
    return tmp_root


async def _run_build_for_video(isolated_extraction_root: Path, target_basename: str) -> Path:
    # Import here to keep startup errors localized to build stage.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from knowledge_build.builder import KnowledgeBuilder  # noqa: WPS433

    builder = KnowledgeBuilder(extraction_dir=str(isolated_extraction_root))
    selected = Path(builder.artifact_dir).name
    if selected != target_basename:
        raise RuntimeError(
            f"Build selected unexpected artifact '{selected}' (expected '{target_basename}')."
        )

    print(f"\n[build] artifact: {selected}")
    print(f"[build] output:   {builder.working_dir}")
    await builder.build()
    return Path(builder.working_dir)


def main() -> int:
    mp4_path = Path(MP4_PATH).expanduser().resolve()
    _validate_paths(mp4_path)

    video_basename = mp4_path.stem

    print("=" * 80)
    print("Single-video pipeline")
    print(f"Video: {mp4_path}")
    print(f"Basename: {video_basename}")
    print("=" * 80)

    # 1) Extraction
    _run([
        sys.executable,
        "-m",
        "knowledge_extraction.extractor",
        "--video-path",
        str(mp4_path),
    ])

    # 2) Pre-build sanitization
    _run([
        sys.executable,
        "-m",
        "knowledge_sanitization.pre_build",
        "--video",
        video_basename,
    ])

    # 3) Build (isolated input so only this video is selected)
    tmp_build_input_root = _copy_sanitized_video_to_temp(video_basename)
    try:
        build_output_dir = asyncio.run(_run_build_for_video(tmp_build_input_root, video_basename))
    finally:
        if CLEANUP_TEMP_BUILD_INPUT:
            shutil.rmtree(tmp_build_input_root, ignore_errors=True)
            print(f"\nRemoved temp build input: {tmp_build_input_root}")

    # 4) Post-build sanitization
    _run([
        sys.executable,
        "-m",
        "knowledge_sanitization.post_build",
        "--video",
        video_basename,
    ])

    print("\n" + "=" * 80)
    print("Pipeline completed successfully")
    print(f"Build cache: /home/gatv-projects/Desktop/project/knowledge_build_cache_{video_basename}")
    print(
        "Sanitized build cache: "
        f"/home/gatv-projects/Desktop/project/knowledge_sanitization/cache/sanitized_build_cache_{video_basename}"
    )
    print(f"Build output seen by runner: {build_output_dir}")
    print(
        "Reports:\n"
        f" - /home/gatv-projects/Desktop/project/knowledge_sanitization/cache/reports/pre_build/{video_basename}.json\n"
        f" - /home/gatv-projects/Desktop/project/knowledge_sanitization/cache/reports/post_build/{video_basename}.json"
    )
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
