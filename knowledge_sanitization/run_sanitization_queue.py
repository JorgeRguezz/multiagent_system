"""Queue runner for knowledge sanitization stages."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = str(
    Path(
        os.environ.get("KNOWLEDGE_PROJECT_ROOT", Path(__file__).resolve().parents[1])
    ).resolve()
)


def _run(module: str, extra_args: list[str]) -> int:
    cmd = [sys.executable, "-m", module] + extra_args
    return subprocess.run(cmd, cwd=PROJECT_ROOT, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pre-build and/or post-build sanitization queues.")
    parser.add_argument("--stage", choices=["pre", "post", "both"], default="both")
    parser.add_argument("--video", default=None, help="Optional video folder name")
    parser.add_argument("--keep-llm-cache", action="store_true")
    args = parser.parse_args()

    rc = 0
    if args.stage in ("pre", "both"):
        cmd_args = []
        if args.video:
            cmd_args += ["--video", args.video]
        rc_pre = _run("knowledge_sanitization.pre_build", cmd_args)
        rc = rc or rc_pre

    if args.stage in ("post", "both"):
        cmd_args = []
        if args.video:
            cmd_args += ["--video", args.video]
        if args.keep_llm_cache:
            cmd_args += ["--keep-llm-cache"]
        rc_post = _run("knowledge_sanitization.post_build", cmd_args)
        rc = rc or rc_post

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
