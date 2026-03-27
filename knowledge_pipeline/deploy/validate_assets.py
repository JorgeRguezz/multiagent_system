from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(os.environ.get("KNOWLEDGE_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()


def _load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _check_entry(root: Path, entry: dict) -> tuple[bool, str]:
    path = root / entry["path"]
    expected_type = entry["type"]
    if expected_type == "file":
        ok = path.is_file()
    elif expected_type == "dir":
        ok = path.is_dir()
    else:
        ok = path.exists()
    return ok, f"{entry['path']} ({expected_type})"


def main() -> int:
    root = _project_root()
    manifest_path = Path(__file__).with_name("assets_manifest.json")
    manifest = _load_manifest(manifest_path)

    print(f"[assets] project_root={root}")
    failures = []

    for entry in manifest.get("required", []):
        ok, label = _check_entry(root, entry)
        if ok:
            print(f"[assets] OK   required: {label}")
        else:
            print(f"[assets] FAIL required: {label}")
            failures.append(label)

    for entry in manifest.get("recommended", []):
        ok, label = _check_entry(root, entry)
        status = "OK   " if ok else "WARN "
        print(f"[assets] {status}recommended: {label}")

    if failures:
        print("[assets] Missing required assets:")
        for label in failures:
            print(f" - {label}")
        return 1

    print("[assets] Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
