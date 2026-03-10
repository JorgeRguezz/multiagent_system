#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VENV_SMOLVLM="${PROJECT_ROOT}/venv_smolvlm"
VENV_INTERNVL="${PROJECT_ROOT}/venv_internVL"
VENV_GPT="${PROJECT_ROOT}/venv_gpt"

REQ_SMOLVLM="${SCRIPT_DIR}/requirements-smolvlm.txt"
REQ_INTERNVL="${SCRIPT_DIR}/requirements-internvl.txt"
REQ_GPT="${SCRIPT_DIR}/requirements-gpt.txt"

PYTHON_BIN="${PYTHON_BIN:-python3}"

log() { echo "[setup] $*"; }

check_cmd() {
  command -v "$1" >/dev/null 2>&1
}

create_or_reuse_venv() {
  local venv_path="$1"
  if [[ ! -d "${venv_path}" ]]; then
    log "Creating venv: ${venv_path}"
    "${PYTHON_BIN}" -m venv "${venv_path}"
  else
    log "Reusing existing venv: ${venv_path}"
  fi
}

install_requirements() {
  local venv_path="$1"
  local req_file="$2"
  log "Installing requirements from ${req_file} into ${venv_path}"
  "${venv_path}/bin/pip" install --upgrade pip setuptools wheel
  # Build prerequisites needed by packages like pycocotools.
  "${venv_path}/bin/pip" install --upgrade "Cython>=0.29" "numpy<2.0" wheel
  "${venv_path}/bin/pip" install -r "${req_file}"
}

validate_env() {
  local venv_path="$1"
  local env_name="$2"
  log "Validating ${env_name} imports"
  "${venv_path}/bin/python" - <<'PY'
import importlib
mods = ["numpy", "torch", "networkx", "nano_vectordb", "sentence_transformers"]
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as exc:
        print(f"WARN import failed: {m}: {exc}")
print("validation-complete")
PY
}

log "Project root: ${PROJECT_ROOT}"
log "Using python: ${PYTHON_BIN}"

if ! check_cmd "${PYTHON_BIN}"; then
  echo "ERROR: ${PYTHON_BIN} not found" >&2
  exit 1
fi

if check_cmd nvidia-smi; then
  log "nvidia-smi detected"
else
  log "WARNING: nvidia-smi not found. GPU runtime may fail."
fi

if check_cmd ffmpeg; then
  log "ffmpeg detected"
else
  log "WARNING: ffmpeg not found. video split/extraction will fail."
fi

create_or_reuse_venv "${VENV_SMOLVLM}"
create_or_reuse_venv "${VENV_INTERNVL}"
create_or_reuse_venv "${VENV_GPT}"

install_requirements "${VENV_SMOLVLM}" "${REQ_SMOLVLM}"
install_requirements "${VENV_INTERNVL}" "${REQ_INTERNVL}"
install_requirements "${VENV_GPT}" "${REQ_GPT}"

validate_env "${VENV_SMOLVLM}" "venv_smolvlm"
validate_env "${VENV_INTERNVL}" "venv_internVL"
validate_env "${VENV_GPT}" "venv_gpt"

log "Done. Next: run smoke checks in pipeline/deploy/SMOKE_TESTS.md"
