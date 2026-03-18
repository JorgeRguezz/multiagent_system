# Smoke Tests (Core System)

Scope covered:
- `knowledge_extraction/`
- `knowledge_build/`
- `knowledge_sanitization/`
- `knowledge_inference/`
- `pipeline/`

## 1) Install Environments

```bash
bash pipeline/deploy/setup_envs.sh
```

## 2) Path Configuration Check (required on new machine)

These files contain absolute paths and must match the new machine:
- `knowledge_extraction/config.py`
- `knowledge_build/_llm.py`
- `pipeline/run_full_queue.py` (PROJECT_ROOT constant)

If your repo is at a different location, update those paths first.

## 3) Basic Import / Runtime Checks

```bash
source venv_smolvlm/bin/activate
python -m compileall knowledge_extraction knowledge_build knowledge_sanitization knowledge_inference pipeline
```

## 4) Pipeline Dry Run (no heavy processing)

```bash
source venv_smolvlm/bin/activate
python -m pipeline.run_full_queue --dry-run
```

Expected:
- videos discovered from `downloads/queue`
- planned extraction/pre-build/build/post-build commands printed

## 5) Single Video End-to-End Queue Test

```bash
source venv_smolvlm/bin/activate
python -m pipeline.run_full_queue --video "How_to_Play_Like_a_PYKE_MAIN_-_ULTIMATE_PYKE_GUIDE" --continue-on-error
```

## 6) Inference Smoke Test (after at least one sanitized build cache exists)

```bash
source venv_smolvlm/bin/activate
python -m knowledge_inference.cli --query "How does Pyke secure kills in this guide?" --debug
```

## Troubleshooting quick checks

```bash
nvidia-smi
ffmpeg -version
source venv_smolvlm/bin/activate && python - <<'PY'
import torch
print('cuda_available=', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu=', torch.cuda.get_device_name(0))
    print('bf16_supported=', torch.cuda.is_bf16_supported())
PY
```
