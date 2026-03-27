# Smoke Tests (Core System)

Scope covered:
- `knowledge_extraction/`
- `knowledge_build/`
- `knowledge_sanitization/`
- `knowledge_inference/`
- `knowledge_pipeline/`

## 1) Install Environments

```bash
bash knowledge_pipeline/deploy/setup_envs.sh
```

## 2) Environment Configuration Check

Copy and review the environment template on a new machine:

```bash
cp knowledge_pipeline/deploy/.env.example .env
```

At minimum, confirm:
- `KNOWLEDGE_PROJECT_ROOT`
- `VENV_SMOLVLM_PYTHON`
- `VENV_VLM_ASR_PYTHON`
- `HF_HOME` if you want a custom model cache location

Then export it for the current shell:

```bash
set -a
source .env
set +a
```

## 3) Asset Validation

```bash
python3 knowledge_pipeline/deploy/validate_assets.py
```

Expected:
- required local directories and files found
- warnings only for optional paths if they are missing

## 4) Model Prefetch

```bash
bash knowledge_pipeline/deploy/download_models.sh
```

Optional Qwen weights:

```bash
DOWNLOAD_OPTIONAL_QWEN=1 bash knowledge_pipeline/deploy/download_models.sh
```

## 5) Basic Import / Runtime Checks

```bash
source venv_smolvlm/bin/activate
python -m compileall knowledge_extraction knowledge_build knowledge_sanitization knowledge_inference knowledge_pipeline
```

## 6) Pipeline Dry Run (no heavy processing)

```bash
source venv_smolvlm/bin/activate
python -m knowledge_pipeline.run_full_queue --dry-run
```

Expected:
- videos discovered from `downloads/queue`
- planned extraction/pre-build/build/post-build commands printed

## 7) Single Video End-to-End Queue Test

```bash
source venv_smolvlm/bin/activate
python -m knowledge_pipeline.run_full_queue --video "How_to_Play_Like_a_PYKE_MAIN_-_ULTIMATE_PYKE_GUIDE" --continue-on-error
```

## 8) Inference Smoke Test (after at least one sanitized build cache exists)

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
