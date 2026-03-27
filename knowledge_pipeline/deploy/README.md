# Deployment Bundle

This folder contains a reproducible dependency bundle for the core pipeline modules:

- `requirements-smolvlm.txt` -> runtime used by queue + extraction/build/sanitization/inference entrypoints
- `requirements-internvl.txt` -> VLM/ASR server environment
- `requirements-gpt.txt` -> segment summarization server environment
- `setup_envs.sh` -> recreate and install all three environments
- `.env.example` -> environment variable template for path overrides and caches
- `download_models.sh` -> prefetch the pinned runtime models before first real run
- `assets_manifest.json` -> required local files/directories that must exist in the repo
- `validate_assets.py` -> checks the asset manifest against the current checkout
- `SMOKE_TESTS.md` -> post-install validation commands

## Usage

```bash
cp knowledge_pipeline/deploy/.env.example .env
set -a
source .env
set +a

bash knowledge_pipeline/deploy/setup_envs.sh
bash knowledge_pipeline/deploy/download_models.sh
python3 knowledge_pipeline/deploy/validate_assets.py
```

If you want the optional Qwen VLM weights downloaded too:

```bash
DOWNLOAD_OPTIONAL_QWEN=1 bash knowledge_pipeline/deploy/download_models.sh
```

Then follow `knowledge_pipeline/deploy/SMOKE_TESTS.md`.
