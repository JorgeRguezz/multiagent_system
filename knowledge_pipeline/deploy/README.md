# Deployment Bundle

This folder contains a reproducible dependency bundle for the core pipeline modules:

- `requirements-smolvlm.txt` -> runtime used by queue + extraction/build/sanitization/inference entrypoints
- `requirements-internvl.txt` -> VLM/ASR server environment
- `requirements-gpt.txt` -> segment summarization server environment
- `setup_envs.sh` -> recreate and install all three environments
- `SMOKE_TESTS.md` -> post-install validation commands

## Usage

```bash
bash pipeline/deploy/setup_envs.sh
```

Then follow `pipeline/deploy/SMOKE_TESTS.md`.
