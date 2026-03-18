# Knowledge Sanitization Pipeline

This module provides two intermediate sanitization phases:

1. `pre_build`: sanitizes extraction artifacts from `knowledge_extraction/cache/extracted_data` into `knowledge_sanitization/cache/sanitized_extracted_data`.
2. `post_build`: sanitizes build caches from `knowledge_build_cache_*` into `knowledge_sanitization/cache/sanitized_build_cache_*`, sanitizes the global graph, and rebuilds vector DB files.

## Run

```bash
python3 -m knowledge_sanitization.run_sanitization_queue --stage both
```

Or run stages separately:

```bash
python3 -m knowledge_sanitization.pre_build
python3 -m knowledge_sanitization.post_build
```

Single video:

```bash
python3 -m knowledge_sanitization.pre_build --video <video_name>
python3 -m knowledge_sanitization.post_build --video <video_name>
```

## Build Pipeline Path Change

`knowledge_build` now reads input strictly from:

- `knowledge_sanitization/cache/sanitized_extracted_data`

Default build queue:

```bash
python3 -m knowledge_build.run_build_queue
```
