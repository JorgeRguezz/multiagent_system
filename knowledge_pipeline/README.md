# Full Knowledge Pipeline (Extraction -> Sanitization -> Build -> Inference)

This document explains the full system implemented across:

- `knowledge_extraction/`
- `knowledge_sanitization/`
- `knowledge_build/`
- `knowledge_inference/`
- orchestrated by `pipeline/run_full_queue.py`

It is written for readers with no prior project context.

---

## 1. What This System Does

This pipeline turns raw gameplay videos into a sanitized, queryable knowledge base.

At a high level:

1. **Extract** structured multimodal data from each video (frames, transcripts, segment summaries).
2. **Sanitize (pre-build)** extraction outputs to remove contamination and normalize fields.
3. **Build** text chunks, vector indexes, and a chunk-entity knowledge graph.
4. **Sanitize (post-build)** build outputs and rebuild retrieval indexes from sanitized artifacts.
5. **Serve inference** over sanitized caches only (retrieval + reranking + grounded answer generation + verification).

Core design choice: **sanitized artifacts are the source of truth** for retrieval and QA.

---

## 2. End-to-End Orchestration

Main orchestrator: `pipeline/run_full_queue.py`

For each video in `downloads/queue/`:

1. Run extraction (`knowledge_extraction.extractor`)
2. Run pre-build sanitization (`knowledge_sanitization.pre_build`)
3. Run build on an isolated temp copy of sanitized extraction data (`knowledge_build.builder.KnowledgeBuilder`)
4. Run post-build sanitization (`knowledge_sanitization.post_build`)

### Idempotency / Skip Logic

`run_full_queue.py` skips videos already "fully processed" unless `--force` is provided.

A video is considered fully processed only if all expected files exist in all stages:

- extraction cache (`knowledge_extraction/cache/extracted_data/<video>/...`)
- sanitized extraction cache (`knowledge_sanitization/cache/sanitized_extracted_data/<video>/...`)
- build cache (`knowledge_build_cache_<video>/...`)
- sanitized build cache (`knowledge_sanitization/cache/sanitized_build_cache_<video>/...`)

### Reporting

A summary JSON report is written to:

- `knowledge_sanitization/cache/reports/full_pipeline_queue_<timestamp>.json`

---

## 3. Stage A: Knowledge Extraction (`knowledge_extraction/`)

Entry point for one video:

- `python -m knowledge_extraction.extractor --video-path <path>`

Queue runner:

- `python -m knowledge_extraction.run_extraction_queue`

### A.1 Extraction workflow inside `extractor.py`

1. **Workspace prep**
- Clears transient cache for the selected video.
- Keeps persistent extracted outputs under `knowledge_extraction/cache/extracted_data/`.

2. **Video segmentation** (`split_video` from `knowledge_build._videoutil.split`)
- Splits video into ~30s segments (configurable).
- Extracts segment audio into cache.
- Samples fixed number of frame timestamps per segment.

3. **Entity extraction phase** (MCP `entity_server` workers)
- Pre-extracts frame images to avoid contention on video reading.
- For each frame, crops configured HUD regions and runs SAM3 + DINOv2 matching.
- Produces per-frame inferred entities:
  - `main_champ`
  - `partners`

4. **ASR + VLM phase** (MCP `vlm_asr_server`)
- ASR: transcribes each segment audio (Whisper).
- VLM: generates frame descriptions conditioned on champion/partners/transcript and previous frame context.
- Produces per-frame multimodal records.

5. **Segment summarization phase** (MCP `segment_summarization_server`)
- Summarizes merged per-frame captions per segment using GPT-OSS (GGUF via llama-cpp).
- Combines with transcript into segment-level content.

6. **Persist extraction artifacts**
- `kv_store_video_segments.json`
- `kv_store_video_frames.json`
- `kv_store_video_path.json`

Output location:

- `knowledge_extraction/cache/extracted_data/<video_name>/`

### A.2 Extraction output schema (conceptual)

- `video_segments`: per segment
  - `time` (e.g. `start-end`)
  - `content` (caption summary + transcript)
  - `transcript`
  - `frame_times`
- `video_frames`: per sampled frame
  - frame metadata + normalized frame-level text (`vlm_output`, transcript)
  - entity cues (`main_champ`, `partners`)
- `video_path`: source video absolute path

---

## 4. Stage B: Pre-Build Sanitization (`knowledge_sanitization/pre_build.py`)

Entry point:

- `python -m knowledge_sanitization.pre_build [--video <video_name>]`

Queue runner:

- `python -m knowledge_sanitization.run_sanitization_queue --stage pre`

### B.1 Purpose

Sanitize extraction outputs before graph/vector construction.

### B.2 Main operations

- Validates required extraction files exist.
- Cleans contaminated text using:
  - meta tag removal (`<|...|>` forms)
  - blocked meta patterns (`spec/blocked_meta_patterns.json`)
  - control char stripping + whitespace normalization
- Validates segment time format and bounds frame times into segment interval.
- Drops invalid or empty records after cleaning.
- Normalizes champion/entity-like names using alias maps:
  - `spec/alias_champions.json`
  - `spec/alias_items.json`
  - `spec/alias_objectives.json`

### B.3 Outputs

Sanitized extraction files:

- `knowledge_sanitization/cache/sanitized_extracted_data/<video_name>/...`

Reports + quarantine:

- reports: `knowledge_sanitization/cache/reports/pre_build/<video_name>.json`
- quarantine jsonl: `knowledge_sanitization/cache/quarantine/pre_build/<video_name>/...`

Status per video is `pass`, `warn`, or `fail`.

---

## 5. Stage C: Knowledge Build (`knowledge_build/`)

Single build entry:

- `python -m knowledge_build.builder`

Queue runner:

- `python -m knowledge_build.run_build_queue`

Build input root defaults to:

- `knowledge_sanitization/cache/sanitized_extracted_data/`

### C.1 Folder selection model

`KnowledgeBuilder` auto-selects the next unbuilt sanitized extraction folder.

A folder is "built" if a matching `knowledge_build_cache_<video_name>` already exists.

### C.2 Build steps

1. Load sanitized extraction artifacts into JSON KV stores.
2. Chunk segment contents (`chunking_by_video_segments`) into chunk records.
3. Embed chunks into `vdb_chunks.json` (dense retrieval index).
4. Extract entities + relationships from chunks using GPT-OSS prompt pipeline:
- base extraction pass
- optional unified glean pass(es)
5. Merge/upsert nodes and edges into graph storage.
6. Embed entities into `vdb_entities.json`.
7. Persist graph and KV artifacts to per-video build cache.
8. Run conservative graph post-cleaning (`unify_entities_conservative`).
9. Merge cleaned per-video graph into global graph (`knowledge_build_cache_global/graph_AetherNexus.graphml`) and update manifest.

### C.3 Build outputs

Per-video output:

- `knowledge_build_cache_<video_name>/`
  - `kv_store_text_chunks.json`
  - `kv_store_video_segments.json`
  - `kv_store_video_frames.json`
  - `kv_store_video_path.json`
  - `vdb_chunks.json`
  - `vdb_entities.json`
  - `graph_chunk_entity_relation.graphml`
  - `graph_chunk_entity_relation_clean.graphml`
  - optional `kv_store_llm_response_cache.json`

Global output:

- `knowledge_build_cache_global/graph_AetherNexus.graphml`
- `knowledge_build_cache_global/aether_manifest.json`

---

## 6. Stage D: Post-Build Sanitization (`knowledge_sanitization/post_build.py`)

Entry point:

- `python -m knowledge_sanitization.post_build [--video <video_name>] [--keep-llm-cache]`

Queue runner:

- `python -m knowledge_sanitization.run_sanitization_queue --stage post`

### D.1 Purpose

Sanitize build artifacts and produce retrieval-ready sanitized caches used by inference.

### D.2 Main operations

- Sanitizes text chunks, segments, frames again (post-build contract enforcement).
- Cleans and normalizes graph nodes/edges:
  - dequote/unescape IDs
  - alias normalization
  - blocked placeholder filtering
  - entity type normalization to allowed set
  - source_id canonicalization against valid chunk IDs
  - edge weight normalization
  - duplicate edge merge policy

### D.3 Retrieval index policy (Chosen Option)

This system **rebuilds both vector indexes from sanitized artifacts**:

- Rebuild `vdb_chunks.json` from sanitized chunk text.
- Rebuild `vdb_entities.json` from sanitized graph entities.

Why this is kept:

- Maintains strict alignment between sanitized data and retrieval indices.
- Matches inference loader assumptions (sanitized stores expect these files).
- Avoids stale embeddings after post-build entity/chunk filtering and merges.

### D.4 Outputs

Per-video sanitized build cache:

- `knowledge_sanitization/cache/sanitized_build_cache_<video_name>/...`

Global sanitized graph:

- `knowledge_sanitization/cache/sanitized_global/graph_AetherNexus.graphml`
- `knowledge_sanitization/cache/sanitized_global/aether_manifest.json`

Reports + quarantine:

- reports: `knowledge_sanitization/cache/reports/post_build/<video_name>.json`
- quarantine: `knowledge_sanitization/cache/quarantine/post_build/<video_name>/...`

---

## 7. Stage E: Inference (`knowledge_inference/`)

Primary runtime path for QA over built knowledge.

CLI:

- `python -m knowledge_inference.cli --query "..." [--debug]`

### E.1 Data loading contract

Inference loads **sanitized** caches only:

- per-video: `knowledge_sanitization/cache/sanitized_build_cache_*`
- global graph: `knowledge_sanitization/cache/sanitized_global/graph_AetherNexus.graphml`

Required per-video files include:

- `kv_store_text_chunks.json`
- `kv_store_video_segments.json`
- `kv_store_video_frames.json`
- `vdb_chunks.json`
- `vdb_entities.json`
- `graph_chunk_entity_relation_clean.graphml`

### E.2 Query flow (`InferenceService`)

1. Analyze query intent (`query_analyzer.py`)
2. Retrieve candidates from multiple branches (`retrievers.py`):
- dense chunk retrieval
- entity->graph retrieval
- global graph lexical retrieval
- visual support retrieval (when visual-detail intent)
3. Deduplicate + rerank (`reranker.py`) with weighted scoring:
- semantic, entity, graph, diversity
4. Build token-budgeted evidence context (`context_builder.py`)
5. Generate answer with grounded prompt (`generator.py`)
6. Verify claims against evidence (`verifier.py`)
7. Produce final answer + confidence + optional debug payload

### E.3 Legacy note

`knowledge_build/_op.py` contains older query helpers (`videorag_query*`), but the active QA path is `knowledge_inference/*`.

---

## 8. Environment and Runtime Prerequisites

This pipeline is GPU-heavy and multi-environment.

### 8.1 Python environments expected by extraction stage

`knowledge_extraction/config.py` references specific interpreters:

- `venv_smolvlm/bin/python3` (entity server)
- `venv_internVL/bin/python3` (VLM + ASR server)
- `venv_gpt/bin/python3` (segment summarization server)

Main pipeline/orchestrator process also needs dependencies from build/sanitization/inference modules.

### 8.2 Major runtime dependencies (non-exhaustive)

- `torch`, `torchvision`
- `moviepy`, `Pillow`, `tqdm`
- `transformers`, `whisper`
- `mcp` client/server interfaces
- `networkx`
- `nano_vectordb`
- `sentence_transformers`
- `llama_cpp`, `huggingface_hub`
- optional/advanced: `vllm`, `imagebind`, `graspologic`

### 8.3 Model/data artifacts expected on disk or via HF

- champion matching DB (`knowledge_extraction/image_matching/*.nvdb`)
- SAM3 + DINOv2 dependencies for entity server
- InternVL/Qwen/Whisper models for VLM-ASR server
- GPT-OSS GGUF model (`unsloth/gpt-oss-20b-GGUF`) for summarization and KG extraction/generation
- embedding model (`all-MiniLM-L6-v2`) for vectorization

### 8.4 Hardware expectations

- CUDA GPU strongly recommended/required for practical throughput.
- The pipeline explicitly unloads models between stages in places to reduce VRAM pressure.

---

## 9. Operational Commands

### Full pipeline

```bash
python -m pipeline.run_full_queue
python -m pipeline.run_full_queue --dry-run
python -m pipeline.run_full_queue --force
python -m pipeline.run_full_queue --video <video_basename>
python -m pipeline.run_full_queue --continue-on-error
```

### Individual stage queues

```bash
python -m knowledge_extraction.run_extraction_queue
python -m knowledge_sanitization.run_sanitization_queue --stage pre
python -m knowledge_build.run_build_queue
python -m knowledge_sanitization.run_sanitization_queue --stage post
```

### Inference

```bash
python -m knowledge_inference.cli --query "What happened around the first dragon fight?" --debug
```

---

## 10. Key Design Contracts

1. **Sanitized-only inference**: inference refuses non-sanitized paths and reads only sanitized build/global caches.
2. **Post-build rebuild of VDBs**: retrieval indices are regenerated from sanitized artifacts to preserve alignment.
3. **Per-video + global graph**: local evidence is combined with global graph support.
4. **Grounded answering**: generator is constrained by provided evidence context and post-verified.
5. **Queue safety**: orchestrator can skip already-processed videos and emits structured reports.

---

## 11. Failure Modes to Watch

- Missing model/runtime dependencies in the currently active environment.
- Missing required files in any stage cache folder.
- GPU memory pressure during extraction/model loading.
- Sanitization dropping too many chunks/entities (can produce low-confidence inference due to sparse evidence).
- Global graph load issues if sanitized global outputs were not produced.

---

## 12. What to Read First in Code

If onboarding quickly, read in this order:

1. `pipeline/run_full_queue.py`
2. `knowledge_extraction/extractor.py`
3. `knowledge_sanitization/pre_build.py`
4. `knowledge_build/builder.py`
5. `knowledge_sanitization/post_build.py`
6. `knowledge_inference/service.py`

Then dive into:

- retrieval/rerank (`knowledge_inference/retrievers.py`, `knowledge_inference/reranker.py`)
- extraction helpers (`knowledge_extraction/entity_server.py`, `knowledge_extraction/vlm_asr_server.py`)
- KG extraction logic (`knowledge_build/_op.py`)
