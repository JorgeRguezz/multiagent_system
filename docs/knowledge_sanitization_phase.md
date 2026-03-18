# Knowledge Sanitization Phase

## Purpose of the module

`knowledge_sanitization/` is the normalization and contamination-removal phase that sits between:

- `knowledge_extraction` and `knowledge_build`, and
- `knowledge_build` and downstream inference/evaluation.

It is explicitly split into two stages:

1. `pre_build`: sanitize extraction artifacts before graph construction.
2. `post_build`: sanitize build outputs after graph construction and rebuild the retrieval artifacts from the sanitized result.

This module therefore serves two distinct purposes:

- it cleans noisy multimodal extraction outputs before they enter graph construction,
- it cleans graph/build artifacts after graph construction so downstream retrieval operates over a stricter, more normalized representation.

## Module structure

The implemented entry points are:

- [`knowledge_sanitization/pre_build.py`](/home/gatv-projects/Desktop/project/knowledge_sanitization/pre_build.py)
- [`knowledge_sanitization/post_build.py`](/home/gatv-projects/Desktop/project/knowledge_sanitization/post_build.py)
- [`knowledge_sanitization/run_sanitization_queue.py`](/home/gatv-projects/Desktop/project/knowledge_sanitization/run_sanitization_queue.py)

Supporting files:

- [`knowledge_sanitization/config.py`](/home/gatv-projects/Desktop/project/knowledge_sanitization/config.py)
- [`knowledge_sanitization/utils.py`](/home/gatv-projects/Desktop/project/knowledge_sanitization/utils.py)
- spec files under [`knowledge_sanitization/spec`](/home/gatv-projects/Desktop/project/knowledge_sanitization/spec)

The phase also writes reports, quarantine files, and sanitized copies of both extraction and build artifacts under:

- [`knowledge_sanitization/cache`](/home/gatv-projects/Desktop/project/knowledge_sanitization/cache)

## High-level role in the full pipeline

The `README` for this module states the intended pipeline order clearly:

- `pre_build` reads from `knowledge_extraction/cache/extracted_data`
- `knowledge_build` then reads from `knowledge_sanitization/cache/sanitized_extracted_data`
- `post_build` reads from `knowledge_build_cache_*`
- `post_build` writes sanitized build caches and a sanitized global graph

So `knowledge_sanitization` is not a single filter pass. It is a pair of staged contracts:

- one contract before graph creation,
- another contract after graph creation.

## Configuration layer

[`knowledge_sanitization/config.py`](/home/gatv-projects/Desktop/project/knowledge_sanitization/config.py) defines the filesystem topology and validation requirements.

Important paths:

- `PROJECT_ROOT`
- `SANITIZATION_ROOT = knowledge_sanitization/cache`
- `EXTRACTION_CACHE_ROOT = knowledge_extraction/cache`
- `EXTRACTED_DATA_ROOT = knowledge_extraction/cache/extracted_data`
- `SANITIZED_EXTRACTED_DATA_ROOT = knowledge_sanitization/cache/sanitized_extracted_data`
- `SANITIZED_GLOBAL_ROOT = knowledge_sanitization/cache/sanitized_global`
- `REPORT_PRE_ROOT = knowledge_sanitization/cache/reports/pre_build`
- `REPORT_POST_ROOT = knowledge_sanitization/cache/reports/post_build`
- `QUAR_PRE_ROOT = knowledge_sanitization/cache/quarantine/pre_build`
- `QUAR_POST_ROOT = knowledge_sanitization/cache/quarantine/post_build`

Important validation constants:

- `ALLOWED_ENTITY_TYPES = {"PERSON", "GEO", "EVENT", "ORGANIZATION", "UNKNOWN"}`
- `REQUIRED_EXTRACTION_FILES`
- `REQUIRED_BUILD_FILES`

This file defines the formal expected inputs and outputs of sanitization.

## Spec-driven sanitization policy

The sanitization module does not hard-code all normalization rules in Python. It uses explicit JSON specs from [`knowledge_sanitization/spec`](/home/gatv-projects/Desktop/project/knowledge_sanitization/spec).

The important policy files are:

- [`alias_champions.json`](/home/gatv-projects/Desktop/project/knowledge_sanitization/spec/alias_champions.json)
- [`alias_items.json`](/home/gatv-projects/Desktop/project/knowledge_sanitization/spec/alias_items.json)
- [`alias_objectives.json`](/home/gatv-projects/Desktop/project/knowledge_sanitization/spec/alias_objectives.json)
- [`blocked_placeholders.json`](/home/gatv-projects/Desktop/project/knowledge_sanitization/spec/blocked_placeholders.json)
- [`blocked_meta_patterns.json`](/home/gatv-projects/Desktop/project/knowledge_sanitization/spec/blocked_meta_patterns.json)

### Alias maps

The alias specs map variant spellings or recognition errors to canonical names.

Examples from the current specs:

- `AHRI` includes aliases such as `ARI`, `HARRY`, `AHIRI`
- `SMOLDER` includes `SMULDER`, `SMOULDER`
- `AURELION SOL` includes `AURELION SOUL`
- `ZHONYA'S HOURGLASS` includes typographic and spacing variants

These alias maps are used by `normalize_name()` to canonicalize champion, item, and objective names.

### Blocked placeholders

`blocked_placeholders.json` includes strings that should never survive as graph/entity names, for example:

- `<ENTITY_NAME>`
- `<ENTITY_TYPE>`
- `<RELATIONSHIP>`
- prompt-channel fragments such as `<|analysis|>`

### Blocked meta patterns

`blocked_meta_patterns.json` contains regex-like strings intended to strip LLM contamination such as:

- `<\|start\|>`
- `<\|end\|>`
- `<\|channel\|>`
- `analysis to=assistant`
- `we need to`
- `let's produce`
- `reasoning:`

This is strong evidence that the sanitization layer is specifically designed to remove prompt leakage and reasoning-channel contamination from the earlier LLM phases.

## Shared utility layer

[`knowledge_sanitization/utils.py`](/home/gatv-projects/Desktop/project/knowledge_sanitization/utils.py) implements the reusable normalization operations used by both sanitization stages.

## 1. Basic filesystem and JSON helpers

The module provides:

- `ensure_dir(path)`
- `load_json(path, default)`
- `save_json(data, path)`
- `append_jsonl(path, obj)`

The use of JSONL for quarantine logs is important: quarantined records are appended incrementally rather than rewritten as one large JSON structure.

## 2. Text normalization functions

### `normalize_unicode`

Normalizes text with:

- `unicodedata.normalize("NFKC", text)`

This standardizes visually similar Unicode variants.

### `strip_diacritics`

Removes combining marks after NFKD decomposition. This is later used by fuzzy name normalization.

### `clean_text`

`clean_text(text, blocked_patterns=None)` is the central contamination-removal primitive.

It performs, in order:

1. non-string guard:
   - returns empty string and `{"non_string": 1}` if the input is not a string.
2. Unicode normalization (`NFKC`).
3. removal of `<|...|>` meta tags using `META_TAG_RE`.
4. removal of any regex patterns from `blocked_patterns`.
5. removal of control characters.
6. newline normalization:
   - `\r\n` and `\r` become `\n`.
7. whitespace normalization:
   - repeated spaces/tabs collapse,
   - indentation after newlines is removed,
   - 3 or more newlines become 2.
8. trim.
9. if the final text is exactly one of:
   - `analysis`
   - `final`
   - `output`
   then replace it with empty string.

The function also returns a `stats` dictionary with counters such as:

- `meta_tags_removed`
- `meta_patterns_removed`
- `control_chars_removed`
- `fully_removed`

These counters are accumulated into the stage reports as contamination statistics.

## 3. Structural validators and normalizers

### `parse_segment_time`

Parses segment `time` fields formatted as:

- `"start-end"`

It rejects invalid formats and rejects ranges where `end <= start`.

### `normalize_name`

This function canonicalizes a name using the merged alias maps.

It:

1. normalizes Unicode,
2. strips surrounding quotes,
3. uppercases the string,
4. builds a reverse alias table from canonical names and aliases,
5. checks direct match,
6. checks accent-stripped match,
7. falls back to `difflib.get_close_matches()` with cutoff `0.92`,
8. otherwise returns the uppercased raw form.

Non-string or empty inputs become:

- `"Unknown"`

This distinction matters because later code sometimes preserves `"Unknown"` from raw defaults and sometimes uppercases normalized outputs into `"UNKNOWN"`.

### `normalize_entity_type`

Normalizes entity types by splitting on `<SEP>` and returning the first member that belongs to `ALLOWED_ENTITY_TYPES`. Otherwise it returns:

- `"UNKNOWN"`

### `canonicalize_source_ids`

Accepts either a string or a list of source IDs, filters them against a valid chunk-ID set, and returns:

- a sorted unique `<SEP>`-joined string

This is the key provenance-pruning function used in `post_build`.

### `should_block_entity_name`

Blocks an entity name if:

- it is empty,
- it matches one of the blocked placeholders,
- it is an angle-bracket placeholder such as `<...>`,
- its length exceeds `120` characters.

This is used only in `post_build`, not in `pre_build`.

## Pre-build sanitization

The first stage is implemented in [`knowledge_sanitization/pre_build.py`](/home/gatv-projects/Desktop/project/knowledge_sanitization/pre_build.py).

Its purpose is to sanitize the raw extraction artifacts before `knowledge_build` consumes them.

## Input and output of `pre_build`

Input root:

- `knowledge_extraction/cache/extracted_data`

Output root:

- `knowledge_sanitization/cache/sanitized_extracted_data`

For each input video folder `<video_name>`, the output is:

- `knowledge_sanitization/cache/sanitized_extracted_data/<video_name>/`

with:

- `kv_store_video_segments.json`
- `kv_store_video_frames.json`
- `kv_store_video_path.json`

It also writes:

- per-video JSON report to `reports/pre_build/<video_name>.json`
- quarantine JSONL files under `quarantine/pre_build/<video_name>/`

## `pre_build` control flow

The main function:

1. ensures the sanitized extraction and report directories exist,
2. resolves either:
   - a single requested `--video`, or
   - all folders under `EXTRACTED_DATA_ROOT`,
3. runs `sanitize_video_folder()` for each folder,
4. prints one line per video:
   - `[pre-build] <video>: <status>`

## `sanitize_video_folder()` behavior

For each input folder:

1. derive `video_name` from the directory name,
2. create:
   - sanitized output dir,
   - quarantine dir,
3. load specs:
   - blocked meta patterns,
   - merged alias maps,
4. initialize a report dictionary,
5. check whether all required extraction files exist,
6. load segment, frame, and video-path JSON files,
7. produce sanitized copies of each.

## Segment sanitization rules in `pre_build`

The source map is:

- `segments.get(video_name, {})`

For each segment record:

1. copy the record into `rec`.
2. parse `rec["time"]` using `parse_segment_time()`.
3. if invalid:
   - append a JSONL quarantine record to `video_segments_invalid.jsonl`,
   - increment dropped count,
   - skip the segment.
4. sanitize:
   - `content`
   - `transcript`
   using `clean_text(..., blocked_meta)`.
5. add contamination counts from the `clean_text` stats into the report.
6. if both cleaned `content` and cleaned `transcript` are empty:
   - quarantine with reason `empty_after_clean`,
   - drop the segment.
7. sanitize `frame_times`:
   - coerce each entry to float if possible,
   - clamp each value into `[start, end]`,
   - sort the resulting list.
8. store the cleaned segment as:
   - normalized `"time"` string,
   - cleaned `"content"`,
   - cleaned `"transcript"`,
   - cleaned `"frame_times"`.

The set of surviving segment IDs becomes `valid_segments`.

## Frame sanitization rules in `pre_build`

The source frame map is:

- `frames.get(video_name, {})`

For each frame record:

1. require that `frame_key` contains `_`.
2. if not:
   - quarantine to `video_frames_invalid.jsonl` with reason `invalid_key`,
   - drop the frame.
3. determine `seg_idx` from:
   - `fr["segment_idx"]`, or
   - the segment prefix of `frame_key`
4. if `seg_idx` is not in `valid_segments`:
   - drop the frame.
5. sanitize:
   - `transcript`
   - `vlm_output`
   with `clean_text()`.
6. normalize `main_champ` using `normalize_name()`.
7. if the normalized main champion differs from the original uppercased form:
   - increment `report["normalizations"]["main_champ"]`
8. normalize `partners`:
   - non-list values become `[]`,
   - each partner is normalized through `normalize_name()`,
   - values equal to `"UNKNOWN"` are dropped,
   - duplicates are removed while preserving order.
9. save the cleaned frame record while preserving all other original fields.

Unlike later stages, `pre_build` does not delete frames with weak semantics. It mainly removes structural invalidity and textual contamination.

## Video path sanitization rules in `pre_build`

`pre_build` only preserves the video path if:

- the top-level key equals `video_name`
- the value is a string

Warnings are recorded if:

- the path is not absolute,
- the path does not exist on disk

If the key is missing or malformed:

- the path entry is dropped,
- the report records a warning,
- the report counts the path file as dropped.

## Pre-build report semantics

The `pre_build` report contains:

- `video`
- `status`
- `input_dir`
- `output_dir`
- per-file `in/out/dropped` counts
- `normalizations`
- `contamination_hits`
- `warnings`

Status logic:

- `fail` if required files are missing
- `warn` if warnings exist but hard failure did not occur
- otherwise `pass`

An actual example report for `The_COMPLETE_SMOLDER_STARTER_GUIDE_-_League_of_Legends` shows:

- 24 input segments, 24 output segments
- 120 input frames, 120 output frames
- 1 input path, 1 output path
- 5 contamination hits
- no warnings

So the reporting is intended as an auditable record of how much cleaning occurred per video.

## Post-build sanitization

The second stage is implemented in [`knowledge_sanitization/post_build.py`](/home/gatv-projects/Desktop/project/knowledge_sanitization/post_build.py).

Its purpose is broader than `pre_build`. It sanitizes already-built caches, sanitizes the graph itself, and rebuilds vector databases from the sanitized content.

## Input and output of `post_build`

Input directories:

- all `knowledge_build_cache_*` directories in the project root
- excluding `knowledge_build_cache_global`

Output directories:

- `knowledge_sanitization/cache/sanitized_build_cache_<video_name>/`

Global output:

- `knowledge_sanitization/cache/sanitized_global/graph_AetherNexus.graphml`
- `knowledge_sanitization/cache/sanitized_global/aether_manifest.json`

Reports:

- `knowledge_sanitization/cache/reports/post_build/<video_name>.json`

Quarantine:

- `knowledge_sanitization/cache/quarantine/post_build/<video_name>/`

## `post_build` control flow

The main function:

1. verifies required packages are importable:
   - `networkx`
   - `numpy`
   - `nano_vectordb`
2. resolves either:
   - a single requested `--video`, or
   - all build cache folders
3. runs `_sanitize_build_cache()` for each
4. runs `_sanitize_global_graph()`
5. returns non-zero only if one or more per-video sanitizations failed

## `_sanitize_build_cache()` behavior

This function sanitizes one `knowledge_build_cache_<video_name>` folder and writes a corresponding:

- `sanitized_build_cache_<video_name>`

folder.

It also accepts:

- `drop_llm_cache: bool = True`

meaning the LLM cache is removed by default unless `--keep-llm-cache` is passed.

## 1. Initial loading and report structure

`_sanitize_build_cache()` loads:

- `kv_store_text_chunks.json`
- `kv_store_video_segments.json`
- `kv_store_video_frames.json`
- `kv_store_video_path.json`

It also loads the same policy specs used earlier, plus:

- blocked placeholders

The report tracks:

- file in/out/dropped counts,
- entity in/out/merged/dropped counts,
- edge in/out/dropped counts,
- contamination hits,
- warnings.

## 2. Chunk sanitization in `post_build`

Chunk sanitization is stricter than in `pre_build`.

For each chunk:

1. sanitize `content` with `clean_text()`.
2. if content becomes empty:
   - drop the chunk,
   - quarantine to `text_chunks_invalid.jsonl` with reason `empty_after_clean`.
3. read `video_segment_id`.
4. require it to be a list.
5. filter the list so each segment reference:
   - is a string,
   - starts with `f"{video_name}_"`.
6. if no valid segment refs remain:
   - drop the chunk,
   - quarantine with reason `no_valid_segment_refs`.
7. store the cleaned chunk with:
   - recomputed approximate token count `max(1, len(content)//4)`,
   - original `chunk_order_index`,
   - filtered `video_segment_id`,
   - `sanitized: True`.

This means `post_build` turns sanitized status into explicit chunk metadata.

## 3. Segment and frame pruning by chunk reachability

After chunk sanitization, `post_build` derives:

- `valid_segment_keys`

from the surviving chunks' segment references.

This is a key design choice: segments and frames are not preserved independently. They are preserved only if they remain reachable from at least one sanitized text chunk.

### Segment rules

For each segment in the build cache:

- if its segment index is not in `valid_segment_keys`, it is silently excluded.
- otherwise `content` and `transcript` are cleaned with `clean_text()`.

### Frame rules

For each frame:

- determine its `seg_idx`,
- if the segment is not preserved in sanitized segments, the frame is excluded,
- otherwise sanitize:
  - `transcript`
  - `vlm_output`
- normalize:
  - `main_champ`
  - `partners`

`partners` in `post_build` are normalized into a sorted set, not order-preserving deduplication as in `pre_build`.

This means partner ordering semantics are weakened at the post-build stage.

## 4. Video path preservation in `post_build`

The path logic is simpler than in `pre_build`:

- if `video_name` exists in the path map, keep it;
- otherwise the sanitized path map is empty.

No warning is generated for non-absolute or missing paths here.

## 5. Graph sanitization in `post_build`

The graph stage is the most important part of `post_build`.

### Graph source selection

The function prefers:

- `graph_chunk_entity_relation_clean.graphml`

and falls back to:

- `graph_chunk_entity_relation.graphml`

If neither exists, it starts with an empty graph.

### Node sanitization rules

For every graph node:

1. dequote and HTML-unescape the node ID via `_dequote_node_id()`.
2. canonicalize the node ID using `normalize_name(alias_map)`.
3. if `should_block_entity_name()` returns true:
   - drop the node.
4. normalize `entity_type` using `normalize_entity_type(...)` and `ALLOWED_ENTITY_TYPES`.
5. clean the node `description` using `clean_text()`.
6. canonicalize `source_id` against the set of valid sanitized chunk IDs.
7. if no valid source IDs remain:
   - drop the node.

Then one of two things happens:

- if the normalized node name already exists in the new graph `H`, merge into the existing node,
- otherwise add a new node.

### Node merge semantics in `post_build`

If two sanitized nodes collapse to the same canonical ID:

- `source_id` values are unioned using `<SEP>` splitting and sorted rejoin,
- `description` values are unioned the same way,
- `entity_type` is upgraded from `UNKNOWN` if the incoming node has a concrete type.

This is a simpler and stricter merge than the one used in `knowledge_build.clean_kg`.

### Edge sanitization rules

For every original edge:

1. normalize source and target node IDs with the same alias logic.
2. if source and target collapse to the same node:
   - drop the edge.
3. if either endpoint did not survive node sanitization:
   - drop the edge.
4. clean the edge description with `clean_text()`.
5. canonicalize `source_id` against valid sanitized chunk IDs.
6. if no valid source IDs remain:
   - drop the edge.
7. parse weight as float, default `1.0`.
8. clamp weight into `[0.0, 10.0]`.
9. parse `order`, default `1`.

If an edge already exists in `H`:

- `weight` becomes the max of the old and new values,
- descriptions are unioned with `<SEP>`,
- source IDs are unioned with `<SEP>`,
- `order` becomes the minimum value.

Otherwise a new edge is inserted.

This means `post_build` normalizes graph structure and also imposes a stricter aggregation policy on duplicated relations.

## 6. Saving sanitized build artifacts

After graph sanitization, `post_build` writes:

- `kv_store_text_chunks.json`
- `kv_store_video_segments.json`
- `kv_store_video_frames.json`
- `kv_store_video_path.json`
- `graph_chunk_entity_relation.graphml`
- `graph_chunk_entity_relation_clean.graphml`

Both graph files are written from the same sanitized graph `H`. In other words, after `post_build`, the "raw" and "clean" graph filenames point to the same already-sanitized graph representation.

## 7. Optional LLM cache retention

By default, `post_build` drops:

- `kv_store_llm_response_cache.json`

If `--keep-llm-cache` is passed, it copies the original build cache file into the sanitized build cache.

This is a deliberate policy choice: the default sanitized cache is meant to preserve retrieval assets and graph content, not necessarily the full intermediate LLM trace.

## 8. Rebuilding vector databases

One of the defining behaviors of `post_build` is that it does not trust the old vector DB files after sanitization. It rebuilds them from the sanitized content.

### Chunk vector DB rebuild

The function:

1. gathers all sanitized chunk texts,
2. embeds them using `_embed_texts()` and `knowledge_build._llm.local_llm_config.embedding_func`,
3. creates a new `NanoVectorDB` at:
   - `vdb_chunks.json`
4. inserts records:
   - `{"__id__": chunk_id, "__vector__": vector}`
5. saves the DB.

### Entity vector DB rebuild

For every sanitized graph node:

1. create an entity ID of the form:
   - `ent-{abs(hash(name)) % (10**32):032d}` truncated to 36 chars
2. create the embedding text:
   - `node_name + " " + description`
3. embed all entity texts,
4. create a new `NanoVectorDB` at:
   - `vdb_entities.json`
5. insert records with:
   - `__id__`
   - `entity_name`
   - `__vector__`

Important detail: these entity IDs are not the same deterministic MD5 IDs used in `knowledge_build`. They are based on Python's `hash()`, which is typically process-randomized between interpreter sessions unless hash seeding is fixed. So post-build entity vector IDs are not guaranteed to be stable across runs.

## 9. Post-build status logic

At the end of `_sanitize_build_cache()`:

- if the sanitized graph has zero nodes, status becomes `fail`
- if sanitized text chunks are zero, status becomes `fail`
- else if warnings exist, status becomes `warn`
- else status remains `pass`

An example report for `The_COMPLETE_SMOLDER_STARTER_GUIDE_-_League_of_Legends` shows:

- 24 input chunks, 24 output chunks
- 117 input entities, 116 output entities
- 146 input edges, 133 output edges
- 24 rebuilt chunk vectors
- 116 rebuilt entity vectors
- no warnings

That is a good concrete illustration of the pruning effect of `post_build`.

## Global graph sanitization

The function `_sanitize_global_graph()` sanitizes the global graph separately from per-video caches.

Input:

- `knowledge_build_cache_global/graph_AetherNexus.graphml`
- `knowledge_build_cache_global/aether_manifest.json`

Output:

- `knowledge_sanitization/cache/sanitized_global/graph_AetherNexus.graphml`
- `knowledge_sanitization/cache/sanitized_global/aether_manifest.json`

### What this function actually does

This is intentionally lightweight compared to per-video `post_build`.

It:

1. loads the global graph,
2. dequotes and HTML-unescapes node IDs,
3. copies all node attributes unchanged,
4. dequotes and HTML-unescapes edge endpoints,
5. copies all edge attributes unchanged,
6. writes the resulting graph,
7. normalizes the manifest into:
   - `{"processed_videos": [...sorted unique strings...]}`

So the sanitized global graph is not fully re-canonicalized using alias maps or source-ID pruning. It is only lightly normalized.

## Queue runner behavior

[`knowledge_sanitization/run_sanitization_queue.py`](/home/gatv-projects/Desktop/project/knowledge_sanitization/run_sanitization_queue.py) is a simple orchestration wrapper.

It supports:

- `--stage pre`
- `--stage post`
- `--stage both`
- optional `--video`
- optional `--keep-llm-cache`

It runs each stage by launching the corresponding module via:

- `sys.executable -m knowledge_sanitization.pre_build`
- `sys.executable -m knowledge_sanitization.post_build`

This script does not add its own logic beyond sequential invocation and return-code aggregation.

## Audit trail and quarantine outputs

One of the strengths of this module is that it is designed as an auditable sanitization system, not just a silent transformer.

It preserves evidence of sanitization through:

- structured JSON reports,
- stage-specific quarantine folders,
- JSONL append-only invalid-record logs,
- contamination counters,
- per-file in/out/dropped statistics.

The quarantine mechanism is used for:

- invalid segment times,
- empty segments after cleaning,
- invalid frame keys,
- empty chunks after cleaning,
- chunks with no valid segment references.

This makes the sanitization phase empirically inspectable, which is useful for both debugging and research reporting.

## Important implementation characteristics

### The phase is deterministic and rule-based

Unlike extraction and build, sanitization does not call a generative model. It is based on explicit string, path, graph, and schema rules.

### Prompt contamination removal is a first-class concern

The presence of blocked meta-patterns such as:

- `<|start|>`
- `we need to`
- `reasoning:`

shows that sanitization is explicitly designed to strip traces of model reasoning/channel leakage from earlier stages.

### `pre_build` and `post_build` have different semantics

`pre_build` is mainly:

- text cleanup,
- structural validation,
- alias normalization on frame-level names.

`post_build` is mainly:

- chunk reachability pruning,
- graph canonicalization,
- placeholder removal,
- provenance repair,
- vector DB regeneration.

### The phase preserves provenance whenever possible

`post_build` does not keep arbitrary source IDs. It filters them through the sanitized chunk ID set. That means every surviving graph node or edge must still be traceable to a surviving sanitized chunk.

### The sanitized build cache is a new canonical retrieval state

Because `post_build` rebuilds both:

- `vdb_chunks.json`
- `vdb_entities.json`

from the sanitized content, the sanitized build cache is not just a cleaned copy. It becomes a new retrieval-ready artifact set.

## Concise conceptual summary

`knowledge_sanitization/` is the system's normalization and quality-control layer.

Its internal logic is:

1. remove prompt/meta contamination from text fields,
2. normalize names through alias maps,
3. validate structure and temporal fields,
4. quarantine invalid records,
5. generate per-video audit reports,
6. sanitize graph nodes and edges after build,
7. prune facts whose provenance no longer maps to valid chunks,
8. rebuild vector indexes from sanitized content,
9. emit sanitized local and global caches.

In the full pipeline, this module is what converts "whatever earlier stages produced" into the stricter artifact contracts that the later stages can rely on.
