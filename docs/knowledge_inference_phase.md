# Knowledge Inference Phase

## Purpose of the module

`knowledge_inference/` is the runtime question-answering layer of the system. Its job is to take a user query and answer it only from the sanitized knowledge artifacts produced by the earlier phases.

Unlike `knowledge_extraction/`, `knowledge_build/`, and `knowledge_sanitization/`, this module does not create new persistent knowledge assets. It is inference-only. It loads already-built sanitized caches, retrieves candidate evidence, reranks and compresses that evidence into a prompt context, generates an answer with a local language model, verifies the answer against the retrieved evidence, and returns both the final answer and the supporting evidence blocks.

At a high level, the inference phase performs seven steps:

1. Load sanitized per-video stores and the sanitized global graph.
2. Analyze the query with a deterministic heuristic intent detector.
3. Retrieve candidate chunks from several retrieval branches in parallel.
4. Merge, deduplicate, and rerank the retrieved chunks with fixed scoring weights.
5. Build a bounded evidence context from the top-ranked chunks.
6. Generate an answer using GPT-OSS through the shared `knowledge_build._llm` runtime.
7. Verify the answer claim-by-claim against the evidence and calibrate a final confidence score.

This module is therefore the serving-time retrieval-augmented generation layer that sits on top of the sanitized build outputs.

## Input contract

The module is deliberately constrained to sanitized artifacts only.

The package `README` and `store_loader.py` enforce two allowed data sources:

- per-video caches:
  - `knowledge_sanitization/cache/sanitized_build_cache_*`
- sanitized global graph:
  - `knowledge_sanitization/cache/sanitized_global/graph_AetherNexus.graphml`

Unsanitized build caches are not read.

This is not just a documentation claim. [`knowledge_inference/store_loader.py`](/home/gatv-projects/Desktop/project/knowledge_inference/store_loader.py) explicitly rejects paths that do not live under `knowledge_sanitization/cache` via `_assert_sanitized_path()`.

## Main entry points

The main runtime entry point is [`knowledge_inference/service.py`](/home/gatv-projects/Desktop/project/knowledge_inference/service.py), which defines `InferenceService`.

Operationally, the module exposes two normal entry points:

- interactive/single-query CLI:
  - `python3 -m knowledge_inference.cli --query "..."`
- offline evaluator:
  - `python3 -m knowledge_inference.eval --dataset /path/to/eval.json`

The CLI runs one query and prints the answer, confidence, top evidence, and optional debug metadata.

The evaluator runs many questions through the same service and writes a JSON report under:

- `knowledge_inference/reports/`

## File-level responsibilities

### `config.py`

[`knowledge_inference/config.py`](/home/gatv-projects/Desktop/project/knowledge_inference/config.py) defines the full serving-time policy surface.

Important path constants:

- `PROJECT_ROOT`
- `SANITIZED_CACHE_ROOT`
- `SANITIZED_BUILD_GLOB = "sanitized_build_cache_*"`
- `SANITIZED_GLOBAL_GRAPH`
- `SANITIZED_GLOBAL_MANIFEST`
- `VIDEO_METADATA_REGISTRY`

Important retrieval and ranking constants:

- `TOP_K_CHUNKS_DENSE = 30`
- `TOP_K_ENTITIES_DENSE = 20`
- `TOP_K_GRAPH_CHUNKS = 30`
- `FINAL_EVIDENCE_K = 14`
- `W_SEMANTIC = 0.50`
- `W_ENTITY = 0.20`
- `W_GRAPH = 0.20`
- `W_DIVERSITY = 0.10`

Important generation and verification constants:

- `MAX_CONTEXT_TOKENS = 10000`
- `MAX_ANSWER_TOKENS = 2500`
- `MIN_EVIDENCE_SCORE = 0.18`
- `MIN_SUPPORTED_CLAIMS_RATIO = 0.45`
- `GEN_TEMPERATURE = 0.1`
- `GEN_TOP_P = 1.0`
- `GEN_TOP_K = 0`
- `GEN_REPEAT_PENALTY = 1.12`
- `VERIFIER_TEMPERATURE = 0.0`
- `VERIFIER_TOP_P = 1.0`
- `VERIFIER_TOP_K = 0`
- `VERIFIER_REPEAT_PENALTY = 1.05`
- `VERIFIER_MAX_TOKENS = 1200`

Important retrieval policy constants:

- `DEFAULT_MAX_PER_VIDEO = 4`
- `CROSS_VIDEO_MAX_PER_VIDEO = 3`
- `ENABLE_INTENT_LLM_FALLBACK = False`

These constants are not passive metadata. They directly determine retrieval depth, answer length, evidence selection, diversity pressure, and the strictness of fallback behavior.

### `types.py`

[`knowledge_inference/types.py`](/home/gatv-projects/Desktop/project/knowledge_inference/types.py) defines the core in-memory data contracts used across the package.

#### `VideoStore`

Represents all loaded artifacts for one sanitized video cache:

- `video_name`
- `chunks_vdb`
- `entities_vdb`
- `chunks_kv`
- `segments_kv`
- `frames_kv`
- `graph`

This structure is the unit of retrieval. All retrieval branches operate over one or more `VideoStore` instances.

#### `QueryIntent`

Encodes heuristic query interpretation:

- `normalized_query`
- `is_cross_video`
- `is_temporal`
- `is_visual_detail`
- `entity_focus_terms`

#### `RetrievalHit`

Represents an intermediate candidate chunk before context construction:

- `chunk_id`
- `video_name`
- `source`
- `chunk_text`
- `segment_ids`
- `score_semantic`
- `score_entity`
- `score_graph`
- `final_score`

#### `EvidenceBlock`

Represents the final evidence units included in the prompt:

- `video_name`
- `time_span`
- `chunk_id`
- `source`
- `text`
- `final_score`

#### `AnswerResult`

Returned by `InferenceService.answer()`:

- `answer`
- `evidence`
- `context`
- `confidence`
- `debug`

### `store_loader.py`

[`knowledge_inference/store_loader.py`](/home/gatv-projects/Desktop/project/knowledge_inference/store_loader.py) is the ingestion layer for sanitized build artifacts.

It performs four responsibilities:

1. discover valid sanitized per-video directories,
2. validate that all reads are under `knowledge_sanitization/cache`,
3. load KV stores, vector stores, and graph files into a `VideoStore`,
4. perform consistency checks and warn about mismatches.

Important behaviors:

- `discover_sanitized_video_dirs()` scans `SANITIZED_CACHE_ROOT` using `SANITIZED_BUILD_GLOB`.
- `_extract_video_name()` strips the `sanitized_build_cache_` prefix from directory names.
- `_load_vdb()` reads the JSON file to determine `embedding_dim`, then constructs a `NanoVectorDB` pointing at that file.
- `load_video_store()` requires these files:
  - `kv_store_text_chunks.json`
  - `kv_store_video_segments.json`
  - `kv_store_video_frames.json`
  - `vdb_chunks.json`
  - `vdb_entities.json`
  - `graph_chunk_entity_relation_clean.graphml`
- `load_global_graph()` loads `graph_AetherNexus.graphml` from the sanitized global directory.

#### Store validation

`_validate_store()` does not reject imperfect stores, but it logs warnings for several integrity problems:

- a video with zero chunks,
- mismatch between `chunks_kv` size and `chunks_vdb` size,
- malformed chunk segment IDs,
- chunk references to segment indices missing from the segment KV,
- empty graph,
- large mismatch between graph node count and entity vector count.

So the loader is strict about path provenance and file existence, but soft about internal consistency: it warns rather than aborting.

### `query_analyzer.py`

[`knowledge_inference/query_analyzer.py`](/home/gatv-projects/Desktop/project/knowledge_inference/query_analyzer.py) implements deterministic heuristic-first query understanding.

There is no active LLM-based query analysis path in the current code. `ENABLE_INTENT_LLM_FALLBACK` exists in configuration, but no fallback is invoked by the runtime.

The analyzer performs the following operations:

1. `_normalize_query()` collapses repeated whitespace.
2. The normalized query is lowercased for pattern checks.
3. Three term lists are matched:
   - `_CROSS_VIDEO_TERMS`
   - `_TEMPORAL_TERMS`
   - `_VISUAL_TERMS`
4. Additional bonus rules are applied:
   - a timestamp like `12:34` or words such as `second`, `seconds`, `minute`, `minutes` increase temporal score,
   - if the query contains at least two entity terms and comparison language such as `compare`, `versus`, `vs`, or `difference`, cross-video score increases,
   - words such as `color`, `icon`, `ui`, `hud`, `left side`, `right side` increase visual score.
5. `_extract_entity_terms()` tokenizes the query, removes stopwords, normalizes several champion aliases, counts term frequency, and returns up to 12 most common tokens.

The alias map is small and explicit:

- `pike -> pyke`
- `ahry -> ahri`
- `smoulder -> smolder`
- `zaahen -> zaahen`

The output thresholds are binary:

- `is_cross_video = cross_score >= 1`
- `is_temporal = temporal_score >= 1`
- `is_visual_detail = visual_score >= 1`

So the query analyzer is intentionally simple, transparent, and deterministic. It does not estimate uncertainty, and it does not produce multi-class intents beyond these three booleans and the extracted entity terms.

### `retrievers.py`

[`knowledge_inference/retrievers.py`](/home/gatv-projects/Desktop/project/knowledge_inference/retrievers.py) implements all retrieval branches.

It uses the shared embedding function from `knowledge_build._llm.local_llm_config`, so the inference phase embeds queries with the same embedding stack used when the build phase created the vector stores.

#### Shared helpers

- `_split_source_ids(source_id)` splits `<SEP>`-joined provenance fields.
- `_tokenize(text)` extracts lowercase alphanumeric tokens.
- `_embed_query(query)` asynchronously embeds the query text.
- `_resolve_chunk_hit(...)` converts a raw chunk ID into a `RetrievalHit` by looking up the chunk KV, chunk content, and attached `video_segment_id` list.

If a chunk ID is not found in the KV store, `_resolve_chunk_hit()` returns `None`, so missing vector/graph references are silently dropped at this step.

#### Dense chunk retrieval

`retrieve_chunks_dense()`:

1. embeds the query once,
2. queries each video's `chunks_vdb` with `top_k = TOP_K_CHUNKS_DENSE`,
3. reads each result row's `__id__` and `__metrics__`,
4. converts matches into `RetrievalHit` objects with:
   - `source = "dense_chunk"`
   - `score_semantic = __metrics__`

This branch is the direct semantic chunk-retrieval path.

#### Entity-graph retrieval

`retrieve_entity_graph()` is the entity-centric retrieval branch. It combines dense entity retrieval with one-hop graph expansion.

For each video store, it:

1. embeds the query,
2. queries the per-video `entities_vdb` with `top_k = TOP_K_ENTITIES_DENSE`,
3. reads `entity_name` and `__metrics__` from each entity result,
4. finds matching graph nodes with `_graph_nodes_for_entity()`,
5. collects chunk IDs from:
   - the matched entity node's `source_id`,
   - each neighbor node's `source_id`,
   - each connecting edge's `source_id`,
6. assigns scores:
   - direct node chunks get `base_score`,
   - neighbor and edge chunks get `neighbor_boost`

The `neighbor_boost` formula is:

- `min(1.0, base_score + log1p(weight) * 0.05)`

where `weight` is read from the graph edge and defaults to `1.0`.

This means the branch does not perform multi-hop traversal. It expands exactly one graph hop away from the matched entity and softly boosts evidence attached to strongly weighted neighboring relations.

The returned hits are labeled:

- `source = "entity_graph"`
- `score_entity = graph_score`
- `score_graph = graph_score`

#### Global graph retrieval

`retrieve_global_graph()` performs lexical matching over the sanitized global graph rather than vector retrieval.

It:

1. tokenizes the query,
2. builds a `chunk_id -> video_name` index from all local stores,
3. scores each global-graph node with `_lexical_match_score(query_tokens, node_name, description)`,
4. keeps the top `TOP_K_GRAPH_CHUNKS` scoring nodes,
5. resolves each node's `source_id` chunk IDs back to local video chunks.

The lexical node score is:

- Jaccard overlap between query tokens and node-text tokens,
- plus a `0.15` bonus if any query token is a substring of the node name.

This branch therefore provides cross-video graph support using lexical entity/description overlap, not embedding similarity.

The returned hits are labeled:

- `source = "global_graph"`
- `score_graph = node_score`

#### Visual-support retrieval

`retrieve_visual_support()` is only active when `intent.is_visual_detail` is `True`.

It does not use an image model. Instead, it reuses text already stored in sanitized frame artifacts.

For each video:

1. it reads `frames_kv[video_name]`,
2. concatenates each frame's:
   - `vlm_output`
   - `transcript`
3. tokenizes this text,
4. computes simple query-token overlap,
5. keeps the top `per_video_k` frames, default `6`,
6. maps each frame's `segment_idx` to a synthetic segment ID:
   - `<video_name>_<segment_idx>`
7. maps segment IDs to chunk IDs using `_segment_to_chunk_ids()`,
8. returns chunk hits for those chunks.

The score is:

- `overlap / max(1, len(query_tokens))`

The returned hits are labeled:

- `source = "visual_support"`
- `score_entity = frame_score`

So "visual retrieval" in the current implementation is really text-over-frame-description retrieval, not direct image retrieval.

#### Parallel branch execution

`retrieve_all()` launches four retrieval branches:

- dense chunk retrieval,
- entity-graph retrieval,
- global-graph retrieval,
- visual-support retrieval.

The first two are async functions; the latter two are wrapped in small async runners. They are executed concurrently with `asyncio.gather(..., return_exceptions=True)`.

Branch failures do not abort the query. Any exception from one branch is logged and that branch contributes no hits.

### `reranker.py`

[`knowledge_inference/reranker.py`](/home/gatv-projects/Desktop/project/knowledge_inference/reranker.py) converts a heterogeneous set of retrieval hits into the final ranked evidence list.

The reranking flow is:

1. deduplicate hits by `(video_name, chunk_id)`,
2. compute or backfill component scores,
3. apply a fixed weighted score,
4. enforce per-video diversity,
5. truncate to `FINAL_EVIDENCE_K`.

#### Deduplication

`dedupe_hits()` merges hits that point to the same chunk in the same video.

When duplicates are merged:

- each component score keeps the maximum value seen,
- `final_score` keeps the maximum value seen,
- `chunk_text` keeps the longer text,
- differing sources are concatenated with `|`
- segment ID lists are unioned and sorted.

This is important because the same chunk can be surfaced independently by several retrieval branches.

#### Component-score completion

`compute_component_scores()` tokenizes the query and chunk text, then computes:

- `lexical_overlap = overlap(query_tokens, chunk_tokens) / len(query_tokens)`

It then updates scores as follows.

For entity score:

- if `intent.entity_focus_terms` exists, it counts how many of those terms appear in the chunk text,
- `entity_bonus = entity_overlap / len(entity_focus_terms)`,
- `score_entity = max(existing_score_entity, 0.65 * lexical_overlap + 0.35 * entity_bonus)`

For graph score:

- if `score_graph <= 0` and `"graph"` appears in `source`,
  - `score_graph = min(1.0, 0.25 + 0.75 * lexical_overlap)`
- otherwise, if still no graph signal,
  - `score_graph = 0.20 * lexical_overlap`

For semantic score:

- if `score_semantic <= 0`,
  - `score_semantic = lexical_overlap`

So even branch outputs without strong native scores receive lexical backoff values, ensuring all hits can be compared under one scoring formula.

#### Weighted final score

`apply_weighted_score()` computes:

- `final_score =
  0.50 * score_semantic +
  0.20 * score_entity +
  0.20 * score_graph +
  0.10 * diversity_component`

`diversity_component` is hardcoded to `1.0` for every hit. So the diversity term is not a dynamic penalty or bonus. It is simply a constant offset shared by all hits.

This means the actual ordering is determined only by semantic, entity, and graph components. Diversity is enforced later by selection limits, not by varying the score itself.

#### Diversity enforcement

`apply_diversity()` sorts hits by `final_score` descending and then selects them greedily.

Selection policy:

- always keep the top-ranked hit,
- then enforce `max_per_video` for subsequent hits,
- stop once `FINAL_EVIDENCE_K` hits are selected.

The `max_per_video` value is chosen in `rerank_hits()`:

- default:
  - `DEFAULT_MAX_PER_VIDEO = 4`
- if `intent.is_cross_video`:
  - `CROSS_VIDEO_MAX_PER_VIDEO = 3`
- if `_infer_single_video_focus(query, available_videos)` is `True`:
  - `max_per_video = FINAL_EVIDENCE_K`

`_infer_single_video_focus()` looks for exact video-name mentions after lowercasing and replacing underscores with spaces. If exactly one video is explicitly mentioned in the query, the diversity cap is effectively removed for that video.

### `context_builder.py`

[`knowledge_inference/context_builder.py`](/home/gatv-projects/Desktop/project/knowledge_inference/context_builder.py) converts ranked hits into prompt-ready evidence blocks.

#### Time-span resolution

`resolve_time_span(store, segment_ids)`:

1. looks up `segments_kv[video_name]`,
2. extracts the numeric segment index from each `segment_id` by splitting at the last underscore,
3. reads each segment's `time` field, expected in `"start-end"` format,
4. parses starts and ends as floats,
5. returns:
   - `min(start)-max(end)`
   formatted as `M:SS` or `H:MM:SS`

If segment IDs are missing or malformed, it returns:

- `"unknown"`

This means chunk time spans are derived from the union of all source segments attached to the chunk, not necessarily from a single atomic timestamp.

#### Budgeted evidence construction

`make_evidence_blocks()` consumes reranked hits under a token budget.

For each hit:

1. stop if `used_tokens >= budget_tokens`,
2. compute remaining budget,
3. assign a per-block budget:
   - `block_budget = min(900, max(240, remaining))`
4. truncate the chunk text with `_truncate_text_to_budget()`

`_truncate_text_to_budget()` uses a rough conversion:

- `1 token ~= 4 characters`

If truncation is needed, it appends:

- `" [truncated]"`

and estimates spent tokens from the clipped character length.

Each accepted hit becomes an `EvidenceBlock` with:

- `video_name`
- resolved `time_span`
- `chunk_id`
- `source`
- truncated `text`
- `final_score`

So context building is not semantic summarization. It is bounded extraction and formatting of raw chunk text.

#### Prompt rendering

`render_context_for_prompt()` formats each evidence block as:

`[video=<name> | time=<range> | chunk=<id> | source=<source>]`

followed by the block text.

Blocks are separated by blank lines.

### `prompts.py`

[`knowledge_inference/prompts.py`](/home/gatv-projects/Desktop/project/knowledge_inference/prompts.py) defines the literal prompt templates used for answer generation and answer verification.

#### QA system prompt

`SYSTEM_GROUNDED_QA` instructs the model to:

- use only the evidence context,
- avoid fabrication,
- state insufficiency or conflict explicitly,
- cite provenance inline in the form `(video=<name>, time=<range>)`,
- prefer concise answers.

#### QA user prompt

`USER_QA_TEMPLATE` inserts:

- the user question,
- the full rendered evidence context,
- a final instruction to answer only from that evidence and acknowledge weak evidence.

#### Verifier prompt

`VERIFIER_TEMPLATE` instructs the verifier to:

- classify each claim as `supported`, `unsupported`, or `uncertain`,
- return strict JSON,
- include a short summary.

### `generator.py`

[`knowledge_inference/generator.py`](/home/gatv-projects/Desktop/project/knowledge_inference/generator.py) is a thin wrapper around the shared local LLM runtime from `knowledge_build._llm`.

`generate_answer(query, context)`:

1. formats the user prompt with `USER_QA_TEMPLATE`,
2. calls `local_llm_config.best_model_func(...)`,
3. passes:
   - `system_prompt = SYSTEM_GROUNDED_QA`
   - `max_tokens = MAX_ANSWER_TOKENS`
   - generation hyperparameters from `config.py`

This means inference does not create its own LLM runtime. It reuses the same `local_llm_config` abstraction already defined for build-time LLM usage.

### `verifier.py`

[`knowledge_inference/verifier.py`](/home/gatv-projects/Desktop/project/knowledge_inference/verifier.py) implements post-generation answer verification.

The verifier does not compare token-level spans or graph provenance directly. It performs a second LLM pass over the answer and the evidence.

#### Claim extraction

`_split_claims(answer)` splits the answer into claims by regex:

- `(?<=[.!?])\s+`

So every sentence-like span ending in `.`, `!`, or `?` becomes a candidate claim.

#### Verifier input construction

`verify_answer(answer, evidence_blocks)`:

1. splits the generated answer into claims,
2. renders evidence blocks as numbered entries,
3. renders claims as numbered entries,
4. inserts both into `VERIFIER_TEMPLATE`,
5. calls `local_llm_config.best_model_func(...)` with verifier hyperparameters.

Notably, this verifier call does not pass a separate system prompt. The full instruction is embedded in the single prompt text.

#### Verifier-output parsing

`_parse_verifier_json(raw)` is intentionally defensive:

- empty output becomes `{"claims": [], "summary": "empty verifier output"}`
- non-JSON output becomes `{"claims": [], "summary": "..."}`
- malformed JSON becomes `{"claims": [], "summary": "verifier JSON parse failure"}`
- missing `claims` becomes an empty list
- missing `summary` becomes an empty string

Then `verify_answer()` aligns the number of returned labels to the number of claims:

- if too few labels are returned, missing labels are filled with `"uncertain"`
- if too many are returned, extras are discarded

#### Soft verification policy

The verifier computes:

- `supported_ratio`
- `unsupported_ratio`
- `uncertain_ratio`

Then it applies a soft answer-adjustment policy:

- if `unsupported_ratio > 0.40`:
  - prefix the full answer with a caution sentence about conflicting or insufficient evidence
- else if there are unsupported claims and at most two unsupported sentences:
  - remove those unsupported sentences via `_prune_unsupported_sentences()`

So the verifier usually preserves the original answer and only trims or cautions it. It does not regenerate the answer from scratch.

### `answer_postprocess.py`

[`knowledge_inference/answer_postprocess.py`](/home/gatv-projects/Desktop/project/knowledge_inference/answer_postprocess.py) performs optional answer formatting based on a local metadata registry.

`load_video_url_registry(path)`:

- loads `knowledge_inference/video_metadata.json` if it exists,
- expects a top-level dictionary,
- extracts only entries that contain a non-empty string `url`.

`prettify_video_name(video_name)`:

- replaces underscores with spaces,
- collapses extra whitespace.

`inject_video_urls(answer, registry)`:

- sorts video names by length descending,
- replaces literal occurrences of a raw video name in the answer with:
  - `<pretty video name> (<url>)`

This is a pure string replacement step. It does not parse citations structurally.

### `service.py`

[`knowledge_inference/service.py`](/home/gatv-projects/Desktop/project/knowledge_inference/service.py) is the main orchestration layer.

`InferenceService` keeps three pieces of state:

- `stores`
- `global_graph`
- `video_url_registry`

It also tracks `_initialized` so store loading happens only once per process.

#### Initialization

`initialize()`:

- loads all sanitized per-video stores and the global graph via `warmup()`,
- caches them on the service instance,
- is idempotent.

The service therefore behaves like a warm in-memory retriever after first use.

#### Sync/async bridge

`answer(query, debug=False)`:

- ensures initialization,
- obtains or creates an event loop via `_always_get_an_event_loop()`,
- runs `_answer_async(...)` to completion.

This makes the public API synchronous even though retrieval and generation are internally async.

#### Full answer flow

`_answer_async()` performs:

1. create `query_id` and `timings`,
2. analyze query,
3. retrieve hits from all retrieval branches,
4. rerank hits,
5. build evidence blocks,
6. render context,
7. if evidence is missing or too weak:
   - return an uncertainty answer immediately,
8. otherwise generate an answer,
9. verify the answer,
10. compute confidence,
11. optionally prepend a caution if support is low,
12. inject video URLs,
13. assemble debug payload,
14. return `AnswerResult`.

#### Early uncertainty fallback

The service returns a low-confidence fallback answer when either:

- `not evidence`
- or `ranked_hits` exists and `ranked_hits[0].final_score < MIN_EVIDENCE_SCORE`

The fallback answer is:

- `"I do not have enough grounded evidence in the sanitized knowledge cache to answer this confidently. Please rephrase the question with more specific entities, time windows, or video context."`

In this path:

- `confidence = 0.2`
- verification debug is synthesized as:
  - `supported_ratio = 0.0`
  - `reason = "insufficient_evidence"`

No generation or verifier call occurs.

#### Confidence calculation

`_compute_confidence()` combines three factors:

1. evidence strength:
   - mean of clamped `final_score` over evidence blocks
2. retrieval agreement:
   - number of distinct first-hop sources among evidence blocks, capped at `3`, then divided by `3`
3. verifier support:
   - `supported_ratio`

The formula is:

- `0.5 * evidence_strength + 0.35 * supported_ratio + 0.15 * retrieval_agreement`

Then several penalties may apply:

- if fewer than 2 evidence blocks:
  - `-0.15`
- if verifier `unsupported_ratio > 0.40`:
  - `-0.25`
- if none of the lowercase query tokens occur in the text of the top 3 evidence blocks:
  - `-0.10`

Finally the value is clamped to `[0.0, 1.0]`.

Confidence bands are:

- `high` for `>= 0.75`
- `medium` for `>= 0.45`
- `low` otherwise

#### Post-verification caution policy

Even after verification, the service adds another user-facing caution if either:

- `supported_ratio < MIN_SUPPORTED_CLAIMS_RATIO`
- or `confidence < 0.45`

In that case it prefixes the answer with:

- `"The available evidence supports only part of the answer. I may be missing additional clips or stronger corroboration."`

This is separate from the verifier's own warning prefix. So, in low-support cases, both verifier-level and service-level cautioning can appear.

#### Logging and debug payload

When `debug=True`, the returned `AnswerResult.debug` contains:

- `query_id`
- `intent`
- `timings`
- `retrieval_counts`
- `final_evidence_count`
- `verification`
- `confidence_band`

The service also logs a compact summary with retrieval counts, evidence count, generation time, verification summary, and total time.

### `cli.py`

[`knowledge_inference/cli.py`](/home/gatv-projects/Desktop/project/knowledge_inference/cli.py) is the simple command-line interface.

It:

- configures process-wide logging,
- parses:
  - `--query`
  - `--debug`
  - `--max-evidence`
- runs `InferenceService.answer(...)`,
- prints:
  - answer text,
  - confidence,
  - top evidence summaries,
  - optional debug JSON.

The CLI does not persist results or cache answers. It is only a thin presentation layer over the service.

### `eval.py`

[`knowledge_inference/eval.py`](/home/gatv-projects/Desktop/project/knowledge_inference/eval.py) implements an offline evaluator for QA performance and latency.

The expected dataset is either:

- a list of case dictionaries,
- or a dictionary with:
  - `"cases": [...]`

Each case may contain:

- `question`
- `expected_answer_keywords`
- `expected_videos`
- `expected_champions`

#### Per-case evaluation flow

For each case:

1. run `InferenceService.answer(question, debug=True)`,
2. measure latency,
3. compute `keyword_hit`:
   - `1.0` if any expected keyword appears in the answer, else `0.0`
4. compute `video_hit`:
   - overlap between expected videos/champions and evidence video names
5. define:
   - `retrieval_proxy = 0.5 * keyword_hit + 0.5 * video_hit`
6. read verifier `supported_ratio` from debug output as a groundedness proxy,
7. store per-case report data.

So the evaluator does not compute exact-match QA, F1, or human judgment. It uses lightweight proxy metrics.

#### Summary metrics

The evaluator reports:

- `cases_total`
- `retrieval_recall_proxy_mean`
- `groundedness_proxy_mean`
- `latency_p50_s`
- `latency_p95_s`

It then writes a timestamped report JSON under:

- `knowledge_inference/reports/`

## End-to-end runtime flow

## 1. Warmup and store loading

The first real step is service initialization through `warmup()`.

This loads:

- all valid sanitized per-video stores,
- the sanitized global graph.

This is done once per process and cached inside `InferenceService`.

## 2. Query analysis

The query is normalized, heuristic flags are set, and entity-focus terms are extracted.

This intent object influences:

- whether visual-support retrieval runs,
- whether cross-video diversity limits are tightened,
- whether entity-term overlap boosts reranking.

## 3. Parallel retrieval

Four retrieval branches run concurrently:

- dense chunk retrieval,
- entity-graph retrieval,
- global graph retrieval,
- visual-support retrieval.

The output is a flat list of `RetrievalHit` objects that may contain duplicates across sources.

## 4. Merge and rerank

The reranker:

- deduplicates identical `(video_name, chunk_id)` hits,
- backfills missing score components lexically,
- applies fixed weighted scoring,
- enforces a per-video diversity cap,
- returns at most `14` final hits.

## 5. Evidence context construction

The top-ranked hits are converted into evidence blocks under the global token budget of `10000`.

Each block includes:

- video name,
- resolved time range,
- chunk ID,
- retrieval source,
- truncated chunk text.

These blocks are then concatenated into the final prompt context.

## 6. Early fallback or answer generation

If no evidence survives, or if the top evidence score is below `0.18`, the system does not call the generator. It returns a fixed uncertainty message.

Otherwise, it sends the query and evidence context to the local GPT-OSS model.

## 7. Claim verification

The generated answer is split into sentence-like claims. A second LLM call classifies each claim as supported, unsupported, or uncertain using the evidence blocks.

Unsupported content is either:

- removed if only a small number of unsupported sentences are detected,
- or preserved with a warning prefix if unsupported content is more substantial.

## 8. Confidence calibration and final formatting

The service combines evidence strength, verifier support, and retrieval-source agreement into a scalar confidence score.

Low-support answers are additionally prefixed with a caution sentence.

Finally, if a local metadata registry maps a video name to a URL, raw video-name mentions in the answer are replaced with a prettified name plus the URL.

## Behavioral details encoded by tests

The unit tests under [`knowledge_inference/tests`](/home/gatv-projects/Desktop/project/knowledge_inference/tests) confirm several implementation details:

- [`knowledge_inference/tests/test_query_analyzer.py`](/home/gatv-projects/Desktop/project/knowledge_inference/tests/test_query_analyzer.py):
  - a query can simultaneously trigger cross-video, temporal, and visual flags
- [`knowledge_inference/tests/test_reranker.py`](/home/gatv-projects/Desktop/project/knowledge_inference/tests/test_reranker.py):
  - duplicate hits from different retrieval branches are merged and preserve combined provenance
- [`knowledge_inference/tests/test_context_builder.py`](/home/gatv-projects/Desktop/project/knowledge_inference/tests/test_context_builder.py):
  - segment time `"0-30"` is rendered as `"0:00-0:30"`
  - long chunk text is truncated and tagged with `[truncated]`
- [`knowledge_inference/tests/test_answer_postprocess.py`](/home/gatv-projects/Desktop/project/knowledge_inference/tests/test_answer_postprocess.py):
  - raw underscore-separated video names are prettified for answer display
  - URL injection only replaces exact known video-name matches

## Architectural characterization

In system terms, `knowledge_inference/` is a deterministic multi-branch retrieval layer wrapped around a local LLM generation-and-verification loop.

Its most important design choices are:

- sanitized-only reads,
- heuristic query understanding rather than LLM planning,
- fixed-weight hybrid retrieval,
- one-hop entity-graph expansion,
- lexical global-graph retrieval,
- text-based visual support rather than direct image retrieval,
- evidence-budgeted prompting,
- soft post-hoc verification rather than hard answer rejection.

These choices make the module relatively transparent and reproducible: most behaviors are controlled by explicit constants and small deterministic functions, while the two LLM-dependent stages are isolated to answer generation and claim verification.
