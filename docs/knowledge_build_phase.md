# Knowledge Build Phase

## Purpose of the module

`knowledge_build/` is the phase that converts sanitized extraction artifacts into the system's persistent retrieval and graph assets.

Where `knowledge_extraction/` produces multimodal evidence as JSON, `knowledge_build/` transforms that evidence into:

- chunked textual units,
- a chunk vector index for naive retrieval,
- an entity vector index for entity-centric retrieval,
- a per-video chunk-entity-relation graph,
- a cleaned per-video graph,
- a global merged graph built across videos,
- cached LLM responses used during graph construction.

In practical terms, this phase is the graph-construction and indexing layer of the system.

## Main entry points

The core implementation centers on [`knowledge_build/builder.py`](/home/gatv-projects/Desktop/project/knowledge_build/builder.py), which defines the `KnowledgeBuilder` class.

Operationally there are two normal entry points:

- single-run builder:
  - `python -m knowledge_build.builder`
- queue runner:
  - `python -m knowledge_build.run_build_queue`

The queue runner repeatedly constructs `KnowledgeBuilder` instances until no unbuilt sanitized extraction folders remain.

## Input contract

The build phase does not read raw extraction output directly. It expects the sanitized handoff from `knowledge_sanitization/`.

`KnowledgeBuilder._resolve_artifact_dir()` requires this directory layout:

`<extraction_dir>/sanitized_extracted_data/<video_name>/`

Inside each video directory, it expects the extraction-format files:

- `kv_store_video_segments.json`
- `kv_store_video_frames.json`
- `kv_store_video_path.json`

By default, `builder.py` uses:

`/home/gatv-projects/Desktop/project/knowledge_sanitization/cache`

as `extraction_dir`.

This is important architecturally: `knowledge_build` is intentionally positioned after sanitization, not directly after extraction.

## File-level responsibilities

### `builder.py`

[`knowledge_build/builder.py`](/home/gatv-projects/Desktop/project/knowledge_build/builder.py) orchestrates the entire build pipeline. It:

- selects the next unbuilt sanitized artifact directory,
- creates the per-video output directory,
- initializes all KV, vector, and graph stores,
- loads sanitized extraction artifacts,
- copies those artifacts into build-local stores,
- chunks segment content,
- inserts chunks into the chunk vector database,
- runs LLM-based entity and relationship extraction,
- upserts graph nodes and edges,
- writes all stores to disk,
- post-cleans the per-video graph,
- merges the cleaned graph into the global graph cache.

### `_op.py`

[`knowledge_build/_op.py`](/home/gatv-projects/Desktop/project/knowledge_build/_op.py) contains the core graph-building logic:

- segment-based chunking,
- chunk ID generation,
- parsing LLM extraction tuples,
- entity normalization,
- relationship normalization,
- node and edge merge policies,
- vector DB upsertion for extracted entities,
- retrieval helpers used later by inference.

### `_llm.py`

[`knowledge_build/_llm.py`](/home/gatv-projects/Desktop/project/knowledge_build/_llm.py) defines:

- `LLMConfig`,
- embedding model setup,
- GPT-OSS completion helpers,
- optional vLLM helpers,
- batched GPT-OSS generation,
- caching hooks,
- cleanup functions for LLM resources.

Although the module still contains vLLM support, the active default build path uses GPT-OSS both for the "best" model and the "cheap" model.

### `prompt.py`

[`knowledge_build/prompt.py`](/home/gatv-projects/Desktop/project/knowledge_build/prompt.py) defines the prompt templates that control:

- base graph extraction,
- unified glean passes,
- entity/relation description summarization,
- retrieval-side query rewriting.

### `clean_kg.py`

[`knowledge_build/clean_kg.py`](/home/gatv-projects/Desktop/project/knowledge_build/clean_kg.py) performs conservative post-hoc entity merging on GraphML graphs. This is the mechanism used both for:

- per-video graph cleaning, and
- global graph merge cleanup.

### Storage backends

`knowledge_build/_storage/` contains the persistence backends actually used by the default builder:

- [`knowledge_build/_storage/kv_json.py`](/home/gatv-projects/Desktop/project/knowledge_build/_storage/kv_json.py): JSON KV storage.
- [`knowledge_build/_storage/vdb_nanovectordb.py`](/home/gatv-projects/Desktop/project/knowledge_build/_storage/vdb_nanovectordb.py): NanoVectorDB-backed vector storage.
- [`knowledge_build/_storage/gdb_networkx.py`](/home/gatv-projects/Desktop/project/knowledge_build/_storage/gdb_networkx.py): NetworkX-backed graph storage persisted as GraphML.

Other storage modules exist (`gdb_neo4j.py`, `vdb_hnswlib.py`) but are not used in the default build configuration.

## High-level build flow

The `build()` method in `KnowledgeBuilder` is the main execution path.

It performs the following sequence:

1. Resolve the next eligible sanitized artifact directory.
2. Create or reuse the per-video build output directory.
3. Initialize storage namespaces.
4. Load sanitized extraction JSON artifacts.
5. Persist those artifacts into build-local KV stores.
6. Call `ainsert()` on the segment data.
7. Post-clean the resulting graph.
8. Merge the cleaned graph into the global graph cache.

## 1. Artifact directory selection

`KnowledgeBuilder.__post_init__()` immediately calls `_resolve_artifact_dir()`.

That function:

- looks under `sanitized_extracted_data`,
- collects all video directories,
- filters to those containing `kv_store_video_segments.json`,
- checks whether a matching build output directory already exists at:
  - `knowledge_build_cache_<video_name>`
- chooses the first unbuilt candidate in sorted order.

So the builder is not parameterized by a specific video directory. Instead, each invocation automatically selects the next unbuilt sanitized folder.

This behavior is why `run_build_queue.py` can simply instantiate `KnowledgeBuilder` in a loop.

## 2. Per-video and global output directories

For a selected artifact directory `<video_name>`, `KnowledgeBuilder` creates:

- per-video build dir:
  - `knowledge_build_cache_<video_name>`
- global build dir:
  - `knowledge_build_cache_global`

The per-video directory holds the full local build result for one video.

The global directory holds:

- `graph_AetherNexus.graphml`
- `aether_manifest.json`

The manifest records which videos have already been merged into the global graph.

## 3. Storage initialization

During `__post_init__()`, the builder constructs several storage namespaces using `config_dict = asdict(self)`.

The default storage set is:

- `video_path_db`: JSON KV store for source video path.
- `video_segments`: JSON KV store for segment-level extraction artifact.
- `video_frames`: JSON KV store for frame-level extraction artifact.
- `text_chunks`: JSON KV store for chunked text units.
- `llm_response_cache`: JSON KV store for cached LLM outputs.
- `entities_vdb`: NanoVectorDB storage for entity embeddings.
- `chunks_vdb`: NanoVectorDB storage for chunk embeddings.
- `chunk_entity_relation_graph`: NetworkX graph store for the extracted knowledge graph.

The configuration flags controlling this are:

- `enable_local = True`
- `enable_naive_rag = True`
- `enable_llm_cache = True`

So, by default, both vector indexes and the LLM cache are enabled.

## 4. Embedding stack

The builder wraps the embedding function through:

- `wrap_embedding_func_with_attrs(...)`
- `limit_async_func_call(...)`

The default embedding configuration comes from [`knowledge_build/_llm.py`](/home/gatv-projects/Desktop/project/knowledge_build/_llm.py):

- model: `all-MiniLM-L6-v2`
- embedding dimension: `384`
- embedding max token size: `512`
- embedding batch size: `32`
- embedding async limit: `4`

These embeddings are used for:

- `vdb_chunks.json`
- `vdb_entities.json`

The vector stores do not use token-level embeddings or cross-encoders. They store one dense vector per chunk or per entity entry.

## 5. Loading sanitized artifacts

Inside `build()`, the builder loads:

- `kv_store_video_segments.json` into `segments_data`
- `kv_store_video_frames.json` into `frames_data`
- `kv_store_video_path.json` into `paths_data`

It then copies them into its own build-local stores:

- `await self.video_segments.upsert(segments_data)`
- `await self.video_frames.upsert(frames_data)`
- `await self.video_path_db.upsert(paths_data)`

This duplication matters: the build phase does not operate directly on the source sanitized files after load. It maintains its own local copies in the build output directory.

## 6. Chunk generation

After loading the artifacts, `build()` calls:

- `await self.ainsert(self.video_segments._data)`

Inside `ainsert()`, the first major operation is:

- `get_chunks(new_videos=new_video_segment, chunk_func=self.chunk_func, max_token_size=self.chunk_token_size)`

The default chunk function is:

- `chunking_by_video_segments`

with:

- `chunk_token_size = 1200`

### Chunking logic

`chunking_by_video_segments()` does not use a tokenizer. It approximates token size using:

- `1 token ~= 4 characters`

Therefore:

- `max_token_size = 1200`
- `max_char_size = 4800`

The function processes ordered segment texts and greedily concatenates adjacent segment contents until the running chunk would exceed `max_char_size`.

For each resulting chunk it stores:

- `tokens`: approximate token count (`len(content) // 4`)
- `content`: concatenated segment text
- `chunk_order_index`
- `video_segment_id`: list of the segment IDs absorbed into the chunk

### Segment ID format inside chunks

`get_chunks()` constructs segment IDs as:

- `<video_name>_<segment_idx>`

This is the bridge key connecting chunk-level graph facts back to segment-level source material.

### Chunk IDs

Each chunk receives a deterministic MD5-based ID:

- `compute_mdhash_id(chunk["content"], prefix="chunk-")`

So chunk identity is content-derived, not position-derived.

## 7. Deduplication before insertion

Before any new chunks are inserted, `ainsert()` calls:

- `self.text_chunks.filter_keys(list(inserting_chunks.keys()))`

This keeps only chunk IDs not already present in the `text_chunks` store.

If no new chunks survive this filter, the builder logs:

- `"All chunks are already in the storage"`

and exits early.

This is the local idempotency mechanism for chunk insertion.

## 8. Chunk vector index

If `enable_naive_rag` is `True`, the builder inserts all new chunks into `chunks_vdb`.

`NanoVectorDBStorage.upsert()`:

1. collects chunk `content` strings,
2. embeds them in batches using the sentence-transformer embedding model,
3. constructs NanoVectorDB records containing:
   - `__id__`
   - `__vector__`
4. saves them to:
   - `vdb_chunks.json`

This index supports later naive retrieval over chunk text.

## 9. Entity and relationship extraction

The core graph-construction logic is `extract_entities()` in [`knowledge_build/_op.py`](/home/gatv-projects/Desktop/project/knowledge_build/_op.py).

Despite the name, it extracts both:

- entities
- relationships

from chunk text.

### 9.1 Prompt setup

The extraction context uses delimiters from [`knowledge_build/prompt.py`](/home/gatv-projects/Desktop/project/knowledge_build/prompt.py):

- tuple delimiter: `<|>`
- record delimiter: `##`
- completion delimiter: `<|COMPLETE|>`

Default allowed entity types are:

- `organization`
- `person`
- `geo`
- `event`

These are uppercased during validation, so the allowed set becomes:

- `ORGANIZATION`
- `PERSON`
- `GEO`
- `EVENT`

Any extracted type outside this set is normalized to `UNKNOWN`.

### 9.2 Base extraction pass

For every chunk, the builder creates a base extraction prompt using:

- `PROMPTS["kg_simple_graph_extraction_template"]`

This prompt asks the model to emit tuples in one of two formats:

```text
("entity"<|><entity_name><|><entity_type><|><entity_description>)
("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relationship_strength>)
```

The builder then calls:

- `oss_llm_batch_generate(simple_prompts, system_prompt=..., max_tokens=3000)`

Important implementation detail: `oss_llm_batch_generate()` is called "batch" generation, but it still loops through prompts sequentially inside an executor. It is not true simultaneous multi-prompt GPU batching.

### 9.3 Parsing extraction output

The builder parses tuple output using `_split_extraction_records()`, which:

- splits the text on record and completion delimiters,
- extracts the text inside parentheses,
- splits each tuple by the tuple delimiter.

Two specialized parsers then convert tuples into Python dictionaries:

- `_handle_single_entity_extraction()`
- `_handle_single_relationship_extraction()`

#### Entity parsing rules

An entity tuple must:

- start with `"entity"`
- contain at least 4 fields

The resulting entity dictionary contains:

- `entity_name`
- `entity_type`
- `description`
- `source_id` = chunk ID

Entity names and types are uppercased and cleaned.

#### Relationship parsing rules

A relationship tuple must:

- start with `"relationship"`
- contain at least 4 fields

The resulting relation dictionary contains:

- `src_id`
- `tgt_id`
- `weight`
- `description`
- `source_id` = chunk ID

If the weight is missing or invalid, it defaults to `1.0`.

### 9.4 Filtering and sanitation rules

`_extract_entities_from_text()` and `_extract_relationships_from_text()` apply several constraints.

Entities are rejected if:

- the entity name is empty,
- the description is empty,
- the same entity name already appeared within the same chunk.

Relationships are rejected if:

- source equals target,
- either endpoint is not in the valid entity set for that chunk,
- the description is empty,
- the undirected source-target pair already appeared in that chunk.

Relationship weights are clamped into:

- minimum `1.0`
- maximum `10.0`

### 9.5 Unified glean pass

After the base extraction pass, the builder performs iterative recovery of missing tuples using:

- `PROMPTS["kg_unified_glean_template"]`

For each chunk it provides the model with:

- a snapshot of currently extracted entities,
- a snapshot of currently extracted relationships,
- the full chunk text,
- instructions to add only missing tuples.

The number of glean rounds is controlled by:

- `entity_extract_max_gleaning = 1`

So in the default configuration the builder performs at most one additional recovery pass after the base extraction.

The glean call uses:

- `use_llm_func = global_config["llm"]["best_model_func"]`

which, by default, is GPT-OSS again.

If the glean pass returns no net-new entities and no net-new relations, the loop stops early.

### 9.6 Current model stack

The default `local_llm_config` sets:

- `best_model_func_raw = oss_llm_complete`
- `cheap_model_func_raw = oss_llm_complete`

So the current build phase uses GPT-OSS for both:

- high-value extraction/summarization calls,
- cheaper auxiliary calls.

The vLLM path exists in code, but it is not the active default path.

## 10. Merging extracted graph facts

The raw extraction output is still chunk-local. The builder then merges chunk-level facts into graph-global nodes and edges.

### 10.1 Node merge policy

Entities are grouped by `entity_name`.

For each entity group, `_merge_nodes_then_upsert()`:

1. checks whether the graph already contains a node with that name,
2. collects prior:
   - `entity_type`
   - `source_id`
   - `description`
3. chooses the entity type by majority frequency,
4. concatenates unique descriptions with `GRAPH_FIELD_SEP = "<SEP>"`,
5. concatenates unique source IDs with `"<SEP>"`,
6. optionally summarizes the long merged description through `_handle_entity_relation_summary()`,
7. upserts the node into the graph.

The final node schema includes:

- `entity_type`
- `description`
- `source_id`

and then the function returns the same plus:

- `entity_name`

### 10.2 Edge merge policy

Relationships are grouped by undirected `(src_id, tgt_id)` pairs.

For each group, `_merge_edges_then_upsert()`:

1. checks whether the graph already has an edge between those nodes,
2. merges previous and incoming edge attributes,
3. sums relationship weights,
4. concatenates unique descriptions with `"<SEP>"`,
5. concatenates unique source IDs with `"<SEP>"`,
6. chooses the minimum `order` if present,
7. creates placeholder `UNKNOWN` nodes if an endpoint is missing,
8. optionally summarizes the merged edge description,
9. upserts the edge.

The final edge schema includes:

- `weight`
- `description`
- `source_id`
- `order`

Even though the graph is undirected, the descriptions and provenance of repeated relation mentions are accumulated across chunks.

## 11. Description summarization during merge

When merged node or edge descriptions become long, `_handle_entity_relation_summary()` can compress them.

The threshold logic is again character-based:

- if description length `< summary_max_tokens * 4`, it is kept as-is

Default:

- `entity_summary_to_max_tokens = 500`

If the threshold is exceeded, the builder calls the cheap LLM with:

- `PROMPTS["summarize_entity_descriptions"]`

This means the build phase uses an LLM twice:

1. to extract graph tuples from chunks,
2. to compress accumulated graph descriptions when they become too large.

## 12. Entity vector index

After nodes are merged into the graph, the builder inserts entity records into `entities_vdb`.

Each entity vector record has ID:

- `ent-<md5(entity_name)>`

and content:

- `entity_name + description`

with metadata:

- `entity_name`

The underlying file is:

- `vdb_entities.json`

This index supports entity-centric retrieval in later inference.

## 13. Persistent stores written to disk

At the start of insertion, `_insert_start()` triggers `index_start_callback()` only on the graph store.

At the end of insertion, `_insert_done()` calls `index_done_callback()` on:

- `text_chunks`
- `llm_response_cache`
- `entities_vdb`
- `chunks_vdb`
- `chunk_entity_relation_graph`
- `video_frames`
- `video_segments`
- `video_path_db`

In the default storage implementation, this causes the following files to be written.

### Copied extraction artifacts

- `kv_store_video_segments.json`
- `kv_store_video_frames.json`
- `kv_store_video_path.json`

### Build-native KV stores

- `kv_store_text_chunks.json`
- `kv_store_llm_response_cache.json`

### Vector indexes

- `vdb_chunks.json`
- `vdb_entities.json`

### Graph outputs

- `graph_chunk_entity_relation.graphml`
- `graph_chunk_entity_relation_clean.graphml`

An actual build directory such as [`knowledge_build_cache_3_Minute_Lux_Guide_-_A_Guide_for_League_of_Legends`](/home/gatv-projects/Desktop/project/knowledge_build_cache_3_Minute_Lux_Guide_-_A_Guide_for_League_of_Legends) contains exactly this file pattern.

## 14. Graph backend behavior

The default graph backend is `NetworkXStorage`.

It stores the graph in:

- `graph_<namespace>.graphml`

For the build phase namespace, that becomes:

- `graph_chunk_entity_relation.graphml`

The graph backend supports:

- node existence checks,
- edge existence checks,
- node/edge degree,
- node and edge retrieval,
- node and edge upsertion,
- optional clustering and node embeddings.

However, the current builder path does not call clustering or node embeddings. It only uses the basic graph CRUD interface.

## 15. Per-video graph cleaning

After `ainsert()` finishes, the builder calls `_post_clean_graphs_and_update_global()`.

This function:

1. loads `graph_chunk_entity_relation.graphml`,
2. runs `unify_entities_conservative(graph)`,
3. writes:
   - `graph_chunk_entity_relation_clean.graphml`

### Conservative unification logic

`unify_entities_conservative()` in [`knowledge_build/clean_kg.py`](/home/gatv-projects/Desktop/project/knowledge_build/clean_kg.py) merges nodes based on normalized keys built from:

- the node ID itself,
- alias candidates mined from the node description.

Normalization includes:

- HTML unescaping,
- quote stripping,
- Unicode normalization,
- case folding,
- accent removal,
- dropping non-alphanumeric characters.

Alias candidates are extracted from description phrases such as:

- `aka`
- `a.k.a.`
- `also called`
- `known as`
- `referred to as`
- `spelled`
- `misspelled`

Two nodes are only unioned if their entity types are compatible:

- same type, or
- one side is `UNKNOWN`, or
- one side has empty type.

For every merged group:

- one representative node is selected,
- node attributes are merged by unique `"<SEP>"` joining,
- edges are remapped to representative nodes,
- self-loops created by merging are dropped.

So the cleaned graph is a conservative alias-collapsed version of the raw per-video graph.

## 16. Global graph update

After the per-video graph is cleaned, the builder merges it into the global graph cache.

The global files are:

- [`knowledge_build_cache_global/graph_AetherNexus.graphml`](/home/gatv-projects/Desktop/project/knowledge_build_cache_global/graph_AetherNexus.graphml)
- [`knowledge_build_cache_global/aether_manifest.json`](/home/gatv-projects/Desktop/project/knowledge_build_cache_global/aether_manifest.json)

### Merge logic

`_update_global_knowledge_graph()`:

1. loads the processed-video manifest,
2. skips the merge if the current video name is already listed,
3. loads the cleaned per-video graph,
4. if no global graph exists, initializes it with this graph,
5. otherwise composes:
   - existing global graph
   - incoming cleaned graph
6. runs `unify_entities_conservative()` again on the merged result,
7. writes the cleaned merged graph back to `graph_AetherNexus.graphml`,
8. adds the video name to the manifest.

This means entity consolidation happens at two levels:

- inside each video's graph,
- again across the union of all videos.

## 17. LLM caching behavior

If `enable_llm_cache` is `True`, the builder creates `kv_store_llm_response_cache.json`.

The cache key is a hash of:

- model identifier,
- system prompt,
- user prompt,
- history messages

This cache is used in:

- description summarization,
- glean extraction passes,
- other GPT-OSS helper calls that pass `hashing_kv`.

This reduces repeated generation work across reruns, although the base batched extraction path `oss_llm_batch_generate()` does not itself use this cache.

## 18. Queue mode behavior

[`knowledge_build/run_build_queue.py`](/home/gatv-projects/Desktop/project/knowledge_build/run_build_queue.py) repeatedly:

1. discovers sanitized candidate directories,
2. instantiates `KnowledgeBuilder`,
3. lets the builder select the next unbuilt folder,
4. runs `asyncio.run(builder.build())`,
5. stops when the builder raises the specific "No unbuilt extraction folders found" condition.

This script stops on the first real failure to avoid infinite retries.

## 19. Important implementation characteristics

### Chunking is segment-concatenation, not sentence splitting

The default builder does not split within segment text using syntax-aware tokenization. It concatenates whole segment records until the approximate token budget is reached.

This preserves segment provenance but also means chunk boundaries are coarse.

### Graph extraction is LLM-native and tuple-based

There is no classical NER pipeline here. The builder delegates both entity extraction and relationship extraction to an LLM and expects structured tuple output back.

### The graph is provenance-aware

Nodes and edges retain `source_id` values pointing back to chunk IDs, and chunk IDs in turn reference `video_segment_id` lists. This creates a traceable chain:

- graph fact
- chunk
- segment
- original video

### Persistence is simple and file-based

The default build stack uses:

- JSON files for KV stores,
- NanoVectorDB JSON files for vectors,
- GraphML for graphs.

That makes the outputs easy to inspect and portable, but it also means build products are directory-based rather than database-service-based.

### The builder is partially idempotent

It skips already-built video folders by directory existence and skips already-inserted chunks by chunk ID. But if corrupted or low-quality outputs already exist, the builder will still treat the folder as built unless the output directory is removed.

## 20. Observed artifact quality implications

Because chunk text comes directly from the extraction phase, build quality depends heavily on extraction quality.

For example, actual built chunk files show that if extraction emits malformed or verbose content, the builder will still ingest it as chunk text. The build phase does not deeply rewrite chunk text before graph extraction; it mostly trusts the sanitized extraction artifact.

So `knowledge_build` is a structured transformation phase, not a heavy semantic repair phase.

## 21. Conceptual summary

`knowledge_build/` takes sanitized per-video extraction artifacts and turns them into the retrieval and graph substrate used by the rest of the system.

Its internal logic is:

1. load sanitized extraction JSON,
2. copy artifacts into build-local stores,
3. group segment text into chunks,
4. embed chunks for naive retrieval,
5. run LLM tuple extraction over chunks,
6. parse and normalize entities and relationships,
7. merge them into a graph,
8. embed entities for entity retrieval,
9. persist all stores,
10. clean the graph conservatively,
11. merge the cleaned graph into the global graph.

This is the phase where the system stops being a collection of extracted descriptions and becomes an explicit, queryable knowledge structure.
