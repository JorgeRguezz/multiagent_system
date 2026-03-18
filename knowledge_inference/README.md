# Knowledge Inference (QA RAG)

This package implements inference-only QA over sanitized build caches.

## Data sources (enforced)
- `knowledge_sanitization/cache/sanitized_build_cache_*`
- `knowledge_sanitization/cache/sanitized_global/graph_AetherNexus.graphml`

Unsanitized caches are not used.

## Pipeline
1. Query analysis (heuristic-first intent detection)
2. Retrieval (dense chunks, entity graph, global graph, optional visual-text support)
3. Reranking (dedupe + weighted ranking + diversity cap)
4. Context building (token budget + provenance formatting)
5. Generation (gpt-oss-20b runtime via `knowledge_build._llm`)
6. Verification (supported/unsupported/uncertain claim check)
7. Confidence calibration and uncertainty fallback

## Defaults chosen for open questions
- Intent method: heuristic-first
- Scoring weights: hardcoded baseline
- Diversity policy: strict per-video cap with intent-aware override
- Verifier strictness: soft verification with cautioning
- Global graph retrieval: lexical matching (v1)
- Visual retrieval depth: text-only visual support (no image model in v1)
- Confidence calibration: explicit heuristic formula
- Max context / answer tokens: `10000` / `2500`
- Conflicting evidence: report both interpretations with provenance
- Eval GT quality: lightweight tagged approach documented in `eval.py`

## CLI
```bash
python3 -m knowledge_inference.cli --query "How does Pyke secure kills in this guide?" --debug
```

## Evaluation
```bash
python3 -m knowledge_inference.eval --dataset /path/to/qa_eval.json
```

Reports are written to `knowledge_inference/reports/`.
