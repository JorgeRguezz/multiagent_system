# Knowledge System Evaluation Phase

## Purpose of the module

`knowledge_system_evaluation/` is the final evaluation workspace of the system. Its role is not to build knowledge or answer questions directly, but to hold evaluation datasets and generated evaluation reports, and to provide a small batch script that fills those reports with outputs from the inference system.

In its current implemented form, this module is much lighter than the earlier phases. It does not define a full evaluation package with its own scoring library, metrics engine, or benchmark runner. Instead, it serves three practical purposes:

1. store manually curated QA evaluation cases,
2. store derived report files in which RAG outputs have been filled in,
3. batch-run the inference system over those cases and write the resulting `rag_answer` and `context` fields back to disk.

So architecturally, this phase is a report-generation and dataset-management layer built on top of `knowledge_inference/`, not an independent model or retrieval module.

## Implemented contents

The directory currently contains:

- one executable script:
  - [`knowledge_system_evaluation/fill_report_eval_test_rag.py`](/home/gatv-projects/Desktop/project/knowledge_system_evaluation/fill_report_eval_test_rag.py)
- several JSON datasets and report snapshots:
  - [`knowledge_system_evaluation/report_eval_test.json`](/home/gatv-projects/Desktop/project/knowledge_system_evaluation/report_eval_test.json)
  - [`knowledge_system_evaluation/report_eval_test_v2.json`](/home/gatv-projects/Desktop/project/knowledge_system_evaluation/report_eval_test_v2.json)
  - [`knowledge_system_evaluation/merged_report_eval_test.json`](/home/gatv-projects/Desktop/project/knowledge_system_evaluation/merged_report_eval_test.json)
  - [`knowledge_system_evaluation/report_eval_test_rag_smoke.json`](/home/gatv-projects/Desktop/project/knowledge_system_evaluation/report_eval_test_rag_smoke.json)
  - [`knowledge_system_evaluation/report_eval_test_rag.json`](/home/gatv-projects/Desktop/project/knowledge_system_evaluation/report_eval_test_rag.json)
  - [`knowledge_system_evaluation/report_eval_test_rag_merged.json`](/home/gatv-projects/Desktop/project/knowledge_system_evaluation/report_eval_test_rag_merged.json)

There is no `__init__.py`, no `config.py`, no `README.md`, and no unit-test suite under this directory.

## File-level responsibilities

### `fill_report_eval_test_rag.py`

[`knowledge_system_evaluation/fill_report_eval_test_rag.py`](/home/gatv-projects/Desktop/project/knowledge_system_evaluation/fill_report_eval_test_rag.py) is the only active programmatic component in this phase.

Its job is simple:

- read a top-level JSON list of evaluation cases,
- run each case's `question` through `knowledge_inference.InferenceService`,
- write the resulting answer into `rag_answer`,
- write the retrieved prompt context into `context`,
- preserve all other fields unchanged,
- continuously save partial progress to an output JSON file.

This script therefore converts a blank or partially filled evaluation sheet into an inference-populated report.

### JSON report files

The JSON files in this directory are not summary dictionaries. Every one of them is a top-level JSON list.

Each object in these lists follows the same core schema:

- `question`
- `answer_gold`
- `context`
- `rag_answer`
- `strong_llm_answer`
- `weak_llm_answer`

The code in this module only writes:

- `context`
- `rag_answer`

It never populates:

- `strong_llm_answer`
- `weak_llm_answer`

So those two fields are placeholders for external or later evaluation workflows, not outputs produced by the current script.

## Evaluation data inventory

The report files differ mainly by dataset size and whether the `context` and `rag_answer` fields have already been populated.

Current observed dataset sizes:

- `report_eval_test.json`:
  - 118 cases
- `report_eval_test_v2.json`:
  - 136 cases
- `merged_report_eval_test.json`:
  - 254 cases
- `report_eval_test_rag_smoke.json`:
  - 118 cases
- `report_eval_test_rag.json`:
  - 118 cases
- `report_eval_test_rag_merged.json`:
  - 254 cases

Current fill status:

- `report_eval_test.json`:
  - `context` empty in all 118 cases
  - `rag_answer` empty in all 118 cases
- `report_eval_test_v2.json`:
  - `context` empty in all 136 cases
  - `rag_answer` empty in all 136 cases
- `merged_report_eval_test.json`:
  - `context` empty in all 254 cases
  - `rag_answer` empty in all 254 cases
- `report_eval_test_rag_smoke.json`:
  - `context` filled in 3 cases
  - `rag_answer` filled in 3 cases
- `report_eval_test_rag.json`:
  - `context` filled in all 118 cases
  - `rag_answer` filled in all 118 cases
- `report_eval_test_rag_merged.json`:
  - `context` filled in all 254 cases
  - `rag_answer` filled in all 254 cases

This pattern shows the intended workflow clearly:

1. start from a blank dataset file,
2. run the filler script,
3. produce a corresponding `*_rag*.json` output with inference results.

## Case schema and semantics

Each evaluation case is a flat dictionary rather than a nested benchmark object.

### `question`

The natural-language query posed to the system.

This is the only field consumed by `fill_report_eval_test_rag.py` when it calls inference.

### `answer_gold`

The manually written reference answer.

This field is not used by the script itself. It exists for later comparison by a human or an external evaluator.

### `context`

The evidence context returned by `InferenceService.answer(question)`.

In practice, this is the fully rendered prompt context assembled by `knowledge_inference/context_builder.py`, meaning it is a concatenation of evidence blocks formatted like:

- `[video=<name> | time=<range> | chunk=<id> | source=<source>]`

followed by chunk text.

### `rag_answer`

The final generated answer returned by the inference system for that question.

This answer has already passed through the inference pipeline's own:

- retrieval,
- reranking,
- context building,
- answer generation,
- answer verification,
- confidence-calibration logic,
- optional video-URL injection.

### `strong_llm_answer`

A placeholder field for an answer from a stronger non-RAG or external LLM baseline.

The current evaluation script never reads or writes it.

### `weak_llm_answer`

A placeholder field for an answer from a weaker baseline model.

The current evaluation script never reads or writes it.

## Dataset variants

## 1. `report_eval_test.json`

[`knowledge_system_evaluation/report_eval_test.json`](/home/gatv-projects/Desktop/project/knowledge_system_evaluation/report_eval_test.json) is a 118-case blank evaluation set.

Its questions are broad gameplay/advice questions such as:

- rune recommendations,
- itemization advice,
- matchup handling,
- champion strengths and weaknesses.

Before filling, all generated-answer fields are empty.

## 2. `report_eval_test_v2.json`

[`knowledge_system_evaluation/report_eval_test_v2.json`](/home/gatv-projects/Desktop/project/knowledge_system_evaluation/report_eval_test_v2.json) is a 136-case blank evaluation set that appears more tightly tied to specific named videos and local temporal spans.

Its questions are more explicitly grounded, for example:

- `"In "14 Actually Useful TIPS for PYKE", around 0-30, what are Pyke's main strengths and weaknesses?"`

This suggests a later evaluation design emphasizing:

- video-specific grounding,
- time-localized evidence,
- repeated question templates across multiple videos.

The file also contains repeated prompts more often than `report_eval_test.json`, which indicates it is being used to probe consistency across examples rather than only coverage across question types.

## 3. `merged_report_eval_test.json`

[`knowledge_system_evaluation/merged_report_eval_test.json`](/home/gatv-projects/Desktop/project/knowledge_system_evaluation/merged_report_eval_test.json) is a 254-case blank dataset.

Its size exactly matches:

- `118 + 136 = 254`

which strongly indicates it is a merged union of `report_eval_test.json` and `report_eval_test_v2.json`.

The filler script uses this merged file as its default input.

## 4. `report_eval_test_rag_smoke.json`

[`knowledge_system_evaluation/report_eval_test_rag_smoke.json`](/home/gatv-projects/Desktop/project/knowledge_system_evaluation/report_eval_test_rag_smoke.json) is a partially filled report.

Only the first three cases contain non-empty:

- `context`
- `rag_answer`

This matches a smoke-test workflow in which the batch process is run on only a small prefix of the dataset to verify correctness and throughput before a full run.

## 5. `report_eval_test_rag.json`

[`knowledge_system_evaluation/report_eval_test_rag.json`](/home/gatv-projects/Desktop/project/knowledge_system_evaluation/report_eval_test_rag.json) is the fully filled RAG-output version of `report_eval_test.json`.

All 118 cases have populated:

- `context`
- `rag_answer`

## 6. `report_eval_test_rag_merged.json`

[`knowledge_system_evaluation/report_eval_test_rag_merged.json`](/home/gatv-projects/Desktop/project/knowledge_system_evaluation/report_eval_test_rag_merged.json) is the fully filled RAG-output version of `merged_report_eval_test.json`.

All 254 cases have populated:

- `context`
- `rag_answer`

This is also the default output path used by the filler script.

## End-to-end control flow

The practical runtime flow of this phase is entirely defined by `run()` in [`knowledge_system_evaluation/fill_report_eval_test_rag.py`](/home/gatv-projects/Desktop/project/knowledge_system_evaluation/fill_report_eval_test_rag.py).

## 1. Import preparation

At module import time, the script:

- computes `PROJECT_ROOT` from the file path,
- inserts that root into `sys.path` if needed,
- imports `InferenceService` from `knowledge_inference`.

This is important because `knowledge_system_evaluation/` is not itself a Python package. The script makes the project root importable manually.

## 2. Default path selection

The script defines:

- `DEFAULT_INPUT = merged_report_eval_test.json`
- `DEFAULT_OUTPUT = report_eval_test_rag_merged.json`

So, unless overridden, the operational assumption is:

- take the merged blank evaluation file,
- generate a merged RAG-filled report.

## 3. Input loading

`_load_cases(path)`:

- opens the input file as UTF-8 JSON,
- requires the top-level object to be a list,
- raises `ValueError` otherwise.

Unlike `knowledge_inference/eval.py`, which accepts either a list or `{"cases": [...]}`, this script accepts only a raw list.

## 4. Service construction

`run(...)` creates one `InferenceService()` instance for the whole batch.

This matters operationally because the inference service caches loaded sanitized stores after first use, so repeated questions reuse the same warm process state instead of reloading stores per question.

## 5. Progress tracking

The script provides a terminal progress bar through:

- `_render_progress(current, total, width=32)`
- `_print_progress(current, total)`

The bar is purely cosmetic. It has no effect on evaluation state.

The printed format is:

- `[########------------------------] 1/254 (  0.4%)`

with carriage-return updates until the final line.

## 6. Optional truncation via `limit`

`run(...)` accepts `limit: int | None`.

It computes:

- `total = len(cases)` if `limit is None`
- otherwise:
  - `min(len(cases), max(0, limit))`

This means negative limits are coerced to zero rather than failing.

Operationally, this supports:

- smoke runs over only the first few cases,
- interrupted or partial report generation.

## 7. Per-case processing

For each case in the selected prefix:

1. create a shallow copy of the case with `item = dict(case)`,
2. read and trim `question`,
3. if `question` is empty:
   - write `rag_answer = ""`
   - write `context = ""`
   - append the item unchanged otherwise,
4. else call:
   - `result = service.answer(question)`
5. write:
   - `item["rag_answer"] = result.answer`
   - `item["context"] = result.context`
6. append the filled item to `result_cases`,
7. immediately write the partially completed result list to disk,
8. update the progress bar.

This means the script is deliberately incremental. It does not wait for the whole batch to finish before writing output.

## 8. Partial-write semantics

The immediate `_write_cases(output_path, result_cases)` call after every processed question is important.

It provides crash tolerance:

- if the run is interrupted, the output file still contains all completed results up to the interruption point.

This is especially relevant because each question invokes the full inference pipeline and may be slow.

## 9. Tail preservation when using `limit`

If `limit < len(cases)`, then after processing the prefix the script executes:

- `result_cases.extend(cases[total:])`

and writes the full list back to disk.

This behavior preserves the unprocessed remainder of the original file after the processed prefix.

Therefore, a smoke-test output still has the same number of rows as the input dataset. The difference is simply that only the first `N` objects have populated `rag_answer` and `context`.

That is exactly the pattern observed in `report_eval_test_rag_smoke.json`, where only the first three cases are filled.

## Command-line interface

`build_parser()` defines three arguments:

- `--input`
- `--output`
- `--limit`

The CLI description is:

- `"Copy report_eval_test.json and fill rag_answer/context via knowledge inference."`

The actual program entry point is `main()`, which resolves the paths and calls `run(...)`.

The normal command shape is therefore:

```bash
python3 -m knowledge_system_evaluation.fill_report_eval_test_rag --input <source.json> --output <dest.json> --limit <N>
```

or, from the file path directly:

```bash
python3 knowledge_system_evaluation/fill_report_eval_test_rag.py
```

because the script manually inserts `PROJECT_ROOT` into `sys.path`.

## Relationship to `knowledge_inference/`

This evaluation phase does not answer questions itself. It delegates all QA behavior to [`knowledge_inference/service.py`](/home/gatv-projects/Desktop/project/knowledge_inference/service.py) through:

- `InferenceService.answer(question)`

As a result, every generated `rag_answer` and `context` in this module inherits the full behavior of the inference phase:

- sanitized-only store loading,
- query analysis,
- parallel retrieval,
- reranking,
- context construction,
- answer generation,
- verification,
- confidence-based cautioning.

The evaluation module records the outputs of that pipeline but does not re-score or reinterpret them.

## What this module does not currently implement

It is equally important to state what is not present in the current code.

This module does not currently implement:

- automatic scoring of `rag_answer` against `answer_gold`,
- exact-match, F1, BLEU, ROUGE, or semantic-similarity metrics,
- pairwise comparison between `rag_answer`, `strong_llm_answer`, and `weak_llm_answer`,
- confidence aggregation,
- latency measurement,
- statistical summaries,
- plotting or chart generation,
- dataset validation beyond top-level JSON-list checking.

Any such analysis would need to be performed by another script or manually outside this directory.

## Architectural characterization

In the full system, `knowledge_system_evaluation/` is best understood as the final experiment-output layer.

Earlier phases do the following:

- `knowledge_extraction/` produces multimodal extraction artifacts,
- `knowledge_sanitization/` cleans them,
- `knowledge_build/` produces retrieval and graph assets,
- `knowledge_inference/` answers queries from those assets.

Then `knowledge_system_evaluation/` takes curated evaluation questions and materializes the inference system's answers and contexts into report files that can be inspected, compared, or scored later.

So the final phase is not an evaluator in the narrow algorithmic sense. It is a batch report-generation interface over the inference system, with stored benchmark datasets and derived RAG outputs.
