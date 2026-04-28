# Knowledge Loop Agent V1 — Implementation Stages

This document breaks the V1 implementation into the main stages that should be developed.

Each stage explains:
- the goal of the stage,
- what it should produce,
- what inputs it depends on,
- how it can be implemented in practice,
- important design notes for the AI coding agent.

These stages are based on the V1 design and should be treated as the main implementation workstreams.

---

# 1. Knowledge Base State Identification

## Goal

The goal of this stage is to build a **compact, machine-readable representation of the current knowledge base state**.

The loop agent cannot decide what to learn next unless it first knows:
- what topics are already present,
- how well those topics are covered,
- how fresh that coverage is,
- whether there are multiple supporting sources,
- whether the knowledge appears narrow or weak.

This stage is the system’s “self-awareness” layer for the KB.

## Inputs

This stage should read from the existing canonical KB artifacts, especially:
- sanitized build caches,
- sanitized global graph,
- optionally prior evaluation summaries,
- optionally prior episode summaries.

## Outputs

It should produce one or more inventory snapshots, for example:
- `knowledge_loop_state/inventory/latest_inventory.json`
- optional historical timestamped snapshots.

The inventory should summarize the KB in terms of:
- `topic_key`
- `facet`
- `video_count`
- `chunk_count`
- `source_redundancy`
- `freshness_days`
- `coverage_score`
- `disagreement_score`

## How to implement it

### Step 1 — Read the canonical KB sources
Create a module such as `knowledge_loop/inventory.py` that loads:
- sanitized build caches,
- sanitized graphs,
- supporting metadata.

### Step 2 — Define stable topic and facet buckets
Introduce a controlled taxonomy for V1.

Examples:
- `CHAMPION::AATROX`
- `MATCHUP::AATROX::FIORA`
- `MACRO::PYKE::ROAMING`

Facets should come from a fixed list, such as:
- `runes`
- `itemization`
- `lane_phase`
- `matchups`
- `macro`
- `teamfighting`
- `pathing`
- `combos`

### Step 3 — Map KB artifacts into topic/facet coverage
Write deterministic logic that maps:
- chunk text,
- graph entities/relations,
- available metadata,
into one or more topic/facet buckets.

This does not need to be perfect in V1. It only needs to be consistent enough to drive gap generation.

### Step 4 — Compute summary metrics
For each topic/facet pair, compute:
- how many videos cover it,
- how many chunks support it,
- how recently it was refreshed,
- how many distinct supporting sources exist,
- whether the support appears narrow or inconsistent.

### Step 5 — Persist inventory snapshot
Save the resulting inventory to disk for later reuse by the supervisor and gap generator.

## Design notes

- This stage should be **deterministic** and primarily script-based.
- It should not rely on LLM reasoning in V1.
- It should use the **existing KB as the source of truth**, not create a second representation.

---

# 2. Knowledge Gap Detection

## Goal

The goal of this stage is to identify **where the knowledge base should improve next**.

A gap is any topic/facet area where the KB is currently weak.

In V1, gaps should be detected primarily through structured signals, not through full query-history reasoning.

## Inputs

This stage should consume:
- the inventory snapshot from Stage 1,
- optionally prior episode outcomes,
- optionally fixed project priorities.

## Outputs

It should generate a ranked set of candidate gaps, each with:
- `gap_id`
- `topic_key`
- `facet`
- gap metrics
- `priority`
- `status`
- short evidence summary

These should be written both to:
- SQLite (`gaps` table), and/or
- a JSON export for debugging.

## How to implement it

### Step 1 — Define gap signals
V1 should use these four main signals:
- **low coverage**
- **stale coverage**
- **missing facet**
- **disagreement / weak redundancy**

### Step 2 — Compute gap metrics from the inventory
For each topic/facet pair, compute values such as:
- `coverage_score`
- `freshness_score`
- `missing_facet_score`
- `disagreement_score`

### Step 3 — Score and rank gaps
Use a simple deterministic formula, for example:

```python
priority = (
    0.40 * (1 - coverage_score) +
    0.25 * freshness_score +
    0.20 * disagreement_score +
    0.15 * missing_facet_score
)
```

### Step 4 — Mark statuses
Each gap should have a lifecycle state, such as:
- `open`
- `active`
- `partially_covered`
- `covered`
- `blocked`

### Step 5 — Pass top-N gaps to the supervisor
The LLM supervisor should not see every possible topic. It should receive only the best structured candidates and choose among them.

## Design notes

- Gap generation should remain mostly **Python-driven** in V1.
- The LLM should select/refine the target gap, but not perform the raw metric computation.
- This stage is where the first explicit link between KB state and loop policy is created.

---

# 3. Targeted New Knowledge Search

## Goal

The goal of this stage is to find **external videos that are likely to fill a selected knowledge gap**.

This stage should be focused and targeted. It is not generic browsing.

## Inputs

This stage should consume:
- one selected target gap,
- active skills that may influence search behavior,
- recent episode memories if relevant.

## Outputs

It should produce a raw candidate pool of videos, including:
- title,
- URL,
- description,
- channel,
- publish date,
- duration,
- metadata needed for the scout stage.

## How to implement it

### Step 1 — Convert target gap into search intents
Translate the selected gap into a few structured search queries.

Examples:
- `MATCHUP::AATROX::FIORA` → “Aatrox vs Fiora guide”, “How to play Aatrox into Fiora”
- `MACRO::PYKE::ROAMING` → “Pyke roaming guide”, “Pyke macro after level 6”

### Step 2 — Allow the supervisor to refine search direction
The LLM supervisor can decide whether the search should emphasize:
- educational guides,
- VOD reviews,
- matchup-specific breakdowns,
- recent patch-focused content,
- diverse creators.

### Step 3 — Retrieve a raw pool of candidates
Use the chosen search backend/API to gather a candidate set.

The output should be intentionally broad enough for later filtering.

### Step 4 — Persist raw candidate metadata
Store the raw candidates for traceability before they are filtered.

## Design notes

- Search should be broad enough to support selection, but still targeted to one gap.
- The system should not move to full ingestion from raw search results directly.
- This stage should prepare the input for transcript-first scouting.

---

# 4. Candidate Filtering and Prioritization

## Goal

The goal of this stage is to turn a broad raw candidate pool into a **small, high-value shortlist**.

This is the main cheap-filter stage before spending the cost of full ingestion.

## Inputs

This stage should consume:
- raw search candidates,
- the selected target gap,
- the current inventory / KB state,
- active skills,
- scouting configuration.

## Outputs

It should produce:
- candidate probe artifacts,
- scored candidate summaries,
- a ranked shortlist recommendation for the supervisor.

## How to implement it

### Step 1 — Run cheap transcript-first scout
For each candidate:
- fetch metadata,
- run cheap/faster ASR,
- store transcript,
- create a compact transcript summary.

### Step 2 — Compute relevance
Estimate how strongly the transcript matches the target gap.

### Step 3 — Compute novelty
Compare the transcript summary or extracted phrases against the current KB state.

The goal is to answer:
- is this likely to add new knowledge,
- or is it just repeating what already exists?

### Step 4 — Compute duplicate risk
Detect whether the candidate is likely too close to:
- already ingested videos,
- already dominant creators,
- other currently shortlisted candidates.

### Step 5 — Compute extractability
Estimate whether the candidate is likely to be well handled by the current pipeline.

Simple features may include:
- transcript quality,
- duration,
- guide-like structure,
- spoken explanatory content.

### Step 6 — Build compact candidate summaries for the LLM
The supervisor should not receive full transcripts.
It should receive structured summaries such as:
- title,
- transcript summary,
- relevance score,
- novelty score,
- duplicate risk,
- extractability,
- “why it might help”.

### Step 7 — Supervisor shortlist + critic pass
The supervisor chooses the best candidates.
The critic then reviews the shortlist and may request one revision.

## Design notes

- This stage is partly tool-driven and partly LLM-directed.
- ASR-first scouting is the default design in V1.
- The hard shortlist size should stay small, ideally 1–3 videos.

---

# 5. Knowledge Acquisition through Video Analysis

## Goal

The goal of this stage is to turn shortlisted candidate videos into **real new KB artifacts** by passing them through the current video-analysis pipeline.

This is the only stage where candidate knowledge becomes part of the actual system knowledge base.

## Inputs

This stage should consume:
- the final shortlisted candidates,
- the local/remote video source paths,
- episode metadata.

## Outputs

It should produce:
- pipeline outputs for the new videos,
- sanitized build caches,
- updated global graph/inference-ready artifacts,
- ingestion reports tied back to the episode and candidate IDs.

## How to implement it

### Step 1 — Create a thin ingest adapter
Build `knowledge_loop/ingest.py` as a thin wrapper.

Its job is to:
- register/download shortlisted videos,
- place them into the correct queue or handoff location,
- call the existing full pipeline,
- track job status.

### Step 2 — Reuse the existing ingestion backend exactly
Do not create a second ingestion path.

Always route through the current sequence:
- extraction,
- pre-build sanitization,
- build,
- post-build sanitization.

### Step 3 — Track candidate-to-output mapping
The loop agent must know which pipeline outputs came from which selected candidate.

### Step 4 — Persist ingest result metadata
Store job status, output paths, and failure reasons.

## Design notes

- This stage should stay mostly script-driven.
- The loop agent decides *what* to ingest, but the pipeline decides *how* knowledge is extracted and built.
- This stage must preserve the rule that sanitized artifacts remain canonical.

---

# 6. New Knowledge Evaluation

## Goal

The goal of this stage is to determine whether the newly ingested videos **actually improved the selected target gap**.

Adding new data is not enough; the system must measure whether the episode produced meaningful improvement.

## Inputs

This stage should consume:
- the target gap,
- the pre-ingestion inventory state,
- the post-ingestion KB state,
- new pipeline outputs,
- evaluation configuration.

## Outputs

It should produce:
- targeted gap probe results,
- regression results,
- structural delta metrics,
- one summarized acceptance/rejection outcome.

## How to implement it

### Step 1 — Run targeted probe evaluation
Create or maintain a small probe set for the selected gap.

Examples for `MACRO::PYKE::ROAMING`:
- When should Pyke leave lane after level 6?
- What makes a roam safe or unsafe?
- What common mistake is described?

These probes should be run after ingestion to test whether the KB is now better equipped to answer them.

### Step 2 — Run light regression evaluation
Use a small fixed question set to ensure the broader system has not degraded significantly.

### Step 3 — Run structural delta analysis
Compare before/after state for the target topic:
- more chunks?
- more supporting videos?
- more source redundancy?
- fresher coverage?
- reduced disagreement?

### Step 4 — Decide outcome
The episode should be marked successful if:
- targeted probe performance improves enough,
- regression does not significantly worsen,
- structural delta is positive.

### Step 5 — Diagnose ambiguous outcomes
If structural coverage improved but answer quality/confidence did not, do not immediately assume acquisition failure.

Instead, mark possible causes such as:
- retrieval-access issue,
- chunking/reranking issue,
- poorly specified gap.

## Design notes

- This stage is essential because it closes the learning loop.
- Without it, the system would only collect videos, not actually know whether it improved.
- In V1, evaluation should remain explicit and bounded.

---

# 7. Feedback, Memory, and Agent Adaptation

## Goal

The goal of this stage is to make the system **learn from past episodes**, even if only in a lightweight way in V1.

The agent should remember:
- what it tried,
- what worked,
- what failed,
- what heuristics appear reusable.

## Inputs

This stage should consume:
- the full episode record,
- ingest results,
- evaluation outcome,
- related prior memories,
- active skill definitions.

## Outputs

It should produce:
- an episodic reflection,
- one or more memory entries,
- optionally a proposed skill update,
- updated skill metadata if validated.

## How to implement it

### Step 1 — Generate an episode reflection
Use the LLM to summarize:
- what the target gap was,
- what candidates were chosen,
- whether the episode improved the KB,
- what likely worked,
- what should change next time.

### Step 2 — Store episodic memory
Persist short memory entries such as:
- “Transcript-rich educational guides improved macro gaps better than montage-style videos.”
- “Using two videos from the same creator reduced novelty for matchup gaps.”

These memories should be queryable later by topic or scope.

### Step 3 — Propose or update skills
A skill is a reusable acquisition procedure.

Example:
- trigger: `facet = macro`
- procedure: prefer transcript-rich educational videos 6–20 minutes long
- validation: at least one shortlisted candidate must exceed a transcript-topic threshold

### Step 4 — Validate before activation
In V1, skill proposals should not be auto-activated blindly.

A skill should only become active when:
- it has appeared in repeated successful episodes,
- recent evidence does not strongly contradict it.

### Step 5 — Use memories and skills in future episodes
When the next episode starts, the supervisor should receive a small number of relevant memories and active skills.

## Design notes

- This stage gives the loop continuity across episodes.
- V1 adaptation should remain lightweight and auditable.
- Skills in V1 are reusable procedures, not code synthesis and not model fine-tuning.

---

# Final implementation note

These seven stages should be developed so that, together, they produce one complete bounded V1 episode:

1. understand current KB state,
2. detect a useful gap,
3. search for relevant new knowledge,
4. filter and prioritize candidate sources,
5. acquire new knowledge through the existing video-analysis pipeline,
6. evaluate whether the target gap improved,
7. remember what happened and adapt slightly for the next episode.

That is the intended implementation path for V1.
