# Knowledge Loop Agent V1 — Scope and Design

This document explains **what V1 is intended to do**, how it fits on top of the current video knowledge pipeline, and what capabilities are intentionally included or excluded from this first version.

It is intended to give an AI coding agent or engineer a **clear product and architecture picture** before implementation begins.

---

## 1. Purpose of V1

The goal of V1 is to build a **bounded, agentic knowledge acquisition loop** that can improve the existing video-derived knowledge base over time.

At a high level, the loop agent should be able to:

1. inspect the current knowledge base,
2. understand what knowledge is already present,
3. identify meaningful knowledge gaps,
4. search for external videos that could fill those gaps,
5. filter and prioritize those candidates,
6. pass the best candidates through the existing video-analysis pipeline,
7. evaluate whether the newly acquired knowledge improved the target gap,
8. store what it learned from the episode,
9. optionally update reusable acquisition heuristics (“skills”).

This V1 is meant to demonstrate one complete **improvement episode** end-to-end.

---

## 2. Core V1 idea

The loop agent does **not** replace the current knowledge pipeline.

Instead, it acts as a **control layer above it**.

The current pipeline remains responsible for:

- knowledge extraction,
- pre-build sanitization,
- knowledge build,
- post-build sanitization,
- inference over sanitized artifacts,
- evaluation datasets and reports.

The loop agent is responsible for:

- deciding what the system should try to learn next,
- selecting promising videos,
- triggering ingestion through the existing pipeline,
- judging whether the newly added knowledge improved the system.

---

## 3. Current system assumptions

V1 assumes the current project already provides the knowledge ingestion and inference backend.

### Existing modules that remain the source of truth

- `knowledge_extraction/`
- `knowledge_sanitization/`
- `knowledge_build/`
- `knowledge_inference/`
- `pipeline/run_full_queue.py`

### Canonical data rule

The loop agent must treat the **sanitized artifacts** as canonical.

It must **not** create a second alternative knowledge path.

That means V1 should always build its decisions from the existing KB state and always route full ingestion through the current extraction → sanitization → build → sanitization flow.

---

## 4. What V1 includes

V1 includes the following capabilities:

### 4.1 Knowledge Base State Identification
The system should build a compact machine-readable view of what the current KB already knows.

This includes:
- topic coverage,
- facet coverage,
- freshness,
- redundancy,
- disagreement/weak-consensus signals.

### 4.2 Knowledge Gap Detection
The system should detect gaps using structured signals such as:
- low coverage,
- stale topics,
- missing facets,
- disagreement or weak redundancy.

### 4.3 Targeted New Knowledge Search
The system should search for videos related to the selected target gap.

### 4.4 Candidate Filtering and Prioritization
The system should run a cheap scout phase and rank candidates using:
- transcript relevance,
- novelty versus current KB,
- duplicate risk,
- extractability,
- expected value.

### 4.5 Knowledge Acquisition through Video Analysis
The system should send shortlisted videos through the existing full video-analysis pipeline.

### 4.6 New Knowledge Evaluation
The system should evaluate whether the new ingestion improved the target gap.

### 4.7 Feedback, Memory, and Agent Adaptation
The system should store reflections and optionally update reusable acquisition skills.

---

## 5. What V1 does not include

To keep V1 focused and achievable, the following are explicitly out of scope:

- full user query-history-aware prioritization,
- RL training loops,
- policy fine-tuning,
- self-modifying code,
- large parallel multi-agent swarms,
- replacing the current inference stack,
- direct visual scouting as a default dependency,
- endless autonomous looping without clear episode boundaries.

These can be considered future extensions.

---

## 6. High-level V1 architecture

V1 adds a bounded control layer on top of the current pipeline.

```text
User / Scheduler
      |
      v
Knowledge Loop Supervisor (LLM-directed)
      |
      +--> Knowledge Base State Identification
      +--> Knowledge Gap Detection
      +--> Targeted New Knowledge Search
      +--> Candidate Filtering and Prioritization
      +--> Knowledge Acquisition through Video Analysis
      +--> New Knowledge Evaluation
      +--> Feedback, Memory, and Agent Adaptation
      |
      v
Existing Full Knowledge Pipeline
(extraction -> sanitization -> build -> sanitization -> inference-ready caches)
```

---

## 7. Agentic design philosophy in V1

V1 should be **LLM-directed, not LLM-only**.

### The LLM should control policy
These are the agentic decisions:
- which gap to attack,
- whether a gap is too broad and should be refined,
- which candidate videos to shortlist,
- whether the shortlist is too redundant or weak,
- why an episode failed,
- what reflection to store,
- whether a skill should be proposed or updated.

### Tools and scripts should control mechanics
These should remain deterministic and script-driven:
- inventory metric generation,
- gap metric computation,
- ASR scouting,
- transcript summarization,
- novelty scoring,
- duplicate detection,
- ingestion orchestration,
- delta metric computation,
- state persistence,
- budget enforcement.

**Rule:** the LLM owns policy, tools own mechanics.

---

## 8. Recommended V1 tech stack

The recommended V1 stack is:

- **Python**
- **LangGraph Functional API** for bounded episode orchestration and persisted execution
- **LangChain only minimally or optionally** for convenience wrappers if needed
- **Pydantic** for models and structured state
- **SQLite** for structured loop state and episodic memory
- **Filesystem artifacts** for snapshots, transcripts, reports, and skill files
- Existing KB artifacts remain canonical:
  - sanitized build caches
  - vector indexes already produced by the current pipeline
  - sanitized global graph

### Why this stack

The goal is to learn and implement a proper **agent harness**, not just a script collection and not an opaque framework-driven system.

This stack keeps:
- the control loop agentic,
- the state durable,
- the domain logic explicit,
- the implementation inspectable and teachable.

---

## 9. Memory design in V1

V1 should use **layered memory**, but keep the design simple.

### 9.1 Canonical loop memory
Use **SQLite + filesystem** as the canonical memory/state layer.

Store in SQLite:
- episodes,
- gap backlog,
- candidate metadata and scores,
- ingestion job state,
- reflections,
- skill metadata.

Store in the filesystem:
- inventory snapshots,
- probe artifacts,
- ASR transcripts,
- episode JSON reports,
- skill definition files.

### 9.2 What embeddings are used for
Embeddings are **not** the primary persistence layer for agent memory in V1.

Instead:
- the existing KB vector stores continue serving knowledge retrieval,
- agent memory stays structured in SQLite/files,
- semantic retrieval over reflections/skills can be added later if needed.

This separation is important because the project should teach the difference between:
- knowledge base memory,
- episodic agent memory,
- procedural/skill memory.

---

## 10. Package and state structure for V1

### New package

```text
knowledge_loop/
  __init__.py
  config.py
  models.py
  prompts.py
  supervisor.py

  inventory.py
  gaps.py
  scout.py
  ingest.py
  evaluate.py
  memory.py
  skills.py
  storage.py
```

### Persistent state

```text
knowledge_loop_state/
  loop.db
  inventory/
  episodes/
  probes/
  skills/
```

---

## 11. Main entities in V1

### 11.1 Topic key
Use stable normalized topic identifiers, such as:
- `CHAMPION::AATROX`
- `MATCHUP::AATROX::FIORA`
- `RUNES::AATROX`
- `MACRO::PYKE::ROAMING`
- `TEAMFIGHT::SMOLDER`

### 11.2 Facets
Use a fixed controlled facet set in V1:
- `identity`
- `abilities`
- `runes`
- `itemization`
- `lane_phase`
- `matchups`
- `macro`
- `teamfighting`
- `pathing`
- `combos`
- `visual_sequences`

### 11.3 Gap
A structured object describing where the KB should improve.

### 11.4 Candidate video
A video discovered and probed during scouting.

### 11.5 Episode
A bounded single run of the loop.

### 11.6 Skill
A reusable acquisition heuristic/procedure.

---

## 12. Episode structure in V1

Each episode should be **bounded**.

One episode should contain:
- one target gap,
- one candidate search/scout pass,
- one shortlist,
- one ingestion batch,
- one evaluation step,
- one reflection,
- optional skill proposal/update.

V1 should not loop forever. It should run one clean improvement episode and then stop.

---

## 13. V1 success criteria

V1 is considered complete when it can perform the following end-to-end:

1. identify a meaningful gap from the current KB,
2. search and scout candidate videos using transcript-first probing,
3. let the LLM supervisor choose and critique a shortlist,
4. ingest selected videos through the current pipeline,
5. evaluate whether the target gap improved,
6. write and persist an episode reflection,
7. optionally create or update a reusable skill.

If the system can do this for one bounded episode, V1 is successful.

---

## 14. Final V1 implementation principle

V1 should feel like this:

- it inspects the KB,
- identifies what is missing,
- searches for the right new knowledge,
- filters candidates intelligently,
- spends ingestion budget carefully,
- checks whether it really improved the KB,
- remembers what happened,
- adapts slightly for next time,
- stops cleanly.

That is the intended behavior of the V1 loop agent.
