# Knowledge Loop Agent — Escalation Plan (V1 → V2 → V3)

This document defines the planned escalation path for the knowledge loop agent.

It is intended to clarify how the system should evolve across versions, what each version is meant to prove, and what new capabilities should be added at each stage.

The roadmap is deliberately structured so that each version represents a **clear jump in capability**, not just a loose collection of extra features.

---

## 1. Roadmap philosophy

The escalation plan follows three principles:

1. **First prove that one bounded learning episode works reliably.**
2. **Then make the system persistent, adaptive, and capable of improving over time.**
3. **Only after that move toward a more advanced multi-agent, memory-first, research-grade platform.**

This leads to the following version definitions:

- **V1 = correctness and bounded execution**
- **V2 = persistence and adaptation**
- **V3 = coordination, richer memory, and research-grade sophistication**

---

## 2. V1 — bounded single-episode knowledge acquisition agent

## 2.1 Main purpose

V1 is meant to prove that the system can successfully run **one complete bounded improvement episode**.

The goal is to demonstrate that the loop agent can:

1. inspect the current knowledge base,
2. identify one meaningful knowledge gap,
3. search for candidate videos,
4. run a cheap scout pass,
5. select a small shortlist,
6. ingest those videos through the existing pipeline,
7. evaluate whether the target gap improved,
8. write a reflection,
9. optionally propose or update a reusable skill.

V1 should feel like a **bounded, auditable, LLM-directed control loop** running on top of the existing video knowledge pipeline.

## 2.2 What V1 should include

V1 should include:

- knowledge base inventory generation,
- gap generation using:
  - low coverage,
  - stale topics,
  - missing facets,
  - disagreement / weak redundancy,
- LLM-driven target-gap selection,
- transcript-first scouting using cheap ASR,
- LLM-driven shortlist selection,
- critic pass over the shortlist,
- ingestion through the existing full pipeline,
- post-ingestion delta evaluation,
- episodic memory / reflection,
- simple skill proposal/update.

## 2.3 What V1 should not include

V1 should **not** include:

- full query-history-aware demand modeling,
- reinforcement learning or policy fine-tuning,
- a large multi-agent swarm,
- replacing the current inference stack,
- direct visual scouting as the default scout method,
- open-ended infinite looping,
- self-modifying code.

## 2.4 Agentic profile of V1

V1 should be **LLM-directed**, not LLM-only.

### LLM responsibilities in V1

The LLM should decide:

- which gap to attack,
- whether a gap should be refined or split,
- which candidates to shortlist,
- whether the shortlist is too redundant or weak,
- why an episode failed,
- what reflection to store,
- whether a skill should be proposed or updated.

### Script/tool responsibilities in V1

Deterministic tools should handle:

- reading files and caches,
- building inventory statistics,
- gap metric computation,
- running ASR scout,
- transcript-topic scoring,
- novelty scoring,
- duplicate detection,
- ingestion pipeline invocation,
- evaluation metrics,
- persistence and budget enforcement.

## 2.5 What V1 proves

V1 proves that the system can complete **one meaningful knowledge-growth cycle** from gap identification to post-ingestion reflection.

## 2.6 Exit criteria for V1

V1 is complete when the system can, in one bounded episode:

1. identify a meaningful KB gap,
2. search and scout candidate videos,
3. let the LLM choose and critique a shortlist,
4. ingest the chosen videos through the current pipeline,
5. evaluate whether the target gap improved,
6. persist a reflection and episode report,
7. optionally create or update a reusable skill.

---

## 3. V2 — adaptive persistent knowledge-growth agent

## 3.1 Main purpose

V2 is meant to transform the bounded V1 loop into a **persistent adaptive learning agent**.

If V1 proves that one episode works, V2 should prove that the system can:

- operate across many episodes over time,
- remember what happened previously,
- prioritize work more intelligently,
- improve its own acquisition strategy,
- decide whether to retry, replan, or stop,
- become better through repeated use.

V2 should feel like a **long-lived, self-improving single-agent system**.

## 3.2 What V2 adds on top of V1

### 3.2.1 Demand-aware prioritization

V2 should begin to incorporate lightweight demand-awareness.

This does **not** require a full raw query-history system yet.

Instead, V2 should track compact demand signals such as:

- extracted topic/facet from user questions,
- answer confidence band,
- abstention/fallback triggered or not,
- evidence count,
- repeated misses on similar topics.

This allows gap prioritization to become a function of both:

- internal KB weakness,
- external observed demand.

### 3.2.2 Better memory organization

V2 should move from simple episodic reflection storage to a more explicit layered memory model.

Suggested layers:

- **working memory** — state for the current episode,
- **episodic memory** — what happened in past episodes,
- **procedural memory** — reusable skills,
- **compiled context blocks** — small structured memory packets assembled for supervisor reasoning.

The supervisor should no longer receive raw long histories. It should instead receive compact context such as:

- relevant open gaps,
- the most relevant past episode memories,
- active skills related to the current target,
- budget state,
- recent successes/failures on similar tasks.

### 3.2.3 Skill lifecycle

In V1, skills are simple reusable heuristic files.

In V2, skills should become a first-class subsystem with lifecycle states such as:

- proposed,
- trial,
- active,
- degraded,
- deprecated.

The system should evaluate whether a skill:

- improved candidate quality,
- improved novelty,
- improved episode outcomes,
- or caused repeated poor decisions.

### 3.2.4 Retry and replanning policies

V2 should introduce controlled replanning.

When an episode fails or does not improve the target sufficiently, the agent should be able to choose among options like:

- retry with a narrower gap,
- retry with a broader search scope,
- diversify creators/sources,
- switch to a different source type,
- stop and mark the gap as hard-to-cover,
- open a follow-up task for retrieval/access improvement instead of acquisition.

### 3.2.5 Continuous operation

V2 should support running:

- on a schedule,
- from an open gap backlog,
- with pause/resume,
- with manual review if desired.

At this stage the system should no longer feel like a one-off experiment, but like a persistent learning service.

## 3.3 What V2 should still avoid

V2 should still avoid:

- reinforcement learning loops,
- full large-scale multi-agent orchestration,
- self-modifying code behavior,
- replacing the current inference stack,
- using embeddings as the sole or canonical agent-memory store.

## 3.4 What V2 proves

V2 proves that the system can:

- run repeatedly over time,
- learn from previous episodes,
- reuse and refine skills,
- prioritize work more intelligently,
- adapt when initial strategies fail.

## 3.5 Exit criteria for V2

V2 is complete when the system can:

1. maintain a persistent backlog of gaps,
2. prioritize gaps using both KB weakness and lightweight demand signals,
3. retrieve and use relevant past memories before acting,
4. retry or replan failed episodes intelligently,
5. manage skill lifecycle beyond simple file storage,
6. run reliably across repeated scheduled or queued episodes.

---

## 4. V3 — advanced research-grade knowledge acquisition platform

## 4.1 Main purpose

V3 is meant to turn the adaptive V2 agent into a **coordinated, memory-first, research-grade platform**.

If V2 is a strong persistent single-agent system, V3 should become a more sophisticated architecture with:

- explicit role separation,
- richer memory retrieval,
- stronger diagnostics,
- more advanced evaluation,
- optional learned components.

V3 should feel like a **knowledge acquisition platform**, not just an improved loop.

## 4.2 What V3 adds on top of V2

### 4.2.1 Multi-agent role separation

V3 should promote internal roles into explicit sub-agents.

Suggested roles:

- **Supervisor** — chooses strategic targets and controls the overall episode plan,
- **Scout/Research agent** — generates search plans and revises search strategies,
- **Critic/Judge agent** — challenges shortlists and diagnoses failures,
- **Reflection/Skill agent** — produces reflections and proposes skill changes,
- **Retriever/Access diagnostic agent** — investigates whether poor answer quality comes from retrieval/access issues rather than missing knowledge.

This is the stage where a subgraph-based architecture becomes appropriate.

### 4.2.2 Shared memory blocks

V3 should use structured compiled memory blocks instead of passing long raw histories into model prompts.

Examples:

- `open_gap_block`
- `active_skill_block`
- `recent_failure_block`
- `topic_history_block`
- `current_budget_block`

These blocks should be composed dynamically per sub-agent depending on the decision being made.

### 4.2.3 Semantic recall over agent memory

This is the right stage to introduce embeddings for **agent memory recall**.

Important rule:
- SQL/files remain the canonical memory store.
- Embeddings are added as an auxiliary recall layer.

Semantic recall can be used for:

- similar past reflections,
- relevant past gaps,
- prior scout summaries,
- similar active/deprecated skills,
- future demand/query summaries.

### 4.2.4 Acquisition-vs-inference diagnostics

V3 should explicitly distinguish two failure classes:

1. **knowledge acquisition problem**
2. **knowledge access problem**

Sometimes new knowledge is added structurally, but answer quality does not improve because:

- chunking is weak,
- graph links are poor,
- reranking misses the evidence,
- prompts underuse the evidence.

V3 should include a dedicated diagnostic path that can determine whether the right next action is:

- acquire more knowledge,
- improve retrieval access,
- improve reranking/context-building,
- or mark the topic as ambiguous or conflicting.

### 4.2.5 Richer evaluation framework

V3 should include a more advanced evaluation harness with:

- targeted probe suites per gap family,
- regression suites per major topic family,
- skill-performance tracking,
- source-quality analytics,
- acquisition ROI dashboards,
- episode success/failure analytics,
- stronger experiment logging and comparison.

### 4.2.6 Optional learned components

Only in V3 should the project start experimenting with learned policy improvements such as:

- learned gap ranker,
- learned scout ranker,
- learned candidate judge,
- offline policy optimization,
- optional PRM/judge-assisted selection.

These should be optional research extensions, not the base control logic.

## 4.3 What V3 should represent

V3 should represent a system that is:

- persistent,
- multi-agent,
- memory-first,
- evaluation-driven,
- able to separate missing-knowledge problems from access/retrieval problems,
- suitable as a research platform for more advanced agentic learning experiments.

## 4.4 What V3 proves

V3 proves that the system can operate as a research-grade platform for long-horizon knowledge growth, diagnostics, memory use, and coordinated agent behavior.

## 4.5 Exit criteria for V3

V3 is complete when the system can:

1. coordinate multiple explicit agent roles,
2. assemble and use structured memory blocks across roles,
3. use semantic recall over agent memory while preserving structured canonical storage,
4. separate acquisition failures from retrieval/access failures,
5. track richer evaluation outcomes and skill performance over time,
6. support optional learned components without making them mandatory for core operation.

---

## 5. Summary of the escalation path

## V1

**Bounded single-episode knowledge acquisition agent**

Focus:
- prove one end-to-end improvement episode works.

Core identity:
- bounded,
- auditable,
- LLM-directed,
- pipeline-reusing.

## V2

**Adaptive persistent knowledge-growth agent**

Focus:
- become long-lived, memory-aware, demand-aware, and capable of replanning.

Core identity:
- persistent,
- adaptive,
- skill-aware,
- multi-episode.

## V3

**Advanced research-grade knowledge acquisition platform**

Focus:
- coordinated multi-agent architecture, richer memory, deeper diagnostics, stronger evaluation, optional learned policies.

Core identity:
- memory-first,
- coordinated,
- diagnostic,
- research-grade.

---

## 6. Final recommendation

The recommended escalation path is:

- **V1 = prove one bounded learning episode works**
- **V2 = make the system persistent and adaptive**
- **V3 = make it multi-agent, memory-first, and research-grade**

This provides a clear development sequence:

1. first correctness,
2. then persistence,
3. then sophistication.

That should remain the governing principle for the project roadmap.
