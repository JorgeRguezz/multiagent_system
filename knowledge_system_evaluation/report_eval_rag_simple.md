# Evaluation Report: `report_eval_test_rag_done_updated.json`

This report reviews only the outputs in `knowledge_system_evaluation/report_eval_test_rag_done_updated.json`.

## Scope

The file contains 118 evaluation items. Each item includes:

- `rag_answer`
- `strong_llm_answer`
- `weak_llm_answer`
- `answer_gold`

All three answer columns are populated for all 118 items, so this file supports a direct comparison.

## High-Level Conclusion

The three systems show a clear tradeoff:

- `rag_answer` is the most evidence-constrained system, but it is too conservative and refuses too often.
- `strong_llm_answer` is the most useful general-answer system and is the strongest overall in this file.
- `weak_llm_answer` is the weakest system because it frequently invents details, mixes up champion abilities, and sounds confident while being wrong.

If I had to rank them as final answer systems for this dataset:

1. `strong_llm_answer`
2. `rag_answer`
3. `weak_llm_answer`

## Summary Signals

- Coverage:
  - `rag_answer`: 118/118
  - `strong_llm_answer`: 118/118
  - `weak_llm_answer`: 118/118
- Average length:
  - `rag_answer`: 196.2 words
  - `strong_llm_answer`: 318.8 words
  - `weak_llm_answer`: 483.8 words
- Apology-style refusals:
  - `rag_answer`: 43
  - `strong_llm_answer`: 0
  - `weak_llm_answer`: 0
- Rough lexical overlap with `answer_gold`:
  - `rag_answer`: 0.152 average
  - `strong_llm_answer`: 0.265 average
  - `weak_llm_answer`: 0.185 average

This does not mean `strong_llm_answer` is more grounded. It means it more often resembles the target answer. The main tradeoff remains groundedness vs. helpfulness.

## System-by-System Judgment

### `rag_answer`

Strengths:

- Usually stays tied to the retrieved evidence.
- Avoids unsupported speculation more often than the other two systems.
- Is more trustworthy when the task is strictly “answer only from available evidence.”

Weaknesses:

- Refuses too often.
- Sometimes spends too much space on caveats and source framing instead of directly answering.
- Can underperform on user usefulness even when a plausible answer is available.

### `strong_llm_answer`

Strengths:

- Most consistently useful and complete.
- Usually gives a coherent, relevant answer even when the retrieval evidence is weak.
- Best overall match to `answer_gold` in this file.

Weaknesses:

- Often answers from general League knowledge rather than from the retrieved evidence.
- Can therefore look better than the RAG system while actually being less faithful to the source material.
- Occasionally misses the specific question and drifts into generic advice.

### `weak_llm_answer`

Strengths:

- Usually verbose and structured.
- Sometimes reaches a partially useful answer on broad gameplay questions.

Weaknesses:

- Frequent hallucinations.
- Wrong ability names, wrong itemization, and fabricated mechanics.
- Confidence is not a reliability signal here; many polished answers are factually poor.

## Specific Examples

## Example 1: `rag_answer` is safer but too conservative

**Question**  
`How do you handle tough matchups like Fiora or Irelia when playing Aatrox?`

**`rag_answer`**

> “I’m sorry, but the provided clips don’t contain any specific advice or examples about how an Aatrox player should deal with tough dueling champions like Fiora or Irelia... Without that information, I can’t give a concrete answer.”

**Why this matters**

This answer is faithful to the evidence constraint, but it is not very useful to a user. It tells us the system is prioritizing source fidelity over answer completeness.

**`strong_llm_answer`**

> “Handling tough matchups like Fiora or Irelia when playing Aatrox requires a combination of strategic gameplay, itemization, and understanding the strengths and weaknesses of both champions...”

It then gives concrete advice about wave management, cooldown windows, itemization, and using Aatrox Q range.

**Judgment**

This is the clearest example of the main tradeoff:

- `rag_answer` is more faithful.
- `strong_llm_answer` is more useful.

## Example 2: `strong_llm_answer` is usually the best final answer

**Question**  
`What are some key runes and item choices for playing Aatrox effectively in the top lane?`

**`strong_llm_answer`**

It gives a coherent build structure:

- `Precision` with `Conqueror`, `Triumph`, `Legend: Tenacity`, `Last Stand`
- `Resolve` with `Second Wind`, `Unflinching`
- Core items like `Goredrinker`, `Death's Dance`, `Black Cleaver`

**Why this matters**

This answer is clear, structured, and useful. It reads like a practical response a player could act on immediately. Across the file, this is the typical pattern for `strong_llm_answer`: it tends to produce the best standalone answer.

**Comparison**

The `rag_answer` for the same question is also relevant, but noisier and more awkwardly tied to extracted evidence. It includes odd or artifact-like details such as:

> “Blue-red Orb”  
> “Boots with blue gem”

That makes the response less polished and less trustworthy as a finished user-facing answer.

## Example 3: `weak_llm_answer` often hallucinates specifics

**Question**  
`How do you handle tough matchups like Fiora or Irelia when playing Aatrox?`

**`weak_llm_answer`**

This answer invents multiple details, including:

- `Q (Darkin’s Requiem)`
- `W (Unforgiven)`
- `E (Death’s Dance)`

These are not valid Aatrox ability names.

It also claims:

> “Aatrox is a bruiser, not a duelist.”

and recommends suspect or mismatched build logic in the same answer.

**Why this matters**

This is not a small naming slip. It shows the weak model can produce polished text that is structurally convincing but mechanically unreliable.

## Example 4: `weak_llm_answer` mixes up champion kits badly

**Question**  
`What strategies should a player consider when using Pyke's abilities during gameplay?`

**`weak_llm_answer`**

The answer says:

- `W (Bone Skewer) to hook enemies`
- `E (Ghostwalk)`
- `Q (Blood Rush) for burst damage`

These are wrong mappings for Pyke’s kit.

**`strong_llm_answer`** for the same question correctly identifies:

- `Bone Skewer (Q)`
- `Ghostwater Dive (W)`
- `Phantom Undertow (E)`
- `Death from Below (R)`

**Why this matters**

This is one of the strongest examples that `weak_llm_answer` is not just lower quality. It is systematically unreliable on champion-specific details.

## Example 5: `rag_answer` can be evidence-faithful even when it is less useful

**Question**  
`Why do you prefer to play Gwen in the jungle rather than the top lane?`

**`rag_answer`**

> “I don’t have any evidence that you prefer to play Gwen in the jungle instead of the top lane... Therefore I can’t answer why you would prefer jungle over top lane based on the supplied material.”

**`strong_llm_answer`**

The strong model gives a plausible explanation around jungle clear, ganking, avoiding counters, and team resource allocation.

**`weak_llm_answer`**

The weak model also gives an explanation, but it invents abilities such as:

- `Gwen’s Glaive`
- `Gwen’s Blade`
- `Gwen’s Fury`

**Judgment**

This is another clean illustration of the overall ranking:

- `rag_answer`: faithful but unhelpful
- `strong_llm_answer`: helpful and plausible
- `weak_llm_answer`: detailed but unreliable

## Example 6: `weak_llm_answer` hallucinates full mechanics, not just names

**Question**  
`What makes Lissandra's abilities unique, and how do they synergize with her overall gameplay?`

**`weak_llm_answer`**

It claims:

- `Q – Frostbite`
- `W – Iceborn Curse`
- `E – Frozen Tomb`
- `R – Glacial Path`

It also describes fake mechanics such as:

> “The slow is stackable”  
> “E – Frozen Tomb... grants a shield and a brief burst of movement speed”  
> “R – Glacial Path... freezes enemies in a line”

**`strong_llm_answer`**

The strong model is much more accurate here. It identifies:

- `Q - Ice Shard`
- `W - Ring of Frost`
- `E - Glacial Path`
- `R - Frozen Tomb`

and explains the control-mage synergy correctly.

**Why this matters**

This is a strong negative example for `weak_llm_answer`. The issue is not wording quality. The issue is fabricated game knowledge.

## Example 7: `strong_llm_answer` can outperform `rag_answer` on utility even when retrieval is weak

**Question**  
`What are the key rune setups I should consider when playing Rengar top, and how do they impact different matchups?`

**`rag_answer`**

> “I’m sorry, but none of the provided videos or transcripts discuss Rengar playing top-lane...”

**`strong_llm_answer`**

The strong model gives a coherent answer centered on:

- `Conqueror`
- `Triumph`
- `Legend: Alacrity`
- `Coup de Grace` / `Last Stand`
- secondary options like `Sudden Impact`, `Ravenous Hunter`, `Bone Plating`, `Second Wind`

It then maps these choices to tanks, poke lanes, burst lanes, and CC-heavy lanes.

**`weak_llm_answer`**

The weak model again produces a polished but dubious answer, including:

- `Predator` as a core top-lane setup
- claims like “gives you a 3-second stealth burst after a kill or assist”

**Judgment**

This example reinforces the same pattern:

- `rag_answer` is bounded by retrieval.
- `strong_llm_answer` is often the best final answer.
- `weak_llm_answer` is too error-prone to trust.

## Final Verdict

Based only on `knowledge_system_evaluation/report_eval_test_rag_done_updated.json`, the systems should be judged as follows:

### Best overall: `strong_llm_answer`

It is the strongest choice if the goal is to return the most useful answer to a user. It consistently gives complete, readable, actionable responses and most often resembles the gold answer.

### Best grounded system: `rag_answer`

It is the safest system if the requirement is strict evidence faithfulness. Its main weakness is over-abstention: it frequently declines to answer where a user would still expect practical guidance.

### Weakest system: `weak_llm_answer`

It is the least reliable system. Its responses are often long and well formatted, but the content frequently contains incorrect champion abilities, invented mechanics, and suspect item or rune advice. It should not be used as a final-answer system without significant improvement.

## Practical Recommendation

If this evaluation is intended to guide product decisions, the best direction is a hybrid:

- keep the grounding discipline of `rag_answer`
- reduce unnecessary refusals
- preserve the readability and completeness of `strong_llm_answer`
- avoid the unsupported fabrication behavior seen in `weak_llm_answer`

In short:

- `strong_llm_answer` is the best user-facing answerer
- `rag_answer` is the best retrieval-faithful answerer
- `weak_llm_answer` is not reliable enough yet
