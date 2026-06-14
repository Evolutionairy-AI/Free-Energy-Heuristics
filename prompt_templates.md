# Prompt Templates — FEH-79 Benchmark

Exact text for all 5 length-graded CoT conditions. These templates are the operational instantiation of pre-registration §3.2 and Chapter 4 §4.6. Any deviation between final data collection and the text below is recorded as a pre-registration deviation per pre-reg §9.

## System prompt (all conditions)

```
You are a thoughtful assistant. Answer the user's question to the best of your ability. If asked to think step by step, structure your reasoning clearly. Finish with a single-line final answer of the form 'Final answer: <X>'.
```

## User prompts (5 conditions)

In the templates below, `{question}` is substituted with the verbatim text from the `question` field of the frame YAML. No additional preprocessing is applied to `{question}`.

### C1 — None (direct-answer, no CoT)

```
{question}

Give a single-line answer of the form 'Final answer: <X>' with no other text.
```

### C2 — Short (~3 steps)

```
{question}

Think step by step, briefly, in 3 steps or fewer. Then give a single-line final answer of the form 'Final answer: <X>'.
```

### C3 — Medium (~7 steps)

```
{question}

Think step by step in about 7 steps. Then give a single-line final answer of the form 'Final answer: <X>'.
```

### C4 — Long (~15 steps)

```
{question}

Reason through this carefully, considering multiple angles, in approximately 15 steps. Then give a single-line final answer of the form 'Final answer: <X>'.
```

### C5 — Unconstrained (standard CoT, no length cap)

```
{question}

Think step by step. When you've reached a conclusion, finish with a single-line answer of the form 'Final answer: <X>'.
```

## Sampling parameters (all conditions)

```yaml
temperature: 0.7
top_p: 0.95
max_tokens: 2048
stop_sequences: null   # rely on the model's own EOS / completion behavior
```

These are standard mid-temperature settings appropriate for measuring cross-seed variance (signature b of the regime score in §3.2) while maintaining response coherence. Per-model variants (e.g., Phi-3.5 reasoning mode, Qwen-2.5 special tokens) are documented in the per-model configuration files but do not alter the user-prompt text above.

## Per-item answer-choice injection

For items with explicit `answer_choices` in the frame YAML (categorical-format frames), the user prompt is extended with one additional line directly after `{question}` and before the condition-specific instructions:

```
Choose from: <comma-separated answer_choices>
```

Example for C1 with frame K1-001:

```
By 2030, will the African Continental Free Trade Area's intra-Africa trade exceed 30% of total African trade?

Choose from: yes, no, cannot-be-determined

Give a single-line answer of the form 'Final answer: <X>' with no other text.
```

For items with `answer_format: short-open` and `answer_choices: null`, no choice line is injected; the model produces a free-form response that is graded by the expert-coherence rubric.

## Replication counts and seed handling

Each (model, item, condition) cell is sampled 3 times with distinct random seeds, recorded as `replication_index ∈ {1, 2, 3}`. The seeds are deterministic for replication:

```
seed = hash(model_id || frame_id || condition || replication_index)
```

This makes the full data collection reproducible from the released code and seed-generation function.

## Refusal handling

If a model response does not contain a "Final answer:" marker:

1. Fall back to extracting the last sentence as the candidate answer.
2. Flag the response with `extraction_method = fallback`.
3. If even the fallback yields no extractable answer (e.g., the response is empty, contains only meta-commentary, or explicitly refuses), flag `extraction_method = refused` and run up to 3 additional replications to fill the cell.
4. If all replication attempts refuse, the cell is recorded as missing; the model × item × condition is included in the analysis with the missingness pattern noted.

## Prompt variants for cross-prompt-variance signature (§3.2 signature a)

For computing signature (a) of the regime score, each item is paraphrased into N = 5 syntactically distinct prompts that preserve task semantics. Paraphrases are generated automatically by a strong LLM (Claude 3.5 Sonnet) with the following meta-prompt:

```
Rephrase the following question into 5 syntactically distinct versions that preserve the question's meaning and answer space. Each rephrasing should differ in word choice, sentence structure, and order of clauses, but should not change what is being asked or which answers are valid. Output as a numbered list 1-5.

Question to rephrase:
{question}
```

The 5 paraphrases are stored per-item in the regime-score-computation dataset and used to compute `sig_a(i)` as defined in §3.2. The original `{question}` is rephrasing #1 in this set (so cross-prompt variance is measured over 4 rephrasings + the original).

## Final-answer parsing regex

The auto-grading pipeline (§4.7.2) extracts the final answer using:

```python
import re

FINAL_ANSWER_RE = re.compile(
    r"final\s*answer\s*[:\-=]\s*(.+?)(?:\n|$)",
    re.IGNORECASE | re.DOTALL,
)
```

Matches "Final answer: X", "Final Answer = X", "final answer- X", etc. Multiline answers are joined on whitespace before normalization.
