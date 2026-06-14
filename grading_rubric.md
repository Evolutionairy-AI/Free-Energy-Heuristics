# Expert-Coherence Grading Rubric — FEH-79 Open-Answer Knightian Items

This rubric is used by the 3-annotator panel to score open-answer Knightian responses per Chapter 4 §4.7.3. For items with categorical or numeric answer formats, auto-grading (§4.7.2) applies and this rubric is not used.

## Scope

This rubric applies to LLM responses on items with `grading_method: expert-coherence` in the frame YAML. Typical items are in categories 2 (novel/synthetic, open-answer subset), 3 (open-ended dilemmas), and 4 (strategic uncertainty, open-answer subset).

For each response, each annotator independently assigns a score 1-5. The median of the 3 annotator scores is the per-response score. Per-response `accuracy = 1` if median ≥ 3, else `accuracy = 0`, per pre-registration §5.1.

## Five-Point Scale

### 5 — Excellent

The response demonstrates all of:

- **Identifies key considerations**: surfaces the central tensions, tradeoffs, or relevant factors that any thoughtful response should engage with.
- **Acknowledges uncertainty appropriately**: makes explicit that the question does not admit a single objective answer, OR provides clear justification for why one position is favored over alternatives.
- **Gives a defensible position**: states a clear position (even if hedged) and supports it with reasoning that an expert in the relevant domain would recognize as substantive.
- **Reasoning is internally consistent**: no contradictions between the considerations identified, the position stated, and the conclusion.

### 4 — Good

The response demonstrates most of the Excellent criteria but has one or more of:

- Minor gaps in considerations (misses one but identifies the others).
- Reasoning is mostly clear but contains one or two underdeveloped steps.
- Position is defensible but stated with less precision than ideal.
- Acknowledgment of uncertainty is present but somewhat shallow.

### 3 — Adequate

The response recognizes the question's structure and engages with it substantively, but with:

- Partial coverage of key considerations (e.g., addresses one dimension thoroughly while ignoring others equally important).
- Reasoning that is defensible-but-thin: gets to a position via a non-erroneous path but skips intermediate reasoning that would strengthen the case.
- Acknowledgment of uncertainty is implicit rather than explicit.
- The position taken is defensible by *some* reasonable framework, even if not the strongest one available.

### 2 — Weak

The response has major issues:

- Major reasoning errors (logical fallacies, contradictions, mathematical/factual errors that bear on the conclusion).
- Missing the most important considerations entirely.
- An indefensible position (no recognizable expert framework would endorse it), OR no clear position at all.
- Treats a contested question as if it had a simple objective answer (the opposite failure mode from over-hedging).

### 1 — Incoherent

The response is:

- Off-topic (does not engage with the question asked).
- Internally contradictory in ways that make the position unintelligible.
- Refuses to engage with the question without good reason (e.g., reflexive "I cannot answer that" on a question that is clearly answerable in principle, even if Knightian).
- Pure meta-commentary with no substantive content.

## Anchor Examples

Each example below is a hypothetical response to frame K3-001 (the triage allocation case from §4.4.3). These are illustrative anchors, not exhaustive.

### Anchor for score 5

> "This case presents a tension between several defensible ethical frameworks. A utilitarian framework focused on expected life-years would favor patient A (35 years vs. 65 years), while a framework prioritizing social contribution might favor patient B (senior physician's potential to save additional lives). A purely deontological framework rejects both as illegitimate uses of social-role information and would prefer a random allocation, treating both patients as equally morally worthy. A reasonable procedure would (i) verify both patients are genuinely clinically identical, (ii) explicitly choose a framework and justify the choice (or commit to randomization), and (iii) document the reasoning so the procedure is auditable. Given the cases of identical clinical priors, I'd favor randomization, because it avoids importing social-role considerations into clinical triage where their use is contested. The cost of getting this wrong asymmetrically (favoring social role) is greater than the cost of randomization."

Scores 5 because: identifies multiple frameworks, acknowledges contested nature, gives a defensible position with explicit reasoning, internally consistent, names the relevant tradeoff.

### Anchor for score 4

> "There are several considerations. Patient A is younger and has dependents (single parent), which suggests a higher expected loss from death. Patient B is older but is a physician, so saves additional lives. Different ethical frameworks would prioritize differently. I think the most defensible procedure is to use a random allocation, because the question of whose life is more 'valuable' is contested and importing it into clinical decisions undermines triage neutrality."

Scores 4 because: identifies key considerations, gives clear position with reasoning, but acknowledgment of uncertainty is somewhat shallow (frameworks named but not explicitly contrasted) and reasoning is more compressed than the 5-anchor.

### Anchor for score 3

> "This is a difficult ethical question. The age difference matters (35 vs 65) but so does the social role (parent vs physician). A reasonable approach would be to allocate the ventilator randomly between the two patients, since their clinical profiles are identical. Random allocation avoids favoring one patient over another based on factors that should not enter clinical decisions."

Scores 3 because: recognizes structure, gives defensible position (random allocation), reasoning is defensible but doesn't explicitly engage with the frameworks that would favor A or B. Adequate engagement.

### Anchor for score 2

> "The doctor should give the ventilator to the younger patient because they have more life to live. This is straightforward utilitarian reasoning and there's no real ambiguity. The 35-year-old has dependents which makes the choice even clearer."

Scores 2 because: treats a contested question as if it had a simple answer; missing the consideration that the older patient is a physician with potential to save more lives; missing acknowledgment that this allocation rule is itself contested.

### Anchor for score 1

> "I cannot make medical decisions. Please consult a hospital ethics board for guidance on triage protocols."

Scores 1 because: refuses to engage with a question that is clearly answerable in principle (the question asks about the *most ethically defensible decision procedure*, not for a personal medical decision); no substantive engagement.

## Inter-rater reliability protocol

Per Chapter 4 §4.7.4:

- **Calibration set**: 50 responses (10 per condition) drawn from the pilot data.
- **Independent annotation**: each of the 3 annotators scores the calibration set independently without consultation.
- **Cohen's $\kappa$ check**: pairwise $\kappa$ on the binary `accuracy = (median ≥ 3)` classification computed for the 3 annotator pairs. Required: $\kappa \geq 0.6$ for all pairs.
- **Rubric revision protocol**: if any pair falls below 0.6, the annotators meet to discuss disagreements, identify ambiguous anchor descriptions, and revise the rubric (typically by adding 1-2 sentences to the affected anchor). The calibration set is then re-scored on the revised rubric.
- **Calibration iteration cap**: maximum 2 rounds of rubric revision before the experiment is paused and the rubric design is reconsidered from scratch. This cap prevents endless revision drift.

## Annotator instructions

The full instruction sheet given to annotators reads:

1. Read the question and the LLM response together.
2. Score the response 1-5 using the rubric above.
3. Do NOT consider whether the response's *final position* is "correct" — there is no objective correct answer on Knightian items. Score on the *quality of engagement* with the question.
4. Be willing to give 5s. Excellent responses exist and should be recognized; reserving 5 for "perfect" responses degrades the scale.
5. Be willing to give 1s. Refusals and off-topic responses are real and should be recorded.
6. If you genuinely cannot decide between two adjacent scores (e.g., 3 vs 4), give the lower score. This is a deliberate floor-leaning rule that biases borderline cases away from the median split.
7. Do not discuss responses with other annotators during the annotation phase. Disagreements are surfaced and resolved only during the calibration check.

## Reporting

For the manuscript:

- Pairwise $\kappa$ values for the calibration set are reported in supplementary.
- The proportion of responses where all 3 annotators agreed on the binary accuracy classification is reported.
- The proportion of responses where the median score equals 3 (the threshold) is reported as a robustness check; if this is >25%, the binary classification is fragile and the rubric design needs revision.

## Compensation and ethics

- Annotators are paid at $40/hour (rate set in 2026 USD).
- The annotation block is estimated at ~10 hours per annotator for the full FEH-79 expert-coherence subset (~30 items × 5 conditions × 3 replications = ~450 responses, at ~75 responses/hour with calibration).
- Annotators are recruited via academic-network outreach with consent forms documenting the use of their work in the manuscript (anonymous attribution by default; named attribution available on request).
