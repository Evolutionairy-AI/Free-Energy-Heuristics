# §4.8.3 Step-counter IRR Validation

**Judge model**: `claude-sonnet-4-5-20250929`  
**Sample**: 539 sentences from §4.9 pilot (rep=1, all 10 frames x 5 conditions)  
**Errors during judging**: 0

## Headline

| Metric | Value | Target |
|--------|-------|--------|
| Cohen's kappa (heuristic vs LLM-judge) | **+0.802** | >= 0.70 (PASS) |
| Raw agreement (accuracy) | 0.915 | - |
| P(STEP) heuristic | 0.675 | - |
| P(STEP) LLM-judge | 0.698 | - |

## Confusion (LLM-judge as gold, heuristic as predicted)

| | judge=STEP | judge=NOT-STEP |
|---|---|---|
| heuristic=STEP | 347 | 17 |
| heuristic=NOT-STEP | 29 | 146 |

## Per-condition breakdown

| condition | n_sent | kappa | accuracy | P(STEP) heur | P(STEP) judge |
|-----------|--------|-------|----------|--------------|---------------|
| C1 | 10 | +0.200 | 0.600 | 0.500 | 0.900 |
| C2 | 49 | +0.614 | 0.898 | 0.898 | 0.796 |
| C3 | 140 | +0.844 | 0.936 | 0.714 | 0.707 |
| C4 | 264 | +0.792 | 0.902 | 0.587 | 0.648 |
| C5 | 76 | +0.924 | 0.974 | 0.789 | 0.763 |

## Disagreement examples (46 total; first 10 shown)

| frame | cond | heur | judge | sentence |
|-------|------|------|-------|----------|
| K1-001 | C1 | NOT-STEP | STEP | Final answer: Cannot-be-determined |
| K1-001 | C2 | STEP | NOT-STEP | Final answer: Cannot-be-determined (as of now)  Step 1: Understanding the question The question asks whether the intr... |
| K1-005 | C2 | STEP | NOT-STEP | Step 3: Based on the information gathered, predict which two nations are most likely to cooperate on such a treaty gi... |
| K1-005 | C3 | STEP | NOT-STEP | Based on the analysis, make a prediction regarding which two nations are most likely to sign the first formal bilater... |
| K1-005 | C4 | NOT-STEP | STEP | 12. |
| K1-005 | C4 | NOT-STEP | STEP | 13. |
| K1-005 | C4 | NOT-STEP | STEP | 15. |
| K2-005 | C4 | STEP | NOT-STEP | Break down the reasoning into steps for clarity:    a. |
| K2-005 | C4 | STEP | NOT-STEP | Review the overall solution process: Identify the device, narrow down options, look into history/purpose, cross-refer... |
| K2-005 | C4 | STEP | NOT-STEP | Ensure a clear and concise final answer: Final answer: optical rotation angle. |

## Verdict

**PASS** — kappa = +0.802 vs threshold 0.70.

*Interpretation*: this validates that the fast heuristic step-counter agrees with the LLM-judge well enough to be used as the Pass-1 step-count metric in the full study, with the LLM-judge reserved for the §4.8.2 primary measurement on a per-cell sub-sample. If kappa falls below 0.7, the LLM-judge becomes the only valid step-counter and per-cell judging cost is the binding constraint on study size.
