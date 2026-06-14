# Pre-Data Calibration Protocol — FEH H1 effect size & σ_g

**Date:** 2026-05-29. **Status:** pre-data feasibility study, **not** the confirmatory test.
**Purpose:** estimate, on real model output, the two numbers the full study currently lacks —
(1) the implied high-regime accuracy drop (the H1 effect size) and (2) σ_g, the cross-model
slope heterogeneity — to decide whether the pre-registered full run (§4.10 Phase 6) is worth
launching, and to replace the *assumed* σ_g = 0.3 in `power_curve.py` with a measured value.

This calibration is exploratory and descriptive. The confirmatory H1 test remains pre-registered
(eq. 6.1, full 5-model panel, `confirmatory_analysis.py`) and is not conducted here.

---

## 1. Why this run exists

- The pilot (1 model, ~2 gold-scorable items) cannot estimate the H1 effect.
- The simulation power analysis (`analysis_validation.md`, `power_curve.md`) showed the design
  is powered (~0.90) only for a **large** reversal (~20 pp) and underpowered (~0.47) for a modest
  one, **under an assumed σ_g = 0.3** — a number no data has yet constrained.
- Therefore the single highest-leverage pre-data action is to measure the real effect size and σ_g
  cheaply, locally, before committing 45 h of compute or writing any results.

## 2. Outcome definition (high regime)

For genuine Knightian items the **normatively correct answer is `cannot-be-determined` (cbd)**.
H1's mechanism (Theorem 2.6.1 cue-truncation past k\*) predicts that additional reasoning pushes the
model *out* of cbd-recognition into confident confabulation → **cbd-correctness falls as steps rise**.

cbd-correctness is **objective, auto-scorable, and free**, and is already the *registered* grading
method (`auto-string-match`) for the 28 high-regime items used here — it is faithful to the
pre-registration, not a deviation. (The 43 `expert-coherence` items, whose outcome is human panel
rating, are **out of scope** for this calibration.)

- **High-regime (regime = 1):** `y = 1` iff the extracted answer normalizes to `cannot-be-determined`.
- **Low-regime (regime = 0):** `y = 1` iff the extracted answer matches the item `gold_answer` (numeric).

## 3. Design

| Factor | Levels |
|---|---|
| **Models** (3, local Ollama) | `mistral:7b-instruct`, `qwen2.5:7b`, `llama3.1:8b` |
| **High-regime items** (28) | All cbd-scorable Knightian items: `auto-string-match` grading **with** a `cannot-be-determined` choice |
| **Low-regime items** (14) | `R-101…R-106` (easy arithmetic-calibration tier) + `A-001…A-008` (aleatory, known probability) |
| **CoT conditions** (5) | C1–C5 per `prompt_templates.md` (single-line → ~15-step) |
| **Replications** (3) | distinct deterministic seed per (model, item, condition, rep) |

Total cells: 3 × 42 × 5 × 3 = **1,890**. Est. ≈ 3 h local (RTX 4070 Super), $0.

**Regime = item category** (K = high, R/A = low) — a deliberate simplification appropriate for an
effect-size estimate. The registered analysis instead bins by the §3.2 regime *score* quartiles; the
calibration does not run that pre-screen.

**High-regime item set (28):**
K1-001…K1-004, K1-006…K1-020 (19); K2-003, K2-005, K2-006, K2-008, K2-012, K2-015, K2-016 (7);
K4-002, K4-003 (2).

**Excluded from this calibration (8 broken in v0.3), now repaired in pool v0.4 (2026-05-29):**
K1-005, K2-007, K2-010, K2-013, K2-018, K4-001, K4-004, K4-008 were tagged `auto-string-match` but
carried **neither** a `gold_answer` **nor** a `cannot-be-determined` choice → not auto-scorable as
written. Resolved in `feh79_item_pool_v0.4.yaml`: cbd added to K1-005/K2-007/K2-018 (now cbd-scorable
→ 31 total); K4-001/004/008 reclassified `expert-coherence`; K2-010/K2-013 cut (a defensible reasoned
answer exists → not truly Knightian). This calibration ran on v0.3's 28 cbd items and is unaffected;
the **full run should use v0.4** (31 cbd items).

## 4. Analysis

`calibration_analysis.py` scores each cell, builds the tidy frame (`y, steps, regime, model, item`),
and runs the **locked** `confirmatory_analysis.run()` (eq. 6.1; with 3 models the `g_m` model-slope
random effect is included, so **σ_g is estimated**). It reports:

- population β1, β3, and the robust implied high-regime pp-drop (`decide` + `decide_amended`);
- **σ_g** posterior median + CI (the model-slope heterogeneity);
- per-item cbd-rate trajectories across C1→C5 (to expose effect heterogeneity, no cherry-picking).

Outputs: `calibration_responses.json`, `calibration_analysis.{json,md}`.

## 5. Decision gate (go / no-go for the full run)

1. **Effect size:** is the measured implied pp-drop ≥ ~20 pp (the range the design is powered for)?
2. **σ_g:** feed the measured σ_g back into `power_curve.py` (replacing the assumed 0.3) and re-read
   power at the measured effect.
3. **Heterogeneity:** how many of the 28 items show the predicted cbd-collapse vs floor/ceiling?
   Informs whether to pre-specify an item-inclusion rule for the full run.

Caveats carried forward: the 3 models are all 7–8 B, so σ_g here is a **lower bound** on the
registered 3 B–32 B panel's; small models may floor/ceiling differently than larger ones. The result
calibrates feasibility, not the confirmatory verdict.
