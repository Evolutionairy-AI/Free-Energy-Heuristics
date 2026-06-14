# Pre-Registration Amendment 1 — Free Energy Heuristics Full Empirical Study

**Date:** 2026-05-29
**Status:** Pre-data amendment (permitted under pre-registration §9.1). **No full-study confirmatory data have been collected.** This amendment is based entirely on (a) the descriptive pilot and (b) simulation on synthetic data; it is filed *before* data collection begins.
**Applies to:** Pre-Registration v0.2 (registered 2026-05-14, OSF project osf.io/9dvzb). All sections not listed below are unchanged.

---

## 0. One-paragraph summary

Before launching data collection, the pre-registered confirmatory analysis pipeline (eq. 6.1) was implemented and validated by synthetic parameter recovery and simulation-based power analysis. The pipeline is sound (clean convergence, faithful coefficient recovery, Type I error controlled at 0). However, the simulation revealed that the **original confirmatory gate (§6.2 / §7.1) has only ~7 % power** at a genuine target effect, and that the **§4.1 power claim of 0.92 was unreliable** (it used a normal approximation that ignored the compound decision rule, the β1/β3 collinearity, and the random-effect uncertainty from a 5-model panel). This amendment (A) reformulates the primary inference statistic to a directional test with a robust magnitude floor, demoting the full-reversal requirement to a reported effect size; (B) replaces the §4.1 power analysis with the simulation-based one; and (C) states honestly the effect-size range the study is powered to confirm. The amendment is *strictly* power-improving and does **not** inflate Type I error.

---

## 1. Motivation and supporting evidence

Three simulation studies were run on synthetic data at the full planned design (5 models × 40 items × 5 conditions × 3 replications ≈ 3,000 observations). Scripts and outputs are archived with the study code:

- `Experiments/confirmatory_analysis.py` — the pre-registered model (eq. 6.1) as runnable code.
- `Experiments/validate_analysis.py` → `analysis_validation.md` — parameter recovery + operating characteristics.
- `Experiments/power_curve.py` → `power_curve.md` — power vs effect size, original vs amended gate.
- `Experiments/design_sweep.py` → `design_sweep.md` — whether adding items closes the power gap.

**Finding 1 — the pipeline is valid.** Across four scenarios (strong reversal, modest reversal, null, opposite-direction), the model converged cleanly (R̂ = 1.000, ESS 1.6 k–2.0 k), the 95 % credible intervals covered the true β3 in every scenario, the decision rule returned the correct verdict at clear effect sizes, and the **false-confirmation rate under the null was 0/30**.

**Finding 2 — the original gate is underpowered; §4.1 was wrong.** At a genuine target effect (β3 = −0.25; implied high-regime accuracy drop ≈ 20 pp min-to-max, ≈ 10 pp over the robust 10th–90th-percentile step range), the original three-part gate confirmed in only **7 %** of simulated studies — not the 0.92 claimed in §4.1. The binding cause is the full-reversal requirement |β3| > |β1|: because the steps main effect and the steps × regime interaction are collinear by construction, 3,000 observations cannot certify |β3| > |β1| with 95 % posterior probability, even though `Pr(β3 < 0) ≈ 1.0`. Directionality is easy; certified full sign-reversal is not.

**Finding 3 — the amended gate recovers most of the lost power, with Type I still 0.** Replacing the gate with `Pr(β3 < 0) > 0.95` plus a robust magnitude floor raised power to **0.47 at the modest effect and 0.90 at a large effect (β3 = −0.40, ≈ 19 pp)**, while holding the false-confirmation rate at **0.00** under the null at every candidate floor.

**Finding 4 — adding items does not close the gap.** A sweep that widens the regime bins (quartile → tercile → median split) and a continuous-regime model (all 79 items) were compared on identical simulated data. More items did **not** improve power (quartile 0.55 ≥ tercile 0.50 ≈ continuous 0.50 > median 0.35): each added item sits nearer the regime-score median and carries a weaker true effect, so dilution cancels the sample-size gain. The within-design levers for modest-effect power are therefore exhausted; the only root-cause remedy would be a larger model panel.

---

## 2. Amendment A — Primary hypothesis (§2.1)

**Was (v0.2 §2.1):**

> **H1:** β3 is negative with magnitude sufficient to reverse the sign of the steps-on-accuracy slope in the high-regime bin: β1 > 0 AND β1 + β3 < 0.

**Now:**

> **H1 (amended).** The interaction coefficient β3 in eq. 6.1 is negative: increasing the number of reasoning steps degrades accuracy in the high-regime (high-meta-uncertainty) bin relative to the low-regime bin, by a substantively non-trivial amount.

The *full sign-reversal* β1 + β3 < 0 is retained as a **reported effect size and a secondary descriptive claim**, not as a gating criterion (see §4 below).

---

## 3. Amendment B — Primary inference statistic (§6.2)

**Was (v0.2 §6.2):** Posterior probability that β3 < 0 AND |β3| > |β1|; confirm if > 0.95.

**Now:** H1 is evaluated by **two jointly-required quantities**:

1. **Directional:** `Pr(β3 < 0 | data) > 0.95`.
2. **Magnitude:** the posterior-median **robust implied accuracy drop** exceeds **6 percentage points**.

The robust implied drop is defined (new, replacing the §6.3 min-to-max definition for gating purposes) as the high-regime accuracy difference between the **10th and 90th percentile** of the within-model standardized step count, evaluated at the **empirical high-regime base rate** and depending only on the posterior of (β1 + β3). This estimand is independent of the global intercept (which is weakly identified with only 5 models) and is robust to outlier maximum-step responses. The 6 pp floor was selected by the power curve (`power_curve.md`): among candidate floors {6, 8, 10} it maximised power at equal — i.e. zero — Type I error.

---

## 4. Amendment C — Confirmation, falsification, and demoted criteria (§7.1, §7.2)

**Confirmation (amended §7.1).** H1 is confirmed iff **both**:
1. `Pr(β3 < 0 | data) > 0.95`, and
2. posterior-median robust implied high-regime accuracy drop > 6 pp.

**Falsification (amended §7.2).** H1 is falsified iff either:
- `Pr(β3 ≥ 0 | data) > 0.95` (interaction in the opposite direction), or
- `Pr(β3 < 0) > 0.95` but the robust implied drop < 3 pp (directional but negligible).

**Inconclusive (§7.3, unchanged in spirit).** Any other outcome.

**Demoted to reported effect sizes (no longer gating; reported for every analysis):**
- the full-reversal probability `Pr(β3 < 0 ∧ |β3| > |β1| | data)` and the original min-to-max implied drop (the v0.2 §6.2 statistic — now descriptive);
- the within-bin slopes of accuracy on step count (low- and high-regime), on the probability scale;
- the sign of the low-regime slope.

---

## 5. Amendment D — Power analysis (§4.1 replaced)

The v0.2 §4.1 normal-approximation power analysis (claiming 0.92 power at d = 0.3) is **withdrawn** as unreliable for the actual hierarchical model and compound decision rule. It is replaced by the following **simulation-based** power analysis (amended gate, 30 simulated studies per effect size at the full design; `power_curve.md`):

| True β3 | Robust implied drop | Confirmation rate (power) | False-confirmation (Type I) |
|---|---|---|---|
| 0.00 (null) | — | — | **0.00** |
| −0.15 | ~3 pp | 0.30 | — |
| −0.25 (modest) | ~10 pp | 0.47 | — |
| −0.40 (large) | ~19 pp | **0.90** | — |

**Powered-for statement (new §4.1a).** The study is **adequately powered (≈ 0.90) to confirm a large reversal** (high-regime accuracy declining by ~20 pp across the step range) and **underpowered (≈ 0.47) for a modest reversal** (~10 pp). This is a deliberate, disclosed limitation, accepted because the FEH framework predicts a *substantive* failure of additional reasoning under genuine meta-uncertainty (cue-truncation past k\* leading to confabulation), not a marginal effect, and the descriptive pilot showed dramatic step-induced reversals on individual Knightian items (e.g. cannot-be-determined rate falling from 1.0 to 0.0 across conditions). A modest true effect will most likely return an **inconclusive** verdict, which will be reported transparently per §7.3. The only design change that would materially raise power for modest effects is a larger model panel (a sweep over items does not — `design_sweep.md`); this is noted as a limitation rather than adopted here.

---

## 6. Caveats disclosed

- **σ_g (model-level slope heterogeneity) is unknown.** The pilot used a single model, so the random-slope SD was assumed (0.3) in simulation; power is sensitive to it.
- **Item-effect heterogeneity.** A realistic simulation in which item effects vary by regime score depressed power relative to a homogeneous-effect simulation, indicating eq. 6.1's item random *intercept* (no item random *slope*) may understate uncertainty. An item random slope will be reported as a pre-specified robustness re-fit (added to §6.4 as R6).
- **Type I** was verified under a null with a positive low-regime step slope; it remains 0 there. A flat (no-step-effect) null was not separately simulated; the 6 pp magnitude floor provides the controlling margin.

---

## 7. Integrity statement

This amendment is filed **before any full-study confirmatory data have been collected**. It is motivated solely by code validation and simulation on synthetic and pilot data, both of which are descriptive-only per §9.1. It supersedes the corresponding parts of §2.1, §4.1, §6.2, §7.1, and §7.2 of v0.2; all other commitments (model eq. 6.1, conditions, items, models, exclusion/stopping rules, robustness checks R1–R5, confound controls) are unchanged. R6 (item random slope) is added to §6.4. The supporting scripts and outputs are archived with the study materials and are reproducible from fixed seeds.
