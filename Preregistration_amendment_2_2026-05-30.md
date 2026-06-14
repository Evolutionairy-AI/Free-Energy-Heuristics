# Pre-Registration Amendment 2 — Free Energy Heuristics Full Empirical Study

**Date:** 2026-05-30
**Status:** Pre-data amendment (permitted under pre-registration §9.1). **No full-study confirmatory data have been collected.** This amendment is based entirely on a separate, pre-registered feasibility/calibration study (`Calibration_protocol.md`), which is exploratory and descriptive by design and is **not** the confirmatory sample. It is filed *before* full data collection begins.
**Applies to:** Pre-Registration v0.2 (registered 2026-05-14, OSF project osf.io/9dvzb), as previously amended by Amendment 1 (2026-05-29) and consolidated in the v0.3 working copy. All sections not listed below are unchanged.
**Author:** Alex Bogdan (EvolutionAIry AI Inc., Toronto, Canada).

---

## 0. One-paragraph summary

A pre-registered calibration study (3 local 7–8 B models × 28 objectively-scorable Knightian items + 14 controls × the 5 length conditions × 3 replications) was run to measure, before committing to the full study, the two quantities the design previously only assumed: the H1 effect size and the cross-model slope heterogeneity. It produced a clear descriptive H1 signal — high-regime "cannot-be-determined" accuracy falls as instructed reasoning increases (≈ 7 pp across the clean C1→C4 range, ≈ 13 pp across C1→C5) while low-regime accuracy is flat — but also revealed that the **registered primary regressor (realized step count, §5.2) is mis-specified for the confirmatory estimand**: it is endogenous, its within-model standardization can invert the interaction sign, and a linear-in-steps form is wrong for what is a threshold effect. On the calibration data the registered model returned "inconclusive" and a variant of it even crossed the falsification threshold, on data where H1 is descriptively true. This amendment (A) changes the **primary confirmatory regressor** from realized steps to the **assigned reasoning-length manipulation** (an intention-to-treat short-vs-long contrast), (B) restates eq. 6.1 on that regressor, (C) demotes realized steps to a secondary instrumented/mediation analysis (new R7), and (D) replaces the power analysis with a simulation-based one calibrated to the measured effect. The Amendment 1 decision gate is retained verbatim and re-targeted. The change is *power-improving and bias-reducing* and does not inflate Type I error (simulated null confirmation 0.01).

---

## 1. Motivation and supporting evidence

The calibration study is documented in `Calibration_protocol.md`; its analysis and diagnostics are archived with the study code (`calibration_analysis.{md,json}`, `calibration_diagnose.py`, `calibration_ablation.py`, `calibration_reconcile.py`, `condition_power.py`). The registered primary regressor (v0.2 §5.2: `steps`, the realized step count, standardized within model) is **mis-specified for the confirmatory estimand**, for three independent and demonstrable reasons. All three are properties of the regressor and the model form, not of the calibration sample; they will recur on the confirmatory data.

**Finding 1 — realized steps is endogenous.** A model emits more steps when it is struggling, so the realized step count is correlated with the latent difficulty of the particular attempt, not only with the experimental manipulation. In the calibration data, wrong answers carried more steps than correct answers in *both* regimes (high +1.8, low +1.4 steps; the easy-arithmetic control items showed a within-family correlation of −0.19 between step count and correctness). Regressing accuracy on realized steps therefore conflates the causal effect of *instructed* deliberation with the confound of *attempt difficulty* — the standard problem of conditioning on a post-treatment, behaviorally-determined mediator.

**Finding 2 — within-model standardization can invert the interaction sign.** β3 is an interaction coefficient and is therefore **not invariant to rescaling** of the step regressor. The registered within-model z-scoring (v0.2 §6.1) applies a different, data-dependent affine transform per model, which reweights each model's contribution to the pooled interaction. On the calibration data the *same* responses yield β3 = −0.013 (raw steps), −0.094 (global z-score), and **+0.174** (within-model z-score, the registered choice) — i.e., the registered standardization flips the interaction to the sign *opposite* H1. An ablation confirmed this is driven by the standardization, not by the random-effect structure (adding the model and item random effects moves β3 *toward* zero, not across it).

**Finding 3 — the effect is a threshold, not a linear slope.** Theorem 2.6.1 (cue-truncation past a critical depth k\*) predicts that crossing into substantial deliberation triggers truncation-and-confabulation, after which additional reasoning does little more. The calibration data show exactly this: high-regime cbd-correctness falls from the direct-answer condition to the deliberation conditions and then plateaus (C1 = 0.71; C2–C5 ≈ 0.58), while low-regime accuracy is flat across all conditions. A linear-in-steps slope is the wrong functional form for a drop-then-plateau and dilutes the effect.

**Consequence.** On data where H1 is descriptively true, the registered realized-steps model returned Pr(β3 < 0) = 0.21 ("inconclusive"), and a pooled variant of it crossed the §7.2 *falsification* threshold (Pr(β3 ≥ 0) > 0.95). The registered analysis can fail to detect, and can even falsify, a true effect. This is a specification failure, not a weak effect — and it is the reason for this amendment.

---

## 2. Why this is a principled correction, not a forking-paths choice

The amendment is adopted before any confirmatory data exist, on a separate feasibility sample, and is justified on a-priori grounds rather than by which specification produced the most favorable statistic:

1. **The defect is structural.** Endogeneity of a post-treatment mediator (Finding 1) and non-invariance of an interaction term to per-group rescaling (Finding 2) are general properties; they are not artifacts of the calibration sample and would hold on any data collected under this design.
2. **The replacement is the more conservative, standard causal choice.** Analyzing the *randomly assigned* manipulation (intention-to-treat) rather than a behaviorally-determined mediator is the textbook remedy for post-treatment confounding. It trades a mechanistic regressor for an unbiased one.
3. **The functional form is theory-derived, not data-fit.** The short-vs-long threshold contrast is the contrast Theorem 2.6.1 predicts (drop at k\*, then plateau); it was not selected to maximize an effect estimate.
4. **It is a re-assignment among already-registered variables, not a new measurement.** The assigned-length conditions C1–C5 are the registered manipulation (v0.2 §3.2), and the condition factor is already a registered model term (v0.2 §5.3: a 5-level fixed-effect control for prompt template). This amendment **promotes** the assigned-length factor from nuisance control to primary causal regressor, and **demotes** realized steps from primary regressor to a secondary mechanistic estimate. No new quantity is measured; the data collected are identical.

---

## 3. Amendment A — Primary reasoning-length regressor (supersedes the realized-steps-as-primary-IV commitment of §3.2 final paragraph, §5.2, §5.3)

The primary reasoning-length regressor becomes the **assigned-length factor**, `long_j ∈ {0, 1}`:

- `long = 0` (short): condition **C1** (direct answer, no deliberation requested).
- `long = 1` (long): conditions **C2–C5** (any instructed step-by-step deliberation).

This is an intention-to-treat (ITT) regressor: it is fixed by random assignment and is independent of the response actually produced, so it is immune to the endogeneity of Finding 1 and to the non-monotone realized-step behavior of C5 (see §6).

The **realized** step count `steps` (v0.2 §5.2) is **retained but demoted to a secondary regressor**, analyzed under the new robustness check R7 (§5), where the assigned condition serves as its instrument (an instrumental-variables / mediation estimate of the per-step effect). The per-condition dose ordering across the monotone portion C1 < C2 < C3 < C4 is reported as a secondary monotone-trend effect size; C5 is excluded from the dose-response (§6).

---

## 4. Amendment B — Primary confirmatory model (supersedes §6.1)

The confirmatory model becomes a hierarchical Bayesian logistic regression on the assigned-length factor:

$$
y_{ij} \sim \mathrm{Bernoulli}(\sigma(\eta_{ij}))
$$
$$
\eta_{ij} = \beta_0 + \beta_1\cdot \mathrm{long}_j + \beta_2\cdot \mathrm{regime}_i + \beta_3\cdot(\mathrm{long}_j \times \mathrm{regime}_i) + \alpha_{m(ij)} + \gamma_{m(ij)}\cdot \mathrm{long}_j + u_i \tag{6.1$'$}
$$

where `long_j ∈ {0,1}` is the assigned-length factor (§3), `regime_i ∈ {0,1}` is the high-regime indicator, $\alpha_m$ and $\gamma_m$ are the model random intercept and the model random slope on `long` (the cross-model heterogeneity of the length effect — the σ_g analogue of the registered model), and $u_i$ is the item random intercept. Priors are unchanged from v0.2 §6.1: $\beta_0, \beta_2 \sim \mathcal{N}(0, 2.5)$; $\beta_1, \beta_3 \sim \mathcal{N}(0, 1)$; random-effect SDs $\sim \mathrm{HalfNormal}(1)$. Sampling and convergence gates are unchanged (PyMC/NUTS, 4 × 2000 draws, target_accept 0.95, non-centered random effects; $\hat R < 1.01$, ESS > 400).

Because `long` is binary, β3 is the **difference-in-differences** of accuracy between the long and short conditions across regimes — the cleanest expression of the H1 dissociation. H1 remains **β3 < 0** (instructed deliberation degrades high-regime accuracy more than low-regime accuracy), unchanged in meaning from Amendment 1.

---

## 5. Amendment C — Decision rule (unchanged from Amendment 1) and robustness battery update (supersedes §6.4)

**Decision rule.** The Amendment 1 gate is retained verbatim, now applied to the β3 of the §4 model:

> **Confirmation:** Pr(β3 < 0 | data) > 0.95 **AND** posterior-median robust implied high-regime accuracy drop > 6 percentage points.
> **Falsification:** Pr(β3 ≥ 0 | data) > 0.95, OR Pr(β3 < 0) > 0.95 with robust implied drop < 3 pp.
> **Inconclusive:** any other outcome.

The robust implied drop is computed as the short→long change in predicted high-regime accuracy at the empirical high-regime base rate (the §6.2/§6.3 estimand of Amendment 1, re-expressed for the binary `long` contrast). The full-reversal probability, the low-regime effect, and the per-condition dose curve are reported but do not gate, as in Amendment 1.

**Robustness battery.** The registered checks R1–R6 are retained. Where a check named the realized-steps regressor it now applies to the secondary realized-steps analysis: **R1** (alternative step segmentation) and **R5** (per-model sign consistency) attach to R7. **R2** (terciles / continuous regime score), **R3** (regime score dropping calibration error), **R4** (held-out items), and **R6** (item random slope) are unchanged and apply to the primary model 6.1′. One check is added:

- **R7 (realized-steps IV / mediation) [ADDED v0.4].** Re-estimate the per-step effect using the assigned condition as an instrument for realized steps (two-stage / mediation estimate), and report the realized-steps interaction from the registered v0.2 model 6.1 for continuity. This keeps the now-secondary mechanistic regressor visible alongside the primary ITT estimand and documents the endogeneity correction of Finding 1.

---

## 6. Handling of condition C5

The calibration revealed that the realized step count for **C5 ("Unconstrained")** is **non-monotone**: mean realized steps were C1 = 0.9, C2 = 3.5, C3 = 10.8, C4 = 17.9, C5 = 6.6 — the unconstrained condition produced fewer *realized* steps than C3, so the realized-step ladder is C1 < C2 < C5 < C3 < C4 rather than the intended C1 < … < C5. A diagnostic (`c5_diagnose.py`) established that this is a **behavioral property of the prompt, not a step-counter artifact**: C5 generated only **44 % of C4's completion tokens** (254 vs 572 mean output tokens; a counter-independent measure of generation length), and fewer than C3 (355) too — i.e. the bare "Think step by step" wording is read by these 7–8 B models as licence for a shorter answer, whereas the explicit "in approximately 15 steps" of C4 actively drives length. Two independent checks confirm there is no C5-specific *measurement* bug. (i) Step *density* per 100 tokens is comparable across the deliberation conditions (C2 = 3.2, C3 = 3.0, C4 = 3.1, C5 = 2.6) — C5's lower step count tracks its lower token count, not a lower flagging rate. (ii) Steps per sentence are in fact *higher* for C5 than for C4 (0.93 vs 0.78), so the classifier is not under-segmenting the C5 format. The entire C5 deficit is that the model *writes less*, not that its reasoning is mis-counted. Two consequences:

- **For the primary ITT model (§4), C5 is unaffected:** an assigned-long trial assigns deliberation regardless of how many steps were realized, so the realized-step non-monotonicity does not bias the ITT estimand. (Indeed C5's accuracy matched C2–C4 despite fewer steps — consistent with the threshold prediction of §1 Finding 3: any deliberation trips the effect, additional depth does not deepen it.)
- **For the secondary realized-steps (R7) and dose-response analyses, the monotone ladder is C1–C4 and C5 is excluded from the ordinal dose contrast** (it remains a valid assigned-long trial in the ITT model). The full run has two pre-data options, to be fixed before launch: (i) retain C5 as an "open-ended" robustness condition outside the dose ladder, or (ii) replace the C5 template with one that reliably induces a higher dose than C4 (e.g. an explicit ≥ 20-step target) to obtain a genuine maximum-deliberation point. Either choice is documented as a pre-data deviation; neither affects the primary ITT estimand.

---

## 7. Amendment D — Power analysis (replaces the §4.1 figures for the new primary model)

**Method.** Full-design Monte Carlo at the planned panel (5 models × 31 high-regime cbd-scorable items [pool v0.4] + 12 low-regime control items × 5 conditions × replications). Data are generated with the **measured** calibration ingredients: the 28 measured per-item high-regime short-condition base rates (full heterogeneity, including floored and ceilinged items), a low-regime base rate of 0.66, model random intercepts (SD 0.4 logit), and a cross-model length-effect spread (SD 0.20 logit). The H1 length effect is applied only in the high regime and tuned to a target mean high-regime short→long accuracy drop. The Amendment 1 gate (§5) is applied per replication (200 replications per effect size). Tool: `condition_power.py`. A fast frequentist IRLS/Wald proxy is used for the gate; a full-PyMC confirmation may be run if the conservative ~7 pp scenario must be pinned exactly.

**Power of the amended condition-ITT analysis** (one-sided directional + ≥ 6 pp magnitude floor), by true high-regime short→long accuracy drop:

| True high-regime drop | Power (3 reps) | Power (5 reps) |
|---|---|---|
| 0 pp (null) | 0.01 *(Type I)* | 0.01 *(Type I)* |
| 5 pp | 0.33 | 0.47 |
| **7 pp** (calibration, clean C1→C4) | **0.65** | **0.73** |
| 10 pp | 0.82 | 0.97 |
| **13 pp** (calibration, full C1→C5) | **0.96** | **1.00** |
| 15 pp | 0.99 | 1.00 |

**Powered-for statement.** The calibration measured the true high-regime drop in the **7–13 pp** range. Across that range the amended analysis has power **0.65–0.96 at 3 replications** and **0.73–1.00 at 5 replications**, with Type I error 0.01. This supersedes the realized-steps power figures of Amendment 1 §5 / v0.3 §4.1 (≈ 0.07 original gate, ≈ 0.47 amended gate at the target), which described the now-secondary regressor; the change of primary regressor is the dominant driver of the recovery. **5 replications is adopted** if the conservative 7 pp scenario is to be powered ≥ 0.8 (it lifts the 7 pp cell from 0.65 to 0.73 and clears 0.8 by ≈ 8 pp); 3 replications remain adequate if the true effect is ≥ 10 pp.

**Caveats (carried forward).** (i) The frequentist Wald test is mildly anti-conservative as a proxy for the Bayesian gate, but feeding the noisy per-item base rates in as ground truth injects variance in the conservative direction; net error is estimated at ±0.1 near the margin. (ii) The calibration panel was 3 small (7–8 B) models on which several high-regime items floored (the model could not recognize "cannot-be-determined" even once); larger models that can recognize the cbd response may exhibit a larger, cleaner effect, moving the true effect toward the high-power rows. (iii) The σ_g analogue (cross-model length-effect spread) was 0.17 [0.01, 0.98] in the calibration — below the 0.30 previously assumed, but weakly constrained by only 3 models and a lower bound on the 3 B–32 B panel's heterogeneity.

---

## 8. What is NOT changed

- The item pool is unchanged in content. (The full run uses pool **v0.4**, which repairs 8 previously un-scorable items and is documented separately; v0.4 changes no registered item's meaning, only its scorability.)
- The five-model panel is unchanged as the registered minimum. (Possible frontier-model additions, motivated by caveat 7(ii), are a separate design decision, not part of this amendment.)
- The manipulation (C1–C5 prompt templates), the prompt protocol, and the step-counting procedure are unchanged.
- The theoretical predictions (H1, H2, H3) and the hypothesis content (H1: β3 < 0) are unchanged.
- The decision-rule *form* (Amendment 1 gate, §5) is unchanged.

---

## 9. Integrity statement

This amendment is filed **before any full-study confirmatory data have been collected**. It is motivated solely by a separate pre-registered feasibility/calibration study, which is descriptive-only and is not part of the confirmatory sample. It supersedes the realized-steps-as-primary-regressor commitment of §3.2 (final paragraph), §5.2–§5.3, the primary model §6.1, the power figures of §4.1, and the robustness list §6.4; it re-targets — and leaves otherwise intact — the Amendment 1 decision-rule form (§6.2 / §7). All other commitments (conditions, items, models, exclusion/stopping rules, confound controls, R1–R6) are unchanged. The registered v0.2 remains the immutable record; v0.2 + Amendment 1 + this Amendment 2 together define the operative pre-registration. A consolidated working copy may fold these changes inline, tagged **[AMENDED v0.4]**, for reading convenience only. Supporting scripts and outputs are archived with the study materials and are reproducible from fixed seeds.
