# Pre-Registration: Free Energy Heuristics Full Empirical Study

**Working title:** Chain-of-Thought Reasoning Degrades LLM Accuracy Under Knightian Uncertainty: A Confirmatory Test of the Meta-Precision Cue-Truncation Theorem

**Principal investigator:** Alex Bogdan (EvolutionAIry AI Inc., Toronto, Canada). Affiliation independent.

**Theoretical pre-print:** Chapter 2 (FEH Theoretical Foundation, Working Draft v0.7) and Chapter 3 (FEH Operationalization, Working Draft v0.1) — to be posted to arXiv prior to data collection and cited by DOI in this pre-registration.

**Pre-registration platform:** OSF Pre-registration (https://osf.io/prereg). AsPredicted format included as Appendix B for users who prefer the lighter-weight registry.

**Submission status:** v0.4 working copy (2026-05-30), incorporating **Amendment 1** (2026-05-29) and **Amendment 2** (2026-05-30) and the pre-data panel/replication design decision (2026-05-30) into the registered v0.2 (OSF 2026-05-14, osf.io/9dvzb). Item pool finalized at FEH-79 **v0.4** (K2-validated, K3-validated; v0.4 repairs 8 previously un-scorable items, changing no registered item's meaning). Hardware environment documented. Outstanding placeholders limited to arXiv DOI + GitHub URL.

> **Changelog — Amendment 1 (2026-05-29, pre-data; permitted under §9.1).** Following implementation and simulation-based validation of the eq. 6.1 pipeline (before any confirmatory data collection), the primary inference statistic (§6.2), confirmation/falsification criteria (§7.1–7.2), primary hypothesis statement (§2.1), and power analysis (§4.1) were revised, and robustness check R6 was added (§6.4). The full-reversal requirement |β3|>|β1| is demoted from a gating criterion to a reported effect size; the primary test becomes directional (Pr(β3<0)>0.95) with a robust magnitude floor. Rationale and evidence: `Preregistration_amendment_1_2026-05-29.md` and the simulation reports (`analysis_validation.md`, `power_curve.md`, `design_sweep.md`). The registered v0.2 remains the immutable record; this consolidated working copy is used by the manuscript. Amended passages are tagged **[AMENDED v0.3]**.

> **Changelog — Amendment 2 (2026-05-30, pre-data; permitted under §9.1).** Following a separate pre-registered feasibility/calibration study (`Calibration_protocol.md`; 3 local models × 28 objectively-scorable Knightian items + 14 controls × 5 conditions × 3 reps) run *before* any confirmatory data collection, the **primary reasoning-length regressor** is changed from the realized step count (endogenous; its within-model standardization can invert the interaction sign; linear-in-steps is the wrong form for a threshold effect) to the **assigned-length intention-to-treat factor** `long ∈ {0,1}` (short = C1; long = C2–C5). This changes the primary regressor (§5.2), the primary model to eq. 6.1′ on `long` (§6.1), the controls framing (§5.3), and the power analysis to a simulation calibrated to the measured effect (§4.1); it adds robustness check **R7** (realized-steps IV/mediation, which retains the old eq. 6.1 as a secondary analysis) and re-attaches R1/R5 to R7 (§6.4). The Amendment-1 decision-gate *form* (§6.2/§7.1–7.2) is retained verbatim and re-targeted to the β3 of eq. 6.1′. Rationale and evidence: `Preregistration_amendment_2_2026-05-30.md`, `calibration_analysis.md`, and the diagnostic scripts. Amended passages are tagged **[AMENDED v0.4: A2]**.

> **Changelog — Panel/replication design decision (2026-05-30, pre-data; permitted under §9.1).** Following a frontier feasibility read (`frontier_read_analysis.md`; Claude Sonnet 4.5 + GPT-4o on the calibration design) that showed H1 holds *off the small-model floor* (pooled assigned short→long high-regime cbd-correctness drop +10.4 pp [+2.1, +18.8]; difference-in-differences +9.2 pp [+0.9, +17.9]; GPT-4o floored on only 3/28 high items vs the local panel's heavy flooring), the model panel is expanded from 5 local to **5 local + 2 frontier (7 models)** and replications from 3 to **5** per cell. Rationale: frontier models are the actual target of the o1/o3 reasoning-scaling counter-narrative and exhibit the effect more cleanly; 5 reps powers per-model confirmation and the conservative ~7 pp scenario. The 5-local panel is retained as the registered minimum/fallback. Affects §3.3, §3.5, §4.1, §4.4. Amended passages are tagged **[AMENDED v0.4: panel]**. A LaTeX-robust numeric-scoring correction to the analysis code (frontier models format fractions as LaTeX under CoT; the registered parser otherwise mis-scores them by condition) is documented in Appendix C; a residual rounding-precision hardening of the scorer is pending before confirmatory collection.

---

## 1. Study Information

### 1.1 Background

The active-inference framework (Friston and colleagues) and the ecological-rationality framework (Gigerenzer and the ABC research group) make apparently incompatible predictions about when additional inference helps versus hurts decision quality. Chapter 2 of the accompanying theoretical pre-print derives a unification: under meta-uncertainty over prior precision (σ²_τ ≥ τ_regime), expected free energy is U-shaped in cue count k, and additional reasoning steps past the optimal truncation k\* strictly increase expected free energy (Theorem 2.6.1).

The framework predicts a specific, novel, falsifiable failure mode of contemporary LLM reasoning systems: under tasks satisfying meta-uncertainty (Knightian conditions, not aleatory risk or simple ambiguity), additional chain-of-thought (CoT) reasoning steps should *decrease* accuracy. This prediction directly contradicts the prevailing "more reasoning = better" narrative around o1/o3-style reasoning models. The present study is the confirmatory test.

### 1.2 Research question

Does additional CoT reasoning degrade LLM accuracy on high-meta-uncertainty items (as predicted by the cue-truncation theorem), and does this effect interact with the regime score (a behavioral proxy for σ²_τ) in the direction predicted by theory?

### 1.3 Headline claim

Within items in the top quartile of the regime score (high meta-uncertainty proxy), accuracy as a function of CoT step count should be *decreasing*. Within items in the bottom quartile, it should be *non-decreasing*. The cross-quartile interaction is the primary test.

---

## 2. Hypotheses

### 2.1 Primary hypothesis (confirmatory)

**H1 [AMENDED v0.3; regressor updated AMENDED v0.4: A2]**: The interaction coefficient $\beta_3$ in the primary analysis model (Section 6, eq. 6.1′) is negative by a substantively non-trivial amount: assigning **long** (instructed step-by-step) rather than **short** (direct-answer) reasoning degrades accuracy in the high-regime (high-meta-uncertainty) bin relative to the low-regime bin.

$$
\beta_3 < 0 \quad \text{(directional)} \quad \text{with substantive magnitude (see §6.2)}
$$

*[AMENDED v0.4: A2] The interaction is now between the high-regime indicator and the **assigned-length** factor `long ∈ {0,1}` (eq. 6.1′, §6.1), not the realized step count. The meaning of H1 — instructed deliberation degrades high-regime accuracy relative to low-regime — is unchanged; only the regressor is changed, from an endogenous behavioral measure to the randomly-assigned manipulation (intention-to-treat). The realized-step interaction of the original eq. 6.1 is retained as the secondary mechanistic analysis R7 (§6.4).*

The stronger *full sign-reversal* condition of the original v0.2 statement — $\beta_1 > 0 \;\text{AND}\; \beta_1 + \beta_3 < 0$, equivalently $\mathrm{slope}_{\mathrm{high-regime}} < 0 < \mathrm{slope}_{\mathrm{low-regime}}$ — is **retained as a reported effect size and secondary descriptive claim**, not as a confirmatory gate. *(Reason: simulation showed the full-reversal gate has ~7% power because $\beta_1$ and $\beta_3$ are collinear; the directional claim with a magnitude floor recovers power to ~0.90 at a large effect while holding Type I at 0 — see §4.1 and the amendment document.)*

### 2.2 Secondary hypotheses (confirmatory)

**H2 (size-scaling)**: The magnitude of the negative interaction $|\beta_3|$ does not systematically decrease with model size. Theory predicts the effect is regime-driven, not capability-driven; a strong negative dependence on model size would falsify the framework's scope claim.

**H3 (regime-validation)**: Items independently classified as "Knightian" by the §3.6 procedure show higher regime scores than items classified as "aleatory" (ground-truth distribution known) or "epistemic" (ground-truth distribution unknown but resolvable). This validates the regime score as picking up meta-uncertainty rather than other uncertainty types.

### 2.3 Exploratory hypotheses (non-confirmatory, reported for transparency)

- E1: Negative interaction is detectable on reasoning models (o1, DeepSeek-R1, Qwen-QwQ) at API-accessible scales, replicating the open-weight finding.
- E2: Token-level analysis of where accuracy degrades within the CoT trace reveals a transition point matching the EFE-predicted k\*.
- E3: The negative interaction generalizes to a separate held-out item set constructed independently of the regime-score pre-screen.

These are reported alongside H1-H3 but do not affect the confirmatory inference.

---

## 3. Study Design

### 3.1 Design type

Within-item × within-model × between-condition experimental design. Items are nested within regime bins; each item is presented to each model under each of 5 conditions, with **[AMENDED v0.4: panel]** 5 replications per (model, item, condition) cell (raised from 3; §4.4).

### 3.2 Conditions (independent variable: CoT length)

Five conditions, length-graded:

- **C1 — None**: direct answer prompt, no CoT allowed (suppression via "Answer in one word only.").
- **C2 — Short**: ~3 reasoning steps target ("Think step by step, briefly, in 3 steps or fewer.").
- **C3 — Medium**: ~7 reasoning steps target ("Think step by step in about 7 steps.").
- **C4 — Long**: ~15 reasoning steps target ("Reason through this carefully in about 15 steps.").
- **C5 — Unconstrained**: standard CoT prompt ("Think step by step.") with no length cap.

The realized step count is recorded per response. **[AMENDED v0.4: A2]** The *primary* independent variable is now the **assigned-length** factor `long ∈ {0,1}` (short = C1; long = C2–C5; §5.2, eq. 6.1′), not the realized step count — the realized count is endogenous (a model emits more steps when struggling) and its analysis is demoted to the secondary instrumented/mediation check R7 (§6.4). The per-condition realized-step ordering (C1 < C2 < C3 < C4) is reported as a secondary monotone dose effect; calibration found C5's realized steps non-monotone (behavioral: the unconstrained prompt yields shorter completions on smaller models), so C5 is excluded from the secondary dose ladder while remaining a valid assigned-`long` trial in the primary ITT model.

### 3.3 Models (between-model factor)

**[AMENDED v0.4: panel]** Seven models: a five-model open-weight local backbone (the registered minimum) plus two frontier API models. The frontier additions are motivated by the feasibility read (`frontier_read_analysis.md`), which found the H1 effect is *cleaner* on capable models that are off the small-model recognition floor and that these are the models the o1/o3 reasoning-scaling counter-narrative is actually about.

*Local backbone (~10× size range, instruction-tuned, open-weight, runnable on NVIDIA RTX 4070 Super, 12GB VRAM, with quantization where required):*
- **M1**: Phi-3.5-mini-instruct (3.8B parameters), bf16
- **M2**: Mistral-7B-Instruct-v0.3 (7B), bf16
- **M3**: Qwen-2.5-7B-Instruct (7B), bf16
- **M4**: Qwen-2.5-14B-Instruct (14B), 4-bit AWQ quantization
- **M5**: Qwen-2.5-32B-Instruct (32B), 4-bit AWQ quantization

*Frontier additions (API, snapshot-pinned for reproducibility):*
- **M6**: Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`)
- **M7**: GPT-4o (`gpt-4o-2024-11-20`)

If a specific model is unavailable at data-collection time, it is substituted by the closest-equivalent instruction-tuned model released before the pre-registration timestamp (closest-size open-weight for M1–M5; closest frontier snapshot for M6–M7). Substitutions are logged in the deviations report. If frontier API access is unavailable, the five-model local backbone is the registered fallback and the frontier additions are reported as a separate confirmatory extension.

### 3.4 Items (within-condition factor, nested in regime bin)

Item pool: 79 frames constructed per §3.6 of the operationalization chapter to span Knightian, ambiguity, and aleatory uncertainty types. Items are pre-screened in the pilot (Section 9 of this pre-registration) and binned by regime score; the primary analysis uses items in the top quartile (high-regime) and bottom quartile (low-regime). Middle 50% items are retained for robustness checks but not used in the primary inference.

Expected sample composition after pre-screen:
- High-regime: ~20 items
- Low-regime: ~20 items
- Middle (excluded from primary): ~39 items

### 3.5 Total observations

**[AMENDED v0.4: panel]** 7 models × 79 frames × 5 conditions × 5 replications = **13,825 observations** total. Primary analysis uses 7 models × ~40 high-or-low-regime items × 5 conditions × 5 replications = **~7,000 observations**. *(Registered-minimum fallback, 5 local models × 3 replications: 5,925 total / ~3,000 primary.)*

---

## 4. Sampling Plan

### 4.1 Sample size justification (power analysis)

**[AMENDED v0.4: A2 — the realized-steps power analysis of Amendment 1 is superseded by a condition-ITT power analysis calibrated to the measured effect.]** The Amendment-1 §4.1 power figures (≈0.90 large / ≈0.47 modest) described the now-secondary realized-steps gate (`power_curve.py`); they are retained for the R7 secondary analysis but no longer describe the primary test. The primary test is now the assigned-length (ITT) interaction of eq. 6.1′, for which power was re-estimated by full-design Monte Carlo (`condition_power.py`) using the **measured** calibration ingredients (28 measured per-item high-regime C1 base rates including floored/ceilinged items, low-regime base rate 0.66, model-intercept SD 0.4 logit, cross-model length-effect SD 0.20 logit; 200 replications per effect size; the amended gate of §6.2/§7.1 applied per replication). Power by true high-regime short→long accuracy drop:

| True high-regime drop | Power (3 reps) | Power (5 reps) |
|---|---|---|
| 0 pp (null) | 0.01 *(Type I)* | 0.01 *(Type I)* |
| 5 pp | 0.33 | 0.47 |
| **7 pp** (calibration, clean C1→C4) | **0.65** | **0.73** |
| 10 pp | 0.82 | 0.97 |
| **13 pp** (calibration, full C1→C5) | **0.96** | **1.00** |
| 15 pp | 0.99 | 1.00 |

**Powered-for statement.** The calibration measured the true high-regime short→long drop in the **7–13 pp** range; the frontier read (off the small-model floor) measured ≈10 pp. **5 replications is adopted** (§4.4): it powers per-model confirmation and lifts the conservative 7 pp cell toward 0.8 (and clears it by ≈8 pp); at the measured 10–13 pp the design has power ≈0.97–1.00. This table is computed for the **5-model** panel and is therefore *conservative* for the adopted **7-model** design (§3.3): adding two frontier models adds data and, per the feasibility read, raises the true effect off the small-model floor — both move power upward. The change of primary regressor (realized steps → assigned length) is the dominant driver of the recovery from the Amendment-1 figures. Caveats (carried forward): feeding noisy per-item base rates as ground truth injects variance in the conservative direction (net ±0.1 near the margin). The remaining concern — that the frequentist Wald proxy might be anti-conservative for the registered Bayesian gate — was **checked directly and resolved**: a full-PyMC confirmation (`condition_power_pymc.py`) re-ran the *actual* registered gate (eq. 6.1′ with the model random intercept **and** slope over 5 clusters, the item random intercept, and the N(0,1) shrinkage prior on β3) on data simulated from the same calibration-anchored DGP, **paired** against the IRLS proxy on identical datasets. The two gates agreed on **100% of simulations** (120/120 across the 0 / 7 / 10 pp cells; 40 sims/cell). At the conservative **7 pp** effect the Bayesian gate's power was **0.82** (identical to the proxy on the same data); at **10 pp** it was **1.00**; Type I at 0 pp was **0.05** (nominal). The Bayesian random-effects machinery therefore introduces no detectable power penalty, and the proxy table above is a faithful stand-in for the registered gate. (This run drew an independent base-rate resample and 40 sims/cell, so its absolute 7 pp level — 0.82 — carries more Monte-Carlo noise, ±~0.06, than the 200-sim proxy table's 0.73; both sit comfortably in the "≈0.8 at 7 pp" range. See `condition_power_pymc.{md,json}`.)

### 4.2 Sampling of items

Items are drawn from a single fixed pool of 79 frames developed in §4 (benchmark design chapter). No replacement, no oversampling. Pilot pre-screen on Mistral-7B + 10 representative frames identifies whether the regime score successfully separates items as predicted; if it does not, the operationalization (§3.2-§3.3) is revised before full data collection and this pre-registration is amended (Section 9).

### 4.3 Sampling of conditions

All 5 conditions are presented for every (model, item) pair. No condition is added or dropped between pilot and full study unless documented as a pre-data deviation.

### 4.4 Replications

**[AMENDED v0.4: panel]** **Five** replications per (model, item, condition) cell (Amendment 1 / v0.2 specified three; raised to five per the §4.1 power analysis, which requires five reps to power per-model confirmation and the conservative ~7 pp scenario). Replications differ in: (a) sampling seed (each replication uses a distinct random seed for the model's temperature-T sampling; frontier API models that do not honor a seed parameter rely on temperature stochasticity, logged per model), (b) prompt phrasing (where signature (a) of the regime score is computed across replications, the prompts are systematically varied; for the primary analysis, the canonical phrasing is used in replication 1, and replications 2–5 use rephrasings).

### 4.5 Exclusion criteria

- Refused responses (model declines to answer): excluded from the analysis cell, replaced by an additional replication where possible.
- Malformed responses (cannot extract an answer): same handling.
- Items with >50% refusal rate across (model, condition) cells: dropped from primary analysis, reported in supplementary.

### 4.6 Stopping rule

Data collection runs to the pre-specified sample size (**[AMENDED v0.4: panel]** 13,825 observations at the adopted 7-model × 5-replication design; 5,925 at the registered-minimum fallback). No optional stopping. No interim analyses on the confirmatory hypothesis until data collection is complete.

---

## 5. Variables

### 5.1 Dependent variable

**Accuracy** ($y_{ij}$): binary indicator of whether the model's extracted answer matches the gold-standard answer for the item. For Knightian items where no single objective ground truth exists, accuracy is replaced by **expert-panel-rated coherence** (3-expert majority on a 5-point coherence scale, binarized at the median). Pre-registered: this replacement applies only to items in the "Knightian" sub-category of the benchmark.

### 5.2 Primary independent variables

**[AMENDED v0.4: A2]** The primary reasoning-length variable is the **assigned-length factor**, not the realized step count:

- **long_j**: binary assigned-length indicator (0 = short, condition C1, direct answer; 1 = long, conditions C2–C5, any instructed step-by-step deliberation). Fixed by random assignment, hence exogenous (intention-to-treat).
- **regime_i**: binary high-regime indicator (1 = top quartile of regime score, 0 = bottom quartile).
- **regime × long**: interaction term (the primary test; β3 of eq. 6.1′).

*Secondary (demoted from primary):* **steps_{ij}**, the realized number of reasoning steps in the model's CoT trace (sentence-level segmentation; paragraph-level robustness in §4.8.4). Retained as the regressor of the secondary instrumented/mediation analysis R7 (§6.4), where the assigned condition serves as its instrument; not used in the primary gate, because the realized count is endogenous (correlated with attempt difficulty: in calibration, wrong answers carried more steps in both regimes).

### 5.3 Controls / random effects

- **model**: **[AMENDED v0.4: panel]** 7-level factor (5 local + 2 frontier) entered as model-level random intercept and random slope on `long`.
- **item**: random intercept (within regime bin).
- **condition / assigned length**: **[AMENDED v0.4: A2]** the assigned-length factor `long` (collapsing C1 vs C2–C5) is now the *primary* causal regressor rather than a nuisance control; the 5-level condition factor is reported as a secondary per-condition dose curve (C1 < C2 < C3 < C4 monotone portion; C5 excluded from the dose ladder per §3.2). This promotes the randomly-assigned manipulation and demotes the realized step count to the R7 mechanistic estimate.
- **replication seed**: not modeled (averaged within cell).

### 5.4 Operational definition of step count

A reasoning step is a sentence in the CoT trace that contains at least one of: (a) inferential connective ("therefore", "so", "because", etc.), (b) intermediate computation, (c) intermediate claim about the task. Sentences that only meta-comment ("Let me think about this") are not counted as steps. Step-counting is automated via a pre-specified regex + LLM-judge pipeline; inter-rater reliability against a 100-item human-coded subsample must exceed Cohen's κ = 0.7 before automated counts are used in the primary analysis.

---

## 6. Analysis Plan

### 6.1 Primary analysis model

**[AMENDED v0.4: A2]** The primary model is a hierarchical Bayesian logistic regression on the **assigned-length** factor `long` (eq. 6.1′). The realized-steps model (eq. 6.1, below) is retained as the secondary analysis R7 (§6.4).

$$
y_{ij} \;\sim\; \mathrm{Bernoulli}\!\left( \sigma(\eta_{ij}) \right)
$$

$$
\eta_{ij} \;=\; \beta_0 + \beta_1 \cdot \mathrm{long}_j + \beta_2 \cdot \mathrm{regime}_i + \beta_3 \cdot (\mathrm{long}_j \times \mathrm{regime}_i) + \alpha_{m(ij)} + \gamma_{m(ij)} \cdot \mathrm{long}_j + u_i \tag{6.1$'$}
$$

where $\sigma$ is the logistic function, `long_j ∈ {0,1}` is the assigned-length factor (§5.2), `regime_i ∈ {0,1}` is the high-regime indicator, $\alpha_m, \gamma_m$ are the model-level random intercept and random slope on `long` (the cross-model heterogeneity of the length effect), $u_i$ is the item-level random intercept, $m(ij)$ indexes the model. Because `long` is binary, β3 is the **difference-in-differences** of accuracy between the long and short conditions across regimes — the cleanest expression of the H1 dissociation.

Priors (unchanged from v0.2 §6.1):
- $\beta_0, \beta_2 \sim \mathcal{N}(0, 2.5)$ (weakly informative on the logit scale)
- $\beta_1, \beta_3 \sim \mathcal{N}(0, 1)$
- Random effect SDs $\sim \mathrm{HalfNormal}(1)$

Implementation: PyMC (Bayesian sampling), non-centered random effects. 4 chains, 2000 warm-up + 2000 sampling iterations each. Convergence: $\hat{R} < 1.01$, ESS > 400 per parameter.

**Secondary model (eq. 6.1; demoted from primary, AMENDED v0.4: A2 — now analysis R7).** The original realized-steps model is retained for the mechanistic/dose analysis, with the realized step count standardized within model:

$$
\eta_{ij} \;=\; \beta_0 + \beta_1 \cdot \mathrm{steps}_{ij} + \beta_2 \cdot \mathrm{regime}_i + \beta_3 \cdot (\mathrm{steps}_{ij} \times \mathrm{regime}_i) + \alpha_{m(ij)} + \gamma_{m(ij)} \cdot \mathrm{steps}_{ij} + u_i \tag{6.1}
$$

In R7 the assigned condition instruments the realized step count (two-stage / mediation estimate), correcting the endogeneity that motivated the primary switch. It is reported, not gated.

### 6.2 Primary inference statistic

**[AMENDED v0.3; re-targeted AMENDED v0.4: A2.]** The decision-gate *form* is unchanged from Amendment 1; it is applied to the β3 of the primary model eq. 6.1′ (assigned-length interaction). H1 is evaluated by **two jointly-required quantities**:

1. **Directional:** $\mathrm{Pr}(\beta_3 < 0 \mid \mathrm{data}) > 0.95$.
2. **Magnitude:** the posterior-median **robust implied accuracy drop** exceeds **6 percentage points**.

The *robust implied accuracy drop* is the high-regime short→long change in predicted accuracy evaluated at the **empirical high-regime base rate**, depending only on the posterior of $(\beta_1 + \beta_3)$. It is independent of the global intercept (weakly identified) and, for the binary `long` contrast, is simply the base-rate-anchored short-vs-long accuracy difference. *(For the secondary realized-steps R7, the same estimand is computed over the 10th–90th percentile of the within-model standardized step count, as in Amendment 1.)* The 6 pp floor was selected by `power_curve.py`: among candidate floors {6, 8, 10} it maximised power at equal (zero) Type I error.

The v0.2 statistic — $\mathrm{Pr}(\beta_3 < 0 \;\text{AND}\; |\beta_3| > |\beta_1| \mid \mathrm{data})$, the full-reversal probability — is **retained as a reported effect size**, not the confirmatory gate.

### 6.3 Effect-size reporting

Reported alongside the posterior probability:
- Posterior median and 95% credible interval for $\beta_3$ and $\beta_1$.
- Implied within-bin slope of accuracy on step count for both high- and low-regime, on the probability scale (computed via posterior-predictive simulation).
- Implied accuracy drop in percentage points from minimum to maximum observed step count in the high-regime bin, for each model.
- **[v0.3]** The robust implied accuracy drop (10th–90th percentile step range at the empirical base rate; the §6.2 gating estimand) with its 95% credible interval, and the full-reversal probability $\mathrm{Pr}(\beta_3<0 \wedge |\beta_3|>|\beta_1|)$.

### 6.4 Sensitivity / robustness checks (pre-specified)

Performed regardless of primary result; reported in supplementary:

- **R1**: Re-fit (6.1) with paragraph-level rather than sentence-level step segmentation.
- **R2**: Re-fit (6.1) using terciles rather than quartiles for regime binning, and on the full continuous regime score (no binning) with a linear interaction term.
- **R3**: Re-fit (6.1) using the $(a) + (b)$ regime score (dropping calibration error), per the §3.7 simulate-and-recover finding.
- **R4**: Re-fit (6.1) on items not used in regime-score calibration (held-out subset), to rule out double-dipping.
- **R5**: Fit the primary model (6.1′) separately per model, report consistency of $\beta_3$ sign across models. **[AMENDED v0.4: A2]** the realized-steps per-model sign consistency attaches to R7.
- **R6 [ADDED v0.3]**: Re-fit the primary model with an added **item-level random slope** on the length factor (the model currently carries only an item random intercept). Simulation indicated that item-to-item effect heterogeneity, if unmodeled, can understate uncertainty; R6 checks the primary result is robust to modeling it. *(Note re R2: simulation found the continuous-regime model does not outperform the quartile-binned primary — `design_sweep.md` — so R2 remains a robustness check, not a primary substitute.)*
- **R7 [ADDED v0.4: A2] — realized-steps IV / mediation.** Re-estimate the per-step effect using the assigned condition as an instrument for the realized step count (two-stage / mediation estimate; the secondary eq. 6.1 of §6.1), and report the realized-steps interaction for continuity with the registered v0.2 model. This keeps the now-secondary mechanistic regressor visible alongside the primary ITT estimand and documents the endogeneity correction. R1 (alternative step segmentation) and the per-model realized-steps sign check (R5) attach here. C5 is excluded from the R7 ordinal dose contrast (non-monotone realized steps; §3.2), but remains an assigned-`long` trial in the primary model.

### 6.5 Multiple comparisons

Primary inference is a single test (the posterior probability for H1). H2 and H3 are confirmatory but logically distinct hypotheses tested separately; no correction is applied across H1/H2/H3 because each addresses a different theoretical claim. Robustness checks R1-R6 are not used to update the confirmatory inference; they probe the conditional invariance of the result.

Exploratory analyses (E1-E3) are reported with Bayes factors and the explicit "exploratory" label; they are not used to support confirmatory claims.

---

## 7. Inference Criteria and Falsification

### 7.1 Confirmation (positive result)

**[AMENDED v0.3; re-targeted AMENDED v0.4: A2 to the β3 of eq. 6.1′.]** H1 is confirmed iff **both**:
1. $\mathrm{Pr}(\beta_3 < 0 \mid \mathrm{data}) > 0.95$, and
2. the posterior-median robust implied high-regime accuracy drop (§6.2) exceeds **6 percentage points**.

The low-regime-slope sign and the full-reversal probability $\mathrm{Pr}(\beta_3<0 \wedge |\beta_3|>|\beta_1|)$ are reported alongside but no longer gate confirmation.

### 7.2 Falsification (negative result that updates the framework)

**[AMENDED v0.3; re-targeted AMENDED v0.4: A2 to the β3 of eq. 6.1′.]** H1 is falsified iff:
- $\mathrm{Pr}(\beta_3 \geq 0 \mid \mathrm{data}) > 0.95$ (the interaction is in the opposite direction with high posterior probability), or
- $\mathrm{Pr}(\beta_3 < 0 \mid \mathrm{data}) > 0.95$ but the robust implied accuracy drop is < **3 percentage points** (the directional effect is real but practically negligible — the framework predicts a substantive failure mode, not a noise-level effect).

### 7.3 Inconclusive

Any other outcome (e.g., $\mathrm{Pr}(\beta_3 < 0) \in [0.5, 0.95]$, or directional effect with 5-10pp implied drop). Reported transparently as inconclusive; framework remains plausible but unconfirmed.

### 7.4 What we will NOT do

- Run additional analyses to "rescue" a non-confirmatory result and report them as confirmatory.
- Re-bin items or change regime thresholds after seeing primary results.
- Cherry-pick subsets of models or items to recover effects.
- Add or remove items, conditions, or models after pre-registration without amendment.

---

## 8. Confound Controls

The three confound mechanisms identified in §3.8 of the operationalization chapter receive paired control item sets:

### 8.1 Prompt-sensitivity baseline (controls signature a)

Reference set of 10 items with well-attested ground truth (textbook arithmetic, well-known historical facts, standard math identities). Pre-registered prediction: reference items do NOT score in the high-regime bin. If they do, signature (a) is contaminated by base-rate prompt sensitivity; we apply per-model prompt-sensitivity correction (subtract per-model baseline cross-prompt variance on reference items) and re-run the primary analysis.

### 8.2 Aleatory control (controls signature b)

Set of 10 items with high inherent stochasticity but no meta-uncertainty (e.g., "predict the outcome of a fair coin flip"). Pre-registered prediction: aleatory items may show high cross-seed variance but should NOT show the CoT-degradation pattern. If they do, the regime score is confounded with aleatory uncertainty.

### 8.3 Per-model calibration baseline (controls signature c)

For each model, calibration error on a fixed 30-item well-defined probe set is computed and subtracted from per-item calibration error before z-standardization. If signature (c) carries weight in the final regime score, this correction prevents per-model calibration quirks from dominating the regime assignment.

### 8.4 Reporting

All three control comparisons are reported in the primary results section as confirmatory tests of the regime score's validity. A failure on any one is reported transparently and the affected signature is either corrected (via the specified procedure) or dropped from the aggregator.

---

## 9. Deviations Protocol

### 9.1 Pilot-driven amendments (permitted)

The §4 pilot may reveal that:
- The regime score does not separate items in the expected direction.
- Step counting is unreliable below the pre-specified κ = 0.7 threshold.
- One or more conditions produces uninterpretable responses.

If any of the above occurs, the operationalization is revised before full data collection, and this pre-registration is **amended** (timestamped, visible on OSF, with the change rationale documented). Amendment occurs *before* the full study begins, not after data is collected.

### 9.2 Model-substitution rule (permitted)

If a specific model is unavailable at data collection (deprecated, weights removed, infrastructure-incompatible), it is substituted by the closest-size instruction-tuned open-weight model released before this pre-registration's timestamp. Substitutions are logged in the deviations report.

### 9.3 Post-data-collection deviations (not permitted for confirmatory analyses)

After full-study data collection begins, no changes to (6.1), the regime bins, the conditions, or the items are allowed for the confirmatory analyses. Any post-collection observation that warrants further investigation is reported as exploratory.

### 9.4 Disclosure

All deviations — pilot-driven or otherwise — are disclosed in the manuscript and on the OSF project page. Even minor deviations (e.g., dropping one item due to data corruption) are logged.

---

## 10. Data and Code Sharing

### 10.1 Code

All code (data collection pipeline, regime-score estimator, step-counting pipeline, analysis scripts, plotting code) released under MIT license. Repository URL to be inserted at first commit (target: `https://github.com/<EvolutionAIry-handle>/free-energy-heuristics`); see Appendix C for the OSF-submission addendum protocol. Release at manuscript submission with immutable git tag `v1.0`.

### 10.2 Data

- Raw LLM responses (per model, per item, per condition, per replication): JSON files, CC-BY-4.0, hosted on OSF.
- Item set with regime scores: CSV, CC-BY-4.0, hosted on OSF.
- Analysis-ready data frame (responses × predictors × outcome): Parquet, CC-BY-4.0, hosted on OSF.
- Hierarchical Bayes posterior samples: NetCDF (via ArviZ), hosted on OSF.

### 10.3 Replicability commitments

- Random seeds for all LLM samplings and Bayesian chains are recorded and released with the data.
- A single-command replication script reproduces all primary and supplementary results from the released data and code.
- Hardware environment (GPU model, CUDA version, library versions) is documented.

---

## 11. Timeline

- T0 (week 0): Pre-registration upload to OSF; theoretical pre-print posted to arXiv.
- T0 + 1 week: §4 pilot data collection (Mistral-7B + 10 frames, all conditions).
- T0 + 2 weeks: Pilot analysis; pre-registration amendment if needed.
- T0 + 3 to 6 weeks: Full data collection across **[AMENDED v0.4: panel]** 7 models (5 local + 2 frontier) × 79 frames × 5 conditions × 5 replications.
- T0 + 7 weeks: Analysis, manuscript draft.
- T0 + 9 weeks: Adversarial review pass.
- T0 + 10 weeks: arXiv full paper.
- T0 + 12 weeks: Submission to peer-reviewed venue (target: NeurIPS / ICML / Cognition / Psychological Review depending on framing).

---

## Appendix A: Theoretical Background (one-paragraph summary for reviewers)

The active-inference framework defines an optimal policy as the minimizer of expected free energy $E[G(a, k)]$ over actions $a$ and cue counts $k$. Under meta-uncertainty — uncertainty over the precision of one's own prior, formalized as a Gamma prior on the inverse-variance scalar $\tau$ — the marginal benefit of an additional cue decomposes into a positive epistemic gain about the state $s$ and a positive meta-cost driven by sharpening of the meta-precision posterior (Lemma 2.4.1). In the high-meta-uncertainty regime $\sigma_\tau^2 \geq \tau_{\mathrm{regime}}$, the meta-cost eventually dominates and additional cues increase expected free energy (Theorem 2.6.1). This predicts a U-shaped accuracy-versus-CoT-length relationship for tasks satisfying meta-uncertainty, in contradiction to the prevailing "more reasoning = better" narrative for LLM reasoning systems. The present study tests the prediction directly.

## Appendix B: AsPredicted-Format Summary (alternative submission)

For users preferring AsPredicted.org's 9-question format:

1. **Have any data been collected for this study already?** No.
2. **What's the main question being asked?** Does additional CoT reasoning degrade LLM accuracy on high-meta-uncertainty items, as predicted by the meta-precision cue-truncation theorem?
3. **Describe the key dependent variable(s) specifying how they will be measured.** Binary accuracy of LLM answer against gold standard (or expert-rated coherence for Knightian items). Step count = sentence count in CoT trace meeting the pre-specified inferential criterion.
4. **How many and which conditions will participants be assigned to?** 5 length-graded CoT conditions × **[AMENDED v0.4: panel]** 7 LLMs (5 local + 2 frontier) × 79 items × 5 replications.
5. **Specify exactly which analyses you will conduct.** **[AMENDED v0.4: A2]** Hierarchical Bayesian logistic regression with **assigned-length (short C1 vs long C2–C5) × regime** interaction (eq. 6.1′ of the full pre-registration). Posterior probability of the directional interaction β3 < 0. The realized-steps × regime interaction (eq. 6.1) is reported as a secondary instrumented/mediation analysis (R7).
6. **Describe exactly how outliers will be defined and handled.** No outlier exclusion. Refused/malformed responses replaced by additional replications; items with >50% refusal across cells dropped, reported.
7. **How will you determine sample size?** **[AMENDED v0.4: panel]** Pre-specified: 13,825 observations total (7 models × 79 items × 5 conditions × 5 reps); ~7,000 in primary analysis. (Registered-minimum fallback 5,925 / ~3,000.) **[AMENDED v0.4: A2]** Simulation-based power for the assigned-length (ITT) primary test (`condition_power.py`), calibrated to the measured effect: at 5 reps, power 0.73 at a conservative 7 pp high-regime drop, ≈0.97–1.00 at the measured 10–13 pp, Type I = 0.01 (see §4.1; conservative for the 7-model panel). A full-PyMC run of the *actual* registered Bayesian gate (`condition_power_pymc.py`), paired against the proxy on identical data, agreed on 100% of simulations (Bayesian power 0.82 at 7 pp, 1.00 at 10 pp, Type I 0.05) — confirming the proxy is not anti-conservative. The Amendment-1 realized-steps figures (≈0.90 large / ≈0.47 modest) now describe the secondary R7 analysis.
8. **Anything else you would like to pre-register?** Three confound controls (prompt sensitivity, aleatory, calibration baselines), all reported regardless of primary result. Five robustness checks. Falsification criteria explicit.
9. **Manuscript ID / Project name:** FEH-2026 — Free Energy Heuristics: Empirical Test of the Cue-Truncation Theorem.

## Appendix C: Pre-Registration Status (2026-05-14)

### Resolved before submission

- **PI / affiliation:** Alex Bogdan, EvolutionAIry AI Inc. (Toronto, Canada). Sole investigator on this study.
- **Item pool:** Finalized as **FEH-79 pool v0.4** (`feh79_item_pool_v0.4.yaml`; **[AMENDED v0.4]** v0.4 repairs 8 previously un-scorable items relative to v0.3 — corrections to answer keys/extractable-answer formatting only, changing no registered item's meaning or regime classification). The v0.3 description below otherwise stands. Originally finalized as **FEH-79 pool v0.3** (`feh79_item_pool_v0.3.yaml`), 141 total items: 79 Knightian (K1×20 non-recurrent forecasting + K2×20 novel-synthetic + K3×20 open-ended dilemma + K4×19 strategic uncertainty) + 50 controls (R×10 reference + A×10 aleatory + CB×30 calibration probe) + 12 R-arithmetic-calibration items (R-101..R-112; 4 difficulty bands × 3 items, added 2026-05-14 per §4.9.4 pilot finding; reported separately as a per-model competence curve, do *not* enter the H1 confirmatory test). Pool validated by K2 cross-model multi-seed pre-screen (4-provider × 5-seed; 36/36 categorical Knightian items pass) and K3 google-floor check (15 K3-pass-clean + 4 K3-pass-marginal + 0 K3-fail-contaminated; one item — K2-015 — flagged contamination_risk medium per pool-metadata note). Validation records: `Experiments/cross_model_results_v0.3_validation.json` and `Experiments/k3_floor_results_v0.3.json`.
- **Hardware environment:** Pilot and full study run on a single workstation: NVIDIA RTX 4070 Super (12 GB VRAM), Windows 11 Home, CUDA 12.x, Ollama runtime (Mistral-7B-Instruct quantized to Q4_K_M for the pilot; full-run model panel served via Ollama or vLLM as documented per-model in the released config).
- **Bibliography:** Consolidated to `bibliography.md` covering §§1–4; ~38 entries, author-year style.

### To be inserted at submission time (do not block OSF upload)

- [ ] **Theoretical pre-print arXiv DOI** — Chapter 2 not yet posted to arXiv. The pre-registration may reference this study without the DOI; the DOI will be inserted into the header upon arXiv posting (anticipated before full-run start).
- [ ] **GitHub repository URL** — Repository to be created at first commit of the data-collection code. The pre-registration commits to MIT-licensed release at manuscript submission with immutable git tag `v1.0` (per §10.1).
- [ ] **OSF project URL** — Auto-generated by OSF at submission time and self-referential.

### To be completed during the pilot (per §4.9.2 P2, not before submission)

- [x] **Step-counting pipeline κ-validation (heuristic vs LLM-judge)** — ✅ **completed 2026-05-14.** The §4.9 pilot generated 150 responses (Mistral-7B × 10 frames × 5 conditions × 3 replications). The fast heuristic step-counter (Pass-1 baseline of `step_counter.py`) was validated against the LLM-judge (Pass-2 primary; Claude `claude-sonnet-4-5-20250929`) on a 539-sentence stratified subsample drawn from the rep=1 cells of the pilot. Result: **Cohen's κ = +0.802** vs threshold 0.70 (raw agreement 91.5%; n = 539; 0 judging errors). Per-condition κ ranges 0.61–0.92 for C2–C5; C1 has only 10 sentences and is uninformative. See `Experiments/step_counter_kappa.py` + `Experiments/kappa_validation.{json,md}`. The heuristic is now validated as the Pass-1 step-counter for the full study; the LLM-judge remains the §4.8.2 primary measurement on a per-cell sub-sample.
- [ ] **Human-coded step-counter validation** — Planned for §4.10 Phase 5 (annotation rubric validation). 100-response subsample (20 per condition), 2 trained annotators, threshold κ ≥ 0.7 against the LLM-judge. Not a precondition for full data collection (the heuristic-vs-LLM-judge κ above is the operational gate).

### Completed before full data collection (Amendment 1, 2026-05-29)

- [x] **Confirmatory analysis pipeline implemented and validated.** Eq. 6.1 implemented in `Experiments/confirmatory_analysis.py`; validated by synthetic parameter recovery and simulation-based operating characteristics (`validate_analysis.py` → `analysis_validation.md`): clean convergence (R̂ = 1.000, ESS 1.6k–2.0k), 95% CrIs cover the true β3 in all scenarios, false-confirmation rate 0/30 under the null.
- [x] **Power analysis upgraded and gate amended (pre-data).** The original §6.2 full-reversal gate had ~7% power at the target effect and the §4.1 normal-approximation (0.92) was unreliable. Amendment 1 (`Preregistration_amendment_1_2026-05-29.md`) reformulates the primary test to directional + robust magnitude floor (power ≈0.90 large / 0.47 modest, Type I 0; `power_curve.md`), demotes the full-reversal requirement to a reported effect size, and adds robustness check R6. A sweep confirmed more items do not raise power (`design_sweep.md`). Filed before any confirmatory data collection, permitted under §9.1.

### Completed before full data collection (Amendment 2 + panel/replication decision, 2026-05-30)

- [x] **Effect size + σ_g calibration run (pre-data).** A separate pre-registered feasibility study (`Calibration_protocol.md`) ran 3 local models × 28 objectively-scorable Knightian items + 14 controls × 5 conditions × 3 reps (1,890 cells, 0 errors; `Experiments/calibration_responses.json`, `calibration_analysis.{md,json}`). It measured a clear descriptive H1 dissociation (high-regime cbd-correctness falls ~7 pp clean C1→C4 / ~13 pp C1→C5 while low-regime accuracy is flat) but revealed the registered realized-steps regressor is mis-specified for the confirmatory estimand (endogenous; within-model standardization can invert the interaction sign; threshold-not-linear). This motivated **Amendment 2** (`Preregistration_amendment_2_2026-05-30.md`): primary regressor → assigned-length ITT (`long`), primary model → eq. 6.1′, simulation-based power (`condition_power.py`), robustness R7. Descriptive-only; not part of the confirmatory sample.
- [x] **Frontier feasibility read (pre-data).** Claude Sonnet 4.5 + GPT-4o on the calibration design (672 cells, 0 errors; `Experiments/frontier_read.py`, `frontier_read_analysis.{md,json}`) confirmed H1 holds *off the small-model floor*: pooled assigned short→long high-regime drop +10.4 pp [+2.1, +18.8], difference-in-differences +9.2 pp [+0.9, +17.9], with GPT-4o floored on only 3/28 high items. This motivated the **panel/replication decision**: panel 5 local → 5 local + 2 frontier (7 models), replications 3 → 5 (§3.3, §3.5, §4.4, §4.1). Manipulation-check finding: GPT-4o complies with the C1 direct-answer instruction (0 realized steps) while Claude reasons under C1 regardless (~4.7 steps) — vindicating the assigned-length (ITT) regressor over realized steps. Descriptive-only; not part of the confirmatory sample.
- [x] **Full-PyMC power confirmation (pre-data).** The §4.1 power table is a fast IRLS/Wald proxy; to verify it is not anti-conservative for the registered Bayesian gate, `Experiments/condition_power_pymc.py` re-ran the *actual* registered gate (eq. 6.1′ with model random intercept + slope, item random intercept, N(0,1) shrinkage on β3) on data from the same calibration-anchored DGP, **paired** against the IRLS proxy on identical datasets (5-model design, reps = 5, 40 sims/cell at 0 / 7 / 10 pp). The two gates **agreed on 100% of simulations** (120/120); Bayesian power 0.82 at 7 pp and 1.00 at 10 pp, Type I 0.05 at the null. The Bayesian random-effects machinery introduces no detectable power penalty. See `Experiments/condition_power_pymc.{md,json}`.
- [x] **LaTeX-robust numeric scoring (analysis-code correction, pre-data).** The frontier read exposed that capable models format numeric answers as LaTeX under CoT (e.g. `\(\frac{1}{4}\)`), which the registered numeric parser could not read — mis-scoring correct answers *by condition* and manufacturing a spurious control-regime effect. The canonical scorer (`Experiments/calibration_analysis.py::numeric_candidates`) now normalizes LaTeX before matching; the change is a strict superset (only adds numeric candidates — verified on calibration data: 15 cells flip wrong→correct, 0 correct→wrong) and is covered by `Experiments/test_latex_scoring.py` (17 assertions). It corrects analysis code only; no registered hypothesis, item, or decision rule changes.
- [x] **Numeric-scorer rounding-precision hardening (analysis-code correction, pre-data).** ✅ **resolved 2026-05-31.** The gold-match relative tolerance (0.1%) was too tight for frontier models that round to varying precision (e.g. 1/505 → 0.001982 vs gold 0.00198) — the *same class* of verbosity-correlated confound as the LaTeX issue. Replaced the blanket tolerance in `Experiments/calibration_analysis.py::gold_match` with **significant-figure matching**: integer golds (R-set arithmetic competence) require exact match (an estimate is a genuine failure); non-integer golds (probabilities stored as roundings of an exact real) match at the coarser of the two operands' significant figures, floored at 3 sig figs (exact fractions get full precision). A blanket tolerance was explicitly rejected after a false-positive audit — the measured minimum cross-gold gap is 0.53% (gold 100 vs 100.531), so a 1% tolerance would cross-match distinct golds, and no fixed tolerance can separate a legitimate 3-sf rounding (0.706 for 0.7063, accept) from a genuine 3rd-sf value error (0.703 for 0.7063, reject). Covered by `Experiments/test_latex_scoring.py` (extended to 27 assertions, all pass). Re-scoring the frontier read with the fix left the difference-in-differences robust (pooled DiD +9.2 → +9.8 pp [+1.8, +18.2]; control regime flatter). Corrects analysis code only; no registered hypothesis, item, or decision rule changes.
