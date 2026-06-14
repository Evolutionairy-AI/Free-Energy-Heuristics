# Pre-Registration: Free Energy Heuristics Full Empirical Study

**Working title:** Chain-of-Thought Reasoning Degrades LLM Accuracy Under Knightian Uncertainty: A Confirmatory Test of the Meta-Precision Cue-Truncation Theorem

**Principal investigator:** Alex Bogdan (EvolutionAIry AI Inc., Toronto, Canada). Affiliation independent.

**Theoretical pre-print:** Chapter 2 (FEH Theoretical Foundation, Working Draft v0.7) and Chapter 3 (FEH Operationalization, Working Draft v0.1) — to be posted to arXiv prior to data collection and cited by DOI in this pre-registration.

**Pre-registration platform:** OSF Pre-registration (https://osf.io/prereg). AsPredicted format included as Appendix B for users who prefer the lighter-weight registry.

**Submission status:** v0.1 draft, to be uploaded after author final review.

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

**H1**: The interaction coefficient $\beta_3$ in the analysis model (Section 6, eq. 6.1) is negative with magnitude sufficient to reverse the sign of the steps-on-accuracy slope in the high-regime bin:

$$
\beta_1 > 0 \quad \text{AND} \quad \beta_1 + \beta_3 < 0
$$

Equivalently: $\mathrm{slope}_{\mathrm{high-regime}} < 0 < \mathrm{slope}_{\mathrm{low-regime}}$.

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

Within-item × within-model × between-condition experimental design. Items are nested within regime bins; each item is presented to each model under each of 5 conditions, with 3 replications per (model, item, condition) cell.

### 3.2 Conditions (independent variable: CoT length)

Five conditions, length-graded:

- **C1 — None**: direct answer prompt, no CoT allowed (suppression via "Answer in one word only.").
- **C2 — Short**: ~3 reasoning steps target ("Think step by step, briefly, in 3 steps or fewer.").
- **C3 — Medium**: ~7 reasoning steps target ("Think step by step in about 7 steps.").
- **C4 — Long**: ~15 reasoning steps target ("Reason through this carefully in about 15 steps.").
- **C5 — Unconstrained**: standard CoT prompt ("Think step by step.") with no length cap.

The realized step count is recorded per response and used as the continuous independent variable in the analysis model (Section 6).

### 3.3 Models (between-model factor)

Five models spanning a ~10× size range, all instruction-tuned, open-weight, runnable on local hardware (NVIDIA RTX 4070 Super, 12GB VRAM) with quantization where required:

- **M1**: Phi-3.5-mini-instruct (3.8B parameters), bf16
- **M2**: Mistral-7B-Instruct-v0.3 (7B), bf16
- **M3**: Qwen-2.5-7B-Instruct (7B), bf16
- **M4**: Qwen-2.5-14B-Instruct (14B), 4-bit AWQ quantization
- **M5**: Qwen-2.5-32B-Instruct (32B), 4-bit AWQ quantization

If a specific model is unavailable at data-collection time, it is substituted by the closest-size instruction-tuned open-weight model released before the pre-registration timestamp. Substitutions are logged in the deviations report.

### 3.4 Items (within-condition factor, nested in regime bin)

Item pool: 79 frames constructed per §3.6 of the operationalization chapter to span Knightian, ambiguity, and aleatory uncertainty types. Items are pre-screened in the pilot (Section 9 of this pre-registration) and binned by regime score; the primary analysis uses items in the top quartile (high-regime) and bottom quartile (low-regime). Middle 50% items are retained for robustness checks but not used in the primary inference.

Expected sample composition after pre-screen:
- High-regime: ~20 items
- Low-regime: ~20 items
- Middle (excluded from primary): ~39 items

### 3.5 Total observations

5 models × 79 frames × 5 conditions × 3 replications = **5,925 observations** total. Primary analysis uses 5 models × ~40 high-or-low-regime items × 5 conditions × 3 replications = **~3,000 observations**.

---

## 4. Sampling Plan

### 4.1 Sample size justification (power analysis)

Effect size of interest: difference of slopes corresponding to ~10 percentage points accuracy change across the observed CoT-step range (from §3.5 of the operationalization chapter). At the per-(model, item, condition, replication) observation level this corresponds to a standardized effect size of approximately Cohen's d = 0.3.

Power for detecting the interaction $\beta_3$ in the hierarchical model (Section 6) is conservatively estimated by treating the model-level random effect as fixed and computing within-item paired-condition power. With N = 40 items, 5 conditions per item, and effect size d = 0.3:

- Expected power at α = 0.05 (two-sided): approximately 0.92
- Expected power at α = 0.01 (two-sided): approximately 0.82

Both exceed the conventional 0.80 threshold. The full hierarchical specification (model-level random effects) gives additional power via partial pooling across models; we therefore consider the study adequately powered to detect the pre-specified effect size.

### 4.2 Sampling of items

Items are drawn from a single fixed pool of 79 frames developed in §4 (benchmark design chapter). No replacement, no oversampling. Pilot pre-screen on Mistral-7B + 10 representative frames identifies whether the regime score successfully separates items as predicted; if it does not, the operationalization (§3.2-§3.3) is revised before full data collection and this pre-registration is amended (Section 9).

### 4.3 Sampling of conditions

All 5 conditions are presented for every (model, item) pair. No condition is added or dropped between pilot and full study unless documented as a pre-data deviation.

### 4.4 Replications

Three replications per (model, item, condition) cell. Replications differ in: (a) sampling seed (each replication uses a distinct random seed for the model's temperature-T sampling), (b) prompt phrasing (where signature (a) of the regime score is computed across replications, the prompts are systematically varied; for the primary analysis, the canonical phrasing is used in replication 1, and replications 2-3 use rephrasings).

### 4.5 Exclusion criteria

- Refused responses (model declines to answer): excluded from the analysis cell, replaced by an additional replication where possible.
- Malformed responses (cannot extract an answer): same handling.
- Items with >50% refusal rate across (model, condition) cells: dropped from primary analysis, reported in supplementary.

### 4.6 Stopping rule

Data collection runs to the pre-specified sample size (5,925 observations). No optional stopping. No interim analyses on the confirmatory hypothesis until data collection is complete.

---

## 5. Variables

### 5.1 Dependent variable

**Accuracy** ($y_{ij}$): binary indicator of whether the model's extracted answer matches the gold-standard answer for the item. For Knightian items where no single objective ground truth exists, accuracy is replaced by **expert-panel-rated coherence** (3-expert majority on a 5-point coherence scale, binarized at the median). Pre-registered: this replacement applies only to items in the "Knightian" sub-category of the benchmark.

### 5.2 Primary independent variables

- **steps_{ij}**: realized number of reasoning steps in the model's CoT trace. Computed by sentence-level segmentation of the response (with paragraph-level robustness check in §3.8).
- **regime_i**: binary high-regime indicator (1 = top quartile of regime score, 0 = bottom quartile).
- **regime × steps**: interaction term (the primary test).

### 5.3 Controls / random effects

- **model**: 5-level factor entered as model-level random intercept and random slope on steps.
- **item**: random intercept (within regime bin).
- **condition**: 5-level fixed effect controlling for prompt template baseline (the "target" CoT length is not the only thing the prompt manipulates; the random slope on the realized step count is the substantive variable).
- **replication seed**: not modeled (averaged within cell).

### 5.4 Operational definition of step count

A reasoning step is a sentence in the CoT trace that contains at least one of: (a) inferential connective ("therefore", "so", "because", etc.), (b) intermediate computation, (c) intermediate claim about the task. Sentences that only meta-comment ("Let me think about this") are not counted as steps. Step-counting is automated via a pre-specified regex + LLM-judge pipeline; inter-rater reliability against a 100-item human-coded subsample must exceed Cohen's κ = 0.7 before automated counts are used in the primary analysis.

---

## 6. Analysis Plan

### 6.1 Primary analysis model

Hierarchical Bayesian model:

$$
y_{ij} \;\sim\; \mathrm{Bernoulli}\!\left( \sigma(\eta_{ij}) \right)
$$

$$
\eta_{ij} \;=\; \beta_0 + \beta_1 \cdot \mathrm{steps}_{ij} + \beta_2 \cdot \mathrm{regime}_i + \beta_3 \cdot (\mathrm{steps}_{ij} \times \mathrm{regime}_i) + \alpha_{m(ij)} + \gamma_{m(ij)} \cdot \mathrm{steps}_{ij} + u_i \tag{6.1}
$$

where $\sigma$ is the logistic function, $\alpha_m, \gamma_m$ are model-level random intercept and random slope, $u_i$ is item-level random intercept, $m(ij)$ indexes the model for observation $(i, j)$.

Priors:
- $\beta_0, \beta_2 \sim \mathcal{N}(0, 2.5)$ (weakly informative on the logit scale)
- $\beta_1, \beta_3 \sim \mathcal{N}(0, 1)$ (weakly informative on the standardized step scale)
- Random effect SDs $\sim \mathrm{HalfNormal}(1)$

Implementation: PyMC (Bayesian sampling). 4 chains, 2000 warm-up + 2000 sampling iterations each. Convergence: $\hat{R} < 1.01$, ESS > 400 per parameter.

### 6.2 Primary inference statistic

**Posterior probability that $\beta_3 < 0$ AND $|\beta_3| > |\beta_1|$** (i.e., the slope reverses sign in the high-regime bin), denoted $\mathrm{Pr}(H1 \mid \mathrm{data})$.

Confirmatory threshold: $\mathrm{Pr}(H1 \mid \mathrm{data}) > 0.95$.

### 6.3 Effect-size reporting

Reported alongside the posterior probability:
- Posterior median and 95% credible interval for $\beta_3$ and $\beta_1$.
- Implied within-bin slope of accuracy on step count for both high- and low-regime, on the probability scale (computed via posterior-predictive simulation).
- Implied accuracy drop in percentage points from minimum to maximum observed step count in the high-regime bin, for each model.

### 6.4 Sensitivity / robustness checks (pre-specified)

Performed regardless of primary result; reported in supplementary:

- **R1**: Re-fit (6.1) with paragraph-level rather than sentence-level step segmentation.
- **R2**: Re-fit (6.1) using terciles rather than quartiles for regime binning, and on the full continuous regime score (no binning) with a linear interaction term.
- **R3**: Re-fit (6.1) using the $(a) + (b)$ regime score (dropping calibration error), per the §3.7 simulate-and-recover finding.
- **R4**: Re-fit (6.1) on items not used in regime-score calibration (held-out subset), to rule out double-dipping.
- **R5**: Fit (6.1) separately per model, report consistency of $\beta_3$ sign across models.

### 6.5 Multiple comparisons

Primary inference is a single test (the posterior probability for H1). H2 and H3 are confirmatory but logically distinct hypotheses tested separately; no correction is applied across H1/H2/H3 because each addresses a different theoretical claim. Robustness checks R1-R5 are not used to update the confirmatory inference; they probe the conditional invariance of the result.

Exploratory analyses (E1-E3) are reported with Bayes factors and the explicit "exploratory" label; they are not used to support confirmatory claims.

---

## 7. Inference Criteria and Falsification

### 7.1 Confirmation (positive result)

H1 is confirmed iff all three of:
1. $\mathrm{Pr}(\beta_3 < 0 \;\mathrm{AND}\; |\beta_3| > |\beta_1| \mid \mathrm{data}) > 0.95$.
2. The implied accuracy drop in the high-regime bin (between min and max observed step count) exceeds 10 percentage points in the posterior median.
3. The implied slope in the low-regime bin is non-negative in the posterior median.

### 7.2 Falsification (negative result that updates the framework)

H1 is falsified iff:
- $\mathrm{Pr}(\beta_3 \geq 0 \mid \mathrm{data}) > 0.95$ (the interaction is in the opposite direction with high posterior probability), or
- $\mathrm{Pr}(\beta_3 < 0 \mid \mathrm{data}) > 0.95$ but the implied accuracy drop is < 5 percentage points (the directional effect is real but practically negligible — the framework predicts a substantive failure mode, not a noise-level effect).

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
- T0 + 3 to 6 weeks: Full data collection across 5 models × 79 frames × 5 conditions × 3 replications.
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
4. **How many and which conditions will participants be assigned to?** 5 length-graded CoT conditions × 5 LLMs × 79 items × 3 replications.
5. **Specify exactly which analyses you will conduct.** Hierarchical Bayesian logistic regression with steps × regime interaction (eq. 6.1 of the full pre-registration). Posterior probability of the directional interaction.
6. **Describe exactly how outliers will be defined and handled.** No outlier exclusion. Refused/malformed responses replaced by additional replications; items with >50% refusal across cells dropped, reported.
7. **How will you determine sample size?** Pre-specified: 5,925 observations total; ~3,000 in primary analysis. Power 0.92 at α=0.05 for d=0.3 effect.
8. **Anything else you would like to pre-register?** Three confound controls (prompt sensitivity, aleatory, calibration baselines), all reported regardless of primary result. Five robustness checks. Falsification criteria explicit.
9. **Manuscript ID / Project name:** FEH-2026 — Free Energy Heuristics: Empirical Test of the Cue-Truncation Theorem.

## Appendix C: Pre-Registration Status (2026-05-14)

### Resolved before submission

- **PI / affiliation:** Alex Bogdan, EvolutionAIry AI Inc. (Toronto, Canada). Sole investigator on this study.
- **Item pool:** Finalized as **FEH-79 pool v0.3** (`feh79_item_pool_v0.3.yaml`), 129 total items: 79 Knightian (K1×20 non-recurrent forecasting + K2×20 novel-synthetic + K3×20 open-ended dilemma + K4×19 strategic uncertainty) + 50 controls (R×10 reference + A×10 aleatory + CB×30 calibration probe). Pool validated by K2 cross-model multi-seed pre-screen (4-provider × 5-seed; 36/36 categorical Knightian items pass) and K3 google-floor check (15 K3-pass-clean + 4 K3-pass-marginal + 0 K3-fail-contaminated; one item — K2-015 — flagged contamination_risk medium per pool-metadata note). Validation records: `Experiments/cross_model_results_v0.3_validation.json` and `Experiments/k3_floor_results_v0.3.json`.
- **Hardware environment:** Pilot and full study run on a single workstation: NVIDIA RTX 4070 Super (12 GB VRAM), Windows 11 Home, CUDA 12.x, Ollama runtime (Mistral-7B-Instruct quantized to Q4_K_M for the pilot; full-run model panel served via Ollama or vLLM as documented per-model in the released config).
- **Bibliography:** Consolidated to `bibliography.md` covering §§1–4; ~38 entries, author-year style.

### To be inserted at submission time (do not block OSF upload)

- [ ] **Theoretical pre-print arXiv DOI** — Chapter 2 not yet posted to arXiv. The pre-registration may reference this study without the DOI; the DOI will be inserted into the header upon arXiv posting (anticipated before full-run start).
- [ ] **GitHub repository URL** — Repository to be created at first commit of the data-collection code. The pre-registration commits to MIT-licensed release at manuscript submission with immutable git tag `v1.0` (per §10.1).
- [ ] **OSF project URL** — Auto-generated by OSF at submission time and self-referential.

### To be completed during the pilot (per §4.9.2 P2, not before submission)

- [ ] **Step-counting pipeline κ-validation** — The §4.9 pilot generates 150 responses (Mistral-7B × 10 frames × 5 conditions × 3 replications). The step-counting pipeline (regex + LLM-judge per `step_counter.py`) is validated against a human-coded subsample of these responses; required threshold κ ≥ 0.7 (per §4.9.2). If κ falls below threshold, the pipeline is revised and the pilot is re-run before the full study begins. This validation is part of the pre-registered pilot protocol, not a precondition for OSF upload.
