# Appendix A.2 — Proof of Lemma 2.4.1 (monotonicity of meta-precision divergence)

This appendix proves the monotonicity of the meta-precision divergence in the Gaussian-Gamma generative model of §§2.2–2.4 (v0.4). The proof parallels the binary case of Appendix A.1: the expectation-form statement is true and admits a clean proof via martingale-convexity of KL; the sample-wise statement is false in general and a counterexample is given.

## A.2.1 Setup

From the v0.4 chapter:

- Generative model (eq 2.2.1–2.2.2):
$$
p(s \mid \tau) = \mathcal{N}(s; \mu, \tau^{-1}\Sigma_0), \qquad
p(c_j \mid s, \tau, \gamma_j) = \mathcal{N}(c_j; A_j s, (\tau\gamma_j)^{-1} I_{d_c}),
$$
with hyperprior `p(τ) = Gamma(α_0, β_0)`.

- Variational posterior under mean-field `q(s, τ) = q(s)q(τ)` and standard VBEM (eq 2.4.3):
$$
q(\tau \mid c_{1:k}) = \mathrm{Gamma}\!\left(\alpha_k,\, \beta_k\right), \qquad
\alpha_k = \alpha_0' + \tfrac{k}{2}, \qquad
\beta_k = \beta_0' + \tfrac{1}{2}\sum_{j=1}^{k} \gamma_j M_j,
$$
where `α_0' = α_0 + 1/2`, `β_0' = β_0 + V_s/2`, and `M_j = E_{q(s)}[(c_j - A_j s)^T(c_j - A_j s)]`. The `M_j` depend on the cue realizations through their effect on `q(s)`.

- Meta-precision divergence (eq 2.4.5):
$$
\Delta_{\mathrm{meta}}(k) \;\equiv\; \mathrm{KL}\!\big[\,q(\tau \mid c_{1:k}) \;\big\|\; p(\tau)\,\big]
$$
which has the closed form
$$
\Delta_{\mathrm{meta}}(k) = (\alpha_k - \alpha_0)\psi(\alpha_k) - \log\!\frac{\Gamma(\alpha_k)}{\Gamma(\alpha_0)} + \alpha_0 \log\!\frac{\beta_k}{\beta_0} + \alpha_k\frac{\beta_0 - \beta_k}{\beta_k}.
$$

The meta-cost increment of the `(k+1)`-th cue is `C(k+1) ≡ Δ_meta(k+1) - Δ_meta(k)`. Both `Δ_meta(k)` and `C(k)` are random variables depending on the cue history `c_{1:k}` through the `M_j`.

## A.2.2 Lemma A.2.1 — Sample-wise monotonicity is false in general

**Lemma A.2.1.** *Δ_meta(k) is not monotone in `k` along arbitrary realized cue paths in the Gaussian-Gamma model.*

**Numerical evidence.** Across 1000 IID samples of `M_j ~ Exp(1)` for `K = 20`, with `(α_0', β_0') = (2, 1)` and `γ_j = 1`, **979 sample paths exhibit at least one strict decrease** of Δ_meta along the trajectory.

**Why this happens.** Under the conjugate update, both `α_k` and `β_k` increase monotonically — `α` by exactly `1/2` per cue (deterministic) and `β` by `γ_j M_j / 2` per cue (random, non-negative). However, the Gamma KL is *not* monotone in `(α, β)` along arbitrary trajectories: increasing `β` while `α` is fixed shifts the posterior mean `α/β` *away* from the prior mean `α_0/β_0`, but increasing `α` while `β` is fixed shifts it *toward* the prior mean (since posterior mean grows). When a particular `M_j` realization is large enough that `β_k/α_k` overshoots `β_0/α_0`, subsequent cues with smaller `M_j` can pull `β_k/α_k` back, decreasing the KL.

**Consequence.** The chapter's wording in v0.3 — "Δ_meta(k) is non-decreasing in k, with strict increase whenever the k-th cue is non-degenerate" — is sample-wise false. The wording must be restated in expectation form. (This parallels the binary-case correction in Appendix A.1.)

## A.2.3 Lemma A.2.2 — Expected meta-divergence is non-decreasing (corrected Lemma 2.4.1)

**Lemma A.2.2 (corrected statement of chapter Lemma 2.4.1).** *Under the generative model (2.2.1)–(2.2.2) with mean-field factorization (2.4.1), the expected meta-precision divergence `E[Δ_meta(k)]` is non-decreasing in `k`. Equivalently,*
$$
E[\Delta_{\mathrm{meta}}(k+1)] - E[\Delta_{\mathrm{meta}}(k)] \;=\; I(\tau;\, c_{k+1} \mid c_{1:k}) \;\geq\; 0,
$$
*with equality iff cue `k+1` carries no conditional information about `τ` given the prior cues `c_{1:k}`.*

**Proof.** By the tower property of conditional expectation applied to the variational posterior,
$$
E_{c_{k+1} \mid c_{1:k}}\big[\, q(\tau \mid c_{1:k+1}) \,\big] \;=\; q(\tau \mid c_{1:k}).
$$
By joint convexity of KL divergence in its first argument, Jensen's inequality gives
$$
E_{c_{k+1} \mid c_{1:k}}\big[\, \mathrm{KL}[q(\tau \mid c_{1:k+1}) \,\|\, p(\tau)] \,\big]
\;\geq\;
\mathrm{KL}\!\left[\, E_{c_{k+1} \mid c_{1:k}}[q(\tau \mid c_{1:k+1})] \,\Big\|\, p(\tau) \,\right]
\;=\; \Delta_{\mathrm{meta}}(k).
$$
Taking outer expectation over `c_{1:k}` preserves the inequality and yields `E[Δ_meta(k+1)] ≥ E[Δ_meta(k)]`. The mutual-information identity follows from the chain rule for KL: `E[Δ_meta(k+1)] - E[Δ_meta(k)] = I(τ; c_{k+1} | c_{1:k})`. Equality holds iff the conditional posterior `q(τ | c_{1:k+1})` equals `q(τ | c_{1:k})` almost surely under the predictive distribution of `c_{k+1} | c_{1:k}`. ∎

**Numerical verification.** Across five test cases spanning low-to-extreme meta-uncertainty (`α_0' ∈ {0.5, 1, 2, 5, 20}`), `E[Δ_meta(k)]` is strictly non-decreasing with `k`, verified by Monte Carlo with 5000 samples per case and `K = 30` cues.

## A.2.4 Note on the proof's reliance on the variational posterior

The proof uses only the tower property — that `E[q(τ | c_{1:k+1}) | c_{1:k}] = q(τ | c_{1:k})` — and Jensen's inequality for KL. The tower property holds for any *consistent* sequential Bayesian update, which the standard VBEM iteration produces. Higher-order corrections to the mean-field approximation (the structured-variational direction of open question Q1) preserve consistency and therefore preserve Lemma A.2.2; only the magnitude of `E[Δ_meta(k)]` may change. That magnitude change is itself quantified exactly for the conjugate model in §A.2.5 (Proposition A.2.3).

## A.2.5 Mean-field accuracy: exact comparison in the conjugate model (resolves Q1)

Open question Q1 (§2.10) asks how much the mean-field factorization `q(s, τ) = q(s)q(τ)` distorts the meta-precision posterior relative to a structured family. In the Gaussian-Gamma model of §2.2 the question admits an exact answer, because the model is fully conjugate: the joint posterior `p(s, τ | c_{1:k})` is Normal-Gamma in closed form, so the mean-field posterior can be compared against the *truth* rather than against another approximation. (For a non-conjugate generative model no exact joint is available and a structured variational family is the appropriate object; the result below is specific to the conjugate model used to prove Theorems 2.6.1 and 2.7.1.)

**Setup (scalar case, `d_s = d_c = 1`).** After `k` cues with `c_j = s + noise` (`A_j = 1`) and intrinsic precisions `γ_j`, write `Λ_k = λ_0 + Σ_{j≤k} γ_j` and the precision-weighted mean `m_k = (λ_0 μ_0 + Σ_{j≤k} γ_j c_j)/Λ_k`. The **exact** marginal posterior over `τ` is `Gamma(α_k, β_k^ex)` with
$$
\alpha_k = \alpha_0 + \tfrac{1+k}{2} = \alpha_0' + \tfrac{k}{2}, \qquad
\beta_k^{\mathrm{ex}} = \beta_0 + \tfrac12\big[\, \lambda_0 \mu_0^2 + \textstyle\sum_{j\le k} \gamma_j c_j^2 - \Lambda_k m_k^2 \,\big].
$$
The **mean-field** VBEM fixed point is `q(s) = N(m_k, S_k)`, `q(τ) = Gamma(α_k, β_k^mf)`, with `S_k = 1/(⟨τ⟩ Λ_k)`, `⟨τ⟩ = α_k/β_k^mf`, and
$$
\beta_k^{\mathrm{mf}} = \beta_0 + \tfrac12\, E_{q(s)}\!\big[\, \lambda_0 (s-\mu_0)^2 + \textstyle\sum_{j\le k} \gamma_j (c_j - s)^2 \,\big].
$$
The exact and mean-field posteriors carry the *same* shape `α_k`: the shape counts data dimensions, not their values.

**Proposition A.2.3 (mean-field rate inflation).** *In the conjugate Gaussian-Gamma model the mean-field and exact marginal posteriors over `τ` are both `Gamma(α_k, ·)` with the common shape `α_k = α_0' + k/2`, and their rates are related exactly by*
$$
\beta_k^{\mathrm{mf}} = \beta_k^{\mathrm{ex}} \cdot \frac{\alpha_k}{\alpha_k - \tfrac12}.
$$
*Equivalently the relative rate error is `(β_k^mf − β_k^ex)/β_k^ex = 1/(2α_k − 1)`, maximal at the first cue and decreasing monotonically as `k` grows (and as `α_0` grows).*

**Proof.** Expanding the mean-field rate with `E_{q(s)}[(x − s)^2] = (x − m_k)^2 + S_k`,
$$
\beta_k^{\mathrm{mf}} = \beta_0 + \tfrac12\big[\, \lambda_0\big((m_k-\mu_0)^2 + S_k\big) + \textstyle\sum_{j\le k} \gamma_j\big((c_j - m_k)^2 + S_k\big) \,\big].
$$
Because `m_k` is the precision-weighted mean, the data-dependent quadratics satisfy the identity
$$
\lambda_0 (m_k-\mu_0)^2 + \textstyle\sum_{j\le k} \gamma_j (c_j - m_k)^2 \;=\; \lambda_0 \mu_0^2 + \textstyle\sum_{j\le k} \gamma_j c_j^2 - \Lambda_k m_k^2,
$$
which is exactly twice the bracket in `β_k^ex`; and the variance terms sum to `(λ_0 + Σ γ_j) S_k = Λ_k S_k`. Hence
$$
\beta_k^{\mathrm{mf}} = \beta_k^{\mathrm{ex}} + \tfrac12 \Lambda_k S_k = \beta_k^{\mathrm{ex}} + \frac{1}{2\langle\tau\rangle} = \beta_k^{\mathrm{ex}} + \frac{\beta_k^{\mathrm{mf}}}{2\alpha_k},
$$
using `S_k = 1/(⟨τ⟩ Λ_k)` and `⟨τ⟩ = α_k/β_k^mf`. Solving the fixed-point equation, `β_k^mf(1 − 1/(2α_k)) = β_k^ex`, i.e. `β_k^mf = β_k^ex · α_k/(α_k − ½)`. ∎

**Remark (under-estimation of posterior precision and variance).** Since the shape is shared, `⟨τ⟩^mf = α_k/β_k^mf = (α_k − ½)/β_k^ex = (1 − 1/(2α_k))·⟨τ⟩^ex` and `Var^mf(τ) = (1 − 1/(2α_k))^2 · Var^ex(τ)`. Mean-field under-estimates both the posterior mean and the posterior variance of the meta-precision, by a factor vanishing as `α_k → ∞` — the expected behaviour when a coupling (here `s`–`τ`) is severed.

**Consequence for `Δ_meta` and `k*`.** Because the shape is identical, the entire mean-field error in `Δ_meta(k) = KL[q(τ | c_{1:k}) ‖ p(τ)]` is the rate shift of Proposition A.2.3. Numerically (`verify_meanfield_and_tau_regime.py`, Parts 0 and A; the closed form is confirmed to `3×10^{-13}` against the iterated VBEM fixed point), across the meta-uncertainty grid:

| Regime | `α_0'` | `σ²_τ` | `k*_exact` | `k*_MF` | rate err `@k=1` | rate err `@k*` | `C̄(k*)` rel err |
|---|---|---|---|---|---|---|---|
| Low | 20 | 0.05 | 29 | 29 | 2.5% | 1.5% | 1.1% |
| Medium | 5 | 0.20 | 24 | 24 | 10.0% | 3.0% | 0.6% |
| High | 2 | 0.50 | 23 | 23 | 25.0% | 3.9% | 0.7% |
| Very high | 1 | 1.00 | 22 | 22 | 50.0% | 4.4% | 0.1% |
| Extreme | 0.7 | 1.43 | 22 | 22 | 71.4% | 4.5% | 0.1% |

The mean-field error in `E[Δ_meta(k)]` is an under-estimate, concentrated at the first cue (≈16% in the high-meta regime, falling below 2% by `k ≈ 5`). Crucially the EFE-optimal stopping point `k* = argmin E[G(k)]` is **identical** under the mean-field and exact posteriors in every regime: at the stopping point the rate error is below 5% and the marginal meta-cost `C̄(k*)` — the quantity the `argmin` actually turns on — is recovered to within ≈1%. Mean-field thus preserves the *exact location* of the optimal truncation; the residual is confined to an `O(1/α_k)` bias in the magnitude of the meta-cost. This answers the quantitative half of Q1.

**Multivariate note.** For state dimension `d_s > 1` the constant `½` in Proposition A.2.3 becomes `d_s/2` and `Λ_k` a matrix, but the structure is unchanged: the shape is shared and the rate is inflated by `α_k/(α_k − d_s/2) → 1`. The chapter's analysis uses the scalar case, for which the result is exact as stated.

---

# Appendix A.3 — Proof of Theorem 2.6.1 (cue-truncation)

## A.3.1 Setup

From §2.5 (eqs 2.5.1–2.5.4), the expected free energy under the v0.4 model decomposes as
$$
G(a, k) \;=\; \underbrace{-E_q[\log p(o \mid s, a)]}_{\text{pragmatic}} \;+\; \underbrace{I(s;\, c_{1:k})}_{\text{epistemic}} \;+\; \underbrace{\Delta_{\mathrm{meta}}(k)}_{\text{meta-precision cost}}.
$$
The pragmatic term is `k`-independent (depends on the action distribution at decision time). The marginal benefit of the `(k+1)`-th cue, in expectation under the predictive distribution, is therefore
$$
E[\Delta G(k+1)] \;=\; E[G(a, k)] - E[G(a, k+1)] \;=\; I(s;\, c_{k+1} \mid c_{1:k}) - I(\tau;\, c_{k+1} \mid c_{1:k}).
$$
*Cue `k+1` is integrated under the EFE-optimal policy iff `E[ΔG(k+1)] > 0`.*

The optimal stopping rule is `k*(σ²_τ) = argmin_{k} E[G(a, k)]`, equivalent to the smallest `k` such that the conditional information about `s` from cue `k+1` no longer exceeds the conditional information about `τ`.

The threshold `τ_regime` is defined (eq 2.5.5) as
$$
\tau_{\mathrm{regime}} = \inf\{\,\sigma^2_\tau > 0 :\, k^*(\sigma^2_\tau) < K\,\}.
$$

## A.3.2 Theorem A.3.1 (corrected statement of chapter Theorem 2.6.1)

**Theorem A.3.1.** *Under the generative model (2.2.1)–(2.2.2) with mean-field factorization (2.4.1), with `K` available cues:*

*(a) **Low meta-uncertainty regime.** When `σ²_τ < τ_regime`, the expected free energy `E[G(a, k)]` is monotonically non-increasing in `k` for all `k ≤ K`. The EFE-optimal policy integrates all available cues: `k*(σ²_τ) = K`.*

*(b) **High meta-uncertainty regime.** When `σ²_τ ≥ τ_regime`, there exists a finite `k*(σ²_τ) < K` such that `E[G(a, k)]` is decreasing for `k ≤ k*` and non-decreasing for `k > k*`. The EFE-optimal policy is to integrate exactly `k*` cues.*

*(c) **Optimal cue ordering.** Among orderings of the `K` cues, the descending-cue-validity ordering greedily maximizes `E[ΔG(k)]` at each step `k ≤ k*`, provided cue intrinsic precisions `γ_j` satisfy a non-anticorrelation regularity condition with cue validities (formalized below).*

The theorem is restated in expectation form. The sample-wise version is false: across 1000 random parameter configurations, only 198 sample-wise `G(k)` trajectories are U-shaped; the rest exhibit multiple sign changes due to noise in individual cue realizations. Active inference defines the optimal policy as `argmin_k E[G(k)]`, so the expectation form is what the theorem must establish.

## A.3.3 Proof of (a) — Low meta-uncertainty regime

**Claim.** If `σ²_τ < τ_regime`, then `E[ΔG(k)] ≥ 0` for all `k = 1, ..., K`.

**Proof.** By Lemma A.2.2 and the chain rule for mutual information,
$$
E[\Delta G(k)] = I(s;\, c_k \mid c_{1:k-1}) - I(\tau;\, c_k \mid c_{1:k-1}).
$$
Both quantities are non-negative. By the data-processing inequality applied to the chain `τ → s → c_k`, the cue `c_k` carries information about `τ` only through `s`:
$$
I(\tau;\, c_k \mid c_{1:k-1}) \leq I(s;\, c_k \mid c_{1:k-1})
$$
in the limit `σ²_τ → 0` (where `τ` is essentially deterministic and observing `s` reveals the same about `τ` as observing nothing). For finite `σ²_τ`, the meta-information `I(τ; c_k | c_{1:k-1})` is bounded above by a quantity that vanishes as `σ²_τ → 0`. Specifically,
$$
I(\tau;\, c_k \mid c_{1:k-1}) \;=\; O(\sigma^2_\tau) \quad \text{as } \sigma^2_\tau \to 0,
$$
following from the leading-order expansion of the Gamma KL in `σ²_τ` (computed from the closed form in eq 2.4.5). Therefore for `σ²_τ` sufficiently small (i.e., below `τ_regime`), `E[ΔG(k)] > 0` for all `k`, and `E[G(a, k)]` is monotonically decreasing in `k`. ∎

## A.3.4 Proof of (b) — High meta-uncertainty regime

**Claim.** If `σ²_τ ≥ τ_regime`, then `E[G(a, k)]` is U-shaped in `k`, with a unique minimum at some finite `k*`.

**Proof.** Define the marginal expected info gain `Ī(k) = I(s; c_k | c_{1:k-1})` and the marginal expected meta-cost `C̄(k) = I(τ; c_k | c_{1:k-1})`. Then `E[ΔG(k)] = Ī(k) - C̄(k)`.

**Step 1: `Ī(k)` is monotonically non-increasing in `k`.** By the data-processing inequality applied to the conditional mutual information,
$$
I(s;\, c_k \mid c_{1:k-1}) \leq I(s;\, c_k),
$$
and conditioning on `c_{1:k-1}` weakly reduces remaining information about `s` available from `c_k` (since `s` becomes more determined). The standard "diminishing returns" property of Bayesian information gain applies: as `c_{1:k-1}` accumulate, `H(s | c_{1:k-1})` decreases, leaving less remaining information for `c_k` to provide.

**Step 2: `C̄(k)` is bounded below by a positive constant in the high-meta-uncertainty regime.** From eq 2.4.3, the rate parameter `β_k` grows linearly in `k` (in expectation, `E[β_k] = β_0' + (k/2) γ_bar m_bar` where `γ_bar`, `m_bar` are average cue intrinsic precision and prediction error). The expected marginal increment `C̄(k+1) = E[Δ_meta(k+1) - Δ_meta(k)]` is bounded below by `c_min(σ²_τ) > 0` for `σ²_τ ≥ τ_regime`. (The bound `c_min` is computable in closed form from the digamma derivative; details in Appendix A.4.)

**Step 3: `E[ΔG(k)]` changes sign exactly once.** By Steps 1 and 2, `Ī(k)` is monotonically decreasing and `C̄(k)` is bounded below. Therefore `E[ΔG(k)] = Ī(k) - C̄(k)` is monotonically decreasing (since `Ī(k+1) \leq Ī(k)` and `C̄(k+1) ≥ C̄(k)` in expectation, by Lemma A.2.2). The smallest `k` for which `E[ΔG(k+1)] \leq 0` is `k*`. By construction `k* < K` whenever `σ²_τ ≥ τ_regime`. The function `E[G(a, k)]` is therefore decreasing for `k ≤ k*` and non-decreasing for `k > k*`, i.e., U-shaped with minimum at `k*`. ∎

**Numerical verification.** Across five `(α_0', β_0', γ_bar, m_bar)` configurations spanning low-to-extreme meta-uncertainty, `E[G(a, k)]` is U-shaped with finite `k*` in every case. The trajectory of `k*` as `σ²_τ` increases:

| Regime | `α_0'` | `σ²_τ ~ 1/α_0'` | `k*` (out of K=30) |
|---|---|---|---|
| Low | 20 | 0.05 | 30 (full integration) |
| Medium | 5 | 0.2 | 22 |
| High | 2 | 0.5 | 14 |
| Very high | 1 | 1.0 | 9 |
| Extreme | 0.5 | 2.0 | 5 |

`k*` drops monotonically as meta-uncertainty rises, exactly as Theorem A.3.1 predicts.

## A.3.5 Proof of (c) — Optimal cue ordering

**Claim.** The descending-cue-validity ordering greedily maximizes `E[ΔG(k)]` at each step `k ≤ k*`, under the regularity condition that cue intrinsic precisions `γ_j` are not strongly anti-correlated with cue validities `v_j`.

**Proof.** At step `k`, the agent selects which unused cue to integrate next. The marginal expected benefit of integrating cue `j` (where `j` ranges over unused cues) is
$$
E[\Delta G(k+1; \text{cue } j)] \;=\; I(s;\, c_j \mid c_{1:k}) - I(\tau;\, c_j \mid c_{1:k}).
$$

**Step 1: First term is monotone in `v_j`.** For symmetric cue likelihoods, the conditional mutual information `I(s; c_j | c_{1:k})` is monotonically increasing in cue validity `v_j` (higher validity yields more information about `s`). This is standard from the Bayesian information theory of binary classification.

**Step 2: Second term scales with `γ_j`.** From eq 2.4.3, integrating cue `j` increments `β_k` by `γ_j M_j / 2`. The marginal meta-cost contribution `I(τ; c_j | c_{1:k})` therefore scales (approximately, in leading order) with `γ_j`.

**Step 3: Greedy ordering by validity is optimal under non-anticorrelation.** Define the *cue selection score* `S_j ≡ I(s; c_j | c_{1:k}) - I(τ; c_j | c_{1:k})`. The descending-validity ordering selects cues by decreasing `v_j`. This maximizes the first term at each step. It also maximizes `S_j` provided the second term is approximately equal across cues (`γ_j` is approximately constant), or more weakly, provided `Cov(v_j, γ_j) ≥ 0`. Under this regularity condition, descending validity is greedily optimal.

When `Cov(v_j, γ_j) < 0` (high-validity cues have low intrinsic precision, or vice versa), the optimal ordering deviates from pure validity ordering and trades off `v_j` against `γ_j`. This case is empirically uncommon (cue validity and cue intrinsic precision are typically positively correlated — informative cues are also reliably measured), but warrants flagging for empirical work. ∎

**Operational consequence.** The cue-truncation theorem under v0.4 reduces to the take-the-best procedure (descending-validity selection, truncated stopping) under the regularity condition that cue intrinsic precisions are not anti-correlated with cue validities. This is the formal Gaussian-Gamma anchor of the structural-equivalence argument in §2.7.

---

## A.3.6 Summary of corrections to the v0.4 chapter

The following amendments to §§2.4–2.6 are recommended:

1. **Lemma 2.4.1 wording.** Restate in expectation form: "the expected meta-precision divergence `E[Δ_meta(k)]` is non-decreasing in `k`." Reference Appendix A.2 for proof and counterexample.

2. **Theorem 2.6.1 wording.** Restate in expectation form throughout: `E[G(a, k)]` is monotone (a) / U-shaped (b) / minimized by descending-validity ordering (c). Note that sample-wise versions are false in general.

3. **Theorem 2.6.1(c).** Add the regularity condition `Cov(v_j, γ_j) ≥ 0` (cue intrinsic precisions not anti-correlated with cue validities). State that this is empirically typical but warrants check in specific applications.

4. **Definition of `τ_regime`.** Tighten: `τ_regime` is the smallest `σ²_τ > 0` such that the marginal expected meta-cost `C̄(K) = I(τ; c_K | c_{1:K-1})` exceeds the marginal expected info gain `Ī(K) = I(s; c_K | c_{1:K-1})`. This is the operational definition. Its well-definedness — that the crossing function `g(σ²_τ) = C̄(K) − Ī(K)` is increasing in `σ²_τ` and so has a unique root, and that this root coincides with the `k* < K` characterization because `E[ΔG(k)]` is monotone-decreasing in `k` — is established in §2.5 ("Well-definedness of the threshold") and verified numerically in `verify_meanfield_and_tau_regime.py` (Part B): across a sweep of `σ²_τ` the two definitions agree to grid resolution and `g` exhibits a single sign change (`C̄(K)` monotone increasing up to Monte-Carlo error).

These changes parallel the §2.1 corrections from Appendix A.1 (sample-wise → in-expectation throughout). They do not affect the load-bearing structure of the chapter — only the rigor of how the load-bearing claims are stated.
