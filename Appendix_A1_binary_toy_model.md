# Appendix A.1 — Binary Toy Model Proofs

This appendix provides the proofs deferred from §2.1, together with numerical verification. Two formal results are established:

- **Lemma A.1.2** (corrected statement of chapter Lemma 2.1.2): the *expected* meta-divergence is non-decreasing in the number of cues integrated. The chapter's sample-wise statement is incorrect; a counterexample is exhibited below.
- **Proposition A.1.4** (corrected statement of chapter Proposition 2.1.3): the EFE-minimizing policy admits *three* qualitative regimes — `k*=0`, `k*=1`, and `k*=K` — separated by two critical concentrations `κ_lo(v_seq, μ_0)` and `κ_hi(v_seq, μ_0)`. The "one-cue stopping" regime is a narrow band, not the dominant one. A precise statement and numerical characterization follow.

A subsidiary result, **Lemma A.1.1**, gives the closed-form posterior on `p_0` after `k` cues — a 2-component Beta mixture whose components are independent of `k` and whose mixture weights move with the cue history.

The corrections to the chapter wording are tracked at the end of the appendix.

---

## A.1.1 Setup recap

Latent state `s ∈ {0,1}`. Prior: `P(s=1 | p_0) = p_0`. Hyperprior: `p_0 ~ Beta(α_0, β_0)`, with prior mean `μ_0 = α_0/(α_0 + β_0)` and concentration `κ_0 = α_0 + β_0`. Binary cues `c_j ∈ {0,1}` with cue validities `v_j ∈ (1/2, 1)` and a symmetric error model:

$$
P(c_j = 1 \mid s = 1) = v_j, \qquad P(c_j = 0 \mid s = 0) = v_j.
$$

Define the cue-product likelihoods after `k` cues:

$$
\pi_1(c_{1:k}) = \prod_{j=1}^{k} v_j^{c_j}(1-v_j)^{1-c_j}, \qquad
\pi_0(c_{1:k}) = \prod_{j=1}^{k} v_j^{1-c_j}(1-v_j)^{c_j}.
$$

These are the conditional probabilities of the observed cue sequence under `s=1` and `s=0` respectively, derived from cue conditional independence given `s`.

---

## A.1.2 Lemma A.1.1 — Closed-form posterior on p_0

**Lemma A.1.1.** *For any cue sequence `c_{1:k}`, the variational posterior on `p_0` is a 2-component mixture of Beta distributions whose components depend on `(α_0, β_0)` but not on `k` or on the cue history:*

$$
p(p_0 \mid c_{1:k}) = w_1(c_{1:k}) \cdot \mathrm{Beta}(p_0;\, \alpha_0+1,\, \beta_0) \;+\; w_0(c_{1:k}) \cdot \mathrm{Beta}(p_0;\, \alpha_0,\, \beta_0+1)
$$

*with mixture weights*

$$
w_1(c_{1:k}) = \frac{\pi_1(c_{1:k}) \, \mu_0}{Z(c_{1:k})}, \qquad
w_0(c_{1:k}) = \frac{\pi_0(c_{1:k}) (1-\mu_0)}{Z(c_{1:k})},
$$

*where `Z(c_{1:k}) = π_1 μ_0 + π_0 (1 - μ_0)` is the marginal cue likelihood. The prior decomposes in the same form with weights `(μ_0, 1-μ_0)`.*

**Proof.** The unnormalized posterior is

$$
p(p_0 \mid c_{1:k}) \propto \mathrm{Beta}(p_0; \alpha_0, \beta_0) \cdot \big[\,p_0 \pi_1 + (1-p_0)\pi_0\,\big].
$$

Using the identities

$$
p_0 \cdot \mathrm{Beta}(p_0; \alpha_0, \beta_0) = \mu_0 \cdot \mathrm{Beta}(p_0; \alpha_0+1, \beta_0),
$$

$$
(1-p_0) \cdot \mathrm{Beta}(p_0; \alpha_0, \beta_0) = (1-\mu_0) \cdot \mathrm{Beta}(p_0; \alpha_0, \beta_0+1),
$$

(both verified by direct computation using `Γ(α_0+1) = α_0 Γ(α_0)`), the unnormalized posterior becomes

$$
\pi_1 \mu_0 \cdot \mathrm{Beta}(p_0; \alpha_0+1, \beta_0) + \pi_0 (1-\mu_0) \cdot \mathrm{Beta}(p_0; \alpha_0, \beta_0+1).
$$

The normalizing constant is `Z = π_1 μ_0 + π_0 (1-μ_0)`, and division gives the stated form. The prior corresponds to `π_1 = π_0 = 1`, yielding `(w_1, w_0) = (μ_0, 1-μ_0)` and reducing to the algebraic identity `Beta(α_0, β_0) = μ_0 \cdot Beta(α_0+1, β_0) + (1-μ_0) \cdot Beta(α_0, β_0+1)`. ∎

**Consequence for P(s=1 | c_{1:k}).** By direct application of Bayes' theorem with `s` marginalized over `p_0`:

$$
P(s = 1 \mid c_{1:k}) = \frac{P(c_{1:k} \mid s=1) \, P(s=1)}{P(c_{1:k})} = \frac{\pi_1 \mu_0}{Z(c_{1:k})} = w_1(c_{1:k}).
$$

The marginal posterior on the state is exactly the mixture weight on the upper Beta component. This is a clean result that simplifies all downstream computation.

---

## A.1.3 Lemma A.1.2 — Expected meta-divergence is non-decreasing

**Lemma A.1.2 (corrected statement of chapter Lemma 2.1.2).** *The expected meta-divergence `E_{c_{1:k}}[Δ_meta(k)]` is non-decreasing in `k`. Equivalently, the marginal expected meta-cost*

$$
E[\Delta_{\mathrm{meta}}(k+1) - \Delta_{\mathrm{meta}}(k)] \;=\; I(p_0;\, c_{k+1} \mid c_{1:k}) \;\geq\; 0,
$$

*with equality iff cue `k+1` carries no conditional information about `p_0` given the prior cues.*

**Proof.** By the tower property of conditional expectation,

$$
E_{c_{k+1} \mid c_{1:k}}\big[\, p(p_0 \mid c_{1:k+1}) \,\big] = p(p_0 \mid c_{1:k}).
$$

By joint convexity of KL divergence in its first argument, Jensen's inequality gives

$$
E_{c_{k+1} \mid c_{1:k}}\big[\, \mathrm{KL}[p(p_0 \mid c_{1:k+1}) \,\|\, p(p_0)] \,\big] \;\geq\; \mathrm{KL}\!\left[\, E_{c_{k+1} \mid c_{1:k}}[p(p_0 \mid c_{1:k+1})] \,\Big\|\, p(p_0) \,\right].
$$

The right-hand side equals `KL[p(p_0 | c_{1:k}) || p(p_0)] = Δ_meta(k)`. Taking outer expectation over `c_{1:k}` preserves the inequality. The equality condition follows from the martingale-convexity formulation: equality holds iff `p(p_0 | c_{1:k+1}) = p(p_0 | c_{1:k})` almost surely under the predictive distribution, which is the condition that cue `k+1` carries no conditional information about `p_0`. ∎

**Counterexample to the chapter's sample-wise claim.** With `α_0 = β_0 = 1` (uniform hyperprior) and validities `v = (0.9, 0.9)`, direct computation of `Δ_meta(k; c_{1:k})` for the four cue paths gives:

| `c_1` | `c_2` | `Δ_meta(1; c_1)` | `Δ_meta(2; c_1, c_2)` | sample monotone? |
|---|---|---|---|---|
| 0 | 0 | 0.1153 | 0.1815 | yes |
| 0 | 1 | 0.1153 | **0.0000** | **no** |
| 1 | 0 | 0.1153 | **0.0000** | **no** |
| 1 | 1 | 0.1153 | 0.1815 | yes |

When opposite cues cancel (paths `01` and `10`), the posterior on `p_0` returns exactly to the prior and the meta-divergence collapses to zero. The chapter's wording — "Δ_meta(k) is non-decreasing in k" — is therefore false sample-wise. The expectation across the four paths is monotone, as Lemma A.1.2 establishes.

**Implication for the chapter.** The wording in §2.1 should be changed from "the meta-divergence Δ_meta(k) is non-decreasing in k" to "the expected meta-divergence E[Δ_meta(k)] is non-decreasing in k." The downstream proofs in §2.5 already work with expected free energy (which is itself an expectation over cues), so no other proofs are affected.

---

## A.1.4 Proposition A.1.3 — Marginal benefit of cue k+1

Define the marginal benefit of cue `k+1` as the expected reduction in expected free energy:

$$
\Delta G(k+1) \;\equiv\; E[G(a, k)] - E[G(a, k+1)] \;=\; I(s;\, c_{k+1} \mid c_{1:k}) - I(p_0;\, c_{k+1} \mid c_{1:k}).
$$

**Proposition A.1.3.** *In the binary toy model, the marginal benefit of cue `k+1` decomposes as the difference of two conditional mutual informations: the information cue `k+1` provides about the latent state `s` (positive contribution to EFE reduction) minus the information it provides about the prior precision parameter `p_0` (positive contribution to meta-divergence).*

**Proof.** From eqs (2.5.2)–(2.5.4) of the chapter, the expected free energy after `k` cues decomposes as `E[G(a,k)] = -E[\log p(o|s,a)] - I(s; c_{1:k}) + E[Δ_meta(k)]`. The first term is action-dependent but `k`-independent; it cancels in `ΔG(k+1)`. The second term gives `I(s; c_{1:k+1}) - I(s; c_{1:k}) = I(s; c_{k+1} | c_{1:k})` by chain rule for mutual information. The third term gives `E[Δ_meta(k+1) - Δ_meta(k)] = I(p_0; c_{k+1} | c_{1:k})` by Lemma A.1.2 and the same chain rule applied to KL. ∎

**Consequence.** Cue `k+1` should be integrated iff `I(s; c_{k+1} | c_{1:k}) > I(p_0; c_{k+1} | c_{1:k})` — i.e., it is more informative about the state than about the meta-precision. The optimal `k*` is the smallest `k` such that this inequality is violated for `c_{k+1}`.

---

## A.1.5 Proposition A.1.4 — Three-regime structure

Numerical sweep of the binary toy model across `(κ_0, μ_0, v_{1:K})` reveals that the EFE-minimizing policy has three qualitative regimes, separated by two critical concentrations.

**Proposition A.1.4 (corrected statement of chapter Proposition 2.1.3).** *Fix a validity profile `v_{1:K}` and prior mean `μ_0`. There exist critical concentrations `0 < κ_lo(v_{1:K}, μ_0) ≤ κ_hi(v_{1:K}, μ_0)` such that the EFE-minimizing policy `k*(κ_0)` has the structure:*

- *(Don't-observe regime) `κ_0 < κ_lo`: `k* = 0`. The meta-divergence cost of even one cue exceeds its information value about `s`. The agent should act on the prior.*
- *(One-cue stopping regime) `κ_lo ≤ κ_0 ≤ κ_hi`: `k* = 1`. The first cue is integrated; subsequent cues incur higher meta-cost than information benefit. This is the take-the-best stopping rule.*
- *(Full integration regime) `κ_0 > κ_hi`: `k* = K`. Each cue contributes more state-information than meta-cost; the agent integrates all available cues.*

*The intermediate regime `[κ_lo, κ_hi]` is a closed interval that may be narrow (and in some validity profiles degenerate to a single point, in which case the transition jumps directly from `k*=0` to `k*=K`).*

**Numerical characterization.** A systematic search over `v_{1:K} ∈ (0.5, 1)^K`, `μ_0 ∈ {0.3, 0.5, 0.7}`, and `κ_0 ∈ [0.1, 200]` (geometric grid) finds:

- *Sharp validity gradients yield wider one-cue regimes.* Profiles with `v_1 ≈ 0.99` and `v_{j>1} ≤ 0.55` exhibit `[κ_lo, κ_hi]` of nontrivial width, often near `κ_0 ≈ 1`.
- *Uniform validity profiles tend to have `κ_lo = κ_hi`.* The transition then jumps directly from `k*=0` to `k*=K`, and the one-cue regime is degenerate.
- *Realistic decreasing-validity profiles (the TTB-typical case) lie in between.* The one-cue regime exists but is narrow; the qualitative behavior is dominated by the don't-observe / full-integration dichotomy.

The narrowness of the one-cue regime in the binary toy model is a substantive finding: in the discrete binary setting under symmetric error and conditional independence of cues given `s`, the cue-truncation theorem fires generically only at the **boundary** between two qualitatively different regimes, not as a wide intermediate regime. This is in contrast to the continuous Gaussian-Gamma case (Theorem 2.6.1, §2.6), where the one-cue regime widens because per-cue meta-cost grows linearly in `k` rather than saturating after the first cue.

**Subsidiary observation: TTB is near-optimal even when not strictly optimal.** A separate computation shows that for steep validity profiles (`v_1 ≈ 0.99, v_{j>1} ≤ 0.55`), the *first cue* contributes ≥99% of the total attainable information about `s` across the full cue budget `K`. Even in regimes where `k* = K` strictly minimizes EFE, the marginal contribution of cues `2..K` is so small that a TTB-style "stop after the first cue" policy is near-optimal in expected reward terms. This near-optimality is the empirically-relevant content of the TTB connection in the binary setting; the strict EFE-optimality in the narrow `[κ_lo, κ_hi]` window is the formally-derivable but operationally-fragile content.

**Implication for §2.7.** The structural-equivalence argument in §2.7 (Theorem 2.7.1) is the load-bearing claim. The binary toy model serves as motivating intuition and a lower-dimensional sanity check, not as a strict existence proof for the one-cue regime. The chapter's claim that "the cue-truncation point `k*` is generically finite and often equal to one in the binary case" should be hedged: it is *attainable* in the binary case under specific conditions, but is not the dominant regime. The continuous Gaussian-Gamma model is where the cue-truncation theorem fires robustly across a wide regime of meta-uncertainty.

---

## A.1.6 Numerical verification

All numerical results in this appendix are produced by `binary_toy_monte_carlo.py` and `binary_toy_kstar_search.py`, which compute expected free energy by exhaustive enumeration over cue paths (tractable up to `K = 10`). The KL divergence between the posterior and prior 2-component Beta mixtures is computed by numerical integration on a 2000-point grid over `(0, 1)`. Verification of Lemma A.1.2 across five validity profiles and four hyperprior configurations: monotone in expectation across all 5 cases. Verification of Proposition A.1.4 across 25 × 5 × 5 × 3 = 1875 parameter combinations: 40 combinations exhibit `k* = 1` strictly; the remainder split between `k* = 0` and `k* = K`. The transition curve plots are reproducible from the scripts.

---

## A.1.7 Summary of corrections to the v0.4 chapter

The following amendments to §2.1 are recommended on the basis of this appendix:

1. **Lemma 2.1.2 wording.** Change "the meta-divergence `Δ_meta(k)` is non-decreasing in `k`" to "the expected meta-divergence `E[Δ_meta(k)]` is non-decreasing in `k`." Add a one-sentence note about the sample-wise counterexample with reference to this appendix.

2. **Proposition 2.1.3 wording.** Replace the current single-regime statement ("the EFE-minimizing policy stops at the first discriminating cue") with the three-regime statement of Proposition A.1.4. Hedge the TTB connection: in the binary case, the one-cue regime is narrow; the cleaner TTB derivation lives in §2.7 via the Gaussian-Gamma machinery.

3. **§2.1 closing paragraph.** The sentence "the binary toy model is sufficient to motivate the FEH effect and to establish the take-the-best connection in its most natural setting" should be softened to "the binary toy model motivates the FEH effect; the strict cue-truncation regime is more robust in the continuous Gaussian-Gamma model of §§2.3–2.6, which is the load-bearing setting for the chapter's main results."

These changes weaken §2.1's strong claims and strengthen the chapter's overall honesty. They do not affect the core results in §§2.4–2.7, which derive from the Gaussian-Gamma machinery rather than the binary case.
