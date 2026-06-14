"""
Binary toy model Monte Carlo — parallels the v0.3 KMM simulation.

Tests two claims of §2.1:
  Lemma 2.1.2 (CORRECTED): E[Delta_meta(k)] is non-decreasing in k.
  Proposition 2.1.3 (RESTATED): for kappa_0 below a critical kappa*, the
    EFE-minimizing policy stops at k*=1 (one-cue stopping = TTB rule).

Setup:
  s in {0,1}, prior P(s=1|p_0) = p_0
  p_0 ~ Beta(alpha_0, beta_0), mu_0 = alpha_0/(alpha_0+beta_0), kappa_0 = alpha_0+beta_0
  c_j in {0,1} with cue validity v_j and symmetric error model:
    P(c_j=1 | s=1) = v_j;  P(c_j=0 | s=0) = v_j

Closed-form posterior on p_0 after k cues (derived in Appendix A.1):
  pi_1(c_{1:k}) = prod_j L_j(c_j; s=1)
  pi_0(c_{1:k}) = prod_j L_j(c_j; s=0)
  Z(c_{1:k})    = pi_1*mu_0 + pi_0*(1-mu_0)
  p(p_0|c_{1:k}) = w_1 * Beta(alpha_0+1, beta_0)  +  w_0 * Beta(alpha_0, beta_0+1)
  where w_1 = pi_1*mu_0/Z, w_0 = pi_0*(1-mu_0)/Z, w_1+w_0 = 1.
  Components are the SAME for every k; only the weights move.

Prior decomposition (verified algebraically):
  Beta(alpha_0, beta_0) = mu_0 * Beta(alpha_0+1, beta_0) + (1-mu_0) * Beta(alpha_0, beta_0+1)
"""

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.special import gammaln, digamma
from itertools import product


# ----------------------------------------------------------------------
# Closed-form posterior over p_0
# ----------------------------------------------------------------------

def cue_likelihood(c, s, v):
    """L_j(c; s) under symmetric error model: v if c==s, (1-v) if c!=s."""
    return v if c == s else (1 - v)


def posterior_weights(c_seq, v_seq, mu_0):
    """Return (w_1, w_0, Z) for posterior p(p_0 | c_seq)."""
    pi_1 = np.prod([cue_likelihood(c, 1, v) for c, v in zip(c_seq, v_seq)])
    pi_0 = np.prod([cue_likelihood(c, 0, v) for c, v in zip(c_seq, v_seq)])
    Z = pi_1 * mu_0 + pi_0 * (1 - mu_0)
    if Z == 0:
        return mu_0, 1 - mu_0, 0  # degenerate: revert to prior
    return pi_1 * mu_0 / Z, pi_0 * (1 - mu_0) / Z, Z


def cue_marginal_prob(c_seq, v_seq, mu_0):
    """P(c_seq) under prior on p_0 (= the normalizer Z)."""
    return posterior_weights(c_seq, v_seq, mu_0)[2]


# ----------------------------------------------------------------------
# Delta_meta(k) — KL of posterior mixture from prior mixture
# ----------------------------------------------------------------------

def kl_mixture_two_component(w_post, w_prior, alpha_0, beta_0, n_grid=2000):
    """
    KL[w * Beta(a+1, b) + (1-w) * Beta(a, b+1) || w' * Beta(a+1, b) + (1-w') * Beta(a, b+1)]

    Computed by numerical integration over a fine grid on (0, 1).
    Two mixture components are FIXED; only the mixture weights differ.
    """
    p = np.linspace(1e-6, 1 - 1e-6, n_grid)
    B1 = beta_dist.pdf(p, alpha_0 + 1, beta_0)
    B0 = beta_dist.pdf(p, alpha_0, beta_0 + 1)
    f_post = w_post * B1 + (1 - w_post) * B0
    f_prior = w_prior * B1 + (1 - w_prior) * B0
    integrand = f_post * (np.log(f_post + 1e-300) - np.log(f_prior + 1e-300))
    return np.trapz(integrand, p)


def delta_meta(c_seq, v_seq, alpha_0, beta_0):
    """KL[p(p_0 | c_seq) || p(p_0)]"""
    mu_0 = alpha_0 / (alpha_0 + beta_0)
    w_1, _, _ = posterior_weights(c_seq, v_seq, mu_0)
    return kl_mixture_two_component(w_1, mu_0, alpha_0, beta_0)


# ----------------------------------------------------------------------
# Information gain about s (epistemic value increment)
# ----------------------------------------------------------------------

def p_s_given_cues(c_seq, v_seq, alpha_0, beta_0):
    """P(s=1 | c_seq).

    Closed form: P(s=1 | c_{1:k}) = pi_1 * mu_0 / Z = w_1, the mixture weight
    on the Beta(alpha_0+1, beta_0) component. Derivation: by direct Bayes,
    P(s=1|c) = P(c|s=1) P(s=1) / P(c) = pi_1 * mu_0 / Z.
    """
    mu_0 = alpha_0 / (alpha_0 + beta_0)
    w_1, _, _ = posterior_weights(c_seq, v_seq, mu_0)
    return w_1


def binary_entropy(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log(p) - (1 - p) * np.log(1 - p)


def expected_state_entropy(k, v_seq, alpha_0, beta_0):
    """E_{c_{1:k}}[H(s | c_{1:k})] under the agent's marginal."""
    mu_0 = alpha_0 / (alpha_0 + beta_0)
    H = 0.0
    for c_seq in product([0, 1], repeat=k):
        Z = cue_marginal_prob(c_seq, v_seq[:k], mu_0)
        if Z == 0:
            continue
        p_s = p_s_given_cues(c_seq, v_seq[:k], alpha_0, beta_0)
        H += Z * binary_entropy(p_s)
    return H


def expected_delta_meta(k, v_seq, alpha_0, beta_0):
    mu_0 = alpha_0 / (alpha_0 + beta_0)
    val = 0.0
    for c_seq in product([0, 1], repeat=k):
        Z = cue_marginal_prob(c_seq, v_seq[:k], mu_0)
        if Z == 0:
            continue
        val += Z * delta_meta(c_seq, v_seq[:k], alpha_0, beta_0)
    return val


# ----------------------------------------------------------------------
# Expected free energy and crossover k*
# ----------------------------------------------------------------------

def expected_free_energy(k, v_seq, alpha_0, beta_0):
    """G(k) = -info_gain_about_s(k) + Delta_meta(k), constant terms dropped."""
    H_prior = binary_entropy(alpha_0 / (alpha_0 + beta_0))
    H_post = expected_state_entropy(k, v_seq, alpha_0, beta_0) if k > 0 else H_prior
    info_gain = H_prior - H_post  # I(s; c_{1:k})
    dm = expected_delta_meta(k, v_seq, alpha_0, beta_0) if k > 0 else 0.0
    return -info_gain + dm  # G(k); drop pragmatic value (action-independent here)


def find_kstar(v_seq, alpha_0, beta_0, K_max=None):
    if K_max is None:
        K_max = len(v_seq)
    G = [expected_free_energy(k, v_seq, alpha_0, beta_0) for k in range(K_max + 1)]
    return int(np.argmin(G)), G


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def test_lemma_2_1_2_in_expectation(v_seq, alpha_0, beta_0, K_max=6):
    """Verify E[Delta_meta(k)] is non-decreasing in k."""
    print(f"\n  Testing Lemma 2.1.2 (corrected, in expectation):")
    print(f"  v_seq = {v_seq[:K_max]}, alpha_0={alpha_0}, beta_0={beta_0}, kappa_0={alpha_0+beta_0}")
    print(f"   k    E[Delta_meta(k)]   monotone?")
    prev = 0.0
    monotone = True
    for k in range(1, K_max + 1):
        dm_k = expected_delta_meta(k, v_seq, alpha_0, beta_0)
        ok = "Y" if dm_k >= prev - 1e-10 else "N (FAIL)"
        if dm_k < prev - 1e-10:
            monotone = False
        print(f"   {k:2d}     {dm_k:.6f}      {ok}")
        prev = dm_k
    return monotone


def test_proposition_2_1_3_kappa_sweep(v_seq, mu_0=0.5, K_max=5, kappas=None):
    """Find kappa* — the threshold above which k* > 1."""
    if kappas is None:
        kappas = [0.5, 1, 2, 5, 10, 20, 50, 100, 500]
    print(f"\n  Testing Proposition 2.1.3:")
    print(f"  v_seq = {v_seq[:K_max]}, mu_0 = {mu_0}, K_max = {K_max}")
    print(f"  kappa_0   alpha_0   beta_0     k*    G(0)..G(K) trajectory")
    kstars = []
    for kappa_0 in kappas:
        a0 = mu_0 * kappa_0
        b0 = (1 - mu_0) * kappa_0
        kstar, G = find_kstar(v_seq[:K_max], a0, b0, K_max=K_max)
        kstars.append(kstar)
        G_str = "  ".join(f"{g:+.3f}" for g in G)
        print(f"   {kappa_0:6.1f}    {a0:5.2f}    {b0:5.2f}     {kstar:2d}    {G_str}")
    return kstars


def main():
    print("=" * 78)
    print("Binary toy model — Monte Carlo verification of §2.1 claims")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Lemma 2.1.2 (corrected — in expectation) — multiple validity profiles
    # ------------------------------------------------------------------
    print("\n[1] Lemma 2.1.2: E[Delta_meta(k)] is non-decreasing in k")
    cases = [
        ("uniform high validity", [0.9, 0.9, 0.9, 0.9, 0.9, 0.9], 1.0, 1.0),
        ("descending validity",   [0.9, 0.8, 0.7, 0.6, 0.55, 0.51], 1.0, 1.0),
        ("low concentration",     [0.85, 0.75, 0.7, 0.65], 0.5, 0.5),
        ("high concentration",    [0.85, 0.75, 0.7, 0.65], 50.0, 50.0),
        ("asymmetric mu",         [0.85, 0.75, 0.7, 0.65], 3.0, 1.0),
    ]
    all_ok = True
    for label, v, a, b in cases:
        print(f"\n  Case: {label}")
        ok = test_lemma_2_1_2_in_expectation(v, a, b, K_max=min(len(v), 5))
        all_ok = all_ok and ok
    print(f"\n  Lemma 2.1.2 verified across all cases: {'YES' if all_ok else 'NO (counterexample!)'}")

    # ------------------------------------------------------------------
    # Proposition 2.1.3 — kappa sweep
    # ------------------------------------------------------------------
    print("\n\n[2] Proposition 2.1.3: k* drops to 1 as kappa_0 decreases")

    print("\n[2a] Descending validities (TTB-natural ordering)")
    test_proposition_2_1_3_kappa_sweep(
        v_seq=[0.95, 0.85, 0.75, 0.65, 0.55],
        mu_0=0.5, K_max=5,
        kappas=[0.5, 1, 2, 5, 10, 20, 50, 200],
    )

    print("\n[2b] Uniform high validities")
    test_proposition_2_1_3_kappa_sweep(
        v_seq=[0.85, 0.85, 0.85, 0.85, 0.85],
        mu_0=0.5, K_max=5,
        kappas=[0.5, 1, 2, 5, 10, 20, 50, 200],
    )

    print("\n[2c] Sharp gradient (very informative first cue, weak rest)")
    test_proposition_2_1_3_kappa_sweep(
        v_seq=[0.95, 0.55, 0.55, 0.55, 0.55],
        mu_0=0.5, K_max=5,
        kappas=[0.5, 1, 2, 5, 10, 20, 50, 200],
    )

    # ------------------------------------------------------------------
    # Sample-wise Delta_meta is NOT monotone — counterexample
    # ------------------------------------------------------------------
    print("\n\n[3] Counterexample: sample-wise Delta_meta is NOT monotone")
    print("    (justifies the need to restate Lemma 2.1.2 in expectation)")
    a, b = 1.0, 1.0
    v_seq = [0.9, 0.9]
    print(f"    Setup: alpha_0={a}, beta_0={b}, v=[0.9, 0.9]")
    print(f"    For each cue path c_{{1:2}}, compute Delta_meta(1) and Delta_meta(2):")
    print(f"     c_1   c_2   Delta_meta(1)   Delta_meta(2)   monotone?")
    for c1 in [0, 1]:
        for c2 in [0, 1]:
            dm1 = delta_meta([c1], v_seq[:1], a, b)
            dm2 = delta_meta([c1, c2], v_seq, a, b)
            mono = "Y" if dm2 >= dm1 else "N (decrease!)"
            print(f"      {c1}     {c2}      {dm1:.4f}          {dm2:.4f}        {mono}")


if __name__ == "__main__":
    main()
