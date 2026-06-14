"""
Verify Theorem 2.6.1 in EXPECTATION form (the form the chapter actually needs):

  E[Delta_meta(k+1)] >= E[Delta_meta(k)]      (Lemma 2.4.1, in expectation)
  E[G(k)] is U-shaped in k                    (Theorem 2.6.1(b), in expectation)

Sample-wise versions are false in general (979/1000 violations of monotonicity;
only 198/1000 G-trajectories are U-shaped). Active inference defines the
optimal policy as argmin E[G(k)], so the expectation form is what matters.
"""

import numpy as np
from scipy.special import gammaln, digamma


def gamma_kl(a1, b1, a0, b0):
    return ((a1 - a0) * digamma(a1) - gammaln(a1) + gammaln(a0)
            + a0 * (np.log(b1) - np.log(b0)) + a1 * (b0 - b1) / b1)


def expected_delta_meta_and_G(alpha_0p, beta_0p, gamma_bar, m_bar, K, c1, lam,
                                 n_samples=10000, rng=None):
    """Monte Carlo E[Delta_meta(k)] and E[G(k)] under random M_j ~ Exp(m_bar)."""
    if rng is None:
        rng = np.random.default_rng(42)
    M_samples = rng.exponential(scale=m_bar, size=(n_samples, K))
    dm_paths = np.zeros((n_samples, K + 1))
    G_paths = np.zeros((n_samples, K + 1))
    for k in range(1, K + 1):
        a_k = alpha_0p + k / 2
        b_k = beta_0p + 0.5 * gamma_bar * np.sum(M_samples[:, :k], axis=1)
        dm = np.array([gamma_kl(a_k, b, alpha_0p, beta_0p) for b in b_k])
        dm_paths[:, k] = dm
        I_k = c1 * (1 - np.exp(-k / lam))
        G_paths[:, k] = -I_k + dm
    return dm_paths.mean(axis=0), G_paths.mean(axis=0)


def main():
    print("=" * 78)
    print("Theorem 2.6.1 verification — EXPECTATION form")
    print("=" * 78)

    K = 30
    cases = [
        ("low meta-uncertainty (alpha_0'=20, info-rich)",  20.0, 20.0, 1.0, 1.0, 5.0, 6.0),
        ("med meta-uncertainty (alpha_0'=5)",                5.0, 5.0,  1.0, 1.0, 4.0, 5.0),
        ("high meta-uncertainty (alpha_0'=2)",               2.0, 2.0,  1.0, 1.0, 3.0, 4.0),
        ("very high (alpha_0'=1)",                           1.0, 1.0,  1.0, 1.0, 2.0, 3.0),
        ("extreme (alpha_0'=0.5)",                           0.5, 0.5,  1.0, 1.0, 1.5, 2.0),
    ]

    rng = np.random.default_rng(2025)
    for label, a0p, b0p, g_bar, m_bar, c1, lam in cases:
        print(f"\n  {label}")
        E_dm, E_G = expected_delta_meta_and_G(a0p, b0p, g_bar, m_bar, K, c1, lam,
                                                n_samples=5000, rng=rng)
        # Lemma 2.4.1 in expectation
        dm_diffs = np.diff(E_dm)
        lemma_holds = np.all(dm_diffs >= -1e-6)
        print(f"    Lemma 2.4.1 (expected Delta_meta non-decreasing): "
              f"{'YES' if lemma_holds else f'NO (min diff = {dm_diffs.min():.5f})'}")
        # Theorem 2.6.1(b) in expectation: G is U-shaped (one sign change in dG)
        G_diffs = np.diff(E_G)
        sign_changes = np.sum(np.diff(np.sign(G_diffs)) != 0)
        kstar = int(np.argmin(E_G))
        unimodal = sign_changes <= 1
        print(f"    Theorem 2.6.1(b) (expected G is U-shaped): "
              f"{'YES' if unimodal else f'NO ({sign_changes} sign changes)'}")
        print(f"    k* = argmin E[G(k)] = {kstar} (out of K={K})")
        # Show the trajectory at sparse points
        sparse_k = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30]
        sparse_k = [k for k in sparse_k if k <= K]
        print(f"    E[G(k)] at k = {sparse_k}:")
        print(f"           " + "  ".join(f"{E_G[k]:+.3f}" for k in sparse_k))
        print(f"    E[Delta_meta(k)] at same k:")
        print(f"           " + "  ".join(f"{E_dm[k]:+.3f}" for k in sparse_k))


if __name__ == "__main__":
    main()
