"""
Verify FFH-Tallying structural equivalence under uniform cue validity.

Setup: comparative judgment between options A and B with K binary cues.
Cue j produces signal d_j in {-1, 0, +1}: +1 favors A, -1 favors B, 0 indifferent.
For uniform validity v_j = v across j, the FFH posterior on "A is better"
depends only on the running tally T_k = sum_{j=1..k} d_j (a sufficient statistic).

Tallying procedure: compute T_K (full sum), decide by sign(T_K).
FFH-tallying: compute T_{k*} (truncated sum), decide by sign(T_{k*}).

Claim 1: Under uniform v, the FFH posterior on s = "A is better" given the
         first k cues is a function of T_k alone, not of the cue ordering.
Claim 2: The FFH-optimal action under symmetric loss is sign(T_{k*}).
Claim 3: When k* = K (low meta-uncertainty), FFH = classical tallying.
         When k* < K (high meta-uncertainty), FFH = truncated tallying.
"""

import numpy as np
from itertools import permutations
from scipy.stats import beta as beta_dist


def cue_likelihood_per_state(d, v):
    """L(d | s = A) for d in {-1, 0, +1}: cue d=+1 favors A with prob v.

    For binary discriminating cues with three values:
      P(d = +1 | s=A) = v;  P(d = -1 | s=A) = 1-v
      P(d = +1 | s=B) = 1-v; P(d = -1 | s=B) = v
      P(d = 0 | s=A or s=B) = 0 (assume non-degenerate cues for clarity)
    """
    if d == 1:
        return (v, 1 - v)  # (P(d|A), P(d|B))
    elif d == -1:
        return (1 - v, v)
    else:
        return (0.5, 0.5)  # noninformative


def posterior_s_given_cues(d_seq, v, mu_0=0.5):
    """P(s = A | d_seq) under uniform validity v and prior mean mu_0."""
    pi_A = 1.0
    pi_B = 1.0
    for d in d_seq:
        L_A, L_B = cue_likelihood_per_state(d, v)
        pi_A *= L_A
        pi_B *= L_B
    Z = pi_A * mu_0 + pi_B * (1 - mu_0)
    if Z == 0:
        return mu_0
    return pi_A * mu_0 / Z


def main():
    print("=" * 78)
    print("FFH-Tallying structural equivalence verification")
    print("=" * 78)
    print()

    # ------------------------------------------------------------------
    # Claim 1: Posterior depends only on tally T, not on ordering
    # ------------------------------------------------------------------
    print("[Claim 1] Posterior on s depends only on tally T = sum(d), not on order.")
    print()
    v = 0.75
    mu_0 = 0.5
    print(f"v = {v}, mu_0 = {mu_0}, K = 5")
    print(f"  Permutations of d_seq with same tally must yield identical posteriors:")
    print()

    for d_seq_canonical in [(1, 1, 1, -1, -1), (1, -1, 1, -1, 1), (1, 1, -1, 1, -1)]:
        T = sum(d_seq_canonical)
        posteriors = set()
        for perm in set(permutations(d_seq_canonical)):
            p = posterior_s_given_cues(list(perm), v, mu_0)
            posteriors.add(round(p, 10))
        check = "PASS" if len(posteriors) == 1 else f"FAIL ({len(posteriors)} distinct values)"
        print(f"  d_seq base = {d_seq_canonical}, T = {T:+d}: "
              f"{len(set(permutations(d_seq_canonical)))} permutations, "
              f"all posteriors = {sorted(posteriors)} ({check})")
    print()

    # ------------------------------------------------------------------
    # Claim 2: Optimal action under symmetric loss is sign(T)
    # ------------------------------------------------------------------
    print("[Claim 2] Optimal action under 0-1 symmetric loss equals sign of tally T.")
    print()
    print(f"  Posterior P(s=A | d_seq) for various tally values T (uniform v={v}, K=5):")
    print(f"  T     P(s=A|d)    sign(T)    optimal_action       match?")
    for T in [-5, -3, -1, 0, 1, 3, 5]:
        n_pos = (T + 5) // 2
        n_neg = 5 - n_pos
        d_seq = [1] * n_pos + [-1] * n_neg
        p_A = posterior_s_given_cues(d_seq, v, mu_0)
        sign_T = 0 if T == 0 else (1 if T > 0 else -1)
        opt_act = 'A' if p_A > 0.5 else ('B' if p_A < 0.5 else 'tie')
        sign_label = {1: 'A', -1: 'B', 0: 'tie'}[sign_T]
        match = "PASS" if opt_act == sign_label else "FAIL"
        print(f"  {T:+d}    {p_A:.4f}      {sign_T:+d}         {opt_act}                    {match}")
    print()

    # ------------------------------------------------------------------
    # Claim 3: FFH = classical tallying iff k* = K; truncated tallying iff k* < K
    # ------------------------------------------------------------------
    print("[Claim 3] FFH reduces to: classical tallying when k* = K;")
    print("           truncated tallying when k* < K.")
    print()
    print("  This follows directly from the cue-truncation theorem applied to")
    print("  uniform-validity cues. Numerical demonstration via the binary toy")
    print("  model is omitted here (would re-run binary_toy_monte_carlo with")
    print("  uniform validities), but the result is structural:")
    print()
    print("  - Under uniform v, the EFE-optimal posterior at k cues depends on")
    print("    T_k alone (Claim 1).")
    print("  - The EFE-optimal action is sign(T_{k*}) (Claim 2 applied at k*).")
    print("  - The EFE-optimal stopping rule k*(sigma_tau^2) is given by the")
    print("    cue-truncation theorem.")
    print("  - Compose: FFH-optimal policy = 'compute T_{k*}, decide by sign'.")
    print()
    print("  This is the truncated tallying procedure. When sigma_tau^2 < tau_regime,")
    print("  k* = K and FFH = classical tallying. When sigma_tau^2 >= tau_regime,")
    print("  k* < K and FFH = truncated tallying, with k* given by the same")
    print("  threshold criterion as for TTB (Theorem 2.6.1).")


if __name__ == "__main__":
    main()
