"""
Verify Theorem A.4.1: exact sample-wise FFH-TTB action identity under the
Descending Dominance (DD) condition.

Setup
-----
Binary discriminating cues d_j in {-1, +1}, validities v_(1) >= ... >= v_(K).
Per-cue log-likelihood-ratio L_j = log(v_j / (1 - v_j)).

(DD)  For all i in {1, ..., K-1}:  L_(i) > sum_{j=i+1}^{K} L_(j).

Agents (uniform prior mu_0 = 1/2):
    TTB action:  sign(d_(1))                                   (one-reason)
    FFH action:  sign(sum_{j=1}^{k*} L_(j) * d_(j))             (integrated)

Claims:
  (C1) Under (DD), for every cue realization c and every k* in {1, ..., K},
       a_FFH(c) = a_TTB(c).                                    [Theorem A.4.1]
  (C2) When (DD) fails, there exists c with a_FFH(c) != a_TTB(c). [Sharpness]
"""

import numpy as np
from itertools import product


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def log_lrs(validities):
    """L_j = log(v_j / (1 - v_j)) for descending-ordered validities."""
    v = np.asarray(validities, dtype=float)
    return np.log(v / (1.0 - v))


def descending_dominance(validities, tol=1e-12):
    """Return (ok, slack) where ok is True iff (DD) holds and
    slack[i] = L_(i) - sum_{j>i} L_(j) for i = 0..K-2."""
    L = log_lrs(validities)
    K = len(L)
    slack = np.array([L[i] - L[i + 1:].sum() for i in range(K - 1)])
    return bool(np.all(slack > tol)), slack


def ttb_action(d):
    """TTB: sign of the first (highest-validity) cue under uniform prior."""
    return int(np.sign(d[0]))


def ffh_action(d, L, k_star):
    """FFH: sign of accumulated signed log-LR over first k_star cues
    under uniform prior."""
    s = float(np.dot(L[:k_star], d[:k_star]))
    return int(np.sign(s)) if s != 0 else 0


def enumerate_paths(K):
    """All 2^K cue realizations in {-1, +1}^K."""
    return np.array(list(product([-1, 1], repeat=K)), dtype=int)


def compare_all_paths(validities, k_star=None):
    """Exhaustive sample-wise FFH-vs-TTB comparison over all 2^K cue paths
    and (optionally) all k_star in {1,...,K}.

    Returns dict with mismatch count and worst-case offending realizations.
    """
    L = log_lrs(validities)
    K = len(L)
    paths = enumerate_paths(K)
    k_range = [k_star] if k_star is not None else list(range(1, K + 1))
    results = {}
    for k in k_range:
        mism = []
        for d in paths:
            a_ttb = ttb_action(d)
            a_ffh = ffh_action(d, L, k)
            if a_ffh != a_ttb:
                mism.append(tuple(d.tolist()))
        results[k] = {
            "n_paths": len(paths),
            "n_mismatch": len(mism),
            "examples": mism[:3],
        }
    return results


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("Theorem A.4.1 verification - exact sample-wise FFH-TTB identity")
    print("=" * 78)
    print()

    # -----------------------------------------------------------------
    # (C1) (DD) satisfied: zero mismatches across all k* and all paths
    # -----------------------------------------------------------------
    print("[C1] Validity profiles satisfying (DD): expect 0 mismatches.")
    print()
    DD_profiles = [
        ("steep geometric:   v=(.95,.75,.60,.55)", [0.95, 0.75, 0.60, 0.55]),
        ("steep geometric:   v=(.99,.80,.65,.55,.52)",
             [0.99, 0.80, 0.65, 0.55, 0.52]),
        ("near-boundary DD:  v=(.90,.70,.58)", [0.90, 0.70, 0.58]),
        ("K=6 geometric:     v=(.97,.78,.65,.58,.54,.52)",
             [0.97, 0.78, 0.65, 0.58, 0.54, 0.52]),
    ]
    for label, v in DD_profiles:
        ok, slack = descending_dominance(v)
        L = log_lrs(v)
        K = len(v)
        results = compare_all_paths(v)
        max_mism = max(r["n_mismatch"] for r in results.values())
        verdict = "PASS" if max_mism == 0 else f"FAIL ({max_mism} mismatches)"
        print(f"  {label}")
        print(f"    L = {np.round(L, 3).tolist()}")
        print(f"    DD slacks (L_(i) - sum L_(j>i)): {np.round(slack, 3).tolist()}"
              f"  -> DD = {ok}")
        print(f"    sample-wise check across all 2^{K} = {2**K} paths "
              f"and k* in 1..{K}:")
        for k, r in results.items():
            print(f"      k* = {k}: {r['n_mismatch']}/{r['n_paths']} mismatches")
        print(f"    verdict: {verdict}")
        print()

    # -----------------------------------------------------------------
    # (C2) (DD) violated: predictable mismatches
    # -----------------------------------------------------------------
    print("[C2] Validity profiles violating (DD): expect nonzero mismatches.")
    print()
    nonDD_profiles = [
        ("shallow (3 equal):   v=(.70,.70,.70)", [0.70, 0.70, 0.70]),
        ("nearly equal pair:   v=(.75,.74,.55)", [0.75, 0.74, 0.55]),
        ("majority overpowers: v=(.80,.70,.70,.70)",
             [0.80, 0.70, 0.70, 0.70]),
    ]
    for label, v in nonDD_profiles:
        ok, slack = descending_dominance(v)
        L = log_lrs(v)
        K = len(v)
        results = compare_all_paths(v, k_star=K)  # full integration
        r = results[K]
        print(f"  {label}")
        print(f"    L = {np.round(L, 3).tolist()}")
        print(f"    DD slacks: {np.round(slack, 3).tolist()}  -> DD = {ok}")
        print(f"    at k* = K = {K}: {r['n_mismatch']}/{r['n_paths']} "
              f"sample-wise mismatches")
        if r["examples"]:
            print(f"    example cue paths where FFH != TTB: {r['examples']}")
        verdict = ("PASS (mismatch as predicted)" if r["n_mismatch"] > 0
                   else "UNEXPECTED (no mismatch despite DD violation)")
        print(f"    verdict: {verdict}")
        print()

    # -----------------------------------------------------------------
    # Robustness: random DD-satisfying profiles
    # -----------------------------------------------------------------
    print("[C1-robust] Random DD-satisfying profiles via geometric construction.")
    print()
    rng = np.random.default_rng(2026)
    total, total_mism, generated = 0, 0, 0
    attempts = 0
    target = 200
    while generated < target and attempts < 5000:
        attempts += 1
        K = int(rng.integers(2, 7))
        v1 = float(rng.uniform(0.90, 0.99))
        # geometric decay of log-LR: L_(j+1) = L_(j) * rho with rho < 1/2
        # guarantees DD: L_(j) = sum_{i>j} L_(i) * (1-rho)/rho > sum since rho<1/2
        rho = float(rng.uniform(0.05, 0.45))
        L1 = np.log(v1 / (1 - v1))
        L_seq = L1 * (rho ** np.arange(K))
        v_seq = 1.0 / (1.0 + np.exp(-L_seq))
        ok, _ = descending_dominance(v_seq)
        if not ok:
            continue
        generated += 1
        for k in range(1, K + 1):
            r = compare_all_paths(v_seq, k_star=k)[k]
            total += r["n_paths"]
            total_mism += r["n_mismatch"]
    print(f"  generated {generated} DD-satisfying profiles "
          f"(K in 2..6, geometric rho in 0.05..0.45)")
    print(f"  total cue-path/k* comparisons: {total}")
    print(f"  total sample-wise mismatches: {total_mism}")
    verdict = "PASS" if total_mism == 0 else f"FAIL ({total_mism} mismatches)"
    print(f"  verdict: {verdict}")
    print()

    # -----------------------------------------------------------------
    # Sanity: TTB always uses cue 1, so on paths where d_1 = +1 the TTB
    # action is +1; on paths where d_1 = -1 the TTB action is -1. Under
    # (DD), the leading signed log-LR L_(1) * d_(1) dominates the rest,
    # so FFH inherits the sign of d_(1). This is the proof's core idea.
    # -----------------------------------------------------------------
    print("[Proof-mechanism check] Under DD, |L_(1)| > sum |L_(j>1)|, so the")
    print("leading signed log-LR dominates the cumulative sum sign:")
    for label, v in DD_profiles[:2]:
        L = log_lrs(v)
        head, rest = abs(L[0]), abs(L[1:]).sum()
        print(f"  {label}:  |L_(1)| = {head:.4f}  vs  sum |L_(j>1)| = "
              f"{rest:.4f}  ({'dominates' if head > rest else 'fails'})")


if __name__ == "__main__":
    main()
