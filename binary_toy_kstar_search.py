"""
Search the parameter space for a k*=1 regime in the binary toy model.

Findings from the basic Monte Carlo: under IID cues with constant validity,
k* = 0 (when meta-cost dominates) or k* = K (when info gain dominates).
There's no clean k*=1 regime — the chapter's Proposition 2.1.3 needs
either restatement or stronger assumptions.

This script probes whether ANY parameter combination in the binary model
gives k*=1 with a strict argmin (G(1) < G(0) AND G(1) < G(2)).
"""

import numpy as np
from itertools import product
from binary_toy_monte_carlo import (
    expected_free_energy,
    find_kstar,
)


def systematic_search():
    """Sweep over (kappa_0, mu_0, validity profile) for k*=1 outcomes."""
    print("=" * 78)
    print("Search for k*=1 regime in binary toy model (intermediate stopping)")
    print("=" * 78)
    print()

    found_kstar1 = []

    for v1 in [0.99, 0.95, 0.9, 0.85, 0.8]:
        for v_rest in [0.51, 0.55, 0.6, 0.65, 0.7]:
            v_seq = [v1, v_rest, v_rest, v_rest, v_rest]
            for kappa in np.geomspace(0.1, 100, 25):
                for mu in [0.3, 0.5, 0.7]:
                    a0 = mu * kappa
                    b0 = (1 - mu) * kappa
                    if a0 < 0.01 or b0 < 0.01:
                        continue
                    kstar, G = find_kstar(v_seq, a0, b0, K_max=5)
                    if kstar == 1 and G[1] < G[0] - 1e-6 and G[1] < G[2] - 1e-6:
                        found_kstar1.append(
                            (v1, v_rest, kappa, mu, kstar, G)
                        )

    if not found_kstar1:
        print("NO parameter combination found where k*=1 with strict argmin.")
        print()
        print("This means: in the binary toy model under symmetric error and IID")
        print("cues, the cue-truncation theorem produces only k*=0 (degenerate)")
        print("or k*=K (full integration). The intermediate one-cue stopping")
        print("regime that Proposition 2.1.3 claims does NOT generically exist.")
    else:
        print(f"Found {len(found_kstar1)} (v1, v_rest, kappa, mu) combinations with k*=1")
        print(f"Sample (first 10):")
        print(f"  v_1   v_rest   kappa     mu     k*     G(0)..G(K)")
        for v1, v_rest, kappa, mu, kstar, G in found_kstar1[:10]:
            G_str = "  ".join(f"{g:+.3f}" for g in G)
            print(f"  {v1:.2f}   {v_rest:.2f}    {kappa:6.2f}   {mu:.1f}    {kstar}    {G_str}")


def transition_curve():
    """Map the (kappa_0 → k*) transition for several validity profiles."""
    print()
    print("=" * 78)
    print("Transition curve: kappa_0 vs k* for various validity profiles")
    print("=" * 78)
    print()

    profiles = {
        "v=[0.95,0.85,0.75,0.65,0.55] (gradient)": [0.95, 0.85, 0.75, 0.65, 0.55],
        "v=[0.85]*5 (uniform)":                    [0.85] * 5,
        "v=[0.99,0.51,0.51,0.51,0.51] (sharp)":    [0.99, 0.51, 0.51, 0.51, 0.51],
        "v=[0.6]*5 (weak)":                         [0.6] * 5,
        "v=[0.51]*5 (very weak)":                   [0.51] * 5,
    }

    print(f"  kappa_0  ", end="")
    for label in profiles:
        print(f"{label[:30]:>32}", end="")
    print()

    for kappa in np.geomspace(0.1, 200, 20):
        a0 = 0.5 * kappa
        b0 = 0.5 * kappa
        print(f"  {kappa:6.2f}  ", end="")
        for label, v_seq in profiles.items():
            kstar, _ = find_kstar(v_seq, a0, b0, K_max=5)
            print(f"{f'k*={kstar}':>32}", end="")
        print()


def alternative_test_one_cue_dominance():
    """
    Alternative test: does the FIRST cue contribute disproportionate info?
    The Gigerenzer/Goldstein TTB literature claims yes — the first
    discriminating cue resolves most of the uncertainty.
    """
    print()
    print("=" * 78)
    print("First-cue dominance: fraction of total info gain from cue 1 only")
    print("=" * 78)
    print()
    print("This is the ECOLOGICAL claim behind TTB — under realistic validity")
    print("gradients, the first cue contributes the majority of attainable info.")
    print()
    print(f"  Profile                                       I(1)/I(K)   I(1)+I(2)/I(K)")
    profiles = {
        "v=[0.99, 0.55, 0.55, 0.55, 0.55] (sharp)":    [0.99, 0.55, 0.55, 0.55, 0.55],
        "v=[0.95, 0.85, 0.75, 0.65, 0.55] (gradient)": [0.95, 0.85, 0.75, 0.65, 0.55],
        "v=[0.85, 0.85, 0.85, 0.85, 0.85] (uniform)":  [0.85] * 5,
        "v=[0.6, 0.6, 0.6, 0.6, 0.6] (weak uniform)":  [0.6] * 5,
    }
    a0, b0 = 1.0, 1.0  # neutral hyperprior
    H_prior = -0.5 * np.log(0.5) - 0.5 * np.log(0.5)
    for label, v_seq in profiles.items():
        from binary_toy_monte_carlo import expected_state_entropy
        H_K = expected_state_entropy(5, v_seq, a0, b0)
        H_1 = expected_state_entropy(1, v_seq, a0, b0)
        H_2 = expected_state_entropy(2, v_seq, a0, b0)
        I_K = H_prior - H_K
        I_1 = H_prior - H_1
        I_2 = H_prior - H_2
        print(f"  {label:46s}    {I_1/I_K:.3f}        {I_2/I_K:.3f}")


if __name__ == "__main__":
    systematic_search()
    transition_curve()
    alternative_test_one_cue_dominance()
