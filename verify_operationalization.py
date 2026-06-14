"""
verify_operationalization.py

Simulate-and-recover validation of the §3 regime_score estimator.

Setup
-----
Binary toy model (Appendix A.1). Agent has Beta(alpha, alpha) hyperprior over
the per-cue discrimination probability p_0, kept symmetric (mean = 0.5). True
meta-uncertainty under this hyperprior:

    sigma2_tau(alpha) = alpha^2 / [(2*alpha)^2 * (2*alpha+1)] = 1 / [4*(2*alpha+1)]

We sweep alpha from highly concentrated (low sigma2_tau) to diffuse (high
sigma2_tau) and verify the three behavioral signatures of §3.2 recover the
rank order of sigma2_tau.

Signatures
----------
  (a) cross-prompt variance:  variance of posterior mean across N=5
                              context perturbations applied to the prior
  (b) cross-seed variance:    variance of posterior mean across M=10
                              cue resamplings (same prompt)
  (c) calibration error:      |mean confidence - mean accuracy| across M=20
                              seeds (Bayesian agent is calibrated by
                              construction, so we expect this signature
                              to be approximately flat in the toy model)
"""

import numpy as np
from scipy.stats import spearmanr


def sigma2_tau(alpha):
    """Var of p_0 under Beta(alpha, alpha) hyperprior (mean = 0.5)."""
    return 1.0 / (4.0 * (2.0 * alpha + 1.0))


# ---------------------------------------------------------------------------
# Three behavioral signatures
# ---------------------------------------------------------------------------

def signature_a_cross_prompt(alpha, p_true, K=10, N_prompts=5, n_ctx=2, rng=None):
    """Variance of posterior mean across N_prompts variants.
    Each 'prompt' conditions on a small context perturbation (n_ctx pseudo-cues
    drawn from Bernoulli(0.5))."""
    if rng is None:
        rng = np.random.default_rng()
    posteriors = []
    for _ in range(N_prompts):
        ctx_s = int(rng.binomial(1, 0.5, size=n_ctx).sum())
        a_p = alpha + ctx_s
        b_p = alpha + (n_ctx - ctx_s)
        cues = rng.binomial(1, p_true, size=K)
        s = int(cues.sum())
        p_post = (a_p + s) / (a_p + b_p + K)
        posteriors.append(p_post)
    return float(np.var(posteriors))


def signature_b_cross_seed(alpha, p_true, K=10, M_seeds=10, rng=None):
    """Variance of posterior mean across M_seeds cue resamplings, fixed prior."""
    if rng is None:
        rng = np.random.default_rng()
    posteriors = []
    for _ in range(M_seeds):
        cues = rng.binomial(1, p_true, size=K)
        s = int(cues.sum())
        p_post = (alpha + s) / (2.0 * alpha + K)
        posteriors.append(p_post)
    return float(np.var(posteriors))


def signature_c_calibration(alpha, p_true, K=10, M_seeds=20, rng=None):
    """|mean confidence - mean accuracy| across M_seeds."""
    if rng is None:
        rng = np.random.default_rng()
    confs, corrects = [], []
    truth = (p_true > 0.5)
    for _ in range(M_seeds):
        cues = rng.binomial(1, p_true, size=K)
        s = int(cues.sum())
        p_post = (alpha + s) / (2.0 * alpha + K)
        conf = max(p_post, 1.0 - p_post)
        answer = (p_post > 0.5)
        confs.append(conf)
        corrects.append(int(answer == truth))
    return float(abs(np.mean(confs) - np.mean(corrects)))


# ---------------------------------------------------------------------------
# Per-alpha aggregation
# ---------------------------------------------------------------------------

def signatures_for_alpha(alpha, n_tasks=200, K=10, N_prompts=5, M_seeds=10,
                          M_calib=20, rng=None):
    """Mean of each signature across n_tasks drawn p_true ~ Beta(alpha, alpha)."""
    if rng is None:
        rng = np.random.default_rng()
    s_a, s_b, s_c = [], [], []
    for _ in range(n_tasks):
        p_true = float(rng.beta(alpha, alpha))
        s_a.append(signature_a_cross_prompt(alpha, p_true, K, N_prompts, rng=rng))
        s_b.append(signature_b_cross_seed(alpha, p_true, K, M_seeds, rng=rng))
        s_c.append(signature_c_calibration(alpha, p_true, K, M_calib, rng=rng))
    return {
        "alpha": alpha,
        "sigma2_tau": sigma2_tau(alpha),
        "mean_sig_a": float(np.mean(s_a)),
        "mean_sig_b": float(np.mean(s_b)),
        "mean_sig_c": float(np.mean(s_c)),
        "sd_sig_a": float(np.std(s_a, ddof=1)),
        "sd_sig_b": float(np.std(s_b, ddof=1)),
        "sd_sig_c": float(np.std(s_c, ddof=1)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(2026)

    print("=" * 78)
    print("§3.7 Simulate-and-recover validation of regime_score estimator")
    print("=" * 78)
    print()
    print("Agent: Beta(alpha, alpha) hyperprior over p_0 (mean=0.5)")
    print("sigma2_tau(alpha) = 1 / [4*(2*alpha+1)]")
    print("Signatures (a) cross-prompt variance, (b) cross-seed variance,")
    print("           (c) calibration error |mean_conf - mean_correct|")
    print(f"Per-alpha aggregation: mean over 200 tasks; K=10 main cues.")
    print()

    alphas = [50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5]

    print(f"  {'alpha':>6} {'sigma2_tau':>12} "
          f"{'sig_a':>10} {'sig_b':>10} {'sig_c':>10}")
    print(f"  {'-'*6} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")

    results = []
    for a in alphas:
        r = signatures_for_alpha(a, n_tasks=200, rng=rng)
        results.append(r)
        print(f"  {r['alpha']:>6.1f} {r['sigma2_tau']:>12.5f} "
              f"{r['mean_sig_a']:>10.5f} {r['mean_sig_b']:>10.5f} "
              f"{r['mean_sig_c']:>10.5f}")
    print()

    # Rank-correlation
    sigma2 = np.array([r["sigma2_tau"] for r in results])
    sig_a = np.array([r["mean_sig_a"] for r in results])
    sig_b = np.array([r["mean_sig_b"] for r in results])
    sig_c = np.array([r["mean_sig_c"] for r in results])

    print("Spearman rank-correlation with true sigma2_tau "
          "(target: positive, near 1):")
    for name, sig in [("sig_a (cross-prompt)", sig_a),
                       ("sig_b (cross-seed)",  sig_b),
                       ("sig_c (calibration)", sig_c)]:
        rho, p = spearmanr(sigma2, sig)
        verdict = ("RECOVERS"          if rho > 0.9 else
                   "RECOVERS-weakly"   if rho > 0.5 else
                   "FAILS")
        print(f"  {name:25s}  rho = {rho:+.3f}  p = {p:.4f}  -> {verdict}")
    print()

    # Aggregated regime_score variants
    def z(x):
        return (x - x.mean()) / x.std(ddof=1)

    regime_abc = (z(sig_a) + z(sig_b) + z(sig_c)) / 3.0
    regime_ab  = (z(sig_a) + z(sig_b)) / 2.0

    print("Aggregated regime_score (z-standardized signatures):")
    for name, score in [("regime_score (a+b+c)", regime_abc),
                          ("regime_score (a+b)",   regime_ab)]:
        rho, p = spearmanr(sigma2, score)
        verdict = ("RECOVERS"        if rho > 0.9 else
                   "RECOVERS-weakly" if rho > 0.5 else
                   "FAILS")
        print(f"  {name:25s}  rho = {rho:+.3f}  p = {p:.4f}  -> {verdict}")
    print()

    # Notes on signature (c)
    print("Note: signature (c) tests whether the Bayesian agent's posterior is")
    print("well-calibrated under the toy-model hyperprior. Because the agent's")
    print("prior is correctly specified (p_true ~ Beta(alpha, alpha) and agent")
    print("uses Beta(alpha, alpha)), the posterior mean is unbiased — calibra-")
    print("tion error is approximately flat across alpha. Real LLMs are not")
    print("calibrated by construction (Desai & Durrett 2020, Jiang et al.")
    print("2021); calibration error retains diagnostic value for LLM evaluation")
    print("but requires LLM-specific validation rather than toy-model recovery.")


if __name__ == "__main__":
    main()
