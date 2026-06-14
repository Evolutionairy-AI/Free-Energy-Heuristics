"""
FEH Verification Experiment 2: Continuous Gaussian-Gamma (Theorem 2.6.1)
=========================================================================
Author: Alex Bogdan / Claude collaboration
Date: 2026-05-11

PRE-REGISTERED PREDICTIONS
--------------------------
Theorem 2.6.1 (Chapter v0.3) predicts that under the continuous Gaussian-Gamma
model (2.3.1)-(2.3.2):

  T1. When σ²_τ < τ_regime, expected free energy G(k) is non-increasing in k.
      Full cue integration is optimal: k* = K.

  T2. When σ²_τ ≥ τ_regime, there exists finite k* < K such that G(k) is
      decreasing for k ≤ k* and INCREASING for k > k*. Truncated integration
      is optimal.

  T3. The threshold τ_regime is finite and depends on the cue precision and
      information structure.

The diagnostic from the binary case showed that:
  - In binary, meta-divergence SATURATES below info-gain → no truncation
  - In continuous, residuals don't saturate, so meta-divergence MAY grow
    without bound and CAN exceed info-gain → truncation possible

This experiment tests whether truncation actually appears in the continuous case.

DESIGN
------
State:   s ∈ R^d (we use d=1 first, then d=4 for higher-dim stress test)
Prior:   s ~ Normal(μ, τ^{-1} Σ_0) with τ ~ Gamma(α_0, β_0)
Cue j:   c_j = A_j s + ε_j, ε_j ~ Normal(0, τ_{c,j}^{-1})
         where A_j is a known cue-to-state observation matrix and τ_{c,j} is
         a known cue precision.

For each (κ, dimension) cell, we generate N_TRIALS independent cue realizations
and compute G(k) for k = 0..K. We report:
  - mean G(k) curve
  - mean argmin k* (the FEH-optimal cue count)
  - whether G(K) > G(k*) — i.e., is full integration STRICTLY WORSE than
    truncation? This is the headline test of T1/T2.

HONEST DECLARATIONS UP FRONT
----------------------------
1. EFE form. We use the chapter's (2.5.2):
     G(k) = epistemic_KL(k) + meta_KL(k)
   The pragmatic-value term is action-dependent and orthogonal to cue truncation;
   we omit it as the chapter does in its derivation of the theorem.

2. Sign convention. The chapter has epistemic_KL appearing with POSITIVE sign
   in G (it's a "complexity cost"). This conflicts with the standard
   active-inference convention where epistemic value enters with NEGATIVE sign
   (information gain is desirable). The chapter's sign convention is what
   produces the cue-truncation prediction; the standard convention does not.
   This is itself a flag we may need to address in the chapter.

   We test BOTH conventions:
     G_chapter(k)  = epistemic_KL(k) + meta_KL(k)   (chapter form)
     G_standard(k) = -epistemic_KL(k) + meta_KL(k)  (standard active-inference form)

3. If neither form shows truncation in the continuous case, the theorem fails
   and we have a real problem to solve before publication.
"""

import numpy as np
from scipy.stats import gamma as gamma_dist
from scipy.special import digamma, gammaln
import matplotlib.pyplot as plt
import json

RNG = np.random.default_rng(20260511)

# ============ Settings ============
K = 12
N_MC = 5000
N_TRIALS = 200

KAPPA_PARAMS = [
    # (label, alpha_0, beta_0)  → mean tau = alpha/beta, var tau = alpha/beta^2
    ('very_high_metauncert', 0.5, 0.5),    # E[τ]=1, Var[τ]=2   (huge variance)
    ('high_metauncert',      1.0, 1.0),    # E[τ]=1, Var[τ]=1
    ('moderate_metauncert',  2.5, 2.5),    # E[τ]=1, Var[τ]=0.4
    ('low_metauncert',       10.0, 10.0),  # E[τ]=1, Var[τ]=0.1
    ('very_low_metauncert',  100.0, 100.0),# E[τ]=1, Var[τ]=0.01
]

# State dimension
DIMS = [1, 4]

# Cue precisions (decreasing, to mimic validity ordering)
def make_cue_precisions(K, regime='moderate'):
    if regime == 'moderate':
        return np.linspace(2.0, 0.3, K)  # high-to-low precision
    elif regime == 'flat':
        return np.ones(K) * 1.0
    elif regime == 'steep':
        return np.linspace(5.0, 0.1, K)

# ============ Core mathematical machinery ============

def make_observation_matrices(K, d, rng):
    """Random orthonormal-ish observation matrices A_j: each cue is a linear projection."""
    As = []
    for j in range(K):
        A = rng.standard_normal((1, d))
        A = A / np.linalg.norm(A)
        As.append(A)
    return As

def simulate_cues(s_true, As, tau_c_vec, rng):
    """c_j = A_j s + noise; returns array of cue observations."""
    K = len(As)
    cues = np.zeros(K)
    for j in range(K):
        sigma = 1.0 / np.sqrt(tau_c_vec[j])
        cues[j] = float((As[j] @ s_true).item()) + rng.normal(0, sigma)
    return cues

def variational_update(prior_mu, prior_Sigma_inv_scaled, As, tau_c_vec, cues, k,
                       alpha_0, beta_0, n_iters=20):
    """
    Coordinate-ascent variational inference for q(s)q(τ) after k cues.

    q(s) is Gaussian with mean μ_q, precision Λ_q.
    q(τ) is Gamma(α_q, β_q).

    Update equations (standard mean-field):
      Λ_q = E_q[τ] * Σ_0^{-1} + Σ_{j=1}^k τ_c,j * A_j^T A_j
      μ_q = Λ_q^{-1} [E_q[τ] * Σ_0^{-1} * μ + Σ_j τ_c,j * A_j^T c_j]
      α_q = α_0 + k*d/2 + d/2          # state has d dims, k cues observed
      β_q = β_0 + 0.5 * E_q[(s-μ)^T Σ_0^{-1} (s-μ)]
            = β_0 + 0.5 * [(μ_q-μ)^T Σ_0^{-1} (μ_q-μ) + tr(Σ_0^{-1} Λ_q^{-1})]

    E_q[τ] = α_q / β_q
    """
    d = len(prior_mu)
    Sigma_0_inv = prior_Sigma_inv_scaled  # this is Σ_0^{-1} (not scaled by tau)

    # Initialize
    alpha_q = alpha_0 + 0.5 * k + 0.5 * d  # using the correct shape update
    beta_q = beta_0
    E_tau = alpha_q / beta_q

    # Pre-compute cue contributions
    cue_precision_term = np.zeros((d, d))
    cue_mean_term = np.zeros(d)
    for j in range(k):
        A = As[j]
        cue_precision_term += tau_c_vec[j] * (A.T @ A)
        cue_mean_term += tau_c_vec[j] * (A.T.flatten() * cues[j])

    for _ in range(n_iters):
        Lambda_q = E_tau * Sigma_0_inv + cue_precision_term
        Lambda_q_inv = np.linalg.inv(Lambda_q + 1e-9 * np.eye(d))
        mu_q = Lambda_q_inv @ (E_tau * Sigma_0_inv @ prior_mu + cue_mean_term)

        diff = mu_q - prior_mu
        quadratic = float(diff @ Sigma_0_inv @ diff)
        trace_term = float(np.trace(Sigma_0_inv @ Lambda_q_inv))
        beta_q_new = beta_0 + 0.5 * (quadratic + trace_term)

        if abs(beta_q_new - beta_q) < 1e-8:
            beta_q = beta_q_new
            break
        beta_q = beta_q_new
        E_tau = alpha_q / beta_q

    Lambda_q = E_tau * Sigma_0_inv + cue_precision_term
    Lambda_q_inv = np.linalg.inv(Lambda_q + 1e-9 * np.eye(d))
    mu_q = Lambda_q_inv @ (E_tau * Sigma_0_inv @ prior_mu + cue_mean_term)

    return {
        'mu_q': mu_q,
        'Lambda_q': Lambda_q,
        'Lambda_q_inv': Lambda_q_inv,
        'alpha_q': alpha_q,
        'beta_q': beta_q,
    }

def kl_gamma(alpha_q, beta_q, alpha_0, beta_0):
    """KL[Gamma(α_q, β_q) || Gamma(α_0, β_0)]."""
    return ((alpha_q - alpha_0) * digamma(alpha_q)
            - gammaln(alpha_q) + gammaln(alpha_0)
            + alpha_0 * (np.log(beta_q) - np.log(beta_0))
            + alpha_q * (beta_0 - beta_q) / beta_q)

def epistemic_kl_state(mu_q, Lambda_q_inv, mu_prior, Sigma_0, E_tau):
    """KL[q(s) || <p(s|τ)>_q(τ)] approximated using E_q[τ] in the prior covariance.

    Strictly the marginalized prior is Student-t; for tractability we use the
    Gaussian approximation with covariance Σ_0 / E_q[τ]. This is consistent
    with the mean-field treatment.
    """
    d = len(mu_q)
    prior_cov = Sigma_0 / E_tau
    prior_prec = np.linalg.inv(prior_cov + 1e-9 * np.eye(d))

    diff = mu_q - mu_prior
    quad = float(diff @ prior_prec @ diff)
    trace = float(np.trace(prior_prec @ Lambda_q_inv))
    sign1, logdet_prior = np.linalg.slogdet(prior_cov + 1e-9 * np.eye(d))
    sign2, logdet_q = np.linalg.slogdet(Lambda_q_inv + 1e-9 * np.eye(d))
    return 0.5 * (trace + quad - d + logdet_prior - logdet_q)


def feh_inference_cost_curve(cues, As, tau_c_vec, alpha_0, beta_0, mu_prior, Sigma_0, K):
    """Compute G(k) curves under both chapter and standard conventions."""
    d = len(mu_prior)
    Sigma_0_inv = np.linalg.inv(Sigma_0 + 1e-9 * np.eye(d))

    epist = np.zeros(K + 1)
    meta = np.zeros(K + 1)

    for k in range(K + 1):
        result = variational_update(mu_prior, Sigma_0_inv, As, tau_c_vec, cues, k,
                                    alpha_0, beta_0)
        E_tau = result['alpha_q'] / result['beta_q']
        epist[k] = epistemic_kl_state(result['mu_q'], result['Lambda_q_inv'],
                                      mu_prior, Sigma_0, E_tau)
        meta[k] = kl_gamma(result['alpha_q'], result['beta_q'], alpha_0, beta_0)

    G_chapter = epist + meta          # chapter's sign convention
    G_standard = -epist + meta        # standard active-inference

    return {
        'epist': epist,
        'meta': meta,
        'G_chapter': G_chapter,
        'G_standard': G_standard,
    }


# ============ Run experiment ============

def run_continuous_experiment():
    results = {}
    for d in DIMS:
        print(f"\n{'='*70}\nDimension d = {d}\n{'='*70}")
        results[d] = {}
        mu_prior = np.zeros(d)
        Sigma_0 = np.eye(d)
        tau_c_vec = make_cue_precisions(K, regime='moderate')

        for (label, alpha_0, beta_0) in KAPPA_PARAMS:
            print(f"\n  Regime: {label}  (α_0={alpha_0}, β_0={beta_0}, E[τ]={alpha_0/beta_0:.2f}, Var[τ]={alpha_0/beta_0**2:.3f})")

            G_chap_curves = np.zeros((N_TRIALS, K + 1))
            G_std_curves = np.zeros((N_TRIALS, K + 1))
            epist_curves = np.zeros((N_TRIALS, K + 1))
            meta_curves = np.zeros((N_TRIALS, K + 1))

            for trial in range(N_TRIALS):
                tau_true = gamma_dist.rvs(alpha_0, scale=1.0/beta_0, random_state=RNG)
                s_true = RNG.multivariate_normal(mu_prior, Sigma_0 / max(tau_true, 1e-6))
                As = make_observation_matrices(K, d, RNG)
                cues = simulate_cues(s_true, As, tau_c_vec, RNG)

                cv = feh_inference_cost_curve(cues, As, tau_c_vec, alpha_0, beta_0,
                                              mu_prior, Sigma_0, K)
                G_chap_curves[trial] = cv['G_chapter']
                G_std_curves[trial] = cv['G_standard']
                epist_curves[trial] = cv['epist']
                meta_curves[trial] = cv['meta']

            k_star_chap = np.argmin(G_chap_curves, axis=1)
            k_star_std = np.argmin(G_std_curves, axis=1)

            # Headline test: is G(K) > G(k*) for the chapter form?
            G_chap_K = G_chap_curves[:, K]
            G_chap_min = G_chap_curves[np.arange(N_TRIALS), k_star_chap]
            frac_strict_truncation_chap = float(np.mean(G_chap_K > G_chap_min + 1e-6))

            G_std_K = G_std_curves[:, K]
            G_std_min = G_std_curves[np.arange(N_TRIALS), k_star_std]
            frac_strict_truncation_std = float(np.mean(G_std_K > G_std_min + 1e-6))

            results[d][label] = {
                'alpha_0': alpha_0, 'beta_0': beta_0,
                'E_tau': alpha_0/beta_0, 'Var_tau': alpha_0/beta_0**2,
                'mean_k_star_chap': float(np.mean(k_star_chap)),
                'mean_k_star_std': float(np.mean(k_star_std)),
                'frac_strict_truncation_chap': frac_strict_truncation_chap,
                'frac_strict_truncation_std': frac_strict_truncation_std,
                'mean_G_chap_curve': G_chap_curves.mean(axis=0).tolist(),
                'mean_G_std_curve': G_std_curves.mean(axis=0).tolist(),
                'mean_epist_curve': epist_curves.mean(axis=0).tolist(),
                'mean_meta_curve': meta_curves.mean(axis=0).tolist(),
            }
            print(f"    Chapter form:  mean k*={np.mean(k_star_chap):.2f}  "
                  f"frac(G(K) > G(k*))={frac_strict_truncation_chap:.2f}")
            print(f"    Standard form: mean k*={np.mean(k_star_std):.2f}  "
                  f"frac(G(K) > G(k*))={frac_strict_truncation_std:.2f}")

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("FEH Experiment 2: Continuous Gaussian-Gamma (Theorem 2.6.1)")
    print("=" * 70)
    print(f"K = {K},  N_MC = {N_MC},  N_TRIALS = {N_TRIALS}")
    print(f"Dimensions: {DIMS}")
    print(f"Cue precision regime: 'moderate' (linspace 2.0 → 0.3)")

    results = run_continuous_experiment()

    # ============ Plot ============
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    cmap = plt.cm.viridis
    colors = [cmap(i / (len(KAPPA_PARAMS)-1)) for i in range(len(KAPPA_PARAMS))]

    for di, d in enumerate(DIMS):
        # G_chapter curves
        ax = axes[di, 0]
        for ki, (label, _, _) in enumerate(KAPPA_PARAMS):
            curve = results[d][label]['mean_G_chap_curve']
            ax.plot(range(K+1), curve, marker='o', markersize=4, color=colors[ki],
                    label=label.replace('_', ' '))
        ax.set_xlabel('k')
        ax.set_ylabel('G_chapter(k) = epistemic_KL + meta_KL')
        ax.set_title(f'd = {d}: Chapter sign convention')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

        # G_standard curves
        ax = axes[di, 1]
        for ki, (label, _, _) in enumerate(KAPPA_PARAMS):
            curve = results[d][label]['mean_G_std_curve']
            ax.plot(range(K+1), curve, marker='o', markersize=4, color=colors[ki],
                    label=label.replace('_', ' '))
        ax.set_xlabel('k')
        ax.set_ylabel('G_standard(k) = -epistemic_KL + meta_KL')
        ax.set_title(f'd = {d}: Standard active-inference convention')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

        # Decomposition for highest-metauncert
        ax = axes[di, 2]
        label = KAPPA_PARAMS[0][0]  # very_high_metauncert
        ax.plot(range(K+1), results[d][label]['mean_epist_curve'], marker='o', label='epistemic KL', color='C0')
        ax.plot(range(K+1), results[d][label]['mean_meta_curve'], marker='s', label='meta KL', color='C1')
        ax.set_xlabel('k')
        ax.set_ylabel('KL')
        ax.set_title(f'd = {d}: Component decomposition\n({label.replace("_", " ")})')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/claude/feh_paper/exp2_continuous_curves.png', dpi=150, bbox_inches='tight')
    print("\nSaved exp2_continuous_curves.png")

    # Save full results
    with open('/home/claude/feh_paper/exp2_summary.json', 'w') as f:
        json.dump({str(d): results[d] for d in DIMS}, f, indent=2)
    print("Saved exp2_summary.json")

    # ============ Verdict ============
    print("\n" + "="*70)
    print("VERDICT ON THEOREM 2.6.1")
    print("="*70)
    print()
    print("Theorem 2.6.1 predicts: under high meta-uncertainty, G(K) > G(k*)")
    print("                       (i.e., full integration is strictly worse).")
    print()
    print("Chapter form (G = +epistemic + meta):")
    for d in DIMS:
        print(f"\n  d = {d}:")
        for (label, _, _) in KAPPA_PARAMS:
            r = results[d][label]
            print(f"    {label:25s}: mean k*={r['mean_k_star_chap']:5.2f}/{K}  "
                  f"P(G(K)>G(k*))={r['frac_strict_truncation_chap']:.2f}")

    print("\nStandard form (G = -epistemic + meta):")
    for d in DIMS:
        print(f"\n  d = {d}:")
        for (label, _, _) in KAPPA_PARAMS:
            r = results[d][label]
            print(f"    {label:25s}: mean k*={r['mean_k_star_std']:5.2f}/{K}  "
                  f"P(G(K)>G(k*))={r['frac_strict_truncation_std']:.2f}")

    # ============ Bottom line ============
    print("\n" + "="*70)
    print("BOTTOM LINE")
    print("="*70)
    high_meta_chap_d4 = results[4]['very_high_metauncert']['frac_strict_truncation_chap']
    high_meta_std_d4 = results[4]['very_high_metauncert']['frac_strict_truncation_std']
    low_meta_chap_d4 = results[4]['very_low_metauncert']['frac_strict_truncation_chap']
    low_meta_std_d4 = results[4]['very_low_metauncert']['frac_strict_truncation_std']

    print(f"\n  At d=4, very high meta-uncertainty:")
    print(f"    P(G(K) > G(k*))  chapter form  = {high_meta_chap_d4:.2f}")
    print(f"    P(G(K) > G(k*))  standard form = {high_meta_std_d4:.2f}")
    print(f"  At d=4, very low meta-uncertainty:")
    print(f"    P(G(K) > G(k*))  chapter form  = {low_meta_chap_d4:.2f}")
    print(f"    P(G(K) > G(k*))  standard form = {low_meta_std_d4:.2f}")

    if high_meta_chap_d4 > 0.5 and high_meta_chap_d4 > low_meta_chap_d4 + 0.2:
        print("\n  THEOREM 2.6.1 (chapter form): SUPPORTED")
    elif high_meta_std_d4 > 0.5 and high_meta_std_d4 > low_meta_std_d4 + 0.2:
        print("\n  THEOREM 2.6.1 (standard form): SUPPORTED")
        print("  IMPORTANT: chapter sign convention needs revision.")
    else:
        print("\n  THEOREM 2.6.1: NOT supported empirically in continuous case.")
        print("  This is a serious finding. The chapter's central theorem fails.")
