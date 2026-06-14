"""
condition_power.py — power of the AMENDED condition-based (ITT) confirmatory
analysis for FEH H1, at the effect size & item heterogeneity MEASURED in the
calibration run.

Why this exists: the registered analysis regresses cbd-correctness on REALIZED
steps (endogenous) with within-model z-scoring (sign-inverting) and a linear
form. The calibration showed it is mis-specified — it returns "inconclusive" or
even crosses the falsification threshold on data where H1 is descriptively true
(high-regime cbd-correctness falls C1->C4 by ~7pp, C1->C5 by ~13pp; low regime
flat). The proposed amendment targets the EXOGENOUS manipulation instead:

  primary contrast = cbd-correctness ~ assigned-LENGTH (short vs long) x regime,
  a short-vs-long THRESHOLD contrast (motivated by Thm 2.6.1 truncation at k*,
  which predicts the observed drop-then-plateau, not a linear slope).

ITT virtues: (a) assignment is randomized -> the regressor is exogenous, immune
to the steps<->struggle endogeneity; (b) the broken C5 mediator (realized steps
collapse) does NOT break ITT — assigned-long is assigned-long regardless of
compliance.

This is a FAST FREQUENTIST PROXY for the Bayesian gate, to decide whether the
amended design is worth the 45h full run BEFORE committing. IRLS logistic with
item + model fixed effects (item FE absorbs the regime main effect and all
item base-rate heterogeneity, incl. floored items); Wald one-sided test on the
length x regime interaction, plus an effect-size floor mirroring the registered
amended gate (MAG_FLOOR_PP = 6, Pr(dir)>0.95 ~ one-sided p<0.05).

DGP is calibrated to the calibration data:
  - high-regime item SHORT-condition base rates = the 28 measured per-item C1
    cbd-rates (full heterogeneity, incl. K2-006=0.0 floored and several at 1.0),
    resampled to the planned 31 v0.4 cbd items;
  - the length effect is a logit shift applied only in the high regime, tuned by
    bisection to a target MEAN high-regime short->long pp-drop;
  - low-regime length effect = 0 (observed was +3pp; 0 is the conservative null
    for the difference-in-differences);
  - model random intercepts + a small model-level length-effect spread
    (heterogeneity across the 5-model panel), sigma on the logit scale.

Caveat (conservative): per-item C1 rates are estimated from only 9 obs each, so
using them as TRUE rates injects extra variance -> if anything UNDER-states
power. Reported power is therefore a lower-ish bound at each effect size.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Measured high-regime per-item SHORT (C1) cbd-rates from calibration_analysis.json.
HIGH_C1 = np.array([
    0.778, 0.667, 1.0, 0.889, 0.667, 0.889, 1.0, 0.778, 0.889, 1.0, 1.0,
    0.889, 0.667, 0.778, 0.889, 0.889, 0.778, 1.0, 0.667,          # K1 (19)
    0.556, 0.333, 0.0, 0.222, 0.556, 0.444, 0.778,                  # K2 (7)
    0.222, 0.667,                                                    # K4 (2)
])  # 28 items

# Full-run design (v0.4 pool + registered panel).
N_MODELS = 5
N_HIGH_ITEMS = 31          # v0.4 cbd-scorable Knightian items
N_LOW_ITEMS = 12           # control items (regime=0)
N_REPS = 3
N_SHORT_CONDS = 1          # C1 (minimal reasoning)
N_LONG_CONDS = 4           # C2..C5 (substantial; C5 broken mediator but valid ITT)

LOW_BASE = 0.66            # measured low-regime short-condition accuracy
SIGMA_MODEL_INT = 0.4      # model random intercept SD (logit)
SIGMA_MODEL_LEN = 0.20     # model-level spread of the length effect (logit)

MAG_FLOOR_PP = 6.0         # matches confirmatory_analysis.MAG_FLOOR_PP
ALPHA_ONE_SIDED = 0.05     # ~ Pr(direction) > 0.95


def _logit(p):
    p = np.clip(p, 0.02, 0.98)
    return np.log(p / (1 - p))


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def irls(X, y, ridge=1e-6, iters=200):
    """Logistic IRLS; returns (beta, cov) with cov = inv(X'WX) at convergence."""
    beta = np.zeros(X.shape[1])
    H = np.eye(X.shape[1])
    for _ in range(iters):
        mu = _sig(X @ beta)
        W = np.clip(mu * (1 - mu), 1e-9, None)
        XtW = X.T * W
        H = XtW @ X + ridge * np.eye(X.shape[1])
        g = X.T @ (y - mu) - ridge * beta
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    cov = np.linalg.inv(H)
    return beta, cov


def build_design():
    """Static design matrix skeleton (item FE + model FE + long + long:regime).
    Returns (X, regime, long, item_idx, model_idx, col_names, idx_int)."""
    rows = []  # (model, item, is_high, is_long)
    # high-regime items
    for it in range(N_HIGH_ITEMS):
        for m in range(N_MODELS):
            for _ in range(N_REPS * N_SHORT_CONDS):
                rows.append((m, it, 1, 0))
            for _ in range(N_REPS * N_LONG_CONDS):
                rows.append((m, it, 1, 1))
    # low-regime items (item ids continue after high)
    for it in range(N_LOW_ITEMS):
        iid = N_HIGH_ITEMS + it
        for m in range(N_MODELS):
            for _ in range(N_REPS * N_SHORT_CONDS):
                rows.append((m, iid, 0, 0))
            for _ in range(N_REPS * N_LONG_CONDS):
                rows.append((m, iid, 0, 1))
    arr = np.array(rows)
    model_idx, item_idx, regime, long = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    n = len(arr)
    n_items = N_HIGH_ITEMS + N_LOW_ITEMS

    # item FE (drop item 0 as reference -> intercept), model FE (drop model 0),
    # long, long:regime.  regime main effect is absorbed by item FE.
    cols = [np.ones(n)]
    names = ["intercept"]
    for it in range(1, n_items):
        cols.append((item_idx == it).astype(float)); names.append(f"item{it}")
    for m in range(1, N_MODELS):
        cols.append((model_idx == m).astype(float)); names.append(f"model{m}")
    cols.append(long.astype(float)); names.append("long")
    cols.append((long * regime).astype(float)); names.append("long:regime")
    X = np.column_stack(cols)
    return X, regime, long, item_idx, model_idx, names


def solve_beta_len(target_pp, p_short_high, tol=0.02):
    """Bisection: find logit shift whose mean high-regime short->long pp-drop
    equals target_pp (averaged over the high items' short base rates)."""
    if target_pp <= 0:
        return 0.0
    lo, hi = 0.0, 6.0
    base = _logit(p_short_high)
    for _ in range(60):
        mid = (lo + hi) / 2
        pp = np.mean((_sig(base) - _sig(base - mid)) * 100.0)
        if abs(pp - target_pp) < tol:
            return mid
        if pp < target_pp:
            lo = mid
        else:
            hi = mid
    return mid


def simulate_once(rng, X, regime, long, item_idx, model_idx, names,
                  p_short_high_items, beta_len):
    """Generate one dataset under the DGP and fit the amended model.
    Returns (p_one_sided, b_int, pp_drop_hat) — caller applies the gate."""
    n = X.shape[0]
    # true per-row probability
    base_logit = np.empty(n)
    # high items
    for it in range(N_HIGH_ITEMS):
        base_logit[item_idx == it] = _logit(p_short_high_items[it])
    for it in range(N_LOW_ITEMS):
        base_logit[item_idx == (N_HIGH_ITEMS + it)] = _logit(LOW_BASE)
    # model random intercept + model-level length effect
    m_int = rng.normal(0, SIGMA_MODEL_INT, N_MODELS)
    m_len = rng.normal(0, SIGMA_MODEL_LEN, N_MODELS)
    eta = base_logit + m_int[model_idx]
    # length effect: high regime gets -beta_len on long; low regime gets 0.
    eta = eta - (beta_len + (-m_len[model_idx])) * (long * regime)
    p = _sig(eta)
    y = (rng.random(n) < p).astype(float)

    beta, cov = irls(X, y)
    j = names.index("long:regime")
    b_int = beta[j]
    se = np.sqrt(max(cov[j, j], 1e-12))
    z = b_int / se
    # one-sided p for H1 (b_int < 0)
    from math import erf, sqrt
    p_one = 0.5 * (1 + erf(z / sqrt(2)))   # P(Z <= z); small when b_int very negative

    # implied high-regime short->long pp-drop at empirical high base rate
    base_hi = _logit(np.mean(p_short_high_items))
    b_long = beta[names.index("long")]
    slope_hi = b_long + b_int
    pp_drop_hat = (_sig(base_hi) - _sig(base_hi + slope_hi)) * 100.0

    return p_one, b_int, pp_drop_hat


def main(n_sims=200, seed=20260530):
    rng = np.random.default_rng(seed)
    X, regime, long, item_idx, model_idx, names = build_design()
    n_high = int(np.sum(regime == 1))
    n_low = int(np.sum(regime == 0))
    print(f"[design] {X.shape[0]} obs ({n_high} high, {n_low} low), "
          f"{X.shape[1]} params, {N_MODELS} models, "
          f"{N_HIGH_ITEMS}+{N_LOW_ITEMS} items, reps={N_REPS}, "
          f"short:long conds = {N_SHORT_CONDS}:{N_LONG_CONDS}")
    print(f"[gate] confirm iff one-sided p<{ALPHA_ONE_SIDED} AND pp-drop>={MAG_FLOOR_PP} "
          f"(mirrors registered amended gate)")
    print(f"[sims] {n_sims} per effect size\n")

    # resample measured high C1 rates to the 31 planned items
    p_short_high = rng.choice(HIGH_C1, size=N_HIGH_ITEMS, replace=True)
    print(f"[items] high-regime short base rates: mean={p_short_high.mean():.3f}, "
          f"floored(<0.10)={int(np.sum(p_short_high < 0.10))}, "
          f"ceiled(>0.90)={int(np.sum(p_short_high > 0.90))}\n")

    targets = [0.0, 5.0, 7.0, 10.0, 13.0, 15.0]
    print(f"{'target pp':>10} | {'beta_len':>8} | {'power(floor6)':>13} | "
          f"{'power(floor0)':>13} | {'median pp_hat':>13} | {'median b_int':>12}")
    print("-" * 84)
    for tp in targets:
        bl = solve_beta_len(tp, p_short_high)
        confirms6, confirms0, ppds, bints = 0, 0, [], []
        for _ in range(n_sims):
            p_one, b_int, ppd = simulate_once(
                rng, X, regime, long, item_idx, model_idx, names,
                p_short_high, bl)
            sig = p_one < ALPHA_ONE_SIDED            # H1 direction, significant
            confirms6 += int(sig and ppd >= MAG_FLOOR_PP)   # registered amended gate
            confirms0 += int(sig)                            # significance only
            ppds.append(ppd); bints.append(b_int)
        print(f"{tp:>10.1f} | {bl:>8.3f} | {confirms6/n_sims:>13.2f} | "
              f"{confirms0/n_sims:>13.2f} | {np.median(ppds):>13.1f} | "
              f"{np.median(bints):>12.3f}")

    print("\nNotes:")
    print("  - target pp = TRUE mean high-regime short->long accuracy drop (DGP).")
    print("  - measured calibration effect: ~7pp (clean C1->C4) to ~13pp (full C1->C5).")
    print("  - floor6 power = registered amended gate (one-sided p<.05 AND pp_hat>=6).")
    print("  - Type I error = power row at target pp = 0.0 (should be small).")


if __name__ == "__main__":
    n = 200
    for a in sys.argv[1:]:
        if a.startswith("--sims="):
            n = int(a.split("=", 1)[1])
        elif a.startswith("--reps="):
            globals()["N_REPS"] = int(a.split("=", 1)[1])
    print(f"[reps={N_REPS}]")
    main(n_sims=n)
