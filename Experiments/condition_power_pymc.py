"""
condition_power_pymc.py — FULL-PyMC confirmation of the condition-ITT power (#52).

WHY THIS EXISTS
---------------
The fast proxy (`condition_power.py`) estimated the power of the amended
condition-ITT design with a FREQUENTIST IRLS logistic: item + model FIXED
effects, a Wald one-sided test on the length x regime interaction, and an
effect-size floor. The REGISTERED gate (Amendment 2, eq. 6.1') is BAYESIAN:

    eta = b0 + b1*long + b2*regime + b3*(long*regime) + a_m + g_m*long + u_i

with the registered priors (b0,b2 ~ N(0,2.5); b1,b3 ~ N(0,1);
sigma_a/g/u ~ HalfNormal(1)), a model random INTERCEPT a_m AND a model random
SLOPE g_m over only ~5 model clusters, an item random intercept u_i, and the
decision rule Pr(b3<0) > 0.95 AND robust high-regime short->long pp-drop > 6.

The original power concern (see project_power_finding) was that a random SLOPE
estimated from only ~5 noisy clusters, together with the N(0,1) shrinkage prior
on b3, inflates the posterior SD of the population interaction -> the Bayesian
gate could have LOWER power than the fixed-effects proxy implies. The proxy
cannot see this. This script does.

WHAT IT DOES
------------
Fits the ACTUAL registered Bayesian model (reusing
`confirmatory_analysis.build_model` / `fit`, which encode the registered priors,
non-centered REs, and sampler) on datasets simulated from the SAME
calibration-anchored DGP used by `condition_power.py`, and applies the ACTUAL
Bayesian gate. To ISOLATE the cost of going Bayesian-RE, each simulated dataset
is ALSO scored with the IRLS proxy gate (a PAIRED comparison on identical data),
using the same empirical short-rate anchor for both so the delta is purely
fixed-effects-Wald vs random-effects-Bayesian.

The binary ITT factor `long` is fed into the registered model's regressor slot
UNSTANDARDIZED (the registered within-model z-scoring is for realized steps; for
a binary factor with an identical short:long ratio across models it would be a
global affine map that does not change Pr(b3<0), but we skip it for clarity and
to keep the b1/b3 scale interpretable in logits).

Outputs `condition_power_pymc.{md,json}`, written incrementally after each cell
so a glitchy console never loses results.
"""
from __future__ import annotations

import json
import sys
import time
from math import erf, sqrt
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import condition_power as cp          # noqa: E402  (DGP + IRLS proxy + constants)
import confirmatory_analysis as ca    # noqa: E402  (registered model + sampler)

OUT_JSON = HERE / "condition_power_pymc.json"
OUT_MD = HERE / "condition_power_pymc.md"

MAG_FLOOR_PP = cp.MAG_FLOOR_PP        # 6.0, matches the registered amended gate
CONFIRM_PROB = 0.95                   # Pr(b3<0) threshold (Amendment 2 gate)
ALPHA_ONE_SIDED = cp.ALPHA_ONE_SIDED  # 0.05 (IRLS proxy)


def simulate_y(rng, long, regime, item_idx, model_idx, p_short_high_items, beta_len):
    """One dataset from the calibration-anchored DGP (identical to
    condition_power.simulate_once, but returns the raw y so both gates can be
    applied to it). High-regime per-item short base rates = measured C1 cbd-rates
    (resampled to N_HIGH_ITEMS); the length effect is a logit drop applied only
    in the high regime, with a per-model spread; low-regime length effect = 0."""
    n = len(long)
    base_logit = np.empty(n)
    for it in range(cp.N_HIGH_ITEMS):
        base_logit[item_idx == it] = cp._logit(p_short_high_items[it])
    for it in range(cp.N_LOW_ITEMS):
        base_logit[item_idx == (cp.N_HIGH_ITEMS + it)] = cp._logit(cp.LOW_BASE)
    m_int = rng.normal(0, cp.SIGMA_MODEL_INT, cp.N_MODELS)
    m_len = rng.normal(0, cp.SIGMA_MODEL_LEN, cp.N_MODELS)
    eta = base_logit + m_int[model_idx]
    eta = eta - (beta_len + (-m_len[model_idx])) * (long * regime)
    p = cp._sig(eta)
    return (rng.random(n) < p).astype(float)


def _empirical_short_base_logit(y, regime, long):
    """logit of the realised high-regime SHORT (long=0) cbd-rate — the anchor for
    the robust short->long pp-drop. Realistic (no oracle); used for BOTH gates."""
    m = (regime == 1) & (long == 0)
    return cp._logit(float(y[m].mean()))


def pymc_gate(y, regime, long, item_idx, model_idx, base_logit_short,
              draws, tune, chains, seed):
    """Fit the registered Bayesian model (eq. 6.1', ITT) and apply the registered
    amended gate. Returns a dict with the gate inputs + convergence."""
    import arviz as az

    data = {
        "y": y.astype("int8"),
        "steps_z": long.astype("float64"),      # binary ITT regressor (UNstandardized)
        "regime": regime.astype("float64"),
        "model_idx": model_idx.astype("int32"),
        "item_idx": item_idx.astype("int32"),
        "n_models": cp.N_MODELS,
        "n_items": cp.N_HIGH_ITEMS + cp.N_LOW_ITEMS,
    }
    model = ca.build_model(data, model_effects=True)
    idata = ca.fit(model, draws=draws, tune=tune, chains=chains,
                   random_seed=seed, progressbar=False)

    b1 = ca._flat(idata, "b1")
    b3 = ca._flat(idata, "b3")
    p_dir = float(np.mean(b3 < 0))
    slope = b1 + b3                               # high-regime short->long logit slope
    ppd = (ca._sigma(base_logit_short)
           - ca._sigma(base_logit_short + slope)) * 100.0
    ppd_med = float(np.median(ppd))

    diag = [v for v in ["b0", "b1", "b2", "b3", "sigma_a", "sigma_g", "sigma_u"]
            if v in idata.posterior]
    summ = az.summary(idata, var_names=diag)
    rhat_max = float(summ["r_hat"].max())
    ess_min = float(summ["ess_bulk"].min())
    converged = (rhat_max < ca.RHAT_MAX) and (ess_min > ca.ESS_MIN)

    confirm = (p_dir > CONFIRM_PROB) and (ppd_med > MAG_FLOOR_PP)
    return dict(confirm=confirm, p_dir=p_dir, ppd_med=ppd_med,
                b3_med=float(np.median(b3)), b1_med=float(np.median(b1)),
                rhat_max=rhat_max, ess_min=ess_min, converged=converged)


def irls_gate(X, names, y, base_logit_short):
    """IRLS proxy gate on the SAME data, SAME empirical anchor as the Bayesian
    gate (so the only difference is fixed-effects-Wald vs random-effects-Bayes)."""
    beta, cov = cp.irls(X, y)
    j = names.index("long:regime")
    b_int = beta[j]
    se = sqrt(max(cov[j, j], 1e-12))
    z = b_int / se
    p_one = 0.5 * (1 + erf(z / sqrt(2)))          # P(b_int<0) ~ one-sided
    b_long = beta[names.index("long")]
    slope_hi = b_long + b_int
    ppd = (cp._sig(base_logit_short) - cp._sig(base_logit_short + slope_hi)) * 100.0
    confirm = (p_one < ALPHA_ONE_SIDED) and (ppd >= MAG_FLOOR_PP)
    return dict(confirm=confirm, p_one=p_one, ppd=float(ppd), b_int=float(b_int))


def run_cell(rng, target_pp, X, regime, long, item_idx, model_idx, names,
             p_short_high, n_sims, draws, tune, chains, base_seed):
    """Power of both gates at one true effect size, paired on identical data."""
    bl = cp.solve_beta_len(target_pp, p_short_high)
    rows = []
    pymc_conf = irls_conf = both = neither = n_div = 0
    t0 = time.time()
    for s in range(n_sims):
        y = simulate_y(rng, long, regime, item_idx, model_idx, p_short_high, bl)
        base = _empirical_short_base_logit(y, regime, long)
        gi = irls_gate(X, names, y, base)
        gp = pymc_gate(y, regime, long, item_idx, model_idx, base,
                       draws, tune, chains, base_seed + s)
        rows.append({"pymc": gp, "irls": gi})
        pymc_conf += int(gp["confirm"])
        irls_conf += int(gi["confirm"])
        both += int(gp["confirm"] and gi["confirm"])
        neither += int((not gp["confirm"]) and (not gi["confirm"]))
        n_div += int(not gp["converged"])
        done = s + 1
        sys.stdout.write(
            f"\r  [{target_pp:>4.1f}pp] sim {done}/{n_sims}  "
            f"pymc={pymc_conf/done:.2f} irls={irls_conf/done:.2f}  "
            f"({(time.time()-t0)/done:.1f}s/fit)")
        sys.stdout.flush()
    sys.stdout.write("\n")
    return dict(
        target_pp=target_pp, beta_len=bl, n_sims=n_sims,
        power_pymc=pymc_conf / n_sims, power_irls=irls_conf / n_sims,
        agree_both=both / n_sims, agree_neither=neither / n_sims,
        n_nonconverged=n_div,
        median_pymc_pdir=float(np.median([r["pymc"]["p_dir"] for r in rows])),
        median_pymc_ppd=float(np.median([r["pymc"]["ppd_med"] for r in rows])),
        median_pymc_b3=float(np.median([r["pymc"]["b3_med"] for r in rows])),
        secs=time.time() - t0,
    )


def write_outputs(meta, cells):
    OUT_JSON.write_text(json.dumps({"meta": meta, "cells": cells}, indent=2),
                        encoding="utf-8")
    L = ["# Condition-ITT power — full-PyMC confirmation (#52)\n"]
    L.append("> Registered Bayesian gate (eq. 6.1', Pr(b3<0)>0.95 AND robust "
             "short->long pp-drop>6) vs the IRLS proxy, PAIRED on identical "
             "calibration-anchored simulated data.\n")
    L.append(f"**Design**: {meta['n_models']} models x "
             f"{meta['n_high']}+{meta['n_low']} items x reps={meta['reps']} x "
             f"short:long {meta['short_conds']}:{meta['long_conds']} = "
             f"{meta['n_obs']} obs/sim. Sampler: draws={meta['draws']}, "
             f"tune={meta['tune']}, chains={meta['chains']}. "
             f"{meta['sims']} sims/cell.\n")
    L.append(f"**High-regime short base rates** (resampled measured C1): "
             f"mean={meta['p_short_mean']:.3f}, "
             f"floored={meta['p_short_floored']}, ceiled={meta['p_short_ceiled']}.\n")
    L.append("| true pp | power (Bayesian gate) | power (IRLS proxy) | "
             "median Pr(b3<0) | median pp-drop | median b3 | non-conv | s/fit |")
    L.append("|---|---|---|---|---|---|---|---|")
    for c in cells:
        L.append(
            f"| {c['target_pp']:.1f} | **{c['power_pymc']:.2f}** | "
            f"{c['power_irls']:.2f} | {c['median_pymc_pdir']:.3f} | "
            f"{c['median_pymc_ppd']:.1f} | {c['median_pymc_b3']:+.3f} | "
            f"{c['n_nonconverged']}/{c['n_sims']} | "
            f"{c['secs']/c['n_sims']:.1f} |")
    L.append("")
    L.append("Notes:")
    L.append("- true pp = TRUE mean high-regime short->long accuracy drop in the DGP.")
    L.append("- measured calibration effect range: ~7pp (clean C1->C4) to ~13pp "
             "(full C1->C5); the registered design is locked at reps=5.")
    L.append("- Bayesian gate = the ACTUAL registered gate (random model "
             "intercept+slope over 5 clusters, item random intercept, N(0,1) "
             "shrinkage on b3). IRLS proxy = condition_power.py's fixed-effects "
             "Wald gate, re-applied on the SAME data with the SAME empirical "
             "short-rate anchor.")
    L.append("- Type I error = the power row at true pp = 0.0 (should be small).")
    L.append("- non-conv = sims with R-hat >= 1.01 or ESS <= 400 (registered "
             "convergence gate); power is over all sims (a real run re-samples "
             "a non-converged fit, so excluding them would only raise power).")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    # CLI
    reps = 5
    sims = 80
    effects = [0.0, 7.0, 10.0]
    draws, tune, chains = 1000, 1000, 2
    smoke = False
    base_seed = 20260531
    for a in sys.argv[1:]:
        if a == "--smoke":
            smoke = True
        elif a.startswith("--reps="):
            reps = int(a.split("=", 1)[1])
        elif a.startswith("--sims="):
            sims = int(a.split("=", 1)[1])
        elif a.startswith("--effects="):
            effects = [float(x) for x in a.split("=", 1)[1].split(",")]
        elif a.startswith("--draws="):
            draws = int(a.split("=", 1)[1])
        elif a.startswith("--tune="):
            tune = int(a.split("=", 1)[1])
        elif a.startswith("--chains="):
            chains = int(a.split("=", 1)[1])

    cp.N_REPS = reps                              # build_design reads this global
    X, regime, long, item_idx, model_idx, names = cp.build_design()
    rng = np.random.default_rng(base_seed)
    p_short_high = rng.choice(cp.HIGH_C1, size=cp.N_HIGH_ITEMS, replace=True)

    meta = dict(
        reps=reps, sims=sims, draws=draws, tune=tune, chains=chains,
        n_models=cp.N_MODELS, n_high=cp.N_HIGH_ITEMS, n_low=cp.N_LOW_ITEMS,
        n_obs=int(X.shape[0]), short_conds=cp.N_SHORT_CONDS,
        long_conds=cp.N_LONG_CONDS,
        p_short_mean=float(p_short_high.mean()),
        p_short_floored=int(np.sum(p_short_high < 0.10)),
        p_short_ceiled=int(np.sum(p_short_high > 0.90)),
    )
    print(f"[design] {X.shape[0]} obs, {X.shape[1]} IRLS params, "
          f"{cp.N_MODELS} models, {cp.N_HIGH_ITEMS}+{cp.N_LOW_ITEMS} items, "
          f"reps={reps}, short:long {cp.N_SHORT_CONDS}:{cp.N_LONG_CONDS}")
    print(f"[gate] Bayesian: Pr(b3<0)>{CONFIRM_PROB} AND pp-drop>{MAG_FLOOR_PP}")
    print(f"[sampler] draws={draws} tune={tune} chains={chains}\n")

    if smoke:
        print("[smoke] one real-setting fit at 7pp to validate wiring + time it")
        bl = cp.solve_beta_len(7.0, p_short_high)
        y = simulate_y(rng, long, regime, item_idx, model_idx, p_short_high, bl)
        base = _empirical_short_base_logit(y, regime, long)
        t0 = time.time()
        gp = pymc_gate(y, regime, long, item_idx, model_idx, base,
                       draws, tune, chains, base_seed)
        dt = time.time() - t0
        gi = irls_gate(X, names, y, base)
        print(f"[smoke] PyMC fit {dt:.1f}s  confirm={gp['confirm']}  "
              f"Pr(b3<0)={gp['p_dir']:.3f}  pp-drop={gp['ppd_med']:.1f}  "
              f"b3={gp['b3_med']:+.3f}  conv={gp['converged']} "
              f"(rhat={gp['rhat_max']:.3f}, ess={gp['ess_min']:.0f})")
        print(f"[smoke] IRLS gate    confirm={gi['confirm']}  "
              f"p_one={gi['p_one']:.3f}  pp-drop={gi['ppd']:.1f}")
        est = dt * sims * len(effects)
        print(f"[smoke] est full run @ {sims} sims x {len(effects)} cells "
              f"~= {est/60:.0f} min ({est/3600:.1f} h)")
        return

    cells = []
    for tp in effects:
        c = run_cell(rng, tp, X, regime, long, item_idx, model_idx, names,
                     p_short_high, sims, draws, tune, chains, base_seed)
        cells.append(c)
        write_outputs(meta, cells)           # incremental: survive a dead console
        print(f"  -> power(Bayes)={c['power_pymc']:.2f}  "
              f"power(IRLS)={c['power_irls']:.2f}  "
              f"nonconv={c['n_nonconverged']}/{c['n_sims']}  "
              f"({c['secs']/60:.1f} min)\n")
    print(f"[done] {OUT_JSON.name} + {OUT_MD.name}")


if __name__ == "__main__":
    main()
