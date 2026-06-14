"""
calibration_ablation.py — locate WHY b3 flips sign between a pooled logistic
(b3 ~= -0.09, H1 direction) and the full registered hierarchical model
(b3 = +0.115, wrong direction) on the SAME calibration data.

Refits eq.6.1 on the calibration data with the hierarchical components added one
at a time, on the identical within-model-standardized step scale the registered
pipeline uses, and prints b3 median + Pr(b3<0) at each stage. Whichever addition
crosses b3 from - to + is the culprit.

  M0  pooled               b0,b1,b2,b3
  M1  + item intercept     + u_i
  M2  + model intercept    + a_m            (no model slope)
  M3  full (REGISTERED)    + model slope g_m

Read-only: touches no registered artifact. Reuses ca.standardize/prepare so the
design matrix is byte-identical to confirmatory_analysis.run().
"""

from __future__ import annotations

import sys
from pathlib import Path

import json
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import calibration_analysis as cal
import confirmatory_analysis as ca

SAMPLE_KW = dict(draws=1000, tune=1000, chains=2, target_accept=0.95)


def build(data, *, item_eff, model_intercept, model_slope):
    import pymc as pm

    y = data["y"]
    steps = data["steps_z"]
    regime = data["regime"]
    midx = data["model_idx"]
    iidx = data["item_idx"]
    M, I = data["n_models"], data["n_items"]

    with pm.Model() as m:
        b0 = pm.Normal("b0", 0.0, 2.5)
        b2 = pm.Normal("b2", 0.0, 2.5)
        b1 = pm.Normal("b1", 0.0, 1.0)
        b3 = pm.Normal("b3", 0.0, 1.0)
        eta = b0 + b1 * steps + b2 * regime + b3 * (steps * regime)

        if model_intercept:
            sa = pm.HalfNormal("sigma_a", 1.0)
            za = pm.Normal("z_a", 0.0, 1.0, shape=M)
            a_m = pm.Deterministic("a_m", sa * za)
            eta = eta + a_m[midx]
        if model_slope:
            sg = pm.HalfNormal("sigma_g", 1.0)
            zg = pm.Normal("z_g", 0.0, 1.0, shape=M)
            g_m = pm.Deterministic("g_m", sg * zg)
            eta = eta + g_m[midx] * steps
        if item_eff and I > 1:
            su = pm.HalfNormal("sigma_u", 1.0)
            zu = pm.Normal("z_u", 0.0, 1.0, shape=I)
            u_i = pm.Deterministic("u_i", su * zu)
            eta = eta + u_i[iidx]

        pm.Bernoulli("y_obs", logit_p=eta, observed=y)
    return m


def fit_b3(data, label, **flags):
    import pymc as pm
    m = build(data, **flags)
    with m:
        idata = pm.sample(cores=1, random_seed=20260514, progressbar=False,
                          idata_kwargs={"log_likelihood": False}, **SAMPLE_KW)
    b1 = idata.posterior["b1"].to_numpy().reshape(-1)
    b3 = idata.posterior["b3"].to_numpy().reshape(-1)
    sg = (idata.posterior["sigma_g"].to_numpy().reshape(-1)
          if "sigma_g" in idata.posterior else None)
    print(f"{label:32s}  b1={np.median(b1):+.3f}  b3={np.median(b3):+.3f}  "
          f"Pr(b3<0)={np.mean(b3<0):.3f}"
          + (f"  sigma_g={np.median(sg):.3f}" if sg is not None else ""))
    return float(np.median(b3)), float(np.mean(b3 < 0))


def main():
    cells = json.loads((HERE / "calibration_responses.json").read_text(encoding="utf-8"))
    pool = cal.load_pool()
    df = cal.score_cells(cells, pool)
    df = ca.standardize_steps_within_model(df[["y", "steps", "regime", "model", "item"]])
    data = ca.prepare(df)

    print(f"[data] {len(df)} rows, {data['n_models']} models, {data['n_items']} items")
    print("=" * 78)
    print("ABLATION: add hierarchical components one at a time; watch b3 cross 0")
    print("  H1 predicts b3 < 0.  Pooled is known ~ -0.09; full registered = +0.115")
    print("=" * 78)
    fit_b3(data, "M0 pooled (no REs)",
           item_eff=False, model_intercept=False, model_slope=False)
    fit_b3(data, "M1 + item intercept",
           item_eff=True, model_intercept=False, model_slope=False)
    fit_b3(data, "M2 + model intercept a_m",
           item_eff=True, model_intercept=True, model_slope=False)
    fit_b3(data, "M3 + model slope g_m (FULL)",
           item_eff=True, model_intercept=True, model_slope=True)
    print("=" * 78)
    print("The line where b3 turns positive identifies the culprit component.")


if __name__ == "__main__":
    main()
