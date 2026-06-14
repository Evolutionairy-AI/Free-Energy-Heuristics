"""
calibration_diagnose.py — why does the descriptive H1 signal (cbd falls C1->C5,
low regime flat) disagree with the registered regression (Pr(b3<0)=0.19)?

Hypothesis: the registered regressor `steps` (realized n_steps_heuristic) is
ENDOGENOUS. On easy (low-regime) items a model emits more steps when it is
struggling, so steps<->wrong correlate WITHIN the low regime, manufacturing a
spurious negative low-regime slope (b1<0) that dominates the regime x steps
interaction b3. The exogenous manipulation is the assigned CONDITION (C1..C5).

This script (descriptive, no PyMC) checks:
  1. mean realized steps by (regime, condition) — does the manipulation move
     steps, and is the steps range compressed/overlapping across regimes?
  2. within-regime association of y with realized steps vs with condition-ordinal
  3. simple population logistic fits: y ~ steps*regime vs y ~ cond*regime, to see
     whether the interaction sign flips when the exogenous regressor is used.
"""

from __future__ import annotations

import sys
from pathlib import Path

import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import calibration_analysis as cal  # reuse loaders + scorer

COND_ORD = {"C1": 0, "C2": 1, "C3": 2, "C4": 3, "C5": 4}


def logit_fit(X: np.ndarray, y: np.ndarray, names, ridge=1e-4, iters=200):
    """Tiny IRLS logistic regression (no statsmodels dependency). Returns dict
    name->coef. X already includes the intercept column."""
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(iters):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = np.clip(mu * (1 - mu), 1e-9, None)
        # IRLS update with tiny ridge for stability
        XtW = X.T * W
        H = XtW @ X + ridge * np.eye(p)
        g = X.T @ (y - mu) - ridge * beta
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return dict(zip(names, beta))


def main():
    cells = json.loads((HERE / "calibration_responses.json").read_text(encoding="utf-8"))
    pool = cal.load_pool()
    df = cal.score_cells(cells, pool)
    df["cond_ord"] = df["condition"].map(COND_ORD)

    print("=" * 70)
    print("1. MEAN REALIZED STEPS by regime x condition (does manipulation move steps?)")
    print("=" * 70)
    piv = df.pivot_table(index="regime", columns="condition", values="steps", aggfunc="mean")
    print(piv.round(2).to_string())
    print("\nsteps range overlap:")
    for r in (0, 1):
        s = df[df["regime"] == r]["steps"]
        lab = "high(K)" if r == 1 else "low(R/A)"
        print(f"  {lab}: steps mean={s.mean():.2f} sd={s.std():.2f} "
              f"min={s.min()} p10={np.percentile(s,10):.0f} "
              f"p90={np.percentile(s,90):.0f} max={s.max()}")

    print("\n" + "=" * 70)
    print("2. WITHIN-REGIME association of y with realized steps vs condition")
    print("=" * 70)
    for r in (1, 0):
        sub = df[df["regime"] == r]
        lab = "high(K)" if r == 1 else "low(R/A)"
        # point-biserial-ish: corr of continuous regressor with binary y
        c_steps = np.corrcoef(sub["steps"], sub["y"])[0, 1]
        c_cond = np.corrcoef(sub["cond_ord"], sub["y"])[0, 1]
        # mean y in bottom vs top steps tercile (within regime)
        q1, q2 = np.percentile(sub["steps"], [33, 66])
        lo = sub[sub["steps"] <= q1]["y"].mean()
        hi = sub[sub["steps"] >= q2]["y"].mean()
        # mean y at C1 vs C5
        yc1 = sub[sub["condition"] == "C1"]["y"].mean()
        yc5 = sub[sub["condition"] == "C5"]["y"].mean()
        print(f"\n  {lab}:")
        print(f"    corr(y, realized steps) = {c_steps:+.3f}   "
              f"| y in low-steps tercile={lo:.3f}  high-steps tercile={hi:.3f}  "
              f"(diff {hi-lo:+.3f})")
        print(f"    corr(y, condition ord)  = {c_cond:+.3f}   "
              f"| y at C1={yc1:.3f}  C5={yc5:.3f}  (diff {yc5-yc1:+.3f})")

    print("\n" + "=" * 70)
    print("3. POPULATION LOGISTIC: y ~ regressor*regime  (interaction = H1 term)")
    print("=" * 70)
    y = df["y"].to_numpy(float)
    regime = df["regime"].to_numpy(float)
    n = len(df)
    ones = np.ones(n)

    for reg_name, reg_col in (("realized steps (REGISTERED)", "steps"),
                              ("condition ord (EXOGENOUS)", "cond_ord")):
        x = df[reg_col].to_numpy(float)
        xs = (x - x.mean()) / x.std()  # standardize for comparability
        X = np.column_stack([ones, xs, regime, xs * regime])
        names = ["b0", "b1_slope", "b2_regime", "b3_interaction"]
        coef = logit_fit(X, y, names)
        low_slope = coef["b1_slope"]
        high_slope = coef["b1_slope"] + coef["b3_interaction"]
        print(f"\n  regressor = {reg_name}")
        print(f"    b1 (low-regime slope)   = {low_slope:+.3f}")
        print(f"    b3 (interaction)        = {coef['b3_interaction']:+.3f}   "
              f"<-- H1 predicts < 0")
        print(f"    high-regime slope b1+b3 = {high_slope:+.3f}")

    print("\n" + "=" * 70)
    print("4. Is realized-steps endogenous in low regime? (steps by correctness)")
    print("=" * 70)
    for r in (1, 0):
        sub = df[df["regime"] == r]
        lab = "high(K)" if r == 1 else "low(R/A)"
        s_wrong = sub[sub["y"] == 0]["steps"].mean()
        s_right = sub[sub["y"] == 1]["steps"].mean()
        print(f"  {lab}: mean steps | correct={s_right:.2f}  wrong={s_wrong:.2f}  "
              f"(wrong-correct {s_wrong-s_right:+.2f})")

    # Also break low regime into R (arithmetic) vs A (aleatory) — A gold-match
    # may be noisy/steps-correlated.
    print("\n  low-regime split (R arithmetic vs A aleatory):")
    lo = df[df["regime"] == 0].copy()
    lo["fam"] = lo["item"].str[0]
    for fam in sorted(lo["fam"].unique()):
        f = lo[lo["fam"] == fam]
        print(f"    {fam}: n={len(f)} acc={f['y'].mean():.3f} "
              f"corr(y,steps)={np.corrcoef(f['steps'], f['y'])[0,1]:+.3f}")


if __name__ == "__main__":
    main()
