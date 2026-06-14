"""
calibration_reconcile.py — definitively explain why the interaction b3 is
NEGATIVE (H1) in some specs and POSITIVE (anti-H1) in others, on the SAME data.

Deterministic IRLS logistic (no sampling, no priors) so estimator noise is
removed and only the SPECIFICATION differs. Decomposes b3 three ways:

  A. pooled b3 under raw / global-z / within-model-z steps  (isolates the
     standardization scheme — the registered pipeline uses within-model-z)
  B. per-model raw within-regime slopes (low vs high) — the assumption-light
     H1 test: is reasoning more harmful in the high regime, model by model?
  C. condition-based check excluding the broken C5 (steps monotone only C1->C4)
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibration_analysis as cal


def irls(X, y, ridge=1e-6, iters=300):
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = 1/(1+np.exp(-(X@beta)))
        W = np.clip(mu*(1-mu), 1e-9, None)
        H = (X.T*W)@X + ridge*np.eye(X.shape[1])
        g = X.T@(y-mu) - ridge*beta
        try: step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError: break
        beta += step
        if np.max(np.abs(step)) < 1e-9: break
    return beta


def b3_pooled(df, steps_col):
    y = df["y"].to_numpy(float)
    x = df[steps_col].to_numpy(float)
    r = df["regime"].to_numpy(float)
    X = np.column_stack([np.ones(len(df)), x, r, x*r])
    b = irls(X, y)
    return b[1], b[3]  # b1 (low slope), b3 (interaction)


def slope(df, steps_col="steps"):
    """within-subset logistic slope of y on raw steps."""
    if df["y"].nunique() < 2 or len(df) < 5:
        return float("nan")
    y = df["y"].to_numpy(float)
    x = df[steps_col].to_numpy(float)
    X = np.column_stack([np.ones(len(df)), x])
    return irls(X, y)[1]


def main():
    cells = json.loads((HERE/"calibration_responses.json").read_text(encoding="utf-8"))
    pool = cal.load_pool()
    df = cal.score_cells(cells, pool)

    # standardizations
    df["steps_global_z"] = (df["steps"]-df["steps"].mean())/df["steps"].std()
    df["steps_within_z"] = df.groupby("model")["steps"].transform(
        lambda g: (g-g.mean())/g.std(ddof=0) if g.std(ddof=0) > 0 else g*0)

    print("="*74)
    print("A. POOLED b3 under three step scalings (H1 predicts b3 < 0)")
    print("="*74)
    for col, lab in [("steps","raw steps"),
                     ("steps_global_z","global z-score"),
                     ("steps_within_z","within-MODEL z-score (REGISTERED)")]:
        b1, b3 = b3_pooled(df, col)
        print(f"  {lab:36s}  b1={b1:+.4f}  b3={b3:+.4f}  "
              f"{'<-- H1' if b3<0 else '<-- ANTI-H1'}")

    print("\n" + "="*74)
    print("B. PER-MODEL raw within-regime slope (assumption-light H1 test)")
    print("   H1 = high-regime slope MORE NEGATIVE than low-regime (per-model b3<0)")
    print("="*74)
    for m in sorted(df["model"].unique()):
        sub = df[df["model"]==m]
        s_low = slope(sub[sub["regime"]==0])
        s_high = slope(sub[sub["regime"]==1])
        print(f"  {m:22s} low={s_low:+.4f}  high={s_high:+.4f}  "
              f"per-model b3(high-low)={s_high-s_low:+.4f}  "
              f"{'H1' if s_high<s_low else 'anti'}")
    # pooled within-regime (raw), all models
    s_low = slope(df[df["regime"]==0]); s_high = slope(df[df["regime"]==1])
    print(f"  {'POOLED (all models)':22s} low={s_low:+.4f}  high={s_high:+.4f}  "
          f"b3={s_high-s_low:+.4f}")

    print("\n" + "="*74)
    print("C. CONDITION-based dissociation, excluding broken C5 (steps monotone C1->C4)")
    print("="*74)
    for r, lab in [(1,"high(K) cbd-correct"),(0,"low(R/A) accuracy")]:
        sub = df[df["regime"]==r]
        vals = {c: round(float(sub[sub["condition"]==c]["y"].mean()),3)
                for c in ["C1","C2","C3","C4","C5"]}
        d14 = vals["C4"]-vals["C1"]
        print(f"  {lab:22s} {vals}  C1->C4 delta={d14:+.3f}")


if __name__ == "__main__":
    main()
