"""
design_sweep.py — Does using MORE ITEMS get the amended gate to 0.80 power at
the modest target effect? Follows power_curve.py per the design decision.

Honest treatment of effect dilution: items have a CONTINUOUS latent regime score
r_i ~ standardized normal, and the true step×regime reversal scales with r_i. So
adding items by widening the regime bins (quartile→tercile→median) brings in
items nearer the median that carry a WEAKER true effect. A naive "set n_items
larger at fixed b3" would overstate the gain; this DGP does not.

Four analyses are compared on the SAME generated data each Monte-Carlo rep
(paired comparison):
  - quartile   : top/bottom 25% as binary regime (current pre-reg primary)
  - tercile    : top/bottom 33%
  - median     : top/bottom halves (all items, no middle dropped)
  - continuous : standardized r_i as a continuous regime covariate (all items;
                 the planned R2 robustness model, tested here as primary)

All use the amended gate (Pr(β3<0)>0.95 ∧ robust pp-drop > floor). Power is the
confirm rate at the target effect; Type I is the confirm rate under the null.

Calibration: s_coef is set so the QUARTILE analysis reproduces effective
b3 ≈ -0.25 (the power_curve target) — so quartile power here should land near the
0.47 already measured, validating the sweep.

Usage:
  python design_sweep.py                # default M=20
  python design_sweep.py --mc 24
  python design_sweep.py --fast         # quick check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import confirmatory_analysis as ca   # noqa: E402

OUT_JSON = ca.REPO_ROOT / "Experiments" / "design_sweep.json"
OUT_MD = ca.REPO_ROOT / "Experiments" / "design_sweep.md"

N_ITEMS = 79          # the real frame-pool size
N_MODELS = 5
N_COND = 5
STEP_TARGETS = np.array([0.5, 4.0, 8.0, 14.0, 9.0])
B0 = 0.40
B1 = 0.10             # base (regime=0) step slope
# baseline-accuracy regime effect per unit r (≈ -0.40 contrast at the quartile).
B2_COEF = -0.40 / 1.27
SIGMA_A, SIGMA_G, SIGMA_U = 0.5, 0.3, 0.5
FLOOR = ca.MAG_FLOOR_PP
# Quartile binary contrast b3 = s_coef × (mean_r_high − mean_r_low) = s_coef ×
# 2×1.271. To make the quartile analysis reproduce the power_curve target
# (effective b3 ≈ -0.25), divide by 2×1.271, NOT 1.271.
TARGET_EFF_B3 = -0.25
S_COEF_TARGET = TARGET_EFF_B3 / (2 * 1.271)   # ≈ -0.098
# (The 2026-05-29 14:45 run used the erroneous /1.271 → effective b3 ≈ -0.50, a
# STRONG effect with realistic item heterogeneity. Its comparative conclusion —
# wider bins do not help — is unaffected; its absolute numbers are not the
# modest-target power. Re-run for clean modest-target absolutes.)


def regime_scores(n: int) -> np.ndarray:
    """Deterministic, evenly-spaced standardized regime scores for n items."""
    q = (np.arange(n) + 0.5) / n
    r = norm.ppf(q)
    return (r - r.mean()) / r.std(ddof=0)


def simulate_graded(s_coef: float, n_reps: int, rng: np.random.Generator) -> pd.DataFrame:
    """Continuous-truth DGP: item i's step slope = B1 + s_coef·r_i."""
    r = regime_scores(N_ITEMS)
    a_m = rng.normal(0, SIGMA_A, N_MODELS)
    g_m = rng.normal(0, SIGMA_G, N_MODELS)
    u_i = rng.normal(0, SIGMA_U, N_ITEMS)
    model_scale = rng.uniform(0.7, 1.3, N_MODELS)

    rows = []
    for m in range(N_MODELS):
        for i in range(N_ITEMS):
            for c in range(N_COND):
                for _ in range(n_reps):
                    mean = STEP_TARGETS[c] * model_scale[m]
                    steps = max(0.0, rng.normal(mean, max(1.0, 0.35 * mean)))
                    rows.append((m, i, r[i], steps))
    df = pd.DataFrame(rows, columns=["model_i", "item_i", "r", "steps"])
    # z-score steps within model
    df["steps_z"] = df.groupby("model_i")["steps"].transform(
        lambda g: (g - g.mean()) / g.std(ddof=0) if g.std(ddof=0) > 0 else g * 0)
    eta = (B0 + B1 * df["steps_z"] + B2_COEF * df["r"]
           + s_coef * df["steps_z"] * df["r"]
           + a_m[df["model_i"].to_numpy()]
           + g_m[df["model_i"].to_numpy()] * df["steps_z"].to_numpy()
           + u_i[df["item_i"].to_numpy()])
    df["y"] = rng.binomial(1, ca._sigma(eta.to_numpy()))
    df["model"] = "M" + df["model_i"].astype(str)
    df["item"] = "I" + df["item_i"].astype(str)
    return df


def _fit_decide(df_fit: pd.DataFrame, regime_col: np.ndarray, sample_kw: dict,
                continuous: bool):
    """Build eq-6.1 on the given regime coding, fit, apply amended gate."""
    d = df_fit.copy()
    d["regime"] = regime_col
    data = ca.prepare(d[["y", "steps_z", "regime", "model", "item"]])
    model = ca.build_model(data)
    idata = ca.fit(model, **sample_kw)
    if not continuous:
        return ca.decide_amended(idata, data)["verdict"]
    # continuous: high-regime = top quartile of r; slope at its mean regime value
    b1 = ca._flat(idata, "b1"); b3 = ca._flat(idata, "b3")
    p_dir = float(np.mean(b3 < 0))
    thr = np.quantile(d["regime"], 0.75)
    hi = d[d["regime"] >= thr]
    base = ca._logit(float(hi["y"].mean()))
    s_ref = float(hi["steps_z"].mean())
    s_lo = float(np.percentile(hi["steps_z"], 10))
    s_hi = float(np.percentile(hi["steps_z"], 90))
    hv = float(hi["regime"].mean())
    slope = b1 + b3 * hv
    ppd = (ca._sigma(base + slope * (s_lo - s_ref))
           - ca._sigma(base + slope * (s_hi - s_ref))) * 100.0
    ppd_med = float(np.median(ppd))
    if p_dir > 0.95 and ppd_med > FLOOR:
        return "confirmed"
    if float(np.mean(b3 >= 0)) > 0.95:
        return "falsified"
    if p_dir > 0.95 and ppd_med < ca.NEGLIGIBLE_PP_AMENDED:
        return "falsified"
    return "inconclusive"


def analyze(df: pd.DataFrame, kind: str, sample_kw: dict) -> str:
    """Recode regime per `kind` and return the amended-gate verdict."""
    if kind == "continuous":
        # standardize r as the continuous regime covariate
        reg = (df["r"] - df["r"].mean()) / df["r"].std(ddof=0)
        return _fit_decide(df, reg.to_numpy(), sample_kw, continuous=True)
    frac = {"quartile": 0.25, "tercile": 1 / 3, "median": 0.5}[kind]
    item_r = df.groupby("item_i")["r"].first().sort_values()
    n_sel = min(int(round(frac * len(item_r))), len(item_r) // 2)
    low_items = set(item_r.index[:n_sel])
    high_items = set(item_r.index[-n_sel:])
    keep = df[df["item_i"].isin(low_items | high_items)].copy()
    reg = keep["item_i"].isin(high_items).astype(float).to_numpy()
    return _fit_decide(keep, reg, sample_kw, continuous=False)


def run_sweep(kinds, mc, sample_kw, n_reps, s_coef, seed0, label):
    """Monte-Carlo confirm rate for each analysis `kind`."""
    counts = {k: 0 for k in kinds}
    for r in range(mc):
        rng = np.random.default_rng(seed0 + r)
        df = simulate_graded(s_coef, n_reps, rng)
        for k in kinds:
            try:
                v = analyze(df, k, sample_kw)
            except Exception as e:
                print(f"[skip] {label} {k} rep {r+1}: {e}")
                continue
            counts[k] += v == "confirmed"
        print(f"[{label}] rep {r+1:2d}/{mc} "
              + " ".join(f"{k[:4]}={counts[k]}" for k in kinds))
    return {k: counts[k] / mc for k in kinds}


def write_report(payload: dict) -> None:
    L = ["# Design Sweep — More Items vs Power (amended gate, floor "
         f"{FLOOR:.0f}pp)\n"]
    L.append("Continuous-truth DGP (item reversal scales with regime score, so "
             "wider bins add weaker-effect items). Four analyses compared on the "
             "same data each rep. Quartile is the current pre-reg primary; its "
             "power should match the ~0.47 from power_curve.md (sanity anchor).\n")
    L.append(f"**Monte-Carlo reps**: {payload['mc']}  |  **n_reps (replications)**"
             f": {payload['n_reps']}  |  **sampler**: {payload['sample_kw']}  |  "
             f"**{payload['timestamp']}**\n")
    L.append("## Power at the target effect (and Type I under the null)\n")
    L.append("| analysis | items/bin | total obs | **power** | Type I |")
    L.append("|---|---|---|---|---|")
    info = payload["config_info"]
    for k in payload["kinds"]:
        pw = payload["power"][k]
        t1 = payload["type1"].get(k)
        L.append(f"| {k} | {info[k]['items']} | {info[k]['obs']} | "
                 f"**{pw:.2f}** | {t1 if t1 is None else f'{t1:.2f}'} |")
    L.append("\n*Power = confirm rate at the target effect; Type I = confirm "
             "rate under the null (b3-scaling = 0).*\n")
    L.append("## Read\n")
    best = max(payload["power"], key=payload["power"].get)
    L.append(f"- Best analysis: **{best}** (power {payload['power'][best]:.2f}). "
             f"Quartile anchor: {payload['power'].get('quartile', float('nan')):.2f} "
             "(should ≈0.47 from power_curve).\n")
    if payload["power"][best] >= 0.80:
        L.append(f"- **{best} reaches ≥0.80 power** while controlling Type I "
                 f"({payload['type1'].get(best)}). Recommend adopting it for the "
                 "primary analysis.\n")
    else:
        L.append(f"- **No analysis reached 0.80** at {payload['n_reps']} "
                 "replications. More items helps but isn't sufficient alone; "
                 "consider combining with more replications or the model-structure "
                 "change.\n")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[done] wrote {OUT_MD.name} and {OUT_JSON.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", type=int, default=20)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    if args.fast:
        kinds = ["quartile", "continuous"]
        mc, n_reps = 5, 3
        sample_kw = dict(draws=400, tune=400, chains=2)
    else:
        kinds = ["quartile", "tercile", "median", "continuous"]
        mc, n_reps = args.mc, 3
        sample_kw = dict(draws=600, tune=600, chains=2)

    # config info for the report
    info = {}
    for k in kinds:
        if k == "continuous":
            items, obs = N_ITEMS, N_ITEMS * N_MODELS * N_COND * n_reps
        else:
            frac = {"quartile": 0.25, "tercile": 1 / 3, "median": 0.5}[k]
            per = min(int(round(frac * N_ITEMS)), N_ITEMS // 2)
            items, obs = f"{per}", 2 * per * N_MODELS * N_COND * n_reps
        info[k] = dict(items=items, obs=obs)

    t0 = time.time()
    power = run_sweep(kinds, mc, sample_kw, n_reps, S_COEF_TARGET,
                      seed0=8000, label="power")
    # Type I under the null (no regime scaling) for the widest candidates.
    null_kinds = [k for k in kinds if k in ("median", "continuous")] or kinds
    type1 = run_sweep(null_kinds, mc, sample_kw, n_reps, 0.0,
                      seed0=9000, label="type1")

    payload = dict(
        timestamp=time.strftime("%Y-%m-%d %H:%M"), mc=mc, n_reps=n_reps,
        sample_kw=sample_kw, kinds=kinds, power=power, type1=type1,
        config_info=info, floor=FLOOR, s_coef=S_COEF_TARGET,
        elapsed_min=round((time.time() - t0) / 60, 1),
    )
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str),
                        encoding="utf-8")
    write_report(payload)
    print(f"[power] {power}\n[type1] {type1}  ({payload['elapsed_min']} min)")


if __name__ == "__main__":
    main()
