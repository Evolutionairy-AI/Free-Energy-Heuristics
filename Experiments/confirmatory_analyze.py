"""
confirmatory_analyze.py — THE registered confirmatory analysis of FEH H1.

Reads the confirmatory data collected by confirmatory_runner.py
(`confirmatory_responses.json`), scores it with the hardened canonical scorer
(calibration_analysis.is_cbd / gold_match — LaTeX- and significant-figure-robust),
and applies the REGISTERED decision rule (pre-reg v0.4, Amendment 2).

PRIMARY TEST — eq. 6.1' (assigned-length ITT):
    eta = b0 + b1*long + b2*regime + b3*(long*regime) + a_m + g_m*long + u_i
  where long = 0 for C1 (short), 1 for C2-C5 (long). This is the registered
  primary regressor (Amendment 2): the EXOGENOUS assigned length, not realized
  steps. Implemented by feeding `long` into the locked confirmatory_analysis
  model's regressor slot UNSTANDARDIZED (binary factor; same construction the
  full-PyMC power confirmation `condition_power_pymc.py` validated).

  GATE (§6.2 / §7.1, re-targeted to the b3 of eq. 6.1'):
    H1 CONFIRMED iff  Pr(b3 < 0 | data) > 0.95  AND  posterior-median robust
    high-regime short->long accuracy drop > 6 pp.
    FALSIFIED iff Pr(b3 >= 0) > 0.95, or Pr(b3<0)>0.95 but drop < 3 pp.
    else INCONCLUSIVE.

SECONDARY — R7 (realized-steps mediation): the OLD registered eq. 6.1 on
within-model z-scored realized steps, reported (not gated), via the registered
`decide()`.

This script does NOT collect data and never writes to confirmatory_responses.json
— it is safe to run at any time (on partial data it prints a completeness
warning). Outputs confirmatory_analysis_results.{md,json}.

Usage:
  python confirmatory_analyze.py            # registered sampler (draws=2000x4)
  python confirmatory_analyze.py --fast     # draws=500x2 (quick look; NOT the
                                            #   registered fit — for dev only)
  python confirmatory_analyze.py --expect=7875   # warn if fewer scored cells
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))

import confirmatory_analysis as ca           # noqa: E402  (locked eq. 6.1 pipeline)
from calibration_analysis import (           # noqa: E402  (hardened scorer)
    is_cbd, gold_match, delatex,             # noqa: F401  (delatex re-export)
)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

RESPONSES_PATH = HERE / "confirmatory_responses.json"
POOL_PATH = REPO_ROOT / "feh79_item_pool_v0.4.yaml"   # REGISTERED pool
OUT_JSON = HERE / "confirmatory_analysis_results.json"
OUT_MD = HERE / "confirmatory_analysis_results.md"
CONDS = ("C1", "C2", "C3", "C4", "C5")


def load_pool() -> dict:
    import yaml
    docs = list(yaml.safe_load_all(POOL_PATH.read_text(encoding="utf-8")))
    return {d["frame_id"]: d for d in docs if d and "frame_id" in d}


def score_cells(cells: list, pool: dict) -> pd.DataFrame:
    """Score each completed cell. high → cbd-correctness; low → gold-match.
    Mirrors calibration_analysis.score_cells (same hardened scorer) + keeps the
    `long` ITT factor and source for the confirmatory model."""
    rows = []
    for c in cells:
        if c.get("error") or not c.get("raw_response"):
            continue   # failed generation: drop, do not count as wrong
        fid = c["frame_id"]
        ans = c.get("extracted_final_answer") or ""
        if c["regime"] == "high":
            y = int(is_cbd(ans))
        else:
            gold = pool.get(fid, {}).get("gold_answer")
            y = int(gold_match(ans, gold))
        rows.append({
            "y": y,
            "steps": c["n_steps_heuristic"],
            "regime": 1 if c["regime"] == "high" else 0,
            "long": 0 if c["condition"] == "C1" else 1,   # assigned-length ITT
            "model": c["model"],
            "item": fid,
            "condition": c["condition"],
            "source": c.get("source", "?"),
        })
    return pd.DataFrame(rows)


def fit_primary_itt(df: pd.DataFrame, sample_kw: dict):
    """Fit the registered eq. 6.1' (assigned-length ITT). Feeds the binary `long`
    factor into the locked model's regressor slot (unstandardized), reusing
    ca.build_model / ca.fit so priors, non-centered REs, and sampler are exactly
    the registered ones. Returns (decision_amended_dict, registered_decision,
    idata, data)."""
    d = df.copy()
    d["steps_z"] = d["long"].astype("float64")   # binary ITT regressor in the slot
    data = ca.prepare(d[["y", "steps_z", "regime", "model", "item"]])
    model = ca.build_model(data, model_effects=(data["n_models"] > 1))
    idata = ca.fit(model, **sample_kw)
    amended = ca.decide_amended(idata, data)      # the GATED rule (Pr(b3<0)+pp-drop)
    registered = ca.decide(idata, data)           # full-reversal stat (descriptive)
    return amended, registered, idata, data


def descriptive(df: pd.DataFrame) -> dict:
    """cbd / accuracy by condition x regime, per-model short->long drops, and the
    difference-in-differences (high drop - low drop). Pure description."""
    out = {}
    hi = df[df["regime"] == 1]
    lo = df[df["regime"] == 0]

    out["high_cbd_by_condition"] = {
        c: round(float(hi[hi["condition"] == c]["y"].mean()), 4)
        for c in CONDS if len(hi[hi["condition"] == c])}
    out["low_acc_by_condition"] = {
        c: round(float(lo[lo["condition"] == c]["y"].mean()), 4)
        for c in CONDS if len(lo[lo["condition"] == c])}

    def short_long_drop(sub):
        s = sub[sub["long"] == 0]["y"]
        l = sub[sub["long"] == 1]["y"]
        if not len(s) or not len(l):
            return None
        return round(float(s.mean() - l.mean()) * 100, 2)   # pp, +ve = falls

    out["high_short_long_drop_pp"] = short_long_drop(hi)
    out["low_short_long_drop_pp"] = short_long_drop(lo)
    if out["high_short_long_drop_pp"] is not None and \
            out["low_short_long_drop_pp"] is not None:
        out["DiD_pp"] = round(out["high_short_long_drop_pp"]
                              - out["low_short_long_drop_pp"], 2)

    out["per_model"] = {}
    for m in sorted(df["model"].unique()):
        sub = df[df["model"] == m]
        out["per_model"][m] = {
            "n": int(len(sub)),
            "high_drop_pp": short_long_drop(sub[sub["regime"] == 1]),
            "low_drop_pp": short_long_drop(sub[sub["regime"] == 0]),
        }
    return out


def main():
    fast = "--fast" in sys.argv[1:]
    expect = 7875
    for a in sys.argv[1:]:
        if a.startswith("--expect="):
            expect = int(a.split("=", 1)[1])

    if not RESPONSES_PATH.exists():
        raise SystemExit(f"no {RESPONSES_PATH.name} yet — run confirmatory_runner.py first")
    cells = json.loads(RESPONSES_PATH.read_text(encoding="utf-8"))
    pool = load_pool()
    df = score_cells(cells, pool)

    n_err = sum(1 for c in cells if c.get("error"))
    n_models = df["model"].nunique()
    complete = len(df) >= expect
    print(f"[data] {len(cells)} raw cells ({n_err} errors) -> {len(df)} scored; "
          f"{n_models} models, {df['item'].nunique()} items")
    if not complete:
        print(f"[WARNING] {len(df)} scored < expected {expect} — this is a "
              f"PARTIAL/INTERIM read, NOT the confirmatory result. The registered "
              f"analysis runs ONCE on the COMPLETE dataset (no interim peeking at "
              f"the H1 estimate). Proceeding for pipeline/health check only.")

    desc = descriptive(df)
    print(f"[desc] high short->long drop = {desc.get('high_short_long_drop_pp')} pp; "
          f"low = {desc.get('low_short_long_drop_pp')} pp; "
          f"DiD = {desc.get('DiD_pp')} pp")

    # Official sampler: the pre-registered design specified 2000x4 @ target_accept
    # 0.95, but that fit leaves ONE nuisance parameter (b2, the high-regime
    # intercept) at R-hat=1.0128, marginally above the strict 1.01 gate (beta3 and
    # all else converge cleanly). Per decision 2026-06-02, the official fit uses a
    # longer chain (4000x4 @ 0.99) which clears R-hat<1.01 for ALL parameters and
    # yields an IDENTICAL verdict. This is a post-hoc draw increase for convergence
    # only — the model, data, regressor, priors, and decision gate are unchanged.
    OFFICIAL_SAMPLER = dict(draws=4000, tune=4000, chains=4, target_accept=0.99)
    sample_kw = dict(draws=500, tune=500, chains=2) if fast else dict(OFFICIAL_SAMPLER)
    print(f"[fit] primary eq. 6.1' ITT "
          f"({'FAST dev sampler' if fast else 'official sampler draws=4000x4 ta=0.99'}) ...")
    amended, registered, idata, data = fit_primary_itt(df, sample_kw)

    print(f"[fit] secondary R7 (realized-steps eq. 6.1) ...")
    df_steps = ca.standardize_steps_within_model(df)
    data_s = ca.prepare(df_steps[["y", "steps_z", "regime", "model", "item"]])
    model_s = ca.build_model(data_s, model_effects=(data_s["n_models"] > 1))
    idata_s = ca.fit(model_s, **sample_kw)
    r7 = ca.decide(idata_s, data_s)

    # ---- Verdict (primary gate) ----
    verdict = amended["verdict"]
    payload = {
        "complete": complete, "n_scored": int(len(df)), "expected": expect,
        "n_errors": n_err, "n_models": int(n_models),
        "models": sorted(df["model"].unique().tolist()),
        "sampler": "fast-dev" if fast else dict(OFFICIAL_SAMPLER),
        "sampler_note": (
            "Official fit uses 4000x4 @ target_accept=0.99 (post-hoc draw increase "
            "from the pre-registered 2000x4 @ 0.95 for convergence only). The 2000x4 "
            "fit left b2 (high-regime intercept) at R-hat=1.0128 > 1.01; the longer "
            "chain clears R-hat<1.01 for all parameters with an identical verdict. "
            "Model, data, regressor, priors, and decision gate are unchanged."),
        "primary_itt": {
            "verdict": verdict,
            "p_b3_neg": amended["p_dir"],
            "p_b3_nonneg": amended["p_b3_nonneg"],
            "b1_median": amended["b1_median"],
            "b3_median": amended["b3_median"],
            "robust_pp_drop_median": amended["pp_drop_robust_median"],
            "robust_pp_drop_ci": list(amended["pp_drop_robust_ci"]),
            "p_full_reversal_descriptive": amended["p_full_reversal"],
            "rhat_max": registered.rhat_max, "ess_min": registered.ess_min,
            "converged": registered.converged,
        },
        "secondary_r7_steps": {
            "verdict": r7.verdict, "p_b3_neg": r7.p_b3_neg,
            "b3_median": r7.b3_median, "b3_ci": list(r7.b3_ci),
            "pp_drop_median": r7.pp_drop_high_median,
            "converged": r7.converged,
        },
        "descriptive": desc,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ---- Markdown ----
    L = ["# Confirmatory Analysis — FEH H1 (registered eq. 6.1' ITT)\n"]
    if not complete:
        L.append(f"> ⚠️ **INTERIM / PARTIAL** ({len(df)}/{expect} cells). NOT the "
                 f"registered confirmatory result; pipeline/health check only.\n")
    L.append(f"**Data**: {len(df)} scored cells ({n_err} gen errors), "
             f"{n_models} models, {df['item'].nunique()} items.\n")
    if not fast:
        L.append("**Sampler**: official fit = 4000×4 @ target_accept=0.99 (post-hoc "
                 "draw increase from pre-registered 2000×4 @ 0.95 for convergence "
                 "only; verdict identical, model/gate unchanged).\n")
    L.append(f"**Convergence**: R̂_max={registered.rhat_max:.4f}, "
             f"ESS_min={registered.ess_min:.0f} → "
             f"{'✅ converged (all R̂<1.01)' if registered.converged else '⚠️ NOT CONVERGED'}.\n")
    L.append("## PRIMARY — eq. 6.1' assigned-length ITT (the registered gate)\n")
    L.append("| quantity | value |")
    L.append("|---|---|")
    L.append(f"| **Pr(β3 < 0 \\| data)** | **{amended['p_dir']:.4f}** (gate: >0.95) |")
    L.append(f"| **robust short→long pp-drop** | **{amended['pp_drop_robust_median']:+.2f}** "
             f"[{amended['pp_drop_robust_ci'][0]:+.2f}, {amended['pp_drop_robust_ci'][1]:+.2f}] (gate: >6) |")
    L.append(f"| β3 median | {amended['b3_median']:+.3f} |")
    L.append(f"| β1 median (low-regime length slope) | {amended['b1_median']:+.3f} |")
    L.append(f"| Pr(β3<0 ∧ \\|β3\\|>\\|β1\\|) (descriptive) | {amended['p_full_reversal']:.4f} |")
    L.append(f"\n### VERDICT (primary): **{verdict.upper()}**\n")
    gate1 = amended["p_dir"] > 0.95
    gate2 = amended["pp_drop_robust_median"] > 6
    L.append(f"- Directional gate Pr(β3<0)>0.95: {'✅ PASS' if gate1 else '❌ fail'} "
             f"({amended['p_dir']:.4f})")
    L.append(f"- Magnitude gate robust pp-drop>6: {'✅ PASS' if gate2 else '❌ fail'} "
             f"({amended['pp_drop_robust_median']:+.2f} pp)\n")
    L.append("## SECONDARY — R7 realized-steps (eq. 6.1, reported not gated)\n")
    L.append(f"- verdict={r7.verdict}; Pr(β3<0)={r7.p_b3_neg:.3f}; "
             f"β3={r7.b3_median:+.3f} [{r7.b3_ci[0]:+.3f},{r7.b3_ci[1]:+.3f}]; "
             f"pp-drop={r7.pp_drop_high_median:+.1f}\n")
    L.append("## Descriptive (the H1 signature)\n")
    L.append("| condition | high cbd-correct | low accuracy |")
    L.append("|---|---|---|")
    for c in CONDS:
        h = desc["high_cbd_by_condition"].get(c)
        lo = desc["low_acc_by_condition"].get(c)
        L.append(f"| {c} | {h if h is not None else '-'} | {lo if lo is not None else '-'} |")
    L.append(f"\nHigh-regime short→long drop **{desc.get('high_short_long_drop_pp')} pp** vs "
             f"low-regime **{desc.get('low_short_long_drop_pp')} pp** → "
             f"difference-in-differences **{desc.get('DiD_pp')} pp**.\n")
    L.append("| model | n | high drop pp | low drop pp |")
    L.append("|---|---|---|---|")
    for m, v in desc["per_model"].items():
        L.append(f"| {m} | {v['n']} | {v['high_drop_pp']} | {v['low_drop_pp']} |")
    L.append("")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"[done] {OUT_JSON.name} + {OUT_MD.name}")
    print(f"  PRIMARY verdict={verdict}  Pr(b3<0)={amended['p_dir']:.4f}  "
          f"robust pp-drop={amended['pp_drop_robust_median']:+.2f}  "
          f"converged={registered.converged}")


if __name__ == "__main__":
    main()
