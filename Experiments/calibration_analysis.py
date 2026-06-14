"""
calibration_analysis.py — score the calibration run and estimate the H1 effect
size + σ_g on real local-model output.

Scores each cell:
  high regime (Knightian): y = 1 iff the extracted answer is cbd
                           ('cannot-be-determined' — the normatively correct
                           response). H1 predicts cbd-correctness FALLS as steps
                           rise.
  low regime (R/A):        y = 1 iff the extracted answer matches gold_answer.

Builds the tidy frame (y, steps, regime, model, item) and runs the LOCKED
confirmatory pipeline (confirmatory_analysis.run, eq. 6.1). With 3 models the
model-slope random effect g_m is included, so σ_g is estimated. Reports:
  - population b1, b3, robust implied high-regime pp-drop (decide + decide_amended)
  - σ_g posterior (the cross-model slope heterogeneity)
  - per-item cbd-rate trajectories C1->C5 (effect heterogeneity, no cherry-picking)

NOT the confirmatory test. See Calibration_protocol.md.

Outputs: calibration_analysis.json, calibration_analysis.md
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))

import confirmatory_analysis as ca  # noqa: E402
from pilot_analysis import CBD_TOKENS  # noqa: E402

RESPONSES_PATH = HERE / "calibration_responses.json"
POOL_PATH = REPO_ROOT / "feh79_item_pool_v0.3.yaml"
OUT_JSON = HERE / "calibration_analysis.json"
OUT_MD = HERE / "calibration_analysis.md"

CONDS = ("C1", "C2", "C3", "C4", "C5")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")
# Frontier models format numeric answers as LaTeX under CoT, e.g.
# "\( \frac{1}{4} \)", "\dfrac{671}{1296}", "$\frac{1}{6}$". The plain numeric
# parser reads {1} and {4} as two separate numbers and the gold match fails —
# and ONLY under the long conditions where the model switches to LaTeX, which
# would manufacture a spurious condition effect. delatex() rewrites \frac{a}{b}
# to "a/b" and strips LaTeX delimiters so numeric_candidates() can read it. It
# is a strict superset (only adds candidates; a string without LaTeX is
# returned unchanged), so it can never flip an existing correct/incorrect match.
_FRAC_RE = re.compile(
    r"\\[dt]?frac\s*\{\s*(-?\d+(?:\.\d+)?)\s*\}\s*\{\s*(-?\d+(?:\.\d+)?)\s*\}"
)


def delatex(s: str) -> str:
    """Normalize LaTeX numeric formatting to plain text for the numeric matcher.
    No-op on strings that contain no LaTeX."""
    if not s or ("\\" not in s and "$" not in s and "{" not in s):
        return s
    s = _FRAC_RE.sub(r"\1/\2", s)            # \frac{a}{b} -> a/b
    for tok in (r"\(", r"\)", r"\[", r"\]", "$", r"\,", r"\!", r"\ ", "{", "}"):
        s = s.replace(tok, " ")
    return s


def load_pool() -> dict:
    import yaml
    docs = list(yaml.safe_load_all(POOL_PATH.read_text(encoding="utf-8")))
    return {d["frame_id"]: d for d in docs if d and "frame_id" in d}


def is_cbd(ans: str) -> bool:
    if not ans:
        return False
    norm = ans.lower().strip().rstrip(".,!? ").strip("'\"")
    return any(tok in norm for tok in CBD_TOKENS)


_BIGSF = 15  # exact fraction candidate: treat as full precision


def _token_sigfigs(tok: str) -> int:
    """Significant figures in a plain decimal/integer numeric string token."""
    s = re.split("[eE]", tok.lstrip("+-"))[0]
    if "." in s:
        intp, frac = s.split(".")
        return max(1, len((intp + frac).lstrip("0")))
    return max(1, len(s.lstrip("0")))


def _numeric_candidates_sf(text: str):
    """List of (value, sig_figs) numeric interpretations of a string. Adds the
    /100 reading only when a percent sign is present. A bare fraction a/b is
    exact (full precision); the percent re-reading inherits the token precision.
    LaTeX (\\frac{a}{b}, \\( \\)) is normalized first so CoT-formatted answers
    parse identically to plain decimals."""
    text = delatex(text)
    has_pct = "%" in text
    out = []
    for m in _NUM_RE.finditer(text.replace(",", "")):
        s = m.group(0)
        try:
            if "/" in s:
                v, sf = float(s.split("/")[0]) / float(s.split("/")[1]), _BIGSF
            else:
                v, sf = float(s), _token_sigfigs(s)
        except (ValueError, ZeroDivisionError):
            continue
        out.append((v, sf))
        if has_pct:
            out.append((v / 100.0, sf))
    return out


def numeric_candidates(text: str) -> set:
    """All numeric interpretations of a string (values only; the precision-aware
    version is _numeric_candidates_sf)."""
    return {v for v, _ in _numeric_candidates_sf(text)}


def _round_sf(x: float, sf: int) -> float:
    if x == 0:
        return 0.0
    return round(x, -int(math.floor(math.log10(abs(x)))) + (sf - 1))


def _sigfigs_of_gold(gold) -> int:
    return _token_sigfigs(gold if isinstance(gold, str) else repr(float(gold)))


def gold_match(extracted: str, gold) -> bool:
    if not extracted:
        return False
    try:
        g = float(gold)
    except (ValueError, TypeError):
        return str(gold).strip().lower() in extracted.lower()
    cands = _numeric_candidates_sf(extracted)
    # INTEGER golds are exact-arithmetic competence items (the R-set): a model
    # that *estimates* ("approximately 1860" for 1857) has genuinely failed, so
    # require an exact integer match.
    if g == round(g):
        return any(abs(v - g) < 0.5 for v, _ in cands)
    # NON-INTEGER golds are probabilities stored as roundings of an exact real
    # (e.g. flush prob 5148/2598960 = 0.0019808, stored as 0.00198). A model may
    # express the same value as an exact fraction (1/6), an over-precise decimal
    # (0.0019807923), or a coarser rounding (0.706 for 0.7063) — and which form it
    # uses correlates with CoT length, so a fixed tolerance mis-scores BY
    # CONDITION. Match if the two agree to the COARSER of their significant
    # figures, floored at 3 sig figs. The 3-sf floor (i) stops a vague "0.5"
    # crediting gold 0.5177, (ii) still accepts genuine 3-sf roundings, and
    # (iii) rejects true value errors whose 3rd significant figure differs
    # (0.703 vs 0.7063). The control golds are all >3-sf-separated, so no two
    # distinct golds can cross-match.
    gsf = _sigfigs_of_gold(gold)
    for v, vsf in cands:
        sf = max(3, min(gsf, vsf))
        a, b = _round_sf(v, sf), _round_sf(g, sf)
        if abs(a - b) <= 1e-9 * max(1.0, abs(b)):
            return True
    return False


def score_cells(cells: list, pool: dict) -> pd.DataFrame:
    rows = []
    for c in cells:
        if c.get("error") or not c.get("raw_response"):
            continue  # failed generation: drop, do not count as wrong
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
            "model": c["model"],
            "item": fid,
            "condition": c["condition"],
        })
    return pd.DataFrame(rows)


def cbd_trajectories(df: pd.DataFrame) -> dict:
    """Per high-regime item: mean cbd-correctness by condition (over models+reps),
    plus C1->C5 direction."""
    hi = df[df["regime"] == 1]
    out = {}
    for fid in sorted(hi["item"].unique()):
        sub = hi[hi["item"] == fid]
        traj = {cond: (round(float(sub[sub["condition"] == cond]["y"].mean()), 3)
                       if len(sub[sub["condition"] == cond]) else None)
                for cond in CONDS}
        c1, c5 = traj["C1"], traj["C5"]
        if c1 is None or c5 is None:
            direction = "?"
        elif c5 < c1 - 0.10:
            direction = "drop"
        elif c5 > c1 + 0.10:
            direction = "rise"
        else:
            direction = "flat"
        out[fid] = {"traj": traj, "direction": direction,
                    "mean_cbd": round(float(sub["y"].mean()), 3)}
    return out


def fmt_ci(ci) -> str:
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def main(sample_kw=None) -> None:
    cells = json.loads(RESPONSES_PATH.read_text(encoding="utf-8"))
    pool = load_pool()
    df = score_cells(cells, pool)

    n_models = df["model"].nunique()
    n_hi = (df["regime"] == 1).sum()
    n_lo = (df["regime"] == 0).sum()
    print(f"[scored] {len(df)} rows  ({n_hi} high, {n_lo} low)  "
          f"{n_models} models  {df['item'].nunique()} items")

    # Per-item cbd trajectories (descriptive; computed before the fit).
    trajs = cbd_trajectories(df)
    n_drop = sum(1 for v in trajs.values() if v["direction"] == "drop")
    n_floor = sum(1 for v in trajs.values() if v["mean_cbd"] < 0.10)
    n_ceil = sum(1 for v in trajs.values() if v["mean_cbd"] > 0.90)

    # Marginal cbd-rate by condition (high regime) and control accuracy by cond.
    hi = df[df["regime"] == 1]
    lo = df[df["regime"] == 0]
    hi_by_cond = {c: round(float(hi[hi["condition"] == c]["y"].mean()), 3)
                  for c in CONDS if len(hi[hi["condition"] == c])}
    lo_by_cond = {c: round(float(lo[lo["condition"] == c]["y"].mean()), 3)
                  for c in CONDS if len(lo[lo["condition"] == c])}

    # ---- Confirmatory pipeline fit (the locked eq. 6.1) ----
    kw = sample_kw or {}
    dec, idata, data = ca.run(df[["y", "steps", "regime", "model", "item"]], **kw)
    amended = ca.decide_amended(idata, data)

    # σ_g posterior (model-slope heterogeneity) + per-model slopes.
    sigma_g = ca._flat(idata, "sigma_g") if "sigma_g" in idata.posterior else None
    sg_med = float(np.median(sigma_g)) if sigma_g is not None else None
    sg_ci = ([float(np.percentile(sigma_g, 2.5)),
              float(np.percentile(sigma_g, 97.5))] if sigma_g is not None else None)
    if "g_m" in idata.posterior:
        gm = idata.posterior["g_m"].to_numpy().reshape(-1, data["n_models"])
        per_model_slope = {lbl: round(float(np.median(gm[:, i])), 3)
                           for i, lbl in enumerate(data["model_labels"])}
    else:
        per_model_slope = {}

    payload = {
        "n_rows": int(len(df)), "n_high": int(n_hi), "n_low": int(n_lo),
        "n_models": int(n_models), "models": sorted(df["model"].unique().tolist()),
        "high_cbd_by_condition": hi_by_cond,
        "low_acc_by_condition": lo_by_cond,
        "n_items_high": len(trajs),
        "n_drop": n_drop, "n_floor": n_floor, "n_ceiling": n_ceil,
        "b1_median": dec.b1_median, "b1_ci": list(dec.b1_ci),
        "b3_median": dec.b3_median, "b3_ci": list(dec.b3_ci),
        "p_b3_neg": dec.p_b3_neg,
        "slope_low_median": dec.slope_low_median,
        "slope_high_median": dec.slope_high_median,
        "pp_drop_registered_median": dec.pp_drop_high_median,
        "pp_drop_registered_ci": list(dec.pp_drop_high_ci),
        "pp_drop_robust_median": amended["pp_drop_robust_median"],
        "pp_drop_robust_ci": list(amended["pp_drop_robust_ci"]),
        "p_full_reversal": amended["p_full_reversal"],
        "verdict_registered": dec.verdict,
        "verdict_amended": amended["verdict"],
        "sigma_g_median": sg_med, "sigma_g_ci": sg_ci,
        "per_model_slope_median": per_model_slope,
        "rhat_max": dec.rhat_max, "ess_min": dec.ess_min,
        "converged": dec.converged,
        "trajectories": trajs,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ---- Markdown report ----
    L = ["# Calibration Analysis — FEH H1 effect size & σ_g (real local-model data)\n"]
    L.append("> Pre-data feasibility estimate, **not** the confirmatory test. "
             "Regime = item category (K = high, R/A = low). "
             "See `Calibration_protocol.md`.\n")
    L.append(f"**Data**: {len(df)} scored responses — {n_hi} high-regime, {n_lo} low-regime — "
             f"across {n_models} models ({', '.join(payload['models'])}).\n")
    L.append(f"**Convergence**: R̂_max = {dec.rhat_max:.3f}, ESS_min = {dec.ess_min:.0f} "
             f"→ {'OK' if dec.converged else '⚠️ NOT converged'}.\n")

    L.append("## Headline numbers\n")
    L.append("| quantity | value |")
    L.append("|---|---|")
    L.append(f"| Pr(β3 < 0) | **{dec.p_b3_neg:.3f}** |")
    L.append(f"| β3 median (95% CI) | {dec.b3_median:+.3f} {fmt_ci(dec.b3_ci)} |")
    L.append(f"| β1 median (95% CI) | {dec.b1_median:+.3f} {fmt_ci(dec.b1_ci)} |")
    L.append(f"| low-regime slope (median β1) | {dec.slope_low_median:+.3f} |")
    L.append(f"| high-regime slope (median β1+β3) | {dec.slope_high_median:+.3f} |")
    L.append(f"| **robust implied high-regime pp-drop** | **{amended['pp_drop_robust_median']:+.1f} pp** "
             f"[{amended['pp_drop_robust_ci'][0]:+.1f}, {amended['pp_drop_robust_ci'][1]:+.1f}] |")
    L.append(f"| registered min→max pp-drop | {dec.pp_drop_high_median:+.1f} pp |")
    L.append(f"| **σ_g (model-slope SD)** | **{sg_med:.3f}** [{sg_ci[0]:.3f}, {sg_ci[1]:.3f}] |"
             if sg_med is not None else "| σ_g | n/a (single model) |")
    L.append(f"| calibration verdict (registered rule) | {dec.verdict} |")
    L.append(f"| calibration verdict (amended rule) | {amended['verdict']} |")
    L.append("")
    L.append(f"*Verdicts are descriptive here — the registered gate applies to the full panel, "
             f"not this 3-model 7–8 B calibration.*\n")

    L.append("## cbd-correctness by CoT condition (the H1 signature)\n")
    L.append("*H1 predicts high-regime cbd-correctness falls C1→C5; low-regime accuracy should not.*\n")
    L.append("| condition | high-regime cbd-correct | low-regime accuracy |")
    L.append("|---|---|---|")
    for c in CONDS:
        h = hi_by_cond.get(c)
        lo_v = lo_by_cond.get(c)
        L.append(f"| {c} | {h if h is not None else '-'} | {lo_v if lo_v is not None else '-'} |")
    L.append("")

    L.append("## Per-model high-regime step slope (median g_m)\n")
    if per_model_slope:
        L.append("| model | median slope contribution g_m |")
        L.append("|---|---|")
        for m, s in per_model_slope.items():
            L.append(f"| {m} | {s:+.3f} |")
    L.append("")

    L.append("## Effect heterogeneity across the 28 high-regime items\n")
    L.append(f"- **{n_drop}** items show the predicted cbd-collapse (C5 < C1 − 0.10)")
    L.append(f"- **{n_floor}** items floored (mean cbd-correct < 0.10 — model never recognized cbd)")
    L.append(f"- **{n_ceil}** items ceilinged (mean cbd-correct > 0.90 — robustly recognized cbd)")
    L.append("")
    L.append("| item | C1 | C2 | C3 | C4 | C5 | mean | direction |")
    L.append("|---|---|---|---|---|---|---|---|")
    for fid, v in trajs.items():
        t = v["traj"]
        cells_str = " | ".join(f"{t[c]:.2f}" if t[c] is not None else "-" for c in CONDS)
        L.append(f"| {fid} | {cells_str} | {v['mean_cbd']:.2f} | {v['direction']} |")
    L.append("")

    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[done] {OUT_JSON.name} + {OUT_MD.name}")
    print(f"  Pr(b3<0)={dec.p_b3_neg:.3f}  robust pp-drop={amended['pp_drop_robust_median']:+.1f}  "
          f"sigma_g={sg_med if sg_med is None else round(sg_med,3)}  "
          f"verdict(amended)={amended['verdict']}")
    print(f"  heterogeneity: {n_drop} drop / {n_floor} floor / {n_ceil} ceiling of {len(trajs)} items")


if __name__ == "__main__":
    fast = "--fast" in sys.argv[1:]
    kw = dict(draws=500, tune=500, chains=2) if fast else {}
    main(sample_kw=kw)
