"""
frontier_analyze.py — score the frontier read and test whether H1 survives OFF
the small-model floor.

Reuses the EXACT calibration scoring (is_cbd / gold_match / pool v0.3) so the
frontier numbers are directly comparable to calibration_analysis.md.

Reports, per model and pooled:
  - high-regime cbd-correctness by condition C1..C4
  - low-regime accuracy by condition (control: should be flat)
  - condition-ITT contrast: assigned-SHORT (C1) vs assigned-LONG (C2-C4)
  - the H1 difference-in-differences: high(short->long drop) - low(short->long drop)
  - item-clustered bootstrap 95% CI on the high-regime short->long drop and the DiD
  - realized steps by condition per model (C1-compliance check: do frontier
    models actually deliberate less under C1?)
  - floor / ceiling item counts (vs the local panel's heavy flooring)

Descriptive + bootstrap only (no PyMC); this is the feasibility read, not the
confirmatory test.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Windows console is cp1252 and crashes on the U+2212 minus sign in the report.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# gold_match is now LaTeX-robust (numeric_candidates calls delatex), so the
# frontier CoT answers (\frac{a}{b}, \( \)) are scored identically to plain
# decimals — no local normalization needed here.
from calibration_analysis import is_cbd, gold_match, load_pool  # noqa: E402

RESP = HERE / "frontier_read_responses.json"
OUT_MD = HERE / "frontier_read_analysis.md"
OUT_JSON = HERE / "frontier_read_analysis.json"
CONDS = ["C1", "C2", "C3", "C4"]
LONG = ["C2", "C3", "C4"]
RNG = np.random.default_rng(20260530)
N_BOOT = 5000


def score(cells, pool) -> pd.DataFrame:
    rows = []
    for c in cells:
        if c.get("error") or not c.get("raw_response"):
            continue
        fid = c["frame_id"]
        ans = c.get("extracted_final_answer") or ""
        if c["regime"] == "high":
            y = int(is_cbd(ans))
        else:
            y = int(gold_match(ans, pool.get(fid, {}).get("gold_answer")))
        rows.append({
            "y": y, "steps": c["n_steps_heuristic"], "tokens_out": c.get("tokens_out") or 0,
            "regime": c["regime"], "model": c["model"], "item": fid, "condition": c["condition"],
        })
    return pd.DataFrame(rows)


def cond_means(sub) -> dict:
    return {cn: (round(float(sub[sub.condition == cn].y.mean()), 3)
                 if len(sub[sub.condition == cn]) else None) for cn in CONDS}


def short_long(sub):
    """(short rate C1, long rate mean over C2-C4 cells, drop = short - long)."""
    s = sub[sub.condition == "C1"].y
    lo = sub[sub.condition.isin(LONG)].y
    if not len(s) or not len(lo):
        return None, None, None
    sr, lr = float(s.mean()), float(lo.mean())
    return sr, lr, sr - lr


def boot_drop(sub):
    """Item-clustered bootstrap of the high short->long drop (pp)."""
    items = sub.item.unique()
    if len(items) < 2:
        return None, None
    by = {it: sub[sub.item == it] for it in items}
    ds = []
    for _ in range(N_BOOT):
        pick = RNG.choice(items, size=len(items), replace=True)
        d = pd.concat([by[it] for it in pick])
        _, _, drop = short_long(d)
        if drop is not None:
            ds.append(drop)
    a = np.array(ds)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def did_boot(hi, lo):
    """Difference-in-differences: high-drop - low-drop, item-clustered bootstrap
    (resample high and low items independently)."""
    hi_items, lo_items = hi.item.unique(), lo.item.unique()
    hby = {it: hi[hi.item == it] for it in hi_items}
    lby = {it: lo[lo.item == it] for it in lo_items}
    vals = []
    for _ in range(N_BOOT):
        hp = pd.concat([hby[it] for it in RNG.choice(hi_items, len(hi_items), True)])
        lp = pd.concat([lby[it] for it in RNG.choice(lo_items, len(lo_items), True)])
        _, _, hd = short_long(hp)
        _, _, ld = short_long(lp)
        if hd is not None and ld is not None:
            vals.append(hd - ld)
    a = np.array(vals)
    return float(a.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def steps_by_cond(sub) -> dict:
    return {cn: round(float(sub[sub.condition == cn].steps.mean()), 1) for cn in CONDS}


def main():
    cells = json.loads(RESP.read_text(encoding="utf-8"))
    pool = load_pool()
    df = score(cells, pool)
    models = sorted(df.model.unique())

    L = ["# Frontier feasibility read — H1 off the small-model floor\n"]
    L.append(f"Scored {len(df)} cells ({(df.regime=='high').sum()} high, "
             f"{(df.regime=='low').sum()} low) from {len(models)} frontier models, "
             f"conditions C1–C4, 2 reps. Scoring identical to calibration "
             f"(is_cbd / gold_match, pool v0.3).\n")
    L.append("**H1 prediction:** high-regime cbd-correctness falls from assigned-SHORT "
             "(C1) to assigned-LONG (C2–C4); low-regime accuracy stays flat. The "
             "difference-in-differences (high drop − low drop) is the H1 effect; "
             "it should be **> 0**.\n")

    out = {"n_cells": len(df), "models": {}, "pooled": {}}

    # ---- per model ----
    for m in models:
        md = df[df.model == m]
        hi, lo = md[md.regime == "high"], md[md.regime == "low"]
        hc, lc = cond_means(hi), cond_means(lo)
        hs, hl, hd = short_long(hi)
        ls, ll, ld = short_long(lo)
        hlo_ci, hhi_ci = boot_drop(hi)
        did, did_lo, did_hi = did_boot(hi, lo)
        # floor/ceiling per high item (mean over conds/reps)
        per_item = hi.groupby("item").y.mean()
        n_floor = int((per_item < 0.10).sum())
        n_ceil = int((per_item > 0.90).sum())
        hsteps = steps_by_cond(hi)

        L.append(f"\n## {m}\n")
        L.append("| regime | C1 | C2 | C3 | C4 | short(C1) | long(C2–4) | short→long drop |")
        L.append("|---|---|---|---|---|---|---|---|")
        L.append(f"| high (cbd-correct) | {hc['C1']} | {hc['C2']} | {hc['C3']} | {hc['C4']} "
                 f"| {hs:.3f} | {hl:.3f} | **{hd:+.3f}** [{hlo_ci:+.3f}, {hhi_ci:+.3f}] |")
        L.append(f"| low (accuracy) | {lc['C1']} | {lc['C2']} | {lc['C3']} | {lc['C4']} "
                 f"| {ls:.3f} | {ll:.3f} | {ld:+.3f} |")
        L.append(f"\n- **H1 difference-in-differences (high drop − low drop): "
                 f"{did:+.3f} [{did_lo:+.3f}, {did_hi:+.3f}]** "
                 f"({'SUPPORTS H1' if did_lo > 0 else 'CI includes 0' if did_hi > 0 else 'against H1'})")
        L.append(f"- floor items (mean cbd<0.10): {n_floor}/28 ; ceiling (>0.90): {n_ceil}/28")
        L.append(f"- realized steps by condition (C1 compliance): "
                 f"C1={hsteps['C1']} C2={hsteps['C2']} C3={hsteps['C3']} C4={hsteps['C4']}")

        out["models"][m] = {
            "high_by_cond": hc, "low_by_cond": lc,
            "high_short": hs, "high_long": hl, "high_drop": hd,
            "high_drop_ci": [hlo_ci, hhi_ci],
            "low_short": ls, "low_long": ll, "low_drop": ld,
            "did": did, "did_ci": [did_lo, did_hi],
            "n_floor": n_floor, "n_ceil": n_ceil, "high_steps": hsteps,
        }

    # ---- pooled ----
    hi, lo = df[df.regime == "high"], df[df.regime == "low"]
    hs, hl, hd = short_long(hi)
    ls, ll, ld = short_long(lo)
    hlo_ci, hhi_ci = boot_drop(hi)
    did, did_lo, did_hi = did_boot(hi, lo)
    L.append("\n## Pooled (both frontier models)\n")
    L.append(f"- high-regime short→long drop: **{hd:+.3f}** [{hlo_ci:+.3f}, {hhi_ci:+.3f}] "
             f"(short {hs:.3f} → long {hl:.3f})")
    L.append(f"- low-regime short→long drop: {ld:+.3f} (short {ls:.3f} → long {ll:.3f})")
    L.append(f"- **H1 difference-in-differences: {did:+.3f} [{did_lo:+.3f}, {did_hi:+.3f}]** "
             f"({'SUPPORTS H1' if did_lo > 0 else 'CI includes 0' if did_hi > 0 else 'against H1'})")
    out["pooled"] = {"high_drop": hd, "high_drop_ci": [hlo_ci, hhi_ci],
                     "low_drop": ld, "did": did, "did_ci": [did_lo, did_hi],
                     "high_short": hs, "high_long": hl}

    L.append("\n## Comparison to the local calibration\n")
    L.append("- Local panel (3× 7–8B): high short→long drop ≈ 7 pp (clean C1→C4); "
             "many K2/K4 items floored; registered realized-steps model returned the "
             "WRONG sign (the reason for Amendment 2).")
    L.append("- Frontier read: see DiD above. If the frontier DiD CI excludes 0 with "
             "fewer floored items, H1 holds off the floor → the central claim is not a "
             "small-model artifact.")

    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[wrote] {OUT_MD.name}, {OUT_JSON.name}")


if __name__ == "__main__":
    main()
