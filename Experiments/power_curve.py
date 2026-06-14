"""
power_curve.py — Set the amended-gate effect-size floor and document operating
characteristics, per the design decision following analysis_validation.md.

For a grid of true interaction sizes (b3), simulate datasets at the full design,
fit the pre-registered model, and evaluate BOTH decision rules on each fit:

  - ORIGINAL gate  (pre-reg §6.2/§7.1): Pr(b3<0 AND |b3|>|b1|)>0.95 AND
                    pp_drop>10 AND low-slope>=0
  - AMENDED gate   (proposed)         : Pr(b3<0)>0.95 AND robust pp-drop > FLOOR,
                    evaluated at several candidate FLOORs (post-hoc on the same
                    posterior — no extra sampling).

Reports, per effect size: confirmation rate (= power; or Type I at b3=0) for the
original gate and each amended floor. Recommends the smallest floor with
Type I <= 0.05 and power >= 0.80 at the target effect (b3 = -0.25).

Usage:
  python power_curve.py                 # default grid, 30 reps
  python power_curve.py --reps 40
  python power_curve.py --fast          # quick check (small grid, few reps)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import confirmatory_analysis as ca       # noqa: E402
import validate_analysis as va           # noqa: E402

OUT_JSON = ca.REPO_ROOT / "Experiments" / "power_curve.json"
OUT_MD = ca.REPO_ROOT / "Experiments" / "power_curve.md"

# Fixed generating params; only b3 varies across the grid.
BASE_PARAMS = dict(b0=0.40, b1=0.10, b2=-0.40,
                   sigma_a=0.5, sigma_g=0.3, sigma_u=0.5)
TARGET_B3 = -0.25          # the pre-registered ~20pp target effect
FLOORS = [6.0, 8.0, 10.0]  # candidate amended-gate magnitude floors (pp)


def true_pp_drop_robust(params: dict, seed: int = 999) -> float:
    """The robust pp-drop computed from TRUE params on one large simulated draw
    (so we know what effect each grid point actually represents)."""
    rng = np.random.default_rng(seed)
    df = va.simulate(params, rng)
    df = ca.standardize_steps_within_model(df)
    hi = df[df["regime"] == 1]
    base = ca._logit(float(hi["y"].mean()))
    s_ref = float(hi["steps_z"].mean())
    s_lo = float(np.percentile(hi["steps_z"], 10))
    s_hi = float(np.percentile(hi["steps_z"], 90))
    slope = params["b1"] + params["b3"]
    p_lo = ca._sigma(base + slope * (s_lo - s_ref))
    p_hi = ca._sigma(base + slope * (s_hi - s_ref))
    return float((p_lo - p_hi) * 100.0)


def run_grid(b3_grid, reps, sample_kw, floors, seed0=5000) -> dict:
    results = {}
    for gi, b3 in enumerate(b3_grid):
        params = dict(BASE_PARAMS, b3=b3)
        true_drop = true_pp_drop_robust(params)
        orig_confirm = 0
        amended = {f: dict(confirm=0, falsify=0, inconcl=0) for f in floors}
        p_dirs, p_h1s, ppd_robusts = [], [], []
        for r in range(reps):
            rng = np.random.default_rng(seed0 + gi * 10_000 + r)
            df = va.simulate(params, rng)
            try:
                dec, idata, data = ca.run(df, **sample_kw)
            except Exception as e:           # skip a rare bad fit, keep going
                print(f"[skip] b3={b3:+.2f} rep {r+1}: {e}")
                continue
            orig_confirm += dec.verdict == "confirmed"
            p_h1s.append(dec.p_h1)
            verdict_key = {"confirmed": "confirm", "falsified": "falsify",
                           "inconclusive": "inconcl"}
            for f in floors:
                a = ca.decide_amended(idata, data, mag_floor_pp=f)
                amended[f][verdict_key[a["verdict"]]] += 1
                if f == floors[0]:
                    p_dirs.append(a["p_dir"])
                    ppd_robusts.append(a["pp_drop_robust_median"])
            print(f"[grid] b3={b3:+.2f} rep {r+1:2d}/{reps} "
                  f"orig={dec.verdict[:5]} p_dir={p_dirs[-1]:.3f} "
                  f"ppd_rob={ppd_robusts[-1]:+.1f}")
        n = len(p_h1s)
        results[f"{b3:+.2f}"] = dict(
            b3=b3, true_pp_drop_robust=round(true_drop, 1), n=n,
            orig_confirm_rate=orig_confirm / n if n else None,
            amended={str(f): dict(confirm_rate=amended[f]["confirm"] / n,
                                  falsify_rate=amended[f]["falsify"] / n,
                                  inconcl_rate=amended[f]["inconcl"] / n)
                     for f in floors} if n else {},
            mean_p_dir=float(np.mean(p_dirs)) if p_dirs else None,
            mean_p_h1=float(np.mean(p_h1s)) if p_h1s else None,
            mean_pp_drop_robust=float(np.mean(ppd_robusts)) if ppd_robusts else None,
        )
    return results


def recommend(results: dict, floors) -> dict:
    """Smallest floor with Type I (b3=0) <= 0.05 AND power (b3=target) >= 0.80."""
    null_key = "+0.00"
    tgt_key = f"{TARGET_B3:+.2f}"
    rec = dict(floor=None, power=None, type1=None, reason="")
    if null_key not in results or tgt_key not in results:
        rec["reason"] = "null or target grid point missing"
        return rec
    for f in sorted(floors):
        t1 = results[null_key]["amended"][str(f)]["confirm_rate"]
        pw = results[tgt_key]["amended"][str(f)]["confirm_rate"]
        if t1 <= 0.05 and pw >= 0.80:
            rec.update(floor=f, power=pw, type1=t1,
                       reason="smallest floor meeting Type I<=0.05 and power>=0.80")
            return rec
    # fallback: best power among floors with Type I<=0.05
    best = None
    for f in sorted(floors):
        t1 = results[null_key]["amended"][str(f)]["confirm_rate"]
        pw = results[tgt_key]["amended"][str(f)]["confirm_rate"]
        if t1 <= 0.05 and (best is None or pw > best[1]):
            best = (f, pw, t1)
    if best:
        rec.update(floor=best[0], power=best[1], type1=best[2],
                   reason="no floor hit 0.80 power; best Type-I-valid floor shown")
    else:
        rec["reason"] = "no floor controlled Type I at 0.05"
    return rec


def write_report(payload: dict) -> None:
    res = payload["results"]
    floors = payload["floors"]
    rec = payload["recommendation"]
    L = ["# Power Curve — Amended Confirmatory Gate\n"]
    L.append("Compares the pre-registered ORIGINAL gate "
             "(Pr(b3<0 ∧ |b3|>|b1|)>0.95 ∧ pp-drop>10 ∧ low-slope≥0) against the "
             "proposed AMENDED gate (Pr(b3<0)>0.95 ∧ robust pp-drop>FLOOR) over a "
             "grid of true interaction sizes. "
             f"{payload['reps']} reps/point at the full design "
             f"(5×40×5×3=3000 obs). Sampler: {payload['sample_kw']}.\n")
    L.append(f"**Generated**: {payload['timestamp']}\n")

    L.append("## Confirmation rate by true effect size\n")
    hdr = "| true b3 | true pp-drop | orig gate | " + \
          " | ".join(f"amended f={f:.0f}" for f in floors) + " | mean Pr(b3<0) |"
    L.append(hdr)
    L.append("|" + "---|" * (3 + len(floors) + 1))
    for k, v in res.items():
        row = (f"| {k} | {v['true_pp_drop_robust']:+.1f} | "
               f"{v['orig_confirm_rate']:.2f} | "
               + " | ".join(f"{v['amended'][str(f)]['confirm_rate']:.2f}"
                            for f in floors)
               + f" | {v['mean_p_dir']:.3f} |")
        L.append(row)
    L.append("\n*Row b3=+0.00 is the null → confirmation rate there is the "
             "Type I error rate. The target effect is b3=-0.25.*\n")

    L.append("## Recommendation\n")
    if rec["floor"] is not None:
        L.append(f"**Adopt magnitude floor = {rec['floor']:.0f} pp.** "
                 f"At this floor the amended gate has **power = {rec['power']:.2f}** "
                 f"at the target effect (b3={TARGET_B3}) and **Type I = "
                 f"{rec['type1']:.2f}** under the null. ({rec['reason']}.)\n")
        L.append("Compare: the original full-reversal gate has power "
                 f"{res[f'{TARGET_B3:+.2f}']['orig_confirm_rate']:.2f} at the same "
                 "target effect.\n")
    else:
        L.append(f"**No candidate floor met the criteria** ({rec['reason']}). "
                 "Consider re-powering (more reps/items) or a different estimand.\n")

    L.append("## Proposed amended primary test (for OSF addendum)\n")
    L.append("> **H1 (amended).** The interaction coefficient β3 in eq. 6.1 is "
             "negative: more reasoning degrades accuracy in the high-regime bin "
             "relative to the low-regime bin.\n>\n"
             "> **Confirmation:** Pr(β3 < 0 | data) > 0.95 AND the posterior "
             f"median robust implied accuracy drop exceeds {rec['floor'] or 'F'} "
             "percentage points (high-regime, 10th→90th percentile step range, "
             "anchored at the empirical high-regime base rate).\n>\n"
             "> **Reported as effect sizes (no longer gating):** the full-reversal "
             "probability Pr(β3<0 ∧ |β3|>|β1|), the within-bin slopes, and the "
             "low-regime slope sign.\n>\n"
             "> **Falsification:** Pr(β3 ≥ 0 | data) > 0.95, or Pr(β3<0)>0.95 with "
             "robust implied drop < 3 pp (directional but negligible).\n")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[done] wrote {OUT_MD.name} and {OUT_JSON.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    if args.fast:
        b3_grid = [0.00, TARGET_B3]
        reps = 6
        sample_kw = dict(draws=400, tune=400, chains=2)
    else:
        b3_grid = [0.00, -0.15, TARGET_B3, -0.40]
        reps = args.reps
        sample_kw = dict(draws=600, tune=600, chains=2)

    t0 = time.time()
    results = run_grid(b3_grid, reps, sample_kw, FLOORS)
    payload = dict(
        timestamp=time.strftime("%Y-%m-%d %H:%M"),
        reps=reps, sample_kw=sample_kw, floors=FLOORS,
        target_b3=TARGET_B3, results=results,
        recommendation=recommend(results, FLOORS),
        elapsed_min=round((time.time() - t0) / 60, 1),
    )
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str),
                        encoding="utf-8")
    write_report(payload)
    rec = payload["recommendation"]
    print(f"[recommend] floor={rec['floor']} power={rec['power']} "
          f"type1={rec['type1']} ({payload['elapsed_min']} min)")


if __name__ == "__main__":
    main()
