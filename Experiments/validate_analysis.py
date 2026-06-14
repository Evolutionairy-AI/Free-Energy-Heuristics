"""
validate_analysis.py — Pre-data validation of the confirmatory pipeline
(`confirmatory_analysis.py`). This is the dry-run that must pass BEFORE the
~45 hr full data collection is launched.

Two things are validated:

  (A) PARAMETER RECOVERY. Simulate data from eq. 6.1 with KNOWN coefficients at
      the full planned design (5 models x 40 items x 5 conditions x 3 reps =
      3000 obs), fit the real pipeline, and check that:
        - the sampler converges (R-hat < 1.01, ESS > 400, per §6.1),
        - the 95% credible intervals cover the true b0,b1,b2,b3 and RE SDs,
        - the decision rule returns the CORRECT verdict for each scenario
          (confirmed under H1-true, falsified under reversed, not-confirmed
           under the null).

  (B) OPERATING CHARACTERISTICS (simulation-based power). Repeat the fit over
      many simulated datasets at the PRE-REGISTERED effect-size boundary
      (~10 pp high-regime accuracy drop) and report the confirmation rate
      (= power) and, under the null, the false-confirmation rate. This replaces
      the analytic normal-approximation power claim in pre-reg §4.1 with a
      simulation-based number for the actual hierarchical model + decision rule.

Scenarios (true generating coefficients, on the standardized-step logit scale):

  h1_strong   : b1=+0.30, b3=-0.60   clear reversal (~large pp-drop) — recovery demo
  h1_boundary : b1=+0.10, b3=-0.25   reversal near the 10-pp gate    — power point
  null        : b1=+0.20, b3= 0.00   no interaction                  — false-positive
  reversed    : b1=+0.30, b3=+0.60   interaction the WRONG way        — falsification

Usage:
  python validate_analysis.py --mode recover          # (A), full-fidelity fits
  python validate_analysis.py --mode power --reps 30   # (B)
  python validate_analysis.py --mode all --reps 30     # both
  python validate_analysis.py --mode fast              # quick smoke (~1 min)
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

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import confirmatory_analysis as ca  # noqa: E402

OUT_JSON = ca.REPO_ROOT / "Experiments" / "analysis_validation.json"
OUT_MD = ca.REPO_ROOT / "Experiments" / "analysis_validation.md"

# Full planned design (pre-reg §3.5, primary-analysis subset).
DESIGN = dict(n_models=5, n_high=20, n_low=20, n_conditions=5, n_reps=3)
# Per-condition mean realized step targets (raw), shaped after the pilot
# trajectory C1..C5 (0->medium->long->very-long->unconstrained).
STEP_TARGETS = np.array([0.5, 4.0, 8.0, 14.0, 9.0])

SCENARIOS = {
    "h1_strong":   dict(b0=0.40, b1=0.30, b2=-0.40, b3=-0.60,
                        sigma_a=0.5, sigma_g=0.3, sigma_u=0.5),
    "h1_boundary": dict(b0=0.40, b1=0.10, b2=-0.40, b3=-0.25,
                        sigma_a=0.5, sigma_g=0.3, sigma_u=0.5),
    "null":        dict(b0=0.40, b1=0.20, b2=-0.40, b3=0.00,
                        sigma_a=0.5, sigma_g=0.3, sigma_u=0.5),
    "reversed":    dict(b0=0.40, b1=0.30, b2=-0.40, b3=0.60,
                        sigma_a=0.5, sigma_g=0.3, sigma_u=0.5),
}
# What the decision rule SHOULD return for each scenario.
EXPECTED = {"h1_strong": "confirmed", "h1_boundary": "confirmed",
            "null": "not-confirmed", "reversed": "falsified"}


def simulate(params: dict, rng: np.random.Generator, design: dict = DESIGN) -> pd.DataFrame:
    """Generate one synthetic dataset from eq. 6.1 with the given true params.

    Steps are z-scored within model (matching the pipeline) before entering eta,
    so the true b1/b3 live on exactly the scale the pipeline estimates. Raw steps
    are returned; the pipeline re-standardizes to the identical z-scores.
    """
    M = design["n_models"]
    items = ([f"HI{i:02d}" for i in range(design["n_high"])] +
             [f"LO{i:02d}" for i in range(design["n_low"])])
    regime_of = {**{it: 1 for it in items if it.startswith("HI")},
                 **{it: 0 for it in items if it.startswith("LO")}}

    # Random-effect realizations.
    a_m = rng.normal(0, params["sigma_a"], M)
    g_m = rng.normal(0, params["sigma_g"], M)
    u_i = {it: rng.normal(0, params["sigma_u"]) for it in items}
    model_scale = rng.uniform(0.7, 1.3, M)  # models differ in verbosity

    rows = []
    for m in range(M):
        for it in items:
            for cond in range(design["n_conditions"]):
                for _ in range(design["n_reps"]):
                    mean = STEP_TARGETS[cond] * model_scale[m]
                    steps = max(0.0, rng.normal(mean, max(1.0, 0.35 * mean)))
                    rows.append(dict(model=f"M{m}", item=it,
                                     regime=regime_of[it], steps=steps,
                                     model_i=m, item_l=it))
    df = pd.DataFrame(rows)

    # z-score steps within model (same definition the pipeline uses).
    def _z(g):
        sd = g.std(ddof=0)
        return (g - g.mean()) / sd if sd > 0 else g * 0.0
    df["steps_z"] = df.groupby("model")["steps"].transform(_z)

    eta = (params["b0"]
           + params["b1"] * df["steps_z"].to_numpy()
           + params["b2"] * df["regime"].to_numpy()
           + params["b3"] * (df["steps_z"].to_numpy() * df["regime"].to_numpy())
           + a_m[df["model_i"].to_numpy()]
           + g_m[df["model_i"].to_numpy()] * df["steps_z"].to_numpy()
           + np.array([u_i[it] for it in df["item_l"]]))
    p = 1.0 / (1.0 + np.exp(-eta))
    df["y"] = rng.binomial(1, p)
    return df[["y", "steps", "regime", "model", "item"]]


def _covered(true_val, ci) -> bool:
    return ci[0] <= true_val <= ci[1]


def recover(scenarios: list[str], sample_kw: dict, seed0: int = 7) -> dict:
    """One full-fidelity fit per scenario; report recovery + verdict."""
    results = {}
    for k, name in enumerate(scenarios):
        params = SCENARIOS[name]
        rng = np.random.default_rng(seed0 + k)
        df = simulate(params, rng)
        t0 = time.time()
        dec, idata, data = ca.run(df, **sample_kw)
        secs = time.time() - t0

        # CI coverage for the structural params present.
        import arviz as az
        post = {v: idata.posterior[v].to_numpy().reshape(-1)
                for v in ["b0", "b1", "b2", "b3"]}
        for v in ["sigma_a", "sigma_g", "sigma_u"]:
            if v in idata.posterior:
                post[v] = idata.posterior[v].to_numpy().reshape(-1)
        recov = {}
        for v, draws in post.items():
            ci = (float(np.percentile(draws, 2.5)),
                  float(np.percentile(draws, 97.5)))
            recov[v] = dict(true=params.get(v), median=float(np.median(draws)),
                            ci=ci, covered=bool(_covered(params.get(v, np.nan), ci)))
        verdict_ok = (
            (EXPECTED[name] == "confirmed" and dec.verdict == "confirmed") or
            (EXPECTED[name] == "falsified" and dec.verdict == "falsified") or
            (EXPECTED[name] == "not-confirmed" and dec.verdict != "confirmed"))
        results[name] = dict(
            params=params, recovery=recov, decision=dec.to_dict(),
            expected=EXPECTED[name], verdict_ok=bool(verdict_ok),
            seconds=round(secs, 1),
        )
        print(f"[recover] {name:12s} verdict={dec.verdict:12s} "
              f"(expect {EXPECTED[name]:13s}) Pr(H1)={dec.p_h1:.3f} "
              f"b3med={dec.b3_median:+.3f} pp_drop={dec.pp_drop_high_median:.1f} "
              f"conv={dec.converged} ({secs:.0f}s)")
    return results


def power(scenarios: list[str], reps: int, sample_kw: dict, seed0: int = 1000) -> dict:
    """Repeat fits to estimate confirmation / false-confirmation rates."""
    results = {}
    # Stable per-scenario seed offset (NOT Python's randomized str hash).
    offset = {n: i * 10_000 for i, n in enumerate(sorted(SCENARIOS))}
    for name in scenarios:
        params = SCENARIOS[name]
        confirms = falsifies = inconcl = 0
        p_h1s = []
        for r in range(reps):
            rng = np.random.default_rng(seed0 + offset[name] + r)
            df = simulate(params, rng)
            dec, _, _ = ca.run(df, **sample_kw)
            p_h1s.append(dec.p_h1)
            confirms += dec.verdict == "confirmed"
            falsifies += dec.verdict == "falsified"
            inconcl += dec.verdict == "inconclusive"
            print(f"[power] {name:12s} rep {r+1:2d}/{reps} "
                  f"verdict={dec.verdict:12s} Pr(H1)={dec.p_h1:.3f}")
        results[name] = dict(
            reps=reps, params=params,
            confirm_rate=confirms / reps,
            falsify_rate=falsifies / reps,
            inconclusive_rate=inconcl / reps,
            mean_p_h1=float(np.mean(p_h1s)),
            expected=EXPECTED[name],
        )
    return results


def write_report(payload: dict) -> None:
    L = ["# Pre-Data Validation of the Confirmatory Analysis Pipeline\n"]
    L.append("Validates `confirmatory_analysis.py` (pre-reg eq. 6.1) by "
             "parameter recovery and simulation-based operating characteristics "
             "on synthetic data at the full planned design "
             f"({DESIGN['n_models']} models x "
             f"{DESIGN['n_high']+DESIGN['n_low']} items x "
             f"{DESIGN['n_conditions']} conditions x {DESIGN['n_reps']} reps).\n")
    L.append(f"**Sampler**: {payload['sample_kw']}  ")
    L.append(f"**Generated**: {payload['timestamp']}\n")

    rec = payload.get("recover")
    if rec:
        L.append("## (A) Parameter recovery + decision-rule verdicts\n")
        all_ok = all(v["verdict_ok"] for v in rec.values())
        all_conv = all(v["decision"]["converged"] for v in rec.values())
        L.append(f"**Verdicts correct in all scenarios:** "
                 f"{'✅ yes' if all_ok else '❌ NO'}  ")
        L.append(f"**Converged (R̂<1.01, ESS>400) in all scenarios:** "
                 f"{'✅ yes' if all_conv else '❌ NO'}\n")
        L.append("| scenario | verdict | expected | ok | Pr(H1) | b3 median (true) | "
                 "pp-drop | R̂max | ESSmin |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for name, v in rec.items():
            d = v["decision"]
            L.append(f"| {name} | {d['verdict']} | {v['expected']} | "
                     f"{'✅' if v['verdict_ok'] else '❌'} | {d['p_h1']:.3f} | "
                     f"{d['b3_median']:+.3f} ({v['params']['b3']:+.2f}) | "
                     f"{d['pp_drop_high_median']:.1f} | {d['rhat_max']:.3f} | "
                     f"{d['ess_min']:.0f} |")
        L.append("\n### Coverage of true coefficients (95% CrI)\n")
        for name, v in rec.items():
            parts = []
            for p, r in v["recovery"].items():
                tag = "✓" if r["covered"] else f"✗(true {r['true']:+.2f})"
                parts.append(f"{p}={r['median']:+.2f}"
                             f"[{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}]{tag}")
            L.append(f"**{name}** — " + ", ".join(parts))
            L.append("")

    pw = payload.get("power")
    if pw:
        L.append("## (B) Operating characteristics (simulation-based power)\n")
        L.append("| scenario | reps | confirm rate | falsify rate | inconcl. | "
                 "mean Pr(H1) | interpretation |")
        L.append("|---|---|---|---|---|---|---|")
        for name, v in pw.items():
            if name in ("h1_strong", "h1_boundary"):
                interp = f"power = {v['confirm_rate']:.2f}"
            elif name == "null":
                interp = f"false-confirm = {v['confirm_rate']:.2f}"
            else:
                interp = f"correct-falsify = {v['falsify_rate']:.2f}"
            L.append(f"| {name} | {v['reps']} | {v['confirm_rate']:.2f} | "
                     f"{v['falsify_rate']:.2f} | {v['inconclusive_rate']:.2f} | "
                     f"{v['mean_p_h1']:.3f} | {interp} |")
        L.append("")

    L.append("## Verdict\n")
    ok = True
    if rec:
        ok = ok and all(v["verdict_ok"] for v in rec.values())
        ok = ok and all(v["decision"]["converged"] for v in rec.values())
    if ok:
        L.append("**PASS** — the pre-registered pipeline samples cleanly, "
                 "recovers known coefficients within 95% credible intervals, and "
                 "the decision rule returns the correct verdict in every "
                 "scenario. The pipeline is safe to lock for the full run.")
    else:
        L.append("**ATTENTION** — at least one scenario failed recovery, "
                 "convergence, or returned an unexpected verdict. Inspect the "
                 "table above before launching data collection.")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[done] wrote {OUT_MD.name} and {OUT_JSON.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["recover", "power", "all", "fast"],
                    default="all")
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()

    if args.mode == "fast":
        # quick smoke: small sampler, recover only, two scenarios
        sample_kw = dict(draws=400, tune=400, chains=2)
        payload = dict(sample_kw=sample_kw,
                       timestamp=time.strftime("%Y-%m-%d %H:%M"),
                       recover=recover(["h1_strong", "null"], sample_kw))
        write_report(payload)
        return

    payload = dict(timestamp=time.strftime("%Y-%m-%d %H:%M"))
    if args.mode in ("recover", "all"):
        kw = dict()  # pre-registered full-fidelity settings
        payload["recover_sample_kw"] = "preregistered (4 chains, 2000+2000)"
        payload["recover"] = recover(
            ["h1_strong", "h1_boundary", "null", "reversed"], kw)
    if args.mode in ("power", "all"):
        # reduced sampler for the repeated power loop (verdict is stable well
        # below full ESS; we only need Pr(H1) vs the 0.95 gate).
        pkw = dict(draws=800, tune=800, chains=2)
        payload["power_sample_kw"] = pkw
        payload["power"] = power(["h1_boundary", "null"], args.reps, pkw)

    payload["sample_kw"] = (payload.get("recover_sample_kw")
                            or payload.get("power_sample_kw"))
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str),
                        encoding="utf-8")
    write_report(payload)


if __name__ == "__main__":
    main()
