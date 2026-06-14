"""
confirmatory_analysis.py — The pre-registered confirmatory analysis pipeline
for the FEH full empirical study.

Implements EXACTLY the locked model of Pre-registration §6.1 (eq. 6.1):

    y_ij ~ Bernoulli(sigma(eta_ij))
    eta_ij = b0 + b1*steps_ij + b2*regime_i + b3*(steps_ij * regime_i)
             + a_m(ij) + g_m(ij)*steps_ij + u_i

  Priors (§6.1):
    b0, b2          ~ Normal(0, 2.5)        # logit scale, weakly informative
    b1, b3          ~ Normal(0, 1)          # standardized step scale
    sigma_a/g/u     ~ HalfNormal(1)         # random-effect SDs
    a_m (model intercept), g_m (model slope), u_i (item intercept) are the REs.

  steps are STANDARDIZED WITHIN MODEL (z-score per model), so b1/b3 live on the
  standardized step scale the priors assume.

  Decision rule:
    §6.2 primary statistic  : Pr(b3 < 0 AND |b3| > |b1| | data)   [confirm > 0.95]
    §7.1 confirmation (all)  : (1) above > 0.95
                               (2) implied high-regime pp-drop (min->max steps) > 10
                               (3) low-regime slope non-negative (median b1 >= 0)
    §7.2 falsification       : Pr(b3 >= 0 | data) > 0.95, OR
                               Pr(b3 < 0) > 0.95 but implied pp-drop < 5
    §7.3 inconclusive        : anything else

Implementation note: random effects use the non-centered parameterization
(a_m = sigma_a * z_a, etc.). This is a sampler-geometry choice that targets the
identical posterior as the centered form in §6.1 — it is not a model change. It
removes the funnel-induced divergences seen under the centered parameterization.

This module is the pre-registered analysis. It is exercised before data
collection by `validate_analysis.py` (synthetic parameter recovery + power) and
can be smoke-tested on the pilot via `load_pilot()`.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Pre-registered sampler settings (§6.1).
PREREG_SAMPLE = dict(draws=2000, tune=2000, chains=4, target_accept=0.95)
# Pre-registered convergence gates (§6.1).
RHAT_MAX = 1.01
ESS_MIN = 400.0
# Decision thresholds (§6.2, §7.1, §7.2).
CONFIRM_PROB = 0.95
PP_DROP_CONFIRM = 10.0   # percentage points
PP_DROP_NEGLIGIBLE = 5.0


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def standardize_steps_within_model(df: pd.DataFrame) -> pd.DataFrame:
    """z-score `steps` within each model (the scale b1/b3 priors assume).

    A model whose steps have zero variance (e.g. only C1 present) gets z=0.
    """
    df = df.copy()

    def _z(g: pd.Series) -> pd.Series:
        sd = g.std(ddof=0)
        return (g - g.mean()) / sd if sd > 0 else g * 0.0

    df["steps_z"] = df.groupby("model")["steps"].transform(_z)
    return df


def prepare(df: pd.DataFrame) -> dict:
    """Factorize model/item labels to contiguous integer codes.

    Expects columns: y (0/1), steps_z, regime (0/1), model, item.
    Returns arrays + the label maps for reporting.
    """
    required = {"y", "steps_z", "regime", "model", "item"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"prepare(): dataframe missing columns {missing}")
    model_codes, model_labels = pd.factorize(df["model"], sort=True)
    item_codes, item_labels = pd.factorize(df["item"], sort=True)
    return {
        "y": df["y"].to_numpy().astype("int8"),
        "steps_z": df["steps_z"].to_numpy().astype("float64"),
        "regime": df["regime"].to_numpy().astype("float64"),
        "model_idx": model_codes.astype("int32"),
        "item_idx": item_codes.astype("int32"),
        "model_labels": list(model_labels),
        "item_labels": list(item_labels),
        "n_models": len(model_labels),
        "n_items": len(item_labels),
        "df": df,
    }


# --------------------------------------------------------------------------- #
# Model (eq. 6.1)
# --------------------------------------------------------------------------- #
def build_model(data: dict, model_effects: bool | None = None):
    """Build the PyMC model implementing eq. 6.1.

    `model_effects`: include a_m / g_m model-level random effects. Auto-disabled
    when only one model is present (variance unidentifiable from a single group)
    — used for the pilot smoke test. Defaults to (n_models > 1).
    """
    import pymc as pm

    if model_effects is None:
        model_effects = data["n_models"] > 1

    y = data["y"]
    steps = data["steps_z"]
    regime = data["regime"]
    model_idx = data["model_idx"]
    item_idx = data["item_idx"]
    M, I = data["n_models"], data["n_items"]

    with pm.Model() as model:
        b0 = pm.Normal("b0", 0.0, 2.5)
        b2 = pm.Normal("b2", 0.0, 2.5)
        b1 = pm.Normal("b1", 0.0, 1.0)
        b3 = pm.Normal("b3", 0.0, 1.0)

        eta = b0 + b1 * steps + b2 * regime + b3 * (steps * regime)

        if model_effects and M > 1:
            sigma_a = pm.HalfNormal("sigma_a", 1.0)
            sigma_g = pm.HalfNormal("sigma_g", 1.0)
            z_a = pm.Normal("z_a", 0.0, 1.0, shape=M)
            z_g = pm.Normal("z_g", 0.0, 1.0, shape=M)
            a_m = pm.Deterministic("a_m", sigma_a * z_a)
            g_m = pm.Deterministic("g_m", sigma_g * z_g)
            eta = eta + a_m[model_idx] + g_m[model_idx] * steps

        if I > 1:
            sigma_u = pm.HalfNormal("sigma_u", 1.0)
            z_u = pm.Normal("z_u", 0.0, 1.0, shape=I)
            u_i = pm.Deterministic("u_i", sigma_u * z_u)
            eta = eta + u_i[item_idx]

        pm.Bernoulli("y_obs", logit_p=eta, observed=y)
    return model


def fit(model, *, draws=None, tune=None, chains=None, target_accept=None,
        cores=1, random_seed=20260514, progressbar=False):
    """Sample the model with the pre-registered settings (overridable for fast
    validation runs). cores=1 by default for Windows multiprocessing safety."""
    import pymc as pm

    kw = dict(PREREG_SAMPLE)
    if draws is not None:
        kw["draws"] = draws
    if tune is not None:
        kw["tune"] = tune
    if chains is not None:
        kw["chains"] = chains
    if target_accept is not None:
        kw["target_accept"] = target_accept
    with model:
        idata = pm.sample(
            cores=cores, random_seed=random_seed, progressbar=progressbar,
            idata_kwargs={"log_likelihood": False}, **kw,
        )
    return idata


# --------------------------------------------------------------------------- #
# Decision rule (§6.2 / §7)
# --------------------------------------------------------------------------- #
def _flat(idata, name) -> np.ndarray:
    return idata.posterior[name].to_numpy().reshape(-1)


def _sigma(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Decision:
    p_h1: float                 # Pr(b3<0 AND |b3|>|b1|)  — §6.2 primary statistic
    p_b3_neg: float             # Pr(b3<0)
    p_b3_nonneg: float          # Pr(b3>=0)
    b1_median: float
    b1_ci: tuple
    b3_median: float
    b3_ci: tuple
    slope_low_median: float     # median(b1)              — low-regime logit slope
    slope_high_median: float    # median(b1+b3)           — high-regime logit slope
    pp_drop_high_median: float  # implied high-regime accuracy drop, min->max steps
    pp_drop_high_ci: tuple
    steps_min: float
    steps_max: float
    rhat_max: float
    ess_min: float
    converged: bool
    verdict: str                # "confirmed" | "falsified" | "inconclusive"
    reasons: list

    def to_dict(self) -> dict:
        return asdict(self)


def decide(idata, data: dict) -> Decision:
    """Apply the pre-registered decision rule to a fitted model."""
    import arviz as az

    b1 = _flat(idata, "b1")
    b3 = _flat(idata, "b3")
    b0 = _flat(idata, "b0")
    b2 = _flat(idata, "b2")

    # §6.2 primary statistic.
    p_h1 = float(np.mean((b3 < 0) & (np.abs(b3) > np.abs(b1))))
    p_b3_neg = float(np.mean(b3 < 0))
    p_b3_nonneg = float(np.mean(b3 >= 0))

    b1_med = float(np.median(b1))
    b3_med = float(np.median(b3))
    b1_ci = (float(np.percentile(b1, 2.5)), float(np.percentile(b1, 97.5)))
    b3_ci = (float(np.percentile(b3, 2.5)), float(np.percentile(b3, 97.5)))

    # Implied high-regime accuracy drop, population (fixed-effect) prediction at
    # regime=1, item RE = 0, model RE = 0 (§6.3). Over the observed standardized
    # step range in the high-regime bin.
    hi = data["df"][data["df"]["regime"] == 1]
    s_min = float(hi["steps_z"].min()) if len(hi) else float(data["steps_z"].min())
    s_max = float(hi["steps_z"].max()) if len(hi) else float(data["steps_z"].max())
    # posterior of predicted accuracy at min/max steps in high regime
    p_at_min = _sigma(b0 + b2 + (b1 + b3) * s_min)
    p_at_max = _sigma(b0 + b2 + (b1 + b3) * s_max)
    pp_drop = (p_at_min - p_at_max) * 100.0  # positive = accuracy falls as steps rise
    pp_drop_med = float(np.median(pp_drop))
    pp_drop_ci = (float(np.percentile(pp_drop, 2.5)),
                  float(np.percentile(pp_drop, 97.5)))

    # Convergence diagnostics on the structural parameters.
    # NB: az.summary rounds R-hat/ESS to 2 dp, which causes a false-negative at the
    # 1.01 gate boundary (e.g. true 1.0056 prints as "1.01" >= RHAT_MAX). Use the
    # unrounded az.rhat / az.ess so the converged flag reflects the actual values.
    diag_vars = [v for v in ["b0", "b1", "b2", "b3",
                             "sigma_a", "sigma_g", "sigma_u"]
                 if v in idata.posterior]
    rhat_ds = az.rhat(idata, var_names=diag_vars)
    ess_ds = az.ess(idata, var_names=diag_vars)
    rhat_max = max(float(rhat_ds[v].max()) for v in diag_vars)
    ess_min = min(float(ess_ds[v].min()) for v in diag_vars)
    converged = (rhat_max < RHAT_MAX) and (ess_min > ESS_MIN)

    # Verdict (§7.1 / §7.2 / §7.3).
    reasons = []
    cond1 = p_h1 > CONFIRM_PROB
    cond2 = pp_drop_med > PP_DROP_CONFIRM
    cond3 = b1_med >= 0.0
    if cond1 and cond2 and cond3:
        verdict = "confirmed"
        reasons.append(f"Pr(H1)={p_h1:.3f}>0.95; pp_drop={pp_drop_med:.1f}>10; "
                       f"low-regime slope median={b1_med:+.3f}>=0")
    elif p_b3_nonneg > CONFIRM_PROB:
        verdict = "falsified"
        reasons.append(f"Pr(b3>=0)={p_b3_nonneg:.3f}>0.95 (interaction opposite)")
    elif p_b3_neg > CONFIRM_PROB and pp_drop_med < PP_DROP_NEGLIGIBLE:
        verdict = "falsified"
        reasons.append(f"Pr(b3<0)={p_b3_neg:.3f}>0.95 but pp_drop={pp_drop_med:.1f}"
                       f"<5 (directional but negligible)")
    else:
        verdict = "inconclusive"
        bits = []
        if not cond1:
            bits.append(f"Pr(H1)={p_h1:.3f}<=0.95")
        if not cond2:
            bits.append(f"pp_drop={pp_drop_med:.1f}<=10")
        if not cond3:
            bits.append(f"low-regime slope median={b1_med:+.3f}<0")
        reasons.append("; ".join(bits) or "no decisive criterion met")

    return Decision(
        p_h1=p_h1, p_b3_neg=p_b3_neg, p_b3_nonneg=p_b3_nonneg,
        b1_median=b1_med, b1_ci=b1_ci, b3_median=b3_med, b3_ci=b3_ci,
        slope_low_median=b1_med, slope_high_median=float(np.median(b1 + b3)),
        pp_drop_high_median=pp_drop_med, pp_drop_high_ci=pp_drop_ci,
        steps_min=s_min, steps_max=s_max,
        rhat_max=rhat_max, ess_min=ess_min, converged=converged,
        verdict=verdict, reasons=reasons,
    )


# --------------------------------------------------------------------------- #
# Amended decision rule (proposed pre-data amendment; see analysis_validation.md
# / power_curve.md). The §6.2 full-reversal gate Pr(b3<0 AND |b3|>|b1|)>0.95 has
# ~7% power at a genuine ~20pp target effect because b1/b3 are collinear and the
# global intercept is weakly identified (5 models). The amendment:
#   confirm  iff  Pr(b3<0) > 0.95  AND  median robust pp-drop > MAG_FLOOR_PP
#   demote   |b3|>|b1| (full reversal) and low-regime slope>=0 to reported
#            effect sizes (still printed, not gated).
# The robust pp-drop anchors to the EMPIRICAL high-regime base rate and the
# 10th-90th percentile step range — independent of b0/b2 and outlier-robust.
# --------------------------------------------------------------------------- #
MAG_FLOOR_PP = 6.0       # locked by the power curve (power_curve.py): dominates
                         # floors 8/10 in power at equal (zero) Type I error.
NEGLIGIBLE_PP_AMENDED = 3.0


def _logit(p: float) -> float:
    p = min(max(p, 0.02), 0.98)
    return float(np.log(p / (1.0 - p)))


def implied_pp_drop_robust(b1, b3, base_logit, s_lo, s_hi, s_ref) -> np.ndarray:
    """Posterior of the implied high-regime accuracy drop from the s_lo to the
    s_hi standardized step level, anchored at `base_logit` (empirical base rate)
    with steps centered at s_ref. Depends only on the (b1+b3) posterior."""
    slope = b1 + b3
    p_lo = _sigma(base_logit + slope * (s_lo - s_ref))
    p_hi = _sigma(base_logit + slope * (s_hi - s_ref))
    return (p_lo - p_hi) * 100.0


def decide_amended(idata, data: dict, *, mag_floor_pp: float = MAG_FLOOR_PP,
                   negligible_pp: float = NEGLIGIBLE_PP_AMENDED,
                   step_pct=(10, 90)) -> dict:
    """Apply the proposed amended decision rule. Returns a plain dict (the
    pre-registered `decide()` remains the registered rule and is untouched)."""
    b1 = _flat(idata, "b1")
    b3 = _flat(idata, "b3")

    p_dir = float(np.mean(b3 < 0))                          # gated
    p_b3_nonneg = float(np.mean(b3 >= 0))
    p_full_reversal = float(np.mean((b3 < 0) & (np.abs(b3) > np.abs(b1))))  # reported

    hi = data["df"][data["df"]["regime"] == 1]
    base = _logit(float(hi["y"].mean()) if len(hi) else 0.5)
    s_ref = float(hi["steps_z"].mean()) if len(hi) else 0.0
    s_lo = float(np.percentile(hi["steps_z"], step_pct[0])) if len(hi) else -1.0
    s_hi = float(np.percentile(hi["steps_z"], step_pct[1])) if len(hi) else 1.0

    ppd = implied_pp_drop_robust(b1, b3, base, s_lo, s_hi, s_ref)
    ppd_med = float(np.median(ppd))
    ppd_ci = (float(np.percentile(ppd, 2.5)), float(np.percentile(ppd, 97.5)))

    if p_dir > CONFIRM_PROB and ppd_med > mag_floor_pp:
        verdict = "confirmed"
    elif p_b3_nonneg > CONFIRM_PROB:
        verdict = "falsified"
    elif p_dir > CONFIRM_PROB and ppd_med < negligible_pp:
        verdict = "falsified"        # directional but negligible
    else:
        verdict = "inconclusive"

    return dict(
        verdict=verdict, p_dir=p_dir, p_b3_nonneg=p_b3_nonneg,
        p_full_reversal=p_full_reversal,
        pp_drop_robust_median=ppd_med, pp_drop_robust_ci=ppd_ci,
        b1_median=float(np.median(b1)), b3_median=float(np.median(b3)),
        slope_high_median=float(np.median(b1 + b3)),
        mag_floor_pp=mag_floor_pp,
    )


def run(df: pd.DataFrame, *, model_effects=None, **sample_kw) -> tuple:
    """End-to-end: standardize -> prepare -> build -> fit -> decide."""
    df = standardize_steps_within_model(df)
    data = prepare(df)
    model = build_model(data, model_effects=model_effects)
    idata = fit(model, **sample_kw)
    return decide(idata, data), idata, data


# --------------------------------------------------------------------------- #
# Pilot-data loader (smoke test only — see module docstring caveats)
# --------------------------------------------------------------------------- #
def load_pilot(responses_path: Path | None = None) -> pd.DataFrame:
    """Build a tidy dataframe from pilot_responses.json for an ingestion smoke
    test. Only R-001 and A-003 carry a binary gold answer, so the DV is sparse;
    the pilot cannot validate the science (1 model, ~2 scorable items). It only
    proves the pipeline ingests the real data schema.
    """
    from pilot_analysis import GOLD_ANSWERS, is_correct, numeric_normalize

    responses_path = responses_path or (REPO_ROOT / "Experiments" /
                                        "pilot_responses.json")
    cells = json.loads(Path(responses_path).read_text(encoding="utf-8"))
    rows = []
    for c in cells:
        fid = c["frame_id"]
        if fid not in GOLD_ANSWERS:        # only gold-scorable items get a y
            continue
        ans = c.get("extracted_final_answer") or ""
        norm = numeric_normalize(ans) or ans
        y = int(is_correct(norm, GOLD_ANSWERS[fid]))
        rows.append({
            "y": y,
            "steps": c["n_steps_heuristic"],
            "regime": 1 if fid.startswith(("K", "A")) else 0,  # crude pilot proxy
            "model": c["model"],
            "item": fid,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Smoke test: ingest pilot data, confirm the pipeline runs end-to-end.
    df = load_pilot()
    print(f"[pilot] loaded {len(df)} scorable rows, "
          f"{df['model'].nunique()} model(s), {df['item'].nunique()} item(s)")
    dec, idata, data = run(df, draws=500, tune=500, chains=2)
    print(f"[pilot] verdict={dec.verdict}  Pr(H1)={dec.p_h1:.3f}  "
          f"converged={dec.converged} (rhat<={dec.rhat_max:.3f}, "
          f"ess>={dec.ess_min:.0f})")
    print("[pilot] NOTE: pilot is a schema smoke test only, not a science test.")
