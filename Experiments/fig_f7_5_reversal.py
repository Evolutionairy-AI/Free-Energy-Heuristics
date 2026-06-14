"""
fig_f7_5_reversal.py — Figure 7.5 (Why the primary is the assigned length).

The endogeneity argument, made visual. The SAME interaction coefficient is
decisively NEGATIVE on the exogenous assigned length (the causal estimand, eq.
6.1') and decisively POSITIVE on the endogenous realized step count (the
post-treatment confound, eq. 6.1, R7). Two posteriors on one zero-centred axis
show the flip at a glance — and a basement lane shows the realized-steps
coefficient REVERTING across zero to negative once the endogeneity is corrected
(per-step IV, and paragraph-segmented steps).

NOTHING is hand-typed into the plot:
  - density CURVES are KDEs of the exported draws
        (confirmatory_posterior_draws.json: primary_b3, r7_b3)
  - annotated SCALARS come from the canonical fits
        (confirmatory_robustness.json: primary_export, r7_export, iv_r7,
         r1_paragraph_steps).
The script cross-checks draws vs canonical and PRINTS both.

Style matches fig_f7_3_posterior.py (the locked template).

Run from Experiments/:  python fig_f7_5_reversal.py
Outputs fig_f7_5_reversal.png (300 dpi) + .pdf (vector — use for the paper).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent
DRAWS = HERE / "confirmatory_posterior_draws.json"
ROBUST = HERE / "confirmatory_robustness.json"

# --------------------------------------------------------------------------- #
#  STYLE  — matches F7.3
# --------------------------------------------------------------------------- #
BG        = "#FDFDFD"
NAVY      = "#0B3A78"
NAVYFILL  = "#D9E5F5"
ORANGE    = "#E87900"
ORANGEFILL= "#FBE3CB"
ORANGEDK  = "#B86200"
NULLCLR   = "#9a9a9a"
INK       = "#0B3A78"
GRID      = "#ececec"

FIGW, FIGH = 12.0, 7.8

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["EB Garamond", "Cormorant Garamond", "Garamond",
                   "Georgia", "DejaVu Serif"],
    "font.size": 13,
    "axes.edgecolor": "#9fb0c8",
    "axes.linewidth": 1.0,
    "mathtext.fontset": "dejavuserif",
    "figure.dpi": 110,
    "savefig.dpi": 300,
})

# --------------------------------------------------------------------------- #
#  LAYOUT constants
# --------------------------------------------------------------------------- #
AX_L, AX_R, AX_T, AX_B = 0.085, 0.955, 0.770, 0.180
FRAME = (0.028, 0.050, 0.944, 0.790)
XMIN, XMAX = -1.75, 0.95
TOP_HEAD = 1.50      # headroom above tallest peak (x ymax)
BASE_DROP = 0.17     # basement depth below baseline (x ymax)


# --------------------------------------------------------------------------- #
#  DATA
# --------------------------------------------------------------------------- #
def load():
    d = json.loads(DRAWS.read_text(encoding="utf-8"))
    a = np.asarray(d["primary_b3"], dtype=float)     # assigned length (causal)
    b = np.asarray(d["r7_b3"], dtype=float)          # realized steps (confound)
    r = json.loads(ROBUST.read_text(encoding="utf-8"))
    pe, re_ = r["primary_export"], r["r7_export"]
    iv, pa = r["iv_r7"], r["r1_paragraph_steps"]
    sc = dict(
        a_med=pe["b3_median"], a_pr=pe["p_b3_neg"],
        b_med=re_["b3_median"], b_pr=re_["p_b3_neg"], b_ci=re_["b3_ci"],
        iv_med=iv["iv_interaction_median"], iv_ci=iv["iv_interaction_ci"],
        pa_med=pa["b3_median"], pa_ci=pa["b3_ci"],
    )
    sc["a_ci"] = [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    print(f"[A assigned] draws med={np.median(a):+.4f} Pr(<0)={(a<0).mean():.4f}"
          f"  canonical med={sc['a_med']:+.4f} Pr={sc['a_pr']:.4f}")
    print(f"[B realized] draws med={np.median(b):+.4f} Pr(<0)={(b<0).mean():.4f}"
          f"  canonical med={sc['b_med']:+.4f} Pr={sc['b_pr']:.4f} CrI={[round(x,3) for x in sc['b_ci']]}")
    print(f"[corrected ] IV {sc['iv_med']:+.3f} {[round(x,2) for x in sc['iv_ci']]}"
          f"   paragraph {sc['pa_med']:+.3f} {[round(x,2) for x in sc['pa_ci']]}")
    assert abs(np.median(a) - sc["a_med"]) < 0.05
    assert abs(np.median(b) - sc["b_med"]) < 0.05
    return a, b, sc


# --------------------------------------------------------------------------- #
#  PLOT
# --------------------------------------------------------------------------- #
def main():
    a, b, sc = load()
    xs = np.linspace(XMIN, XMAX, 600)
    ya = gaussian_kde(a)(xs)
    yb = gaussian_kde(b)(xs)
    ymax = max(ya.max(), yb.max())

    fig, ax = plt.subplots(figsize=(FIGW, FIGH))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(-BASE_DROP * ymax, TOP_HEAD * ymax)

    # ---- the two posteriors --------------------------------------------- #
    def band(xs, ys, ci, fill, line, edge):
        lo, hi = ci
        ax.fill_between(xs, ys, color=fill, alpha=0.45, zorder=2)
        m = (xs >= lo) & (xs <= hi)
        ax.fill_between(xs[m], ys[m], color=fill, alpha=1.0, zorder=3)
        ax.plot(xs, ys, color=line, lw=2.4, zorder=5)

    band(xs, ya, sc["a_ci"], NAVYFILL, NAVY, NAVY)
    band(xs, yb, sc["b_ci"], ORANGEFILL, ORANGE, ORANGE)
    ax.axvline(sc["a_med"], color=NAVY, ls=":", lw=1.3, zorder=5,
               ymin=0, ymax=(ya.max() + BASE_DROP * ymax) / ((TOP_HEAD + BASE_DROP) * ymax))
    ax.axvline(sc["b_med"], color=ORANGE, ls=":", lw=1.3, zorder=5,
               ymin=0, ymax=(yb.max() + BASE_DROP * ymax) / ((TOP_HEAD + BASE_DROP) * ymax))

    # null reference + baseline
    ax.axvline(0, color=NULLCLR, ls="--", lw=1.6, zorder=4)
    ax.axhline(0, color="#d6dbe3", lw=1.1, zorder=1)
    ax.text(0, 0.965, "  null (0)", transform=ax.get_xaxis_transform(),
            color=NULLCLR, fontsize=12, ha="left", va="center", style="italic")

    # ---- corrected estimates: basement lane (revert across zero) -------- #
    yiv = -0.055 * ymax
    ypa = -0.110 * ymax
    for med, ci, yy, lab in (
        (sc["iv_med"], sc["iv_ci"], yiv,
         f"IV-instrumented   {sc['iv_med']:+.2f}  [{sc['iv_ci'][0]:.2f}, {sc['iv_ci'][1]:.2f}]"),
        (sc["pa_med"], sc["pa_ci"], ypa,
         f"paragraph-segmented   {sc['pa_med']:+.2f}  [{sc['pa_ci'][0]:.2f}, {sc['pa_ci'][1]:.2f}]"),
    ):
        ax.errorbar(med, yy, xerr=[[med - ci[0]], [ci[1] - med]], fmt="o",
                    color=ORANGEDK, ecolor=ORANGEDK, elinewidth=2.0, capsize=4,
                    capthick=1.8, markersize=7, zorder=6)
        ax.text(ci[1] + 0.05, yy, lab, ha="left", va="center",
                fontsize=10, color=ORANGEDK)

    # arrow: realized posterior -> corrected (reverts negative)
    ax.annotate("", xy=(sc["pa_med"], 0.015 * ymax),
                xytext=(0.14, 0.62 * ymax),
                arrowprops=dict(arrowstyle="-|>", color=ORANGEDK, lw=1.6,
                                ls="--", connectionstyle="arc3,rad=0.32"),
                zorder=6)
    ax.text(-0.02, 0.40 * ymax, "corrected for\nendogeneity",
            ha="center", va="center", fontsize=11, color=ORANGEDK, style="italic")

    # ---- axes cosmetics -------------------------------------------------- #
    ax.set_xlabel(r"Interaction  $\beta_3$  (log-odds)", fontsize=17, color=INK)
    ax.set_ylabel("Posterior density", fontsize=17, color=INK)
    ax.set_yticks([])
    ax.tick_params(axis="x", colors="#5a6b82", labelsize=13)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, lw=0.8)

    # ---- per-series stat boxes (inline legend) -------------------------- #
    ax.text(0.030, 0.945,
            "Primary ITT — assigned length\n(exogenous · causal estimand)\n"
            f"median {sc['a_med']:.2f} · $\\Pr(\\beta_3{{<}}0)$ = {sc['a_pr']:.4f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=13, color=NAVY,
            bbox=dict(boxstyle="round,pad=0.55", facecolor=BG, edgecolor=NAVY, lw=1.0))
    ax.text(0.970, 0.945,
            "Secondary R7 — realized steps\n(endogenous · confounded)\n"
            f"median +{sc['b_med']:.2f} · $\\Pr(\\beta_3{{<}}0)$ = {sc['b_pr']:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=13, color=ORANGEDK,
            bbox=dict(boxstyle="round,pad=0.55", facecolor=BG, edgecolor=ORANGE, lw=1.0))

    # ---- layout: place axes, then fig-coord chrome ---------------------- #
    ax.set_position([AX_L, AX_B, AX_R - AX_L, AX_T - AX_B])

    fig.add_artist(FancyBboxPatch((FRAME[0], FRAME[1]), FRAME[2], FRAME[3],
                   boxstyle="round,pad=0,rounding_size=0.02",
                   transform=fig.transFigure, fill=False, edgecolor=NAVY,
                   linewidth=1.4, zorder=0, mutation_aspect=FIGW / FIGH))

    fig.text(0.5, 0.955, "Why the Primary Is the Assigned Length",
             ha="center", fontsize=26, color=NAVY)
    dvy = 0.915
    fig.add_artist(Line2D([0.380, 0.470], [dvy, dvy], color=ORANGE, lw=1.8,
                          transform=fig.transFigure))
    fig.add_artist(Line2D([0.530, 0.620], [dvy, dvy], color=ORANGE, lw=1.8,
                          transform=fig.transFigure))
    fig.add_artist(Line2D([0.5], [dvy], marker="o", color=ORANGE, markersize=9,
                          linestyle="none", transform=fig.transFigure))
    fig.text(0.5, 0.892,
             "The same interaction is decisively negative on the exogenous "
             "assigned length (the causal estimand)\nbut positive on the "
             "endogenous realized step count — and reverts negative once that "
             "endogeneity is corrected.",
             ha="center", va="top", fontsize=14, color=NAVY, style="italic")

    fig.text(0.5, 0.072,
             "The flip is endogeneity, not a benefit of more steps: instrumenting "
             "or paragraph-segmenting the realized count restores the negative sign.",
             ha="center", va="center", fontsize=12.5, color=INK, style="italic")

    for ext in ("png", "pdf", "svg"):
        out = HERE / f"fig_f7_5_reversal.{ext}"
        try:
            fig.savefig(out, facecolor=BG)
            print(f"[saved] {out}")
        except PermissionError:
            print(f"[skip]  {out} is locked (close it in your viewer); re-run to write it.")


if __name__ == "__main__":
    main()
