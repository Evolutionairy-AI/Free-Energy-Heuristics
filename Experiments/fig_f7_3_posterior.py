"""
fig_f7_3_posterior.py — Figure 7.3 (Posterior of the interaction) for the FEH paper.

The headline confirmatory picture: the entire posterior of the assigned-length
interaction beta_3 (eq. 6.1') sits left of zero, and the implied high-regime
accuracy drop clears the pre-registered 6-point gate.

NOTHING is hand-typed into the plot:
  - density CURVES are KDEs of the exported posterior draws
        (confirmatory_posterior_draws.json: primary_b3, primary_robust_pp_drop)
  - annotated SCALARS (median, Pr, drop median + CI) come from the canonical
        full-posterior fit (confirmatory_analysis_results.json -> primary_itt).
The script cross-checks the draws against the canonical scalars and PRINTS both;
the only legit difference is the drop CI (the figure draws are a 4000-sample
downsample; canonical CI [7.68, 25.55]). We shade/annotate the canonical CI.

Layout (corner inset, fixed): single beta_3 panel; the implied-drop density is a
BOXED inset parked in the far upper-right where the beta_3 curve is already near
zero, so the inset box never touches the curve; the null line is clipped below
the inset; the "clears the gate" line lives in a footer (not over the curve).
matplotlib owns every coordinate -> gate at exactly 6, CI edges exact.

Run from Experiments/:  python fig_f7_3_posterior.py
Outputs fig_f7_3_posterior.png (300 dpi) + .pdf (vector — use for the paper).
Edit the STYLE block to restyle; layout constants are grouped under LAYOUT.
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
RESULTS = HERE / "confirmatory_analysis_results.json"
GATE_PP = 6.0

# --------------------------------------------------------------------------- #
#  STYLE  — edit freely
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
#  LAYOUT constants (axes fraction unless noted)
# --------------------------------------------------------------------------- #
AX_L, AX_R, AX_T, AX_B = 0.085, 0.955, 0.795, 0.195   # main axes position
INSET_BOX = (0.685, 0.32, 0.300, 0.630)               # x0,y0,w,h of inset BORDER
INSET_AX  = (0.700, 0.420, 0.275, 0.440)              # x0,y0,w,h of inset plot
NULL_YMAX = 0.27                                        # clip null line below inset
FRAME = (0.028, 0.050, 0.944, 0.785)                  # fig-fraction x0,y0,w,h


# --------------------------------------------------------------------------- #
#  DATA
# --------------------------------------------------------------------------- #
def load():
    d = json.loads(DRAWS.read_text(encoding="utf-8"))
    b3 = np.asarray(d["primary_b3"], dtype=float)
    drop = np.asarray(d["primary_robust_pp_drop"], dtype=float)
    p = json.loads(RESULTS.read_text(encoding="utf-8"))["primary_itt"]
    sc = dict(b3_med=p["b3_median"], p_b3_neg=p["p_b3_neg"],
              drop_med=p["robust_pp_drop_median"], drop_ci=p["robust_pp_drop_ci"])
    print(f"[b3]   draws median={np.median(b3):+.4f}  Pr(<0)={(b3 < 0).mean():.4f}")
    print(f"[b3]   canonical median={sc['b3_med']:+.4f}  Pr(<0)={sc['p_b3_neg']:.4f}")
    print(f"[drop] draws median={np.median(drop):.3f}  canonical={sc['drop_med']:.3f}"
          f"  CI={[round(x,2) for x in sc['drop_ci']]}")
    assert abs(np.median(b3) - sc["b3_med"]) < 0.05
    assert abs(np.median(drop) - sc["drop_med"]) < 0.5
    sc["b3_ci"] = [float(np.percentile(b3, 2.5)), float(np.percentile(b3, 97.5))]
    return b3, drop, sc


def kde_curve(samples, pad=0.08, n=400, x_extra=None):
    lo, hi = samples.min(), samples.max()
    span = hi - lo
    x0, x1 = lo - pad * span, hi + pad * span
    if x_extra is not None:
        x0, x1 = min(x0, x_extra[0]), max(x1, x_extra[1])
    xs = np.linspace(x0, x1, n)
    return xs, gaussian_kde(samples)(xs)


# --------------------------------------------------------------------------- #
#  PLOT
# --------------------------------------------------------------------------- #
def main():
    b3, drop, sc = load()
    fig, ax = plt.subplots(figsize=(FIGW, FIGH))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # ---- main panel: posterior of beta_3 ---------------------------------- #
    xs, ys = kde_curve(b3, x_extra=(b3.min() - 0.05, 0.20))
    lo, hi = sc["b3_ci"]
    ax.fill_between(xs, ys, color=NAVYFILL, alpha=0.45, zorder=2)
    m = (xs >= lo) & (xs <= hi)
    ax.fill_between(xs[m], ys[m], color=NAVYFILL, alpha=1.0, zorder=3)
    ax.plot(xs, ys, color=NAVY, lw=2.4, zorder=5)

    ax.axvline(0, ymin=0, ymax=NULL_YMAX, color=NULLCLR, ls="--", lw=1.5, zorder=4)
    ax.axvline(sc["b3_med"], color=NAVY, ls=":", lw=1.4, zorder=5)
    ax.text(0, 0.205, "  null (0)", transform=ax.get_xaxis_transform(),
            color=NULLCLR, fontsize=12, ha="left", va="center", style="italic")

    ax.set_xlabel(r"Interaction  $\beta_3$  (log-odds)", fontsize=17, color=INK)
    ax.set_ylabel("Posterior density", fontsize=17, color=INK)
    ax.set_ylim(bottom=0)
    ax.tick_params(colors="#5a6b82", labelsize=13)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.yaxis.grid(True, color=GRID, lw=0.8)

    ax.text(0.035, 0.955,
            f"median $\\beta_3$ = {sc['b3_med']:.2f}\n"
            f"$\\Pr(\\beta_3 < 0 \\mid \\mathrm{{data}})$ = {sc['p_b3_neg']:.4f}\n"
            f"95% CrI [{lo:.2f}, {hi:.2f}]",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=14.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.55", facecolor=BG,
                      edgecolor=NAVY, linewidth=1.0))

    # ---- inset: implied accuracy drop (boxed, far upper-right) ------------- #
    axin = ax.inset_axes(list(INSET_AX))
    axin.set_facecolor(BG)
    xd, yd = kde_curve(drop, x_extra=(GATE_PP - 1, drop.max() + 1))
    dlo, dhi = sc["drop_ci"]
    axin.fill_between(xd, yd, color=ORANGEFILL, alpha=0.5, zorder=2)
    md = (xd >= dlo) & (xd <= dhi)
    axin.fill_between(xd[md], yd[md], color=ORANGEFILL, alpha=1.0, zorder=3)
    axin.plot(xd, yd, color=ORANGE, lw=2.0, zorder=5)
    axin.axvline(GATE_PP, color=ORANGE, ls="--", lw=1.7, zorder=4)
    axin.axvline(sc["drop_med"], color=ORANGEDK, ls=":", lw=1.2, zorder=5)

    axin.set_xlim(-1, 34)
    axin.set_xticks([0, 10, 20, 30])
    axin.set_title("Implied accuracy drop (pp)", fontsize=12.5, color=INK, pad=5)
    axin.set_yticks([])
    axin.tick_params(axis="x", colors="#5a6b82", labelsize=11)
    for sp in ("top", "right", "left"):
        axin.spines[sp].set_visible(False)
    axin.set_ylim(bottom=0)

    ymax = axin.get_ylim()[1]
    axin.text(GATE_PP - 0.8, ymax * 0.90, "6 pp gate", color=ORANGE, fontsize=10.5,
              ha="right", va="top", style="italic")
    axin.text(0.97, 0.85, f"median {sc['drop_med']:.1f} pp\n[{dlo:.1f}, {dhi:.1f}]",
              transform=axin.transAxes, ha="right", va="top",
              fontsize=10.5, color=INK)

    # ---- layout: position axes, then fig-coord chrome --------------------- #
    ax.set_position([AX_L, AX_B, AX_R - AX_L, AX_T - AX_B])
    axpos = ax.get_position()

    # rounded box around the inset (computed from the placed axes)
    bx = axpos.x0 + INSET_BOX[0] * axpos.width
    by = axpos.y0 + INSET_BOX[1] * axpos.height
    bw = INSET_BOX[2] * axpos.width
    bh = INSET_BOX[3] * axpos.height
    fig.add_artist(FancyBboxPatch((bx, by), bw, bh,
                   boxstyle="round,pad=0,rounding_size=0.018",
                   transform=fig.transFigure, fill=False, edgecolor=NAVY,
                   linewidth=1.1, zorder=6, mutation_aspect=FIGW / FIGH))

    # outer rounded frame (generous clearance below x-label + footer)
    fig.add_artist(FancyBboxPatch((FRAME[0], FRAME[1]), FRAME[2], FRAME[3],
                   boxstyle="round,pad=0,rounding_size=0.02",
                   transform=fig.transFigure, fill=False, edgecolor=NAVY,
                   linewidth=1.4, zorder=0, mutation_aspect=FIGW / FIGH))

    # title + orange divider (line · dot · line) + subtitle
    fig.text(0.5, 0.955, "Posterior of the Interaction",
             ha="center", fontsize=27, color=NAVY)
    dvy = 0.915
    fig.add_artist(Line2D([0.355, 0.475], [dvy, dvy], color=ORANGE, lw=1.8,
                          transform=fig.transFigure))
    fig.add_artist(Line2D([0.525, 0.645], [dvy, dvy], color=ORANGE, lw=1.8,
                          transform=fig.transFigure))
    fig.add_artist(Line2D([0.5], [dvy], marker="o", color=ORANGE, markersize=9,
                          linestyle="none", transform=fig.transFigure))
    fig.text(0.5, 0.890,
             "The entire posterior of the assigned-length interaction sits left "
             "of zero,\nand the implied high-regime accuracy drop clears the "
             "pre-registered six-point gate.",
             ha="center", va="top", fontsize=14.5, color=NAVY, style="italic")

    # footer line (its clean home — no curve underneath)
    fig.text(0.52, 0.092,
             "Entire 95% CrI of the implied drop clears the 6 pp gate.",
             ha="center", va="center", fontsize=13, color=INK, style="italic")

    for ext in ("png", "pdf"):
        out = HERE / f"fig_f7_3_posterior.{ext}"
        try:
            fig.savefig(out, facecolor=BG)
            print(f"[saved] {out}")
        except PermissionError:
            print(f"[skip]  {out} is locked (close it in your viewer); re-run to write it.")


if __name__ == "__main__":
    main()
