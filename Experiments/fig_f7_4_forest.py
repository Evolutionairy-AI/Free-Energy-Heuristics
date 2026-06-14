"""
fig_f7_4_forest.py — Figure 7.4 (Per-model interaction, R5) for the FEH paper.

The honesty figure. The pooled Pr(beta_3<0)=1.0000 invites a false picture of a
uniform law; refit separately per model (R5), the interaction is DECISIVE in the
three Qwen systems, DIRECTIONAL-BUT-INCONCLUSIVE in the two frontier models, and
REVERSED only in phi3.5 (Mistral inconclusive). A horizontal forest plot, one row
per model.

NOTHING is hand-typed into the plot. Every median, CrI and Pr(beta_3<0) is read
from confirmatory_robustness.json -> r5_per_model. Rows are ordered by the data
(most negative median at top). Colour category is derived from each CrI:
  CrI entirely < 0  -> NAVY   (degradation, theory-consistent)
  CrI spans 0       -> GREY   (inconclusive)
  CrI entirely > 0  -> ORANGE (reversed)

phi3.5 carries a dagger: its reversal partly reflects a collapsing control arm
(low-regime accuracy fell ~24 pp under reasoning), not a clean Knightian benefit.

Style matches fig_f7_3_posterior.py (the locked template).

Run from Experiments/:  python fig_f7_4_forest.py
Outputs fig_f7_4_forest.png (300 dpi) + .pdf (vector — use for the paper).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
ROBUST = HERE / "confirmatory_robustness.json"

# --------------------------------------------------------------------------- #
#  STYLE  — matches F7.3
# --------------------------------------------------------------------------- #
BG       = "#FDFDFD"
NAVY     = "#0B3A78"
ORANGE   = "#E87900"
GREY     = "#8c93a0"          # inconclusive (CrI crosses 0)
NULLCLR  = "#9a9a9a"
INK      = "#0B3A78"
GRID     = "#ececec"
ROWLINE  = "#eef0f3"

FIGW, FIGH = 12.0, 8.4

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
AX_L, AX_R, AX_T, AX_B = 0.200, 0.580, 0.745, 0.250   # main axes (fig fraction)
COL1_X = 1.05        # right column: beta_3 [CrI]   (axes-fraction x via yaxis-tf)
COL2_X = 1.58        # right column: Pr(beta_3<0)
FRAME  = (0.022, 0.045, 0.956, 0.800)                 # outer frame (fig fraction)

DISPLAY = {
    "qwen2.5:32b": "Qwen2.5 32B",
    "qwen2.5:7b": "Qwen2.5 7B",
    "qwen2.5:14b": "Qwen2.5 14B",
    "gpt-4o-2024-11-20": "GPT-4o",
    "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
    "mistral:7b-instruct": "Mistral 7B",
    "phi3.5": "Phi-3.5",
}
DAGGER = {"phi3.5"}


# --------------------------------------------------------------------------- #
#  DATA
# --------------------------------------------------------------------------- #
def load():
    r = json.loads(ROBUST.read_text(encoding="utf-8"))["r5_per_model"]
    rows = []
    for key, d in r.items():
        lo, hi = d["b3_ci"]
        cat = "neg" if hi < 0 else ("pos" if lo > 0 else "cross")
        rows.append(dict(key=key,
                         name=DISPLAY.get(key, key) + (" †" if key in DAGGER else ""),
                         med=d["b3_median"], lo=lo, hi=hi,
                         pr=d["p_b3_neg"], cat=cat))
    rows.sort(key=lambda x: x["med"])          # most negative first -> top
    for w in rows:
        print(f"  {w['key']:<28} b3={w['med']:+.2f}  CrI=[{w['lo']:+.2f},{w['hi']:+.2f}]"
              f"  Pr<0={w['pr']:.3f}  [{w['cat']}]")
    return rows


# --------------------------------------------------------------------------- #
#  PLOT
# --------------------------------------------------------------------------- #
def main():
    rows = load()
    n = len(rows)
    ys = np.arange(n)[::-1]                     # row 0 -> top
    cmap = {"neg": NAVY, "cross": GREY, "pos": ORANGE}

    fig, ax = plt.subplots(figsize=(FIGW, FIGH))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    xmin, xmax = -4.7, 2.2
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.7, n - 0.3)

    # faint per-row guide lines (name -> bar -> numbers)
    for y in ys:
        ax.hlines(y, xmin, xmax, color=ROWLINE, lw=8, zorder=0)

    # zero (null) reference
    ax.axvline(0, color=NULLCLR, ls="--", lw=1.5, zorder=2)

    # forest rows
    yt = ax.get_yaxis_transform()              # x: axes-fraction, y: data
    for w, y in zip(rows, ys):
        c = cmap[w["cat"]]
        ax.errorbar(w["med"], y, xerr=[[w["med"] - w["lo"]], [w["hi"] - w["med"]]],
                    fmt="o", color=c, ecolor=c, elinewidth=2.4, capsize=5,
                    capthick=2.0, markersize=9, markeredgecolor=c, zorder=5)
        ax.text(COL1_X, y, f"{w['med']:+.2f}  [{w['lo']:+.2f}, {w['hi']:+.2f}]",
                transform=yt, ha="left", va="center", fontsize=12.5,
                color=INK, clip_on=False)
        ax.text(COL2_X, y, f"{w['pr']:.3f}", transform=yt, ha="left",
                va="center", fontsize=12.5, color=c, clip_on=False)

    # column headers (just above the top row)
    yhdr = n - 0.42
    ax.text(COL1_X, yhdr, r"$\beta_3$  [95% CrI]", transform=yt, ha="left",
            va="center", fontsize=12.5, color="#5a6b82", style="italic")
    ax.text(COL2_X, yhdr, r"Pr($\beta_3$<0)", transform=yt, ha="left",
            va="center", fontsize=12.5, color="#5a6b82", style="italic")

    # y ticks = model names
    ax.set_yticks(ys)
    ax.set_yticklabels([w["name"] for w in rows], fontsize=14.5, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors="#5a6b82", labelsize=12.5)

    ax.set_xlabel(r"Per-model interaction  $\beta_3$  (log-odds)   "
                  r"$-$   negative = degradation under instructed reasoning",
                  fontsize=14.5, color=INK, labelpad=10)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, lw=0.8)

    # legend (horizontal strip below the axis — clean, no overlap with rows)
    handles = [
        Line2D([0], [0], marker="o", color=NAVY, lw=2.4, markersize=8,
               label="CrI below 0  —  degradation"),
        Line2D([0], [0], marker="o", color=GREY, lw=2.4, markersize=8,
               label="CrI crosses 0  —  inconclusive"),
        Line2D([0], [0], marker="o", color=ORANGE, lw=2.4, markersize=8,
               label="CrI above 0  —  reversed"),
    ]
    leg = ax.legend(handles=handles, loc="upper center",
                    bbox_to_anchor=(0.5, -0.215), ncol=3, fontsize=11.5,
                    frameon=False, handletextpad=0.5, columnspacing=2.4)
    for t in leg.get_texts():
        t.set_color(INK)

    # ---- layout: place axes, then fig-coord chrome ------------------------ #
    ax.set_position([AX_L, AX_B, AX_R - AX_L, AX_T - AX_B])

    fig.add_artist(FancyBboxPatch((FRAME[0], FRAME[1]), FRAME[2], FRAME[3],
                   boxstyle="round,pad=0,rounding_size=0.02",
                   transform=fig.transFigure, fill=False, edgecolor=NAVY,
                   linewidth=1.4, zorder=0, mutation_aspect=FIGW / FIGH))

    fig.text(0.5, 0.955, "Per-Model Interaction",
             ha="center", fontsize=27, color=NAVY)
    dvy = 0.915
    fig.add_artist(Line2D([0.380, 0.470], [dvy, dvy], color=ORANGE, lw=1.8,
                          transform=fig.transFigure))
    fig.add_artist(Line2D([0.530, 0.620], [dvy, dvy], color=ORANGE, lw=1.8,
                          transform=fig.transFigure))
    fig.add_artist(Line2D([0.5], [dvy], marker="o", color=ORANGE, markersize=9,
                          linestyle="none", transform=fig.transFigure))
    fig.text(0.5, 0.892,
             "Refit separately per model (R5), the assigned-length interaction is "
             "decisive in the three Qwen systems,\ndirectional-but-inconclusive in "
             "the two frontier models, and reversed only in Phi-3.5, with Mistral "
             "inconclusive.",
             ha="center", va="top", fontsize=14.5, color=NAVY, style="italic")

    # footer: dagger note + scale caveat
    fig.text(0.5, 0.092,
             "† Phi-3.5's reversal partly reflects a collapsing control arm "
             "(low-regime accuracy fell ~24 pp under reasoning), not a clean "
             "Knightian benefit.",
             ha="center", va="center", fontsize=11.5, color=INK, style="italic")
    fig.text(0.5, 0.062,
             r"Unpooled per-model $\beta_3$ run larger than the pooled "
             r"$\beta_3 = -0.69$ (§7.3), which holds one interaction across all "
             "seven systems.",
             ha="center", va="center", fontsize=11, color="#5a6b82", style="italic")

    for ext in ("png", "pdf", "svg"):
        out = HERE / f"fig_f7_4_forest.{ext}"
        try:
            fig.savefig(out, facecolor=BG)
            print(f"[saved] {out}")
        except PermissionError:
            print(f"[skip]  {out} is locked (close it in your viewer); re-run to write it.")


if __name__ == "__main__":
    main()
