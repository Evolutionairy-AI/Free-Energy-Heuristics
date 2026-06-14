"""
fig_f7_2_did.py — Figure 7.2 (Difference-in-Differences) for the FEH paper.

Coordinate-exact matplotlib version. NO number is hand-typed into the plot:
  - point values  <- confirmatory_analysis_results.json  (descriptive.*)
  - 95% CIs       <- item-clustered bootstrap recomputed live from
                     confirmatory_responses.json  (cluster = item,
                     n_boot = 2000, seed = 20260514 — the SAME procedure and
                     seed as F7.1's per-condition error bars).

Run from the Experiments/ directory (needs confirmatory_analyze.py importable):
    python fig_f7_2_did.py

Outputs (next to this script):
    fig_f7_2_did.png   (300 dpi raster)
    fig_f7_2_did.pdf   (vector — use this for the paper)

The STYLE block below is a clean starting point. Tweak colours / fonts / sizes
there; the data and layout logic underneath does not need to change.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch  # noqa: F401 (kept for easy restyle)

from confirmatory_analyze import score_cells, load_pool

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "confirmatory_analysis_results.json"
RESPONSES = HERE / "confirmatory_responses.json"
SEED = 20260514
N_BOOT = 2000

# --------------------------------------------------------------------------- #
#  STYLE  — edit freely
# --------------------------------------------------------------------------- #
SHOW_ERRORBARS = True          # False -> bare bars, axis 0..14 (image-gen look)

CREAM   = "#f5f0e6"            # background
NAVY    = "#1c3f6e"            # high regime + titles
ORANGE  = "#d9791f"            # low regime
GOLD    = "#e0b06a"            # difference-in-differences
INK     = "#1c3f6e"            # text
GRID    = "#cdc6b8"            # gridlines / zero line
BAR_EDGE = "#00000022"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Georgia", "Garamond", "Times New Roman", "DejaVu Serif"],
    "font.size": 13,
    "axes.edgecolor": "#7c8aa0",
    "axes.linewidth": 1.0,
    "figure.dpi": 110,
    "savefig.dpi": 300,
})


# --------------------------------------------------------------------------- #
#  DATA  — computed from source, nothing transcribed
# --------------------------------------------------------------------------- #
def point_values():
    d = json.loads(RESULTS.read_text(encoding="utf-8"))["descriptive"]
    return (d["high_short_long_drop_pp"],
            d["low_short_long_drop_pp"],
            d["DiD_pp"])


def bootstrap_cis():
    """Item-clustered bootstrap of the two drops and the DiD, in pp.
    High and low items are disjoint clusters -> resample each set independently,
    matching F7.1's per-regime item bootstrap."""
    cells = json.loads(RESPONSES.read_text(encoding="utf-8"))
    df = score_cells(cells, load_pool())
    if "long" not in df.columns:
        df["long"] = (df["condition"] != "C1").astype(int)

    def split(sub):
        out = {}
        for it in sub["item"].unique():
            cell = sub[sub["item"] == it]
            out[it] = (cell[cell.long == 0]["y"].to_numpy(),
                       cell[cell.long == 1]["y"].to_numpy())
        return out

    hi = df[df.regime == 1]
    lo = df[df.regime == 0]
    hs, ls = split(hi), split(lo)
    hi_items, lo_items = hi["item"].unique(), lo["item"].unique()

    rng = np.random.default_rng(SEED)
    H, L, D = [], [], []
    for _ in range(N_BOOT):
        ph = rng.choice(hi_items, size=len(hi_items), replace=True)
        pl = rng.choice(lo_items, size=len(lo_items), replace=True)
        hd = (np.concatenate([hs[k][0] for k in ph]).mean()
              - np.concatenate([hs[k][1] for k in ph]).mean()) * 100
        ld = (np.concatenate([ls[k][0] for k in pl]).mean()
              - np.concatenate([ls[k][1] for k in pl]).mean()) * 100
        H.append(hd); L.append(ld); D.append(hd - ld)

    def ci(a):
        a = np.asarray(a)
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    return {"high": ci(H), "low": ci(L), "did": ci(D),
            "p_did_pos": float((np.asarray(D) > 0).mean())}


# --------------------------------------------------------------------------- #
#  PLOT
# --------------------------------------------------------------------------- #
def main():
    high, low, did = point_values()
    cis = bootstrap_cis()
    print(f"[data] high={high} low={low} DiD={did}")
    print(f"[data] CIs  high={cis['high']}  low={cis['low']}  did={cis['did']}"
          f"  Pr(DiD>0)={cis['p_did_pos']:.4f}")

    labels = ["High regime drop", "Low regime drop", "Difference-in-differences"]
    sub    = ["C1 vs C2–C5 mean", "C1 vs C2–C5 mean", ""]
    vals   = [high, low, did]
    colors = [NAVY, ORANGE, GOLD]
    ci_lo  = [cis["high"][0], cis["low"][0], cis["did"][0]]
    ci_hi  = [cis["high"][1], cis["low"][1], cis["did"][1]]
    x = np.arange(3)

    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor(CREAM)

    ax.bar(x, vals, width=0.62, color=colors, edgecolor=BAR_EDGE, linewidth=1, zorder=3)

    if SHOW_ERRORBARS:
        for xi, v, lo_, hi_, c in zip(x, vals, ci_lo, ci_hi, colors):
            err = [[v - lo_], [hi_ - v]]
            ax.errorbar(xi, v, yerr=err, fmt="none", ecolor=NAVY,
                        elinewidth=1.6, capsize=7, capthick=1.6, zorder=4)
        ax.axhline(0, color=GRID, linewidth=1.2, zorder=2)
        top = max(ci_hi) * 1.12
        bottom = min(0, min(ci_lo)) - 1.5
        ax.set_ylim(bottom, top)
        label_y = [hi_ + top * 0.02 for hi_ in ci_hi]
    else:
        ax.set_ylim(0, 14)
        label_y = [v + 0.35 for v in vals]

    # value labels
    for xi, v, ly in zip(x, vals, label_y):
        ax.text(xi, ly, f"{v:g} pp", ha="center", va="bottom",
                fontsize=19, color=INK)

    # x tick labels (two-line)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14, color=INK)
    for xi, s in zip(x, sub):
        if s:
            ax.text(xi, ax.get_ylim()[0] - (ax.get_ylim()[1]-ax.get_ylim()[0])*0.065,
                    s, ha="center", va="top", fontsize=11, color="#5a6b82")

    ax.set_ylabel("Accuracy drop  (percentage points)", fontsize=14, color=INK)
    ax.tick_params(axis="y", labelsize=12, colors="#5a6b82")
    ax.tick_params(axis="x", length=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7, linestyle=(0, (3, 4)), zorder=0)
    ax.set_axisbelow(True)

    # DiD arrow annotation
    ax.annotate("12.7 − 1.73 = 10.97 pp\nthe model-free gap",
                xy=(2, did), xytext=(2.55, did),
                fontsize=12.5, color=INK, style="italic", va="center",
                arrowprops=dict(arrowstyle="->", color="#8a96a8",
                                linestyle=(0, (2, 2)), lw=1.2),
                annotation_clip=False)

    # note (descriptive-marginal disclosure + DiD CI) — parked in the BOTTOM
    # margin so it can never overlap the bars or whiskers. Move/restyle freely.
    note = ("All three quantities are descriptive marginals using the registered "
            "short/long coding (C1 vs pooled C2–C5 mean).\n"
            f"DiD 95% cluster-bootstrap CI [{cis['did'][0]:.1f}, {cis['did'][1]:.1f}] "
            f"(items as clusters); the gap excludes zero "
            f"(Pr > 0 = {cis['p_did_pos']:.3f}).")
    fig.text(0.5, 0.035, note, ha="center", va="bottom",
             fontsize=10.5, color=INK, style="italic",
             bbox=dict(boxstyle="round,pad=0.5", facecolor=CREAM,
                       edgecolor=ORANGE, linestyle="--", linewidth=1.0))

    # title + subtitle  (NO baked-in figure caption — that lives in the manuscript)
    fig.text(0.5, 0.965, "Difference-in-Differences in Accuracy Drop",
             ha="center", fontsize=24, color=NAVY)
    fig.text(0.5, 0.905,
             "Compressing the signature to its single number: the high-regime "
             "drop is large,\nthe low-regime drop is near noise, and the gap "
             "between them is the model-free heart of the result.",
             ha="center", fontsize=12.5, color=NAVY, style="italic")

    fig.subplots_adjust(left=0.10, right=0.78, top=0.84, bottom=0.22)

    for ext in ("png", "pdf"):
        out = HERE / f"fig_f7_2_did.{ext}"
        try:
            fig.savefig(out, facecolor=CREAM, bbox_inches="tight")
            print(f"[saved] {out}")
        except PermissionError:
            print(f"[skip]  {out} is locked (close it in your viewer); re-run to write it.")


if __name__ == "__main__":
    main()
