"""
fig_f5_1_design.py — Figure 5.1 (Design schematic) for the FEH paper.

Orients the reader to the factorial before the results: the shape and size of the
confirmatory run. Pure structure, no estimated quantities — an infographic of the
crossing  7 models x 45 items x 5 conditions x 5 replications = 7,875 responses.

NOTHING is hand-typed: every count is COMPUTED from confirmatory_responses.json
(total, per-model, per-condition, regime split, replications, and the item
category breakdown from frame-id prefixes). Only the descriptive labels
("Knightian", "open-weight", model display names) are fixed strings.

Style matches the locked house template (fig_f7_3_posterior.py).

Run from Experiments/:  python fig_f5_1_design.py
Outputs fig_f5_1_design.png (300 dpi) + .pdf + .svg (both vector).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
RESPONSES = HERE / "confirmatory_responses.json"

# --------------------------------------------------------------------------- #
#  STYLE  — matches the locked template
# --------------------------------------------------------------------------- #
BG       = "#FDFDFD"
NAVY     = "#0B3A78"
ORANGE   = "#E87900"
ORANGEDK = "#B86200"
INK      = "#0B3A78"
SUBINK   = "#3a5680"      # softer navy for card body text
CARDFILL = "#EEF3FB"      # very light navy
RESULTFILL = "#D9E5F5"    # the payoff card, a touch stronger
DIVIDER  = "#ececec"

FIGW, FIGH = 13.5, 7.2

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["EB Garamond", "Cormorant Garamond", "Garamond",
                   "Georgia", "DejaVu Serif"],
    "font.size": 13,
    "mathtext.fontset": "dejavuserif",
    "figure.dpi": 110,
    "savefig.dpi": 300,
})

FRAME = (0.028, 0.050, 0.944, 0.790)
AXPOS = (0.050, 0.060, 0.900, 0.740)     # frame interior (below subtitle)


# --------------------------------------------------------------------------- #
#  DATA  — every count computed from source
# --------------------------------------------------------------------------- #
def load():
    cells = json.loads(RESPONSES.read_text(encoding="utf-8"))
    total = len(cells)
    n_err = sum(1 for c in cells if c.get("error"))
    per_model = Counter(c["model"] for c in cells)
    per_cond = Counter(c["condition"] for c in cells)
    per_regime = Counter(c["regime"] for c in cells)
    per_rep = Counter(c["replication"] for c in cells)
    per_source = Counter(c["source"] for c in cells)

    frames = {}
    for c in cells:
        frames.setdefault(c["frame_id"], c["regime"])
    n_items = len(frames)
    hi_items = [f for f, rg in frames.items() if rg == "high"]
    lo_items = [f for f, rg in frames.items() if rg == "low"]
    pre = lambda fs: Counter(f.split("-")[0] for f in fs)
    hp, lp = pre(hi_items), pre(lo_items)

    n_models = len(per_model)
    cell_per_model = per_model.most_common(1)[0][1]
    # each model is single-source -> count models per source
    msrc = {}
    for c in cells:
        msrc[c["model"]] = c["source"]
    n_local = sum(1 for v in msrc.values() if v == "local")
    n_frontier = sum(1 for v in msrc.values() if v == "frontier")

    assert total == 7875 and n_err == 0
    assert len(set(per_model.values())) == 1                 # balanced models
    assert len(set(per_cond.values())) == 1                  # balanced conditions
    assert n_local + n_frontier == n_models

    d = dict(
        total=total, n_err=n_err, n_models=n_models, per_model=cell_per_model,
        n_local=n_local, n_frontier=n_frontier,
        n_items=n_items, n_hi=len(hi_items), n_lo=len(lo_items),
        k1=hp["K1"], k2=hp["K2"], k4=hp["K4"], arith=lp["R"], aleat=lp["A"],
        n_cond=len(per_cond), per_cond=per_cond.most_common(1)[0][1],
        n_rep=len(per_rep), per_rep=per_rep.most_common(1)[0][1],
        high=per_regime["high"], low=per_regime["low"],
    )
    print("[counts]", json.dumps(d))
    return d


# --------------------------------------------------------------------------- #
#  PLOT
# --------------------------------------------------------------------------- #
def card(ax, cx, n_big, label, lines, w=0.205, y_t=0.985, y_b=0.43):
    left, right = cx - w / 2, cx + w / 2
    ax.add_patch(FancyBboxPatch((left, y_b), w, y_t - y_b,
                 boxstyle="round,pad=0.004,rounding_size=0.018",
                 transform=ax.transData, facecolor=CARDFILL, edgecolor=NAVY,
                 linewidth=1.3, zorder=2, mutation_aspect=0.55))
    ax.text(cx, y_t - 0.085, n_big, ha="center", va="center",
            fontsize=42, color=NAVY, zorder=3)
    ax.text(cx, y_t - 0.175, label, ha="center", va="center",
            fontsize=15.5, color=NAVY, style="italic", zorder=3)
    ax.add_artist(Line2D([left + 0.02, right - 0.02], [y_t - 0.205] * 2,
                         color=ORANGE, lw=1.4, transform=ax.transData, zorder=3))
    ax.text(cx, y_t - 0.245, "\n".join(lines), ha="center", va="top",
            fontsize=10, color=SUBINK, linespacing=1.55, zorder=3)


def main():
    d = load()
    fig = plt.figure(figsize=(FIGW, FIGH))
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes(list(AXPOS))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # card x-geometry: 4 cards, 3 gaps, symmetric about 0.5
    w, area_l, area_r = 0.205, 0.030, 0.970
    gap = ((area_r - area_l) - 4 * w) / 3
    lefts = [area_l + i * (w + gap) for i in range(4)]
    centers = [l + w / 2 for l in lefts]
    gap_centers = [lefts[i] + w + gap / 2 for i in range(3)]

    cards = [
        (f"{d['n_models']}", "models", [
            f"{d['n_local']} local · open-weight",
            "Phi-3.5 · Mistral 7B",
            "Qwen2.5 7B / 14B / 32B",
            "",
            f"{d['n_frontier']} frontier · API",
            "Claude Sonnet 4.5 · GPT-4o"]),
        (f"{d['n_items']}", "items", [
            f"{d['n_hi']} high-regime · Knightian",
            f"K1 ×{d['k1']} · K2 ×{d['k2']} · K4 ×{d['k4']}",
            "",
            f"{d['n_lo']} low-regime · controls",
            f"arithmetic ×{d['arith']} · aleatory ×{d['aleat']}"]),
        (f"{d['n_cond']}", "conditions", [
            "C1  direct answer",
            "C2 · C3 · C4",
            "C5  unconstrained",
            "",
            "reasoning-length",
            "ladder"]),
        (f"{d['n_rep']}", "replications", [
            "independent draws",
            "of every",
            "model × item ×",
            "condition cell",
            "",
            "fixed seeds"]),
    ]
    for cx, (nb, lab, lines) in zip(centers, cards):
        card(ax, cx, nb, lab, lines)

    for gc in gap_centers:
        ax.text(gc, 0.705, "×", ha="center", va="center", fontsize=30,
                color=ORANGE, zorder=4)

    # result band (the payoff)
    bl, br, bb, bt = 0.275, 0.725, 0.115, 0.355
    ax.add_patch(FancyBboxPatch((bl, bb), br - bl, bt - bb,
                 boxstyle="round,pad=0.004,rounding_size=0.03",
                 transform=ax.transData, facecolor=RESULTFILL, edgecolor=NAVY,
                 linewidth=1.5, zorder=2, mutation_aspect=0.55))
    ax.text(0.5, 0.275, f"=  {d['total']:,}", ha="center", va="center",
            fontsize=46, color=ORANGE, zorder=3)
    ax.text(0.5, 0.165,
            f"responses collected · zero generation failures",
            ha="center", va="center", fontsize=15, color=NAVY,
            style="italic", zorder=3)

    # margins strip
    ax.text(0.5, 0.035,
            f"{d['per_model']:,} per model      ·      {d['per_cond']:,} per condition"
            f"      ·      {d['high']:,} high-regime      ·      {d['low']:,} low-regime controls",
            ha="center", va="center", fontsize=11.5, color=SUBINK, zorder=3)

    # ---- fig-coord chrome (title / divider / subtitle / frame) ----------- #
    fig.add_artist(FancyBboxPatch((FRAME[0], FRAME[1]), FRAME[2], FRAME[3],
                   boxstyle="round,pad=0,rounding_size=0.02",
                   transform=fig.transFigure, fill=False, edgecolor=NAVY,
                   linewidth=1.4, zorder=0, mutation_aspect=FIGW / FIGH))

    fig.text(0.5, 0.955, "Anatomy of the Run", ha="center",
             fontsize=27, color=NAVY)
    dvy = 0.915
    fig.add_artist(Line2D([0.420, 0.475], [dvy, dvy], color=ORANGE, lw=1.8,
                          transform=fig.transFigure))
    fig.add_artist(Line2D([0.525, 0.580], [dvy, dvy], color=ORANGE, lw=1.8,
                          transform=fig.transFigure))
    fig.add_artist(Line2D([0.5], [dvy], marker="o", color=ORANGE, markersize=9,
                          linestyle="none", transform=fig.transFigure))
    fig.text(0.5, 0.890,
             "A fully-crossed factorial — every model answers every item under "
             "every condition, five times.",
             ha="center", va="top", fontsize=15, color=NAVY, style="italic")

    for ext in ("png", "pdf", "svg"):
        out = HERE / f"fig_f5_1_design.{ext}"
        try:
            fig.savefig(out, facecolor=BG)
            print(f"[saved] {out}")
        except PermissionError:
            print(f"[skip]  {out} is locked (close it in your viewer); re-run to write it.")


if __name__ == "__main__":
    main()
